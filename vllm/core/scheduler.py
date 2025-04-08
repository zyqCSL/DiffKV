import enum
import time
import os
from typing import Dict, Iterable, List, Optional, Tuple, Union

from vllm.config import ModelConfig, CacheConfig, SchedulerConfig, ParallelConfig
from vllm.core.block_manager import AllocStatus
from vllm.core.orchestrator import Orchestrator
from vllm.core.policy import PolicyFactory
from vllm.logger import init_logger
from vllm.sequence import (Sequence, SequenceData, SequenceGroup,
                           SequenceGroupMetadata, SequenceStatus)

logger = init_logger(__name__)


class PreemptionMode(enum.Enum):
    """Preemption modes.

    1. Swapping: Swap out the blocks of the preempted sequences to CPU memory
    and swap them back in when the sequences are resumed.
    2. Recomputation: Discard the blocks of the preempted sequences and
    recompute them when the sequences are resumed, treating the sequences as
    new prompts.
    """
    SWAP = enum.auto()
    RECOMPUTE = enum.auto()


class SchedulerOutputs:

    def __init__(
        self,
        scheduled_seq_groups: List[SequenceGroup],
        prompt_run: bool,
        num_batched_tokens: int,
        mem_pending: bool,
        ignored_seq_groups: List[SequenceGroup],
    ) -> None:
        self.scheduled_seq_groups = scheduled_seq_groups
        self.prompt_run = prompt_run
        self.num_batched_tokens = num_batched_tokens
        self.mem_pending = mem_pending
        # Swap in and swap out should never happen at the same time.
        # assert not (blocks_to_swap_in and blocks_to_swap_out)
        self.ignored_seq_groups = ignored_seq_groups

    def is_empty(self) -> bool:
        # NOTE: We do not consider the ignored sequence groups.
        return (not self.scheduled_seq_groups and not self.mem_pending)


class MemoryManager:
    ''' Memory space manager on the scheduler side
        Decide whether a new sequence can be allocated and
        wether a token can be appended to a running sequence
    '''

    def __init__(
        self,
        orchestrator: Orchestrator,
        num_kv_heads: int,
        num_layers: CacheConfig,
        quant_block_size: Dict[Tuple[int, int], int],
        num_gpu_blocks: List[int],
        num_cpu_blocks: List[int],
        watermark: float = 0.01,
        sliding_window: Optional[int] = None,
        max_kv_slots: Optional[int] = None,
    ) -> None:
        # cluster orchestraion
        self.orchestrator = orchestrator
        # model params
        self.num_kv_heads = num_kv_heads    # per worker/device
        self.num_layers = num_layers        # per worker/device
        # each heads has its own kv cache
        self.num_kv_caches = self.num_kv_heads * self.num_layers

        self.sliding_window = sliding_window    # depreciated by max_kv_slots
        self.max_kv_slots = max_kv_slots
        # if max_kv_slots is not None:
        #     assert max_kv_slots % block_size == 0, (max_kv_slots, block_size)
        #     self.max_kv_blocks = max_kv_slots // block_size

        # memory related params
        self.quant_to_block_size = quant_block_size
        self.watermark = watermark
        assert watermark >= 0.0

        # total mem space on each worker
        self.total_gpu_blocks = num_gpu_blocks.copy()
        self.total_cpu_blocks = num_cpu_blocks.copy()

        # watermark blocks for each worker
        self.watermark_blocks = [int(watermark * x) for x in self.total_gpu_blocks]

        # free mem space on each worker
        self.free_gpu_blocks = num_gpu_blocks.copy()
        self.free_cpu_blocks = num_cpu_blocks.copy()

        # tmp stats for accurately estimating available memory
        # estimated number of required blocks
        self._estimated_num_blocks = 0
        # estimated free
        self._estimated_free_blocks = 0

    def reset_estimated_free_blocks(self):
        self._estimated_free_blocks = min(self.free_gpu_blocks)

    def update_estimated_free_blocks(self, delta: int):
        self._estimated_free_blocks += delta

    def update_free_gpu_blocks(self, gpu_blocks: List[int]) -> None:
        ''' Called in LLMEngine.step(), after the forward pass completes '''
        assert len(self.free_gpu_blocks) == len(gpu_blocks)
        self.free_gpu_blocks = gpu_blocks
        self._estimated_num_blocks = 0

    def update_free_cpu_blocks(self, cpu_blocks: List[int]) -> None:
        assert len(self.free_cpu_blocks) == len(cpu_blocks)
        self.free_cpu_blocks = cpu_blocks

    def revert_allocate_estimate(self) -> None:
        assert self._estimated_num_blocks > 0
        for i in range(len(self.free_gpu_blocks)):
            self.free_gpu_blocks[i] += self._estimated_num_blocks
        self._estimated_num_blocks = 0

    def revert_append_estimate(self) -> None:
        for i in range(len(self.free_gpu_blocks)):
            self.free_gpu_blocks[i] += self.num_kv_caches

    def total_num_free_gpu_blocks(self) -> int:
        return sum(self.free_gpu_blocks)

    def total_num_free_cpu_blocks(self) -> int:
        return sum(self.free_cpu_blocks)

    def min_num_free_gpu_blocks(self) -> int:
        return min(self.free_gpu_blocks)

    def get_free_gpu_blocks(self) -> List[int]:
        return self.free_gpu_blocks.copy()
    # can_swap_in/out is implemented on the worker side
    # in BlockSpaceManager

    def can_allocate(
        self,
        seq_group: SequenceGroup,
    ) -> AllocStatus:
        ''' Args
        num_allocated_blocks: number of blocks already allocated for scheduled seq groups
        Returns: [alloc decision, accumulated number of blocks]
        '''
        # reset estimation
        self._estimated_num_blocks = 0

        # FIXME(woosuk): Here we assume that all sequences in the group share
        # the same prompt. This may not be true for preempted sequences.
        seq = seq_group.get_seqs(status=SequenceStatus.WAITING)[0]
        quant_config = seq_group.quant_configs[:2]
        assert quant_config in self.quant_to_block_size, f'quant {quant_config} not supported'
        block_size = self.quant_to_block_size[quant_config]
        num_prompt_tokens = seq.data.get_len()
        # Each head has its own list of blocks
        # NOTE: we need 1 additional block for low precision
        if self.max_kv_slots is None:
            num_prompt_blocks = _divide_round_up(num_prompt_tokens, block_size) + 1
        else:
            num_prompt_blocks = _divide_round_up(
                min(num_prompt_tokens, self.max_kv_slots), block_size) + 1

        num_required_blocks = num_prompt_blocks * self.num_kv_caches
        # print('estimated num_required_blocks = ', num_required_blocks)
        # Use watermark to avoid frequent cache eviction.
        never_flags = [self.total_gpu_blocks[x] - num_required_blocks < self.watermark_blocks[x]
                       for x in range(len(self.total_gpu_blocks))]
        if any(never_flags):
            return AllocStatus.NEVER
        else:
            # NOTE: memory space is updated here for estimation (imprecise due to KV compression)
            # it be updated with ground truth in llm_engine.step after model execution
            ok_flags = [self.free_gpu_blocks[x] - num_required_blocks >= self.watermark_blocks[x]
                        for x in range(len(self.free_gpu_blocks))]
            if all(ok_flags):
                self._estimated_num_blocks = num_required_blocks
                for x in range(len(self.total_gpu_blocks)):
                    self.free_gpu_blocks[x] -= num_required_blocks
                return AllocStatus.OK
            else:
                return AllocStatus.LATER

    def can_append_slot(
        self,
        seq_group: SequenceGroup,
    ) -> bool:
        # NOTE: Simple heuristic (depreciated): If there is at least one free block
        # for each sequence, we can append (worst case for sparsity).
        # NOTE: the heuristic can be too pessimistic.
        # For llama2 70b w. 80 layers & 64 heads running on 8 gpus with block_size = 16
        # we waste 8192 * 80 * 64 / 8 * 16 * 2 * 2 / 1024 / 1024 = 320 mb per sequence per device
        # NOTE: for shared mem the heuristics can be true. When pruning the kv cache
        # a different shared block (wrt different tokens) in prefix might be modified
        # thus requiring a new block to be allocated
        # TODO: We need to tune the block size for larger models
        # NOTE: here free_gpu_blocks are update for estimation (imprecise)
        # it is updated w. ground truth in llm_engine.step after model execution
        num_seqs = seq_group.num_seqs(status=SequenceStatus.RUNNING)
        # TODO: in the worse case we need 2 blocks for each head, but that is overly pessimistic and rare
        num_required_blocks = num_seqs * self.num_kv_caches
        # NOTE: only when the sequence just finished prompt processing, it is possible that each head might
        # need 2 blocks (1 for each precision). During decode, each time only one token is appended (at most),
        # so we only need 1 block for each head
        if not seq_group.is_decode:
            num_required_blocks *= 2
            seq_group.is_decode = True
        if self._estimated_free_blocks >= num_required_blocks:
            self._estimated_free_blocks -= num_required_blocks
            return True
        else:
            return False

    def can_swap_in(self, seq_group: SequenceGroup) -> bool:
        seq_ids = seq_group.get_seqs(status=SequenceStatus.SWAPPED)
        return self.orchestrator.run_workers_bool_all(
            "can_swap_in_seqs",
            seq_ids=seq_ids)

    def can_swap_out(self, seq_ids: List[int]) -> bool:
        return self.orchestrator.run_workers_bool_all(
            "can_swap_out_seqs",
            seq_ids=seq_ids,
        )


class Scheduler:

    def __init__(
        self,
        model_config: ModelConfig,
        scheduler_config: SchedulerConfig,
        cache_config: CacheConfig,
        parallel_config: ParallelConfig,
        orchestrator: Orchestrator,
    ) -> None:
        self.model_config = model_config
        self.scheduler_config = scheduler_config
        self.cache_config = cache_config
        self.parallel_config = parallel_config
        self.orchestrator = orchestrator

        self.prompt_limit = min(self.scheduler_config.max_model_len,
                                self.scheduler_config.max_num_batched_tokens)

        # Instantiate the scheduling policy.
        self.policy = PolicyFactory.get_policy(policy_name="fcfs")
        # Create the memory manager.
        # NOTE: memory_manager only decides if a sequence can be allocated
        # allocation is performed by per-worker block_manager
        self.memory_manager = MemoryManager(
            orchestrator=self.orchestrator,
            num_kv_heads=self.model_config.get_num_kv_heads(self.parallel_config),
            num_layers=self.model_config.get_num_layers(self.parallel_config),
            quant_block_size=self.cache_config.quantized_block_num_tokens,
            num_gpu_blocks=self.cache_config.num_gpu_blocks,
            num_cpu_blocks=self.cache_config.num_cpu_blocks,
            sliding_window=self.cache_config.sliding_window,
            max_kv_slots=self.cache_config.max_kv_slots,
        )

        # TODO(zhuohan): Use deque instead of list for better performance.
        # Sequence groups in the WAITING state.
        self.waiting: List[SequenceGroup] = []
        # Sequence groups in the RUNNING state.
        self.running: List[SequenceGroup] = []
        # Sequence groups in the SWAPPED state.
        self.swapped: List[SequenceGroup] = []

        # runtime stats
        self.num_finished_seqs = 0
        # prompt + generated
        self.num_processed_tokens = 0
        # number of kv blocks (per head) required with full FP16 KV cache
        # 2 for key and value cache
        self.baseline_block_size = cache_config.block_size // (2 * self.model_config.get_head_size())
        print(f'FP16 baseline block size = {self.baseline_block_size}')
        self.baseline_num_blocks = 0

    def add_seq_group(self, seq_group: SequenceGroup) -> None:
        # Add sequence groups to the waiting queue.
        self.waiting.append(seq_group)

    def abort_seq_group(self, request_id: Union[str, Iterable[str]]) -> None:
        if isinstance(request_id, str):
            request_id = (request_id, )
        request_ids = set(request_id)
        for state_queue in [self.waiting, self.running, self.swapped]:
            # We need to reverse the list as we are removing elements
            # from it as we iterate over it. If we don't do it,
            # indices will get messed up and we will skip over elements.
            for seq_group in reversed(state_queue):
                if seq_group.request_id in request_ids:
                    # Remove the sequence group from the state queue.
                    state_queue.remove(seq_group)
                    for seq in seq_group.get_seqs():
                        if seq.is_finished():
                            continue
                        seq.status = SequenceStatus.FINISHED_ABORTED
                        self.free_seq(seq)
                    request_ids.remove(seq_group.request_id)
                    if not request_ids:
                        return

    def has_unfinished_seqs(self) -> bool:
        return self.waiting or self.running or self.swapped

    def get_num_unfinished_seq_groups(self) -> int:
        return len(self.waiting) + len(self.running) + len(self.swapped)

    def _schedule(self) -> SchedulerOutputs:
        # NOTE: Blocks that need to be swapped or copied before model execution
        # are now kept by each worker's block_manager
        # blocks_to_swap_in: Dict[int, int] = {}
        # blocks_to_swap_out: Dict[int, int] = {}
        # blocks_to_copy: Dict[int, List[int]] = {}

        # Fix the current time.
        now = time.monotonic()
        mem_pending = False

        # Join waiting sequences if possible.
        if not self.swapped:
            ignored_seq_groups: List[SequenceGroup] = []
            scheduled: List[SequenceGroup] = []
            # The total number of sequences on the fly, including the
            # requests in the generation phase.
            num_curr_seqs = sum(seq_group.get_max_num_running_seqs()
                                for seq_group in self.running)
            seq_lens: List[int] = []

            # Optimization: We do not sort the waiting queue since the preempted
            # sequence groups are added to the front and the new sequence groups
            # are added to the back.
            while self.waiting:
                seq_group = self.waiting[0]

                waiting_seqs = seq_group.get_seqs(
                    status=SequenceStatus.WAITING)
                assert len(waiting_seqs) == 1, (
                    "Waiting sequence group should have only one prompt "
                    "sequence.")
                num_prompt_tokens = waiting_seqs[0].get_len()
                if num_prompt_tokens > self.prompt_limit:
                    logger.warning(
                        f"Input prompt ({num_prompt_tokens} tokens) is too long"
                        f" and exceeds limit of {self.prompt_limit}")
                    for seq in waiting_seqs:
                        seq.status = SequenceStatus.FINISHED_IGNORED
                    ignored_seq_groups.append(seq_group)
                    self.waiting.pop(0)
                    continue

                # If the sequence group cannot be allocated, stop.
                can_allocate = self.memory_manager.can_allocate(seq_group)
                if can_allocate == AllocStatus.LATER:
                    break
                elif can_allocate == AllocStatus.NEVER:
                    logger.warning(
                        f"Input prompt ({num_prompt_tokens} tokens) is too long"
                        f" and exceeds the capacity of memory_manager")
                    for seq in waiting_seqs:
                        seq.status = SequenceStatus.FINISHED_IGNORED
                    ignored_seq_groups.append(seq_group)
                    self.waiting.pop(0)
                    continue

                # If the number of batched tokens exceeds the limit, stop.
                new_seq_lens = seq_lens + [num_prompt_tokens]
                num_batched_tokens = len(new_seq_lens) * max(new_seq_lens)
                if (num_batched_tokens >
                        self.scheduler_config.max_num_batched_tokens):
                    self.memory_manager.revert_allocate_estimate()
                    break

                # The total number of sequences in the RUNNING state should not
                # exceed the maximum number of sequences.
                num_new_seqs = seq_group.get_max_num_running_seqs()
                if (num_curr_seqs + num_new_seqs >
                        self.scheduler_config.max_num_seqs):
                    self.memory_manager.revert_allocate_estimate()
                    break

                num_paddings = num_batched_tokens - sum(new_seq_lens)
                if num_paddings > self.scheduler_config.max_paddings:
                    self.memory_manager.revert_allocate_estimate()
                    break
                seq_lens = new_seq_lens

                seq_group = self.waiting.pop(0)
                self.running.append(seq_group)
                num_curr_seqs += num_new_seqs
                scheduled.append(seq_group)

            if scheduled:
                # allocate all prompts in one pass to avoid memory management overheads
                self._batch_allocate(scheduled)
            if scheduled or ignored_seq_groups:
                scheduler_outputs = SchedulerOutputs(
                    scheduled_seq_groups=scheduled,
                    prompt_run=True,
                    num_batched_tokens=len(seq_lens) *
                    max(seq_lens) if seq_lens else 0,
                    mem_pending=mem_pending,
                    ignored_seq_groups=ignored_seq_groups,
                )
                return scheduler_outputs

        # NOTE(woosuk): Preemption happens only when there is no available slot
        # to keep all the sequence groups in the RUNNING state.
        # In this case, the policy is responsible for deciding which sequence
        # groups to preempt.
        self.running = self.policy.sort_by_priority(now, self.running)

        # Reserve new token slots for the running sequence groups.
        running: List[SequenceGroup] = []
        preempted: List[SequenceGroup] = []
        running_seq_ids: List[int] = []
        self.memory_manager.reset_estimated_free_blocks()
        sched_quant_config = None
        while self.running:
            seq_group = self.running.pop(0)
            while not self.memory_manager.can_append_slot(seq_group):
                if self.running:
                    # Preempt the lowest-priority sequence groups.
                    victim_seq_group = self.running.pop(-1)
                    mem_pending = self._preempt(victim_seq_group) or mem_pending
                    preempted.append(victim_seq_group)
                else:
                    # No other sequence groups can be preempted.
                    # Preempt the current sequence group.
                    mem_pending = self._preempt(seq_group) or mem_pending
                    preempted.append(seq_group)
                    break
            else:
                # Append new slots to the sequence group.
                running_seq_ids.extend(seq_group.get_seq_ids(status=SequenceStatus.RUNNING))
                running.append(seq_group)
                if not sched_quant_config:
                    sched_quant_config = seq_group.quant_configs
        self._append_slot(
            seq_ids=running_seq_ids,
            kbits_high=sched_quant_config[0],
            vbits_high=sched_quant_config[1],
            kbits_low=sched_quant_config[2],
            vbits_low=sched_quant_config[3],
        )
        self.running = running

        # Swap in the sequence groups in the SWAPPED state if possible.
        self.swapped = self.policy.sort_by_priority(now, self.swapped)
        if not preempted:
            num_curr_seqs = sum(seq_group.get_max_num_running_seqs()
                                for seq_group in self.running)

            while self.swapped:
                seq_group = self.swapped[0]

                # If the sequence group cannot be swapped in, stop.
                if not self.memory_manager.can_swap_in(seq_group):
                    break

                # The total number of sequences in the RUNNING state should not
                # exceed the maximum number of sequences.
                num_new_seqs = seq_group.get_max_num_running_seqs()
                if (num_curr_seqs + num_new_seqs >
                        self.scheduler_config.max_num_seqs):
                    break
                mem_pending = True
                seq_group = self.swapped.pop(0)
                self._swap_in(seq_group)
                self._append_slot(seq_group)
                num_curr_seqs += num_new_seqs
                self.running.append(seq_group)

        # Each sequence in the generation phase only takes one token slot.
        # Therefore, the number of batched tokens is equal to the number of
        # sequences in the RUNNING state.
        num_batched_tokens = sum(
            seq_group.num_seqs(status=SequenceStatus.RUNNING)
            for seq_group in self.running)

        scheduler_outputs = SchedulerOutputs(
            scheduled_seq_groups=self.running,
            prompt_run=False,
            num_batched_tokens=num_batched_tokens,
            mem_pending=mem_pending,
            ignored_seq_groups=[],
        )
        return scheduler_outputs

    def schedule(self) -> Tuple[List[SequenceGroupMetadata], SchedulerOutputs]:
        # Schedule sequence groups.
        # This function call changes the internal states of the scheduler
        # such as self.running, self.swapped, and self.waiting.
        scheduler_outputs = self._schedule()

        # Create input data structures.
        seq_group_metadata_list: List[SequenceGroupMetadata] = []
        for seq_group in scheduler_outputs.scheduled_seq_groups:
            seq_data: Dict[int, SequenceData] = {}
            # NOTE: we omit block_table & kv_lens here as it should be per-device
            for seq in seq_group.get_seqs(status=SequenceStatus.RUNNING):
                seq_id = seq.seq_id
                seq_data[seq_id] = seq.data

            seq_group_metadata = SequenceGroupMetadata(
                request_id=seq_group.request_id,
                is_prompt=scheduler_outputs.prompt_run,
                seq_data=seq_data,
                sampling_params=seq_group.sampling_params,
                num_bits_k_high=seq_group.quant_configs[0],
                num_bits_v_high=seq_group.quant_configs[1],
                num_bits_k_low=seq_group.quant_configs[2],
                num_bits_v_low=seq_group.quant_configs[3],
            )
            seq_group_metadata_list.append(seq_group_metadata)
        return seq_group_metadata_list, scheduler_outputs

    # TODO: the following functions all need revisiting
    # actual mem ops should be performed inside each worker
    def fork_seq(self, parent_seq: Sequence, child_seq: Sequence) -> None:
        self.orchestrator.run_workers(
            "fork_seq",
            parent_seq_id=parent_seq.seq_id,
            child_seq_id=child_seq.seq_id,
        )

    def free_seq(self, seq: Sequence) -> None:
        if seq.is_finished():
            self.num_finished_seqs += 1
            self.num_processed_tokens += seq.data.get_len()
            self.baseline_num_blocks += \
                (seq.data.get_len() + self.baseline_block_size - 1) // self.baseline_block_size

        self.memory_manager.update_free_gpu_blocks(
            self.orchestrator.run_workers(
                "free_seq",
                seq_id=seq.seq_id,
                is_finished=seq.is_finished(),
                get_all_outputs=True))

    def free_finished_seq_groups(self) -> None:
        self.running = [
            seq_group for seq_group in self.running
            if not seq_group.is_finished()
        ]

    def log_stats(self, log_path: str) -> None:
        assert os.path.isdir(log_path)
        self.orchestrator.run_workers(
            "log_stats",
            log_path=log_path,
        )

    def _allocate(
        self,
        seq_group: SequenceGroup,
    ) -> None:
        seq = seq_group.get_seqs(status=SequenceStatus.WAITING)[0]
        num_prompt_tokens = seq.data.get_len()
        seq_ids = [seq.seq_id for seq in seq_group.get_seqs(status=SequenceStatus.WAITING)]
        # quant_config = [list(seq_group.quant_configs[0]) + list(seq_group.quant_configs[1])]
        compress_config = [seq_group.compress_configs]

        self.orchestrator.run_workers(
            "allocate_seqs",
            seq_ids=seq_ids,
            num_prompt_tokens=num_prompt_tokens,
            kbits_high=seq_group.quant_configs[0],
            vbits_high=seq_group.quant_configs[1],
            kbits_low=seq_group.quant_configs[2],
            vbits_low=seq_group.quant_configs[3],
            compress_config=compress_config,
        )
        for seq in seq_group.get_seqs(status=SequenceStatus.WAITING):
            seq.status = SequenceStatus.RUNNING

    def _batch_allocate(self, scheduled_seq_groups: List[SequenceGroup]) -> None:
        # allocate all scheduled sequence groups in one pass
        batch_seq_ids: List[List[int]] = []
        batch_num_prompt_tokens: List[int] = []
        # batch_quant_configs: List[List[int]] = []
        batch_compress_configs: List[List[int]] = []

        for seq_group in scheduled_seq_groups:
            waiting_seqs = seq_group.get_seqs(status=SequenceStatus.WAITING)
            seq = waiting_seqs[0]
            batch_num_prompt_tokens.append(seq.data.get_len())
            # batch_quant_configs.append(
            #     list(seq_group.quant_configs[0]) + list(seq_group.quant_configs[1])
            # )
            batch_compress_configs.append(seq_group.compress_configs)

            seq_ids = []
            for seq in waiting_seqs:
                seq_ids.append(seq.seq_id)
                seq.status = SequenceStatus.RUNNING
            batch_seq_ids.append(seq_ids)

        seq_group = scheduled_seq_groups[0]
        self.orchestrator.run_workers(
            "allocate_batch_seqs",
            batch_seq_ids=batch_seq_ids,
            batch_num_prompt_tokens=batch_num_prompt_tokens,
            kbits_high=seq_group.quant_configs[0],
            vbits_high=seq_group.quant_configs[1],
            kbits_low=seq_group.quant_configs[2],
            vbits_low=seq_group.quant_configs[3],
            batch_compress_configs=batch_compress_configs,
        )

    def _append_slot_to_seq_group(
        self,
        seq_group: SequenceGroup,
    ) -> bool:
        seq_ids = [seq.seq_id for seq in
                   seq_group.get_seqs(status=SequenceStatus.RUNNING)]
        return self.orchestrator.run_workers_bool_any(
            "append_slot_to_seqs",
            seq_ids=seq_ids,
            kbits_high=seq_group.quant_configs[0],
            vbits_high=seq_group.quant_configs[1],
            kbits_low=seq_group.quant_configs[2],
            vbits_low=seq_group.quant_configs[3],
        )

    def _append_slot(
        self,
        seq_ids: List[int],
        kbits_high: int,
        vbits_high: int,
        kbits_low: int,
        vbits_low: int,
    ) -> bool:
        return self.orchestrator.run_workers_bool_any(
            "append_slot_to_seqs",
            seq_ids=seq_ids,
            kbits_high=kbits_high,
            vbits_high=vbits_high,
            kbits_low=kbits_low,
            vbits_low=vbits_low,
        )

    def _preempt(
        self,
        seq_group: SequenceGroup,
        preemption_mode: Optional[PreemptionMode] = None,
    ) -> bool:
        # If preemption mode is not specified, we determine the mode as follows:
        # We use recomputation by default since it incurs lower overhead than
        # swapping. However, when the sequence group has multiple sequences
        # (e.g., beam search), recomputation is not currently supported. In
        # such a case, we use swapping instead.
        # FIXME(woosuk): This makes our scheduling policy a bit bizarre.
        # As swapped sequences are prioritized over waiting sequences,
        # sequence groups with multiple sequences are implicitly prioritized
        # over sequence groups with a single sequence.
        # TODO(woosuk): Support recomputation for sequence groups with multiple
        # sequences. This may require a more sophisticated CUDA kernel.

        # return if mem swap is needed
        if preemption_mode is None:
            if seq_group.get_max_num_running_seqs() == 1:
                preemption_mode = PreemptionMode.RECOMPUTE
            else:
                preemption_mode = PreemptionMode.SWAP
        if preemption_mode == PreemptionMode.RECOMPUTE:
            self._preempt_by_recompute(seq_group)
            return False
        elif preemption_mode == PreemptionMode.SWAP:
            self._preempt_by_swap(seq_group)
            return True
        else:
            raise AssertionError("Invalid preemption mode.")

    def _preempt_by_recompute(
        self,
        seq_group: SequenceGroup,
    ) -> None:
        seqs = seq_group.get_seqs(status=SequenceStatus.RUNNING)
        assert len(seqs) == 1
        for seq in seqs:
            seq.status = SequenceStatus.WAITING
            # NOTE: update estimated free blocks so that the scheduler can make decisions on appending
            _init_free_blocks = self.memory_manager.min_num_free_gpu_blocks()
            self.memory_manager.update_free_gpu_blocks(
                self.orchestrator.run_workers(
                    "free_seq",
                    seq_id=seq.seq_id,
                    is_finished=False,
                    get_all_outputs=True))
            _final_free_blocks = self.memory_manager.min_num_free_gpu_blocks()
            self.memory_manager.update_estimated_free_blocks(_final_free_blocks - _init_free_blocks)

        # NOTE: For FCFS, we insert the preempted sequence group to the front
        # of the waiting queue.
        self.waiting.insert(0, seq_group)

    def _preempt_by_swap(
        self,
        seq_group: SequenceGroup,
    ) -> None:
        # TODO: Implement the swapping mechanism.
        # TODO: note we should carefully handle the estimated memory here, as in _preempty_by_compute
        self._swap_out(seq_group)
        self.swapped.append(seq_group)

    def _swap_in(
        self,
        seq_group: SequenceGroup,
    ) -> None:
        seqs = seq_group.get_seqs(status=SequenceStatus.SWAPPED)
        seq_ids = [seq.seq_id for seq in seqs]
        self.orchestrator.run_workers(
            "swap_in_seqs",
            seq_ids=seq_ids,
        )
        for seq in seqs:
            seq.status = SequenceStatus.RUNNING

    def _swap_out(
        self,
        seq_group: SequenceGroup,
    ) -> None:
        seqs = seq_group.get_seqs(status=SequenceStatus.RUNNING)
        seq_ids = [seq.seq_id for seq in seqs]
        # can_swap_out = self.orchestrator.run_workers_bool_all(
        #     "can_swap_out_seqs",
        #     seq_ids=seq_ids,
        # )
        if not self.memory_manager.can_swap_out(seq_ids):
            # FIXME(woosuk): Abort the sequence group instead of aborting the
            # entire engine.
            raise RuntimeError(
                "Aborted due to the lack of CPU swap space. Please increase "
                "the swap space to avoid this error.")
        # determine the blocks to be swapped out
        # TODO: have swap_out_seqs return #free gpu blocks
        self.memory_manager.update_free_gpu_blocks(
            self.orchestrator.run_workers(
                "swap_out_seqs",
                seq_ids=seq_ids,
                get_all_outputs=True
        ))
        for seq in seqs:
            seq.status = SequenceStatus.SWAPPED

    def _poll_free_blocks(self) -> List[int]:
        return self.orchestrator.run_workers(
            "get_free_gpu_blocks",
            get_all_outputs=True)

def _divide_round_up(x: int, y: int) -> int:
    return (x + y - 1) // y