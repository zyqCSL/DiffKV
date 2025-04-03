"""A block manager that manages token blocks."""
import enum
import torch
import numpy as np
from typing import Dict, List, Deque, Optional, Set, Tuple, Union
from collections import deque
import time
import copy

from vllm._C import mem_mgt_ops
from vllm.sequence import Sequence, SequenceGroup, SequenceStatus, SequenceGroupMetadata
from vllm.utils import Device

# # Mapping: layer, head, logical block number -> physical block.
# BlockTable = List[List[List[PhysicalTokenBlock]]]
# # length of each sparse KV head. Mapping: layer, head -> seq_len
# KVLenTable = List[List[int]]
_PAD_LEN = 0
_PAD_BLOCK = -1

# TODO: make this allocator per-device
class BlockAllocator:
    """Manages free physical token blocks for a device.

    The allocator maintains a list of free blocks and allocates a block when
    requested. When a block is freed, its reference count is decremented. If
    the reference count becomes zero, the block is added back to the free list.
    """

    def __init__(
        self,
        device: Device,
        num_blocks: int,
        max_num_seqs: int,
    ) -> None:
        self.device = device
        self.num_blocks = num_blocks
        self.max_num_seqs = max_num_seqs

        # preallocated metadata resources
        if self.device == Device.GPU:
            device = 'cuda'
        else:
            device = 'cpu'
        # blocks within start_block_pos & end_block_pos are free
        self.free_blocks = torch.arange(self.num_blocks, dtype=torch.int32, device=device)
        self.start_block_pos = 0   # closed
        self.end_block_pos = self.num_blocks - 1   # closed
        # reference counters indexed by block id
        self.block_refs = torch.zeros((self.num_blocks,), dtype=torch.int32, device=device)
        # self.free_blocks: Deque[np.int32] = deque(np.arange(num_blocks, dtype=np.int32))

    def allocate(self, num_blocks: int) -> Tuple[int, int]:
        ''' Return
        [start_block_pos, end_block_pos, List[slot_id]]
        '''
        num_free_blocks = self.get_num_free_blocks()
        if num_free_blocks < num_blocks:
            raise ValueError(f"Out of memory! {num_blocks} blocks required but only {num_free_blocks} are free.")
        start_pos = self.start_block_pos
        end_pos = (self.start_block_pos + num_blocks) % self.num_blocks
        self.start_block_pos = end_pos
        return start_pos, end_pos

    # def free(self, end_pos: int) -> None:
    #     ''' TODO (yanqi): compaction of the fragmented mem region is performed in cuda kernels
    #     TODO: revisit this, do we need such an interface?
    #     '''
    #     self.end_block_pos = end_pos

    def get_num_free_blocks(self) -> int:
        if self.end_block_pos >= self.start_block_pos:
            return self.end_block_pos - self.start_block_pos + 1
        else:
            return self.end_block_pos + self.num_blocks - self.start_block_pos + 1

    def _can_allocate(self, num_blocks: int) -> bool:
        return self.get_num_free_blocks() >= num_blocks

    def update_start_block_pos(self, start_pos: int) -> None:
        self.start_block_pos = (start_pos + self.num_blocks) % self.num_blocks
        assert self.start_block_pos >= 0

    def update_end_block_pos(self, end_pos: int) -> None:
        self.end_block_pos = (end_pos + self.num_blocks) % self.num_blocks
        assert self.end_block_pos >= 0

    def get_next_end_pos(self):
        return (self.end_block_pos + 1) % self.num_blocks


class AllocStatus(enum.Enum):
    """Result for BlockSpaceManager.can_allocate

    1. Ok: seq_group can be allocated now.
    2. Later: seq_group cannot be allocated.
      The capacity of allocator is larger than seq_group required.
    3. Never: seq_group can never be allocated.
      The seq_group is too large to allocated in GPU.
    """
    OK = enum.auto()
    LATER = enum.auto()
    NEVER = enum.auto()


class BlockSpaceManager:
    """Manages the mapping between logical and physical token blocks."""

    def __init__(
        self,
        max_num_seqs: int,
        block_bytes: int,
        num_gpu_blocks: int,
        num_cpu_blocks: int,
        quantized_kv_bits: List[Tuple[int, int]],
        quantized_block_num_tokens: Dict[Tuple[int, int], int],
        num_layers: int,
        num_heads: int,
        num_kv_heads: int,
        head_size: int,
        max_model_len: int,
        watermark: float = 0.01,
        sliding_window: Optional[int] = None,
        max_kv_slots: Optional[int] = None,
        log_enabled: bool = True,
    ) -> None:
        ''' Args
        max_num_seqs: max number of seqs that can reside concurrently on GPU
        '''
        self.max_num_seqs = max_num_seqs
        assert self.max_num_seqs > 0
        self.block_bytes = block_bytes

        self.num_total_gpu_blocks = num_gpu_blocks
        self.num_total_cpu_blocks = num_cpu_blocks

        # kv quantization configs
        self.quantized_kv_bits = copy.deepcopy(quantized_kv_bits)
        self.quant_to_block_token_num = copy.deepcopy(quantized_block_num_tokens)
        # self.quant_to_block_token_num = {
        #     _config: self.block_bytes // self.quant_to_kv_head_bytes[_config]
        #     for _config in quantized_kv_bits
        # }
        min_block_size = min(self.quant_to_block_token_num.values())
        assert min_block_size >= 1
        # print(f'quant_to_block_token_num = {self.quant_to_block_token_num}')

        self.num_layers = num_layers    # per worker
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads    # per worker
        self.head_size = head_size
        self.max_model_len = max_model_len  # max context length of the model
        # NOTE: we need to reserve one block in case that only 1 token is quantized to low precision,
        # requiring one additional block, while the high precision block number remains
        # NOTE: we also need a another block to make room for append
        self.max_block_table_len = _divide_round_up(self.max_model_len, min_block_size) + 2
        
        print(f'DEBUG: max_block_table_len = {self.max_block_table_len}')
        
        # Allocate for the worst case: each head requires a new block
        self.num_kv_caches = self.num_layers * self.num_kv_heads

        # print(f'BlockSpaceManager.max_model_len = {self.max_model_len}')

        self.sliding_window = sliding_window    # sliding window is depreciated for max_kv_slots
        self.quant_to_max_kv_blocks = None
        if max_kv_slots is not None:
            self.quant_to_max_kv_blocks = {
                # reserve one additional block for low precision, and one for append
                _config: _divide_round_up(max_kv_slots, self.quant_to_block_token_num[_config]) + 2
                for _config in quantized_kv_bits
            }
            self.max_block_table_len = min(self.max_block_table_len,
                                           max(self.quant_to_max_kv_blocks.values()))

        self.watermark = watermark
        assert watermark >= 0.0

        self.watermark_blocks = int(watermark * num_gpu_blocks)
        self.gpu_allocator = BlockAllocator(Device.GPU, num_gpu_blocks, max_num_seqs)
        self.cpu_allocator = BlockAllocator(Device.CPU, num_cpu_blocks, max_num_seqs)

        # metadata resource slot ids
        self.free_slot_ids = deque(range(self.max_num_seqs))
        # mapping: seq_id -> metadata slot_id
        self.seq_to_slot: Dict[int, int] = {}
        # block tables
        self.block_tables = torch.full(
            (self.max_num_seqs, self.num_layers, self.num_kv_heads, self.max_block_table_len),
            _PAD_BLOCK, dtype=torch.int32, device='cuda')
        # number of saved kv tokens of each seq on GPU
        # the last dim 2 is for 2 quant configs, the 1st config grows from left to right
        # and the 2nd config grows from right to left
        self.kv_len_tables = torch.full(
            (self.max_num_seqs, self.num_layers, self.num_kv_heads, 2),
            _PAD_LEN, dtype=torch.int32, device='cuda')
        # number of allocated blocks of each seq on GPU
        # the last dim 2 is for 2 quant configs, the 1st config grows from left to right
        # and the 2nd config grows from right to left
        self.block_num_tables = torch.full(
            (self.max_num_seqs, self.num_layers, self.num_kv_heads, 2),
            _PAD_LEN, dtype=torch.int32, device='cuda')
        # (compress ratio, quant ratio) of each sequence
        self.compress_config_tables = torch.full(
            (self.max_num_seqs, 2),
            1, dtype=torch.float32, device='cuda')

        self.sparsity_tables = torch.full(
            (self.max_num_seqs, self.num_layers, self.num_heads),
            0, dtype=torch.uint64, device='cuda')

        # quant configs of each sequence. NOTE: now we only keep it in host memory
        self.quant_config_tables = torch.zeros(
            (self.max_num_seqs, 4), dtype=torch.int16, device='cpu')
        # self.quant_config_tables = torch.zeros(
        #     (self.max_num_seqs, 4), dtype=torch.int16, device='cuda')
        
        # metadata of swapped out sequences
        self.free_swapped_slot_ids = deque(range(self.max_num_seqs))
        self.swapped_seq_to_slot: Dict[int, int]
        self.swapped_block_tables = torch.full_like(
            self.block_tables, _PAD_BLOCK, dtype=torch.int32, device='cpu')
        self.swapped_kv_len_tables = torch.full_like(
            self.kv_len_tables, _PAD_LEN, dtype=torch.int32, device='cpu')
        self.swapped_block_num_tables = torch.full_like(
            self.block_num_tables, _PAD_LEN, dtype=torch.int32, device='cpu')
        # self.swapped_quant_config_tables = torch.zeros_like(
        #     self.quant_config_tables, dtype=torch.int16, device='cpu')
        self.swapped_compress_config_tables = torch.zeros_like(
            self.compress_config_tables, dtype=torch.float32, device='cpu')

        # TODO (yanqi): revisit, do we need this?
        # seq_id -> if seq is in gpu
        self.seq_in_gpu: Dict[int, bool] = {}

        # pending memory operation lists
        self.blocks_to_swap_in: Dict[int, int] = {}
        self.blocks_to_swap_out: Dict[int, int] = {}
        self.blocks_to_copy: Dict[int, List[int]] = {}

        # sparse kv related metadata
        self.log_enabled = log_enabled
        self.num_finished_seqs = 0
        self.accum_kv_lens = torch.zeros((self.num_layers, self.num_kv_heads, 2), 
                                         dtype=torch.int64, device='cuda')
        self.accum_block_nums = torch.zeros_like(self.accum_kv_lens)
        # self.accum_num_critical_tokens = torch.zeros((self.num_layers, self.num_heads),
        #                                              dtype=torch.uint64, device='cuda')
        self.accum_num_critical_tokens = np.zeros((self.num_layers, self.num_heads),
                                                  dtype=np.uint64)

    def allocate_seq_group(self, seq_group: SequenceGroup) -> None:
        # NOTE: Here we assume that all sequences in the group have the same
        # prompt.
        seq = seq_group.get_seqs(status=SequenceStatus.WAITING)[0]
        num_prompt_tokens = seq.data.get_len()
        seq_ids = [seq.seq_id for seq in seq_group.get_seqs(status=SequenceStatus.WAITING)]
        self.allocate(seq_ids, [num_prompt_tokens] * len(seq_ids))

    def allocate(
        self,
        batch_seq_ids: List[List[int]],
        batch_num_prompt_tokens: List[int],
        # batch_quant_configs: List[List[int]],
        kbits_high: int,
        vbits_high: int,
        kbits_low: int,
        vbits_low: int,
        batch_compress_configs: List[List[float]],
    ) -> None:
        ''' NOTE: All sequences in the batch should share the same quant configs, so that we 
            can use templates for cuda kernels. Otherwise the latency of attention kernel will suffer
        '''
        # Allocate all seqs in potentially multiple groups
        # NOTE: Here we assume that all sequences in the group have the same prompt.
        # Allocate new physical token blocks that will store the prompt tokens.

        # TODO
        # t0 = time.time()
        assert len(batch_seq_ids) == len(batch_num_prompt_tokens)
        if len(self.free_slot_ids) < len(batch_seq_ids):
            raise ValueError(f"Out of metadata slots! {len(batch_seq_ids)} seqs required but only {len(self.free_slot_ids)} are free.")

        # check quant configs
        assert (kbits_high, vbits_high) in self.quantized_kv_bits
        assert (kbits_low, vbits_low) in self.quantized_kv_bits
        # we use high precision when allocating memory for prompt
        quant = (kbits_high, vbits_high)
        block_size = self.quant_to_block_token_num[quant]
        
        slot_ids = [-1] * len(batch_seq_ids)
        total_blocks_per_head = 0
        batch_num_blocks_per_head = [-1] * len(batch_num_prompt_tokens)
        for i, seq_ids in enumerate(batch_seq_ids):
            assert len(seq_ids) == 1, "All seqs in a group should share a single prompt"
            seq_id = seq_ids[0]
            assert seq_id not in self.seq_to_slot
            slot_id = self.free_slot_ids.popleft()
            self.seq_to_slot[seq_id] = slot_id
            # print(f'BlockSpaceManager.allocate seq_id: {seq_id}, slot_id: {slot_id}')
            self.seq_in_gpu[seq_id] = True
            slot_ids[i] = slot_id

            # compute required number of kv blocks
            # NOTE: during prompt, allocate blocks only according to the high precision
            # the total blocks used by high & low precisions must be no larger than
            # the blocks required by high precision only
            num_prompt_tokens = batch_num_prompt_tokens[i]
            # if (batch_quant_configs[i][2], batch_quant_configs[i][3]) != (0, 0):
            if (kbits_high, vbits_high) != (kbits_low, vbits_low):
                # NOTE: with both high & low precisions enabled, we need to reserve one block for low precision
                num_blocks_per_head = _divide_round_up(num_prompt_tokens, block_size) + 1
            else:
                num_blocks_per_head = _divide_round_up(num_prompt_tokens, block_size)
            if self.quant_to_max_kv_blocks is not None:
                num_blocks_per_head = min(num_blocks_per_head, self.quant_to_max_kv_blocks[quant])
            total_blocks_per_head += num_blocks_per_head
            batch_num_blocks_per_head[i] = num_blocks_per_head

        # fill in quant configs
        self.quant_config_tables[slot_ids] = torch.tensor(
            [kbits_high, vbits_high, kbits_low, vbits_low], 
            dtype=torch.int16, device=self.quant_config_tables.device)
         # fill in compress_config
        self.compress_config_tables[slot_ids] = torch.tensor(
            batch_compress_configs, dtype=torch.float32, device=self.compress_config_tables.device)

        # print(f'*** batch_num_blocks_per_head = {batch_num_blocks_per_head}')

        start_block_pos, end_block_pos = self.gpu_allocator.allocate(
            total_blocks_per_head * self.num_kv_caches)

        # TODO: debug, remove later
        self.block_tables[slot_ids] = _PAD_BLOCK
        self.block_num_tables[slot_ids] = _PAD_LEN
        for slot_id in slot_ids:
            self.sparsity_tables[slot_id] = 0

        # requires reset as it's used to compute max_context_len
        self.kv_len_tables[slot_ids] = _PAD_LEN

        # t1 = time.time()
        # print(f'*** start_block_pos = {start_block_pos}, end_block_pos = {end_block_pos}, '
        #       f'start block_id = {self.gpu_allocator.free_blocks[start_block_pos]}, '
        #       f'last block_id = {self.gpu_allocator.free_blocks[end_block_pos]}')

        # TODO (yanqi): cuda kernel to fill metadata
        # TODO (yanqi): fill self.block_tables, kv_len_tables & block_num_tables
        mem_mgt_ops.allocate_seqs(
            torch.tensor(slot_ids, dtype=torch.int32, device='cuda'),
            torch.tensor(batch_num_prompt_tokens, dtype=torch.int32, device='cuda'),
            torch.tensor(batch_num_blocks_per_head, dtype=torch.int32, device='cuda'),
            self.block_tables,
            self.kv_len_tables,
            self.block_num_tables,
            self.gpu_allocator.free_blocks,
            self.gpu_allocator.block_refs,
            start_block_pos,
            end_block_pos)

        # tz = time.time()
        # print(f'*** BlockSpaceManager.batch_allocate latency = {1000 * (tz - t0)} ms, '
        #       f'cuda latency = {1000 * (tz - t1)} ms')

    # TODO: need to distinguish layers & heads. Specify which layer & head needs block in inputs
    def append_slot_to_seq(self, seq: Sequence) -> bool:
        ''' Return true if mem copy is required '''
        return self.append_slot([seq.seq_id])

    def append_slot(
        self,             
        seq_ids: List[int],
        kbits_high: int,
        vbits_high: int,
        kbits_low: int,
        vbits_low: int,
    ) -> bool:
        """Allocate a physical slot for a new token.
        Return true if mem copy is required
        """
        # t0 = time.time()
        slot_ids = [None] * len(seq_ids)
        for i, seq_id in enumerate(seq_ids):
            assert self.seq_in_gpu[seq_id]
            slot_ids[i] = self.seq_to_slot[seq_id]

        # check all seqs share the same quant config
        ref_quant_config = torch.tensor(
            [kbits_high, vbits_high, kbits_low, vbits_low], 
            dtype=torch.int16, device=self.quant_config_tables.device)
        assert torch.equal(
            self.quant_config_tables[slot_ids],
            ref_quant_config.reshape((1, -1)).expand(len(slot_ids), 4))
        
        block_size_high = self.quant_to_block_token_num[(kbits_high, vbits_high)]
        block_size_low = self.quant_to_block_token_num[(kbits_low, vbits_low)]
        
        # * 2 for two precisions
        free_block_pos = torch.zeros(
            (len(slot_ids) * self.num_layers * self.num_kv_heads * 2 + 1,),
            dtype=torch.int32, device='cuda')
        # fill the 1st slot for prefix sum later
        free_block_pos[0] = self.gpu_allocator.start_block_pos 
        mem_mgt_ops.prepare_append_seqs(
            torch.tensor(slot_ids, dtype=torch.int32, device='cuda'),
            self.kv_len_tables,
            self.block_num_tables,
            kbits_high,
            vbits_high,
            kbits_low,
            vbits_low,
            block_size_high,
            block_size_low,
            free_block_pos)
        # print(free_block_pos)

        # t1 = time.time()
        # prefix sum: compute end block pos for each head
        free_block_pos.cumsum_(dim=0)
        # t2 = time.time()
        
        # NOTE: check that there are enough blocks for append
        required_blocks = free_block_pos[-1].item() - self.gpu_allocator.start_block_pos
        assert required_blocks <= self.gpu_allocator.get_num_free_blocks(), (
            f'Out of memory! {required_blocks} blocks required, '
            f'but only {self.gpu_allocator.get_num_free_blocks()} available'
        )

        # print(free_block_pos)

        # update gpu_allocator pointers
        # NOTE: the pointer might exceed free_blocks length
        # TODO: add a test for the cross-boundary corner case
        # NOTE: we don't need to subtract 1 here as in free & free_prompt
        # since here we are updating the start_block_pos
        self.gpu_allocator.update_start_block_pos(free_block_pos[-1].item())

        # TODO: later we need the kernel to return which blocks require copying
        mem_mgt_ops.append_seqs(
            torch.tensor(slot_ids, dtype=torch.int32, device='cuda'),
            self.block_tables,
            self.kv_len_tables,
            self.block_num_tables,
            self.gpu_allocator.free_blocks,
            self.gpu_allocator.block_refs,
            free_block_pos)

        # tz = time.time()
        # print(f'*** append_slot latency = {1000 * (tz - t0)} ms, '
        #       f'prepare = {1000 * (t1 - t0)} ms, '
        #       f'cumsum = {1000 * (t2 - t1)} ms, '
        #       f'append = {1000 * (tz - t2)} ms')

        copy_required = False
        return copy_required

    def fork_seq(self, parent_seq: Sequence, child_seq: Sequence) -> None:
        raise NotImplementedError
        self.fork(parent_seq.seq_id, child_seq.seq_id)

    def fork(self, parent_seq_id: int, child_seq_id: int) -> None:
        raise NotImplementedError
        # NOTE: fork does not allocate a new physical block.
        # Thus, it is always safe from OOM.
        src_block_table = self.block_tables[parent_seq_id]
        src_kv_len_table = self.kv_len_tables[parent_seq_id]
        src_block_num_table = self.block_num_tables[parent_seq_id]

        self.block_tables[child_seq_id] = np.copy(src_block_table)
        self.kv_len_tables[child_seq_id] = np.copy(src_kv_len_table)
        self.block_num_tables[child_seq_id] = np.copy(src_block_num_table)
        # FIXME: revisit this, is it possible to parallelize this part?
        for layer in range(self.num_layers):
            for head in range(self.num_kv_heads):
                head_block_num = src_block_num_table[layer, head]
                for block in src_block_table[layer, head, :head_block_num]:
                    self.gpu_allocator.inc_ref(block)

    def _get_physical_blocks(
            self, seq_ids: List[int]) -> List[int]:
        raise NotImplementedError
        # NOTE: Here, we assume that the physical blocks are only shared by
        # the sequences in the same group.
        blocks: Set[int] = set()
        for seq_id in seq_ids:
            block_table = self.block_tables[seq_id]
            block_nums = self.block_num_tables[seq_id]
            # assert seq.seq_id in self.kv_len_tables
            for layer in range(self.num_layers):
                for head in range(self.num_kv_heads):
                    head_block_num = block_nums[layer, head]
                    blocks.update(block_table[layer, head, :head_block_num])
        return list(blocks)

    def can_swap_in_seq_group(self, seq_group: SequenceGroup) -> bool:
        raise NotImplementedError
        # NOTE: here we assume that a sequence group is always swapped out as a whole
        seq_ids = seq_group.get_seqs(status=SequenceStatus.SWAPPED)
        return self.can_swap_in(seq_ids)

    def can_swap_in(self, seq_ids: List[int]) -> bool:
        raise NotImplementedError
        blocks = self._get_physical_blocks(seq_ids)
        # num_swapped_seqs = seq_group.num_seqs(status=SequenceStatus.SWAPPED)
        # NOTE: here we assume that a sequence group is always swapped out as a whole
        num_swapped_seqs = len(seq_ids)
        num_free_blocks = self.gpu_allocator.get_num_free_blocks()
        # NOTE: Conservatively, we assume that every sequence will allocate
        # at least one free block (for each head) right after the swap-in.
        # NOTE: This should match the logic in can_append_slot().
        num_required_blocks = len(blocks) + num_swapped_seqs * self.num_kv_caches
        return num_free_blocks - num_required_blocks >= self.watermark_blocks

    def swap_in_seq_group(self, seq_group: SequenceGroup) -> bool:
        raise NotImplementedError
        seqs = seq_group.get_seqs(status=SequenceStatus.SWAPPED)
        seq_ids = [seq.seq_id for seq in seqs]
        return self.swap_in(seq_ids)

    def swap_in(self, seq_ids: List[int]) -> None:
        raise NotImplementedError
        # CPU block -> GPU block.
        mapping: Dict[int, int] = {}
        for seq_id in seq_ids:
            assert not self.seq_in_gpu[seq_id]
            new_block_table = np.full((self.num_layers, self.num_kv_heads, self.max_block_table_len),
                                      _PAD_BLOCK, dtype=np.int32)
            block_table = self.block_tables[seq_id]
            block_nums = self.block_num_tables[seq_id]

            for layer in range(self.num_layers):
                for head in range(self.num_kv_heads):
                    head_block_num = block_nums[layer, head]
                    for pos, cpu_block in enumerate(block_table[layer, head, :head_block_num]):
                        if cpu_block in mapping:
                            gpu_block = mapping[cpu_block]
                            self.gpu_allocator.inc_ref(gpu_block)
                        else:
                            gpu_block = self.gpu_allocator.allocate()
                            mapping[cpu_block] = gpu_block
                        new_block_table[layer, head, pos] = gpu_block
                        # Free the CPU block swapped in to GPU.
                        self.cpu_allocator.free(cpu_block)
            self.block_tables[seq_id] = new_block_table
            self.seq_in_gpu[seq_id] = True

        self.blocks_to_swap_in.update(mapping)
        return len(mapping) > 0

    def can_swap_out_seq_group(self, seq_group: SequenceGroup) -> bool:
        raise NotImplementedError
        seqs = seq_group.get_seqs(status=SequenceStatus.RUNNING)
        seq_ids = [seq.seq_id for seq in seqs]
        return self.can_swap_out(seq_ids)

    def can_swap_out(self, seq_ids: List[int]) -> bool:
        raise NotImplementedError
        blocks = self._get_physical_blocks(seq_ids)
        return len(blocks) <= self.cpu_allocator.get_num_free_blocks()

    def swap_out_seq_group(self, seq_group: SequenceGroup) -> bool:
        raise NotImplementedError
        seqs = seq_group.get_seqs(status=SequenceStatus.RUNNING)
        seq_ids = [seq.seq_id for seq in seqs]
        return self.swap_out(seq_ids)

    def swap_out(self, seq_ids: List[int]) -> bool:
        raise NotImplementedError
        # GPU block -> CPU block.
        mapping: Dict[int, int] = {}
        for seq_id in seq_ids:
            assert self.seq_in_gpu[seq_id]
            new_block_table = np.full((self.num_layers, self.num_kv_heads, self.max_block_table_len),
                                      _PAD_BLOCK, dtype=np.int32)
            block_table = self.block_tables[seq_id]
            block_nums = self.block_num_tables[seq_id]

            for layer in range(self.num_layers):
                for head in range(self.num_kv_heads):
                    head_block_num = block_nums[layer, head]
                    for pos, gpu_block in enumerate(block_table[layer, head, :head_block_num]):
                        if gpu_block in mapping:
                            cpu_block = mapping[gpu_block]
                            self.cpu_allocator.inc_ref(cpu_block)
                        else:
                            cpu_block = self.cpu_allocator.allocate()
                            mapping[gpu_block] = cpu_block
                        new_block_table[layer, head, pos] = cpu_block
                        # Free the GPU block swapped out to CPU.
                        self.gpu_allocator.free(gpu_block)
            self.block_tables[seq_id] = new_block_table
            self.seq_in_gpu[seq_id] = False

        self.blocks_to_swap_out.update(mapping)
        return len(mapping) > 0

    '''
    def _free_block_table(self, seq_id: int) -> None:
        # blocks should not be shared within a sequence
        block_table = self.block_tables[seq_id]
        block_nums = self.block_num_tables[seq_id]
        if self.seq_in_gpu[seq_id]:
            allocator = self.gpu_allocator
        else:
            allocator = self.cpu_allocator
        for layer in range(self.num_layers):
            for head in range(self.num_kv_heads):
                head_block_num = block_nums[layer, head]
                allocator.batch_free(block_table[layer, head, :head_block_num])
    '''

    def free_seq(self, seq: Sequence) -> None:
        self.free(seq.seq_id, seq.is_finished())

    def free(self, seq_id: int, is_finished: bool) -> int:
        # t0 = time.time()

        # NOTE (yanqi): here we assume block sharing is disabled
        assert seq_id in self.seq_to_slot
        assert self.seq_in_gpu[seq_id]
        slot_id = self.seq_to_slot[seq_id]
        del self.seq_to_slot[seq_id]
        del self.seq_in_gpu[seq_id]
        self.free_slot_ids.append(slot_id)
        
        if self.log_enabled and is_finished:
            # TODO: sparsity stats, comment off for throughput tests
            self.num_finished_seqs += 1
            self.accum_kv_lens += self.kv_len_tables[slot_id]
            self.accum_block_nums += self.block_num_tables[slot_id]
            # self.accum_num_critical_tokens += self.sparsity_tables[slot_id]
            self.accum_num_critical_tokens += np.array(self.sparsity_tables[slot_id].cpu())
            
            # TODO: remove later
            # print('freed kv_len = ', torch.sum(self.kv_len_tables[slot_id], dim=(0, 1)))
            # print('freed block_num = ', torch.sum(self.block_num_tables[slot_id], dim=(0, 1)))
            # print(f'seq_id {seq_id} mean critical tokens = ', torch.mean(self.sparsity_tables[slot_id].float()))

        # write freed blocks to free_blocks
        # NOTE: 2 for two quant configs
        free_block_pos = torch.zeros(
            (self.num_layers * self.num_kv_heads * 2 + 1,),
            dtype=torch.int32, device='cuda')
        # NOTE: we need to add 1 here because the end_block_pos is closed,
        # meaning that it already contains a usable block.
        # If we want to extend usable free blocks,
        # we need to start from the next poistion, otherwise we overwrite a usable block
        free_block_pos[0] = self.gpu_allocator.get_next_end_pos()
        # prepare the memory layout
        mem_mgt_ops.prepare_free_seqs(
            torch.tensor([slot_id], dtype=torch.int32, device='cuda'),
            self.block_num_tables,
            free_block_pos)
        # prefix sum: compute offsets to write pruned blocks for each head
        free_block_pos.cumsum_(dim=0)
        # NOTE: we need to subtract 1 here to compensate the +1 when initializing free_block_pos
        # so as to keep the end_block_pos a closed interval
        self.gpu_allocator.update_end_block_pos(free_block_pos[-1].item() - 1)

        # put freed blocks to free_blocks
        mem_mgt_ops.free_seqs(
            torch.tensor([slot_id], dtype=torch.int32, device='cuda'),
            self.block_tables,
            self.block_num_tables,
            self.gpu_allocator.free_blocks,
            free_block_pos,
            self.gpu_allocator.block_refs)
        # requires reset as it's used to compute max_context_len
        self.kv_len_tables[slot_id].fill_(_PAD_LEN)
        # must start from 0
        self.sparsity_tables[slot_id].fill_(0)

        # self.block_tables[slot_id].fill_(_PAD_BLOCK)
        # self.block_num_tables[slot_id].fill_(_PAD_LEN)

        # # TODO: debug, remove later
        # print(torch.mean(self.accum_block_nums / self.num_finished_seqs,
        #                  dim=1))

        # tz = time.time()
        # print(f'free latency = {1000 * (tz - t0)} ms')        
        return self.get_num_free_gpu_blocks()

    def free_prompt(
        self, 
        seq_ids: List[int],
        kbits_high: int,
        vbits_high: int,
        kbits_low: int,
        vbits_low: int,
    ) -> None:
        ''' NOTE: kv_len_tables should be updated directly in the attention kernel
        '''
        # # TODO
        # t0 = time.time()

        # NOTE: kv_lens in the parameters should be allocated a new tensor
        # and should not overwrite self.kv_len_tables entries

        slot_ids = [-1] * len(seq_ids)
        for i, seq_id in enumerate(seq_ids):
            assert seq_id in self.seq_to_slot
            slot_id = self.seq_to_slot[seq_id]
            slot_ids[i] = slot_id
        
        # check all seqs share the same quant config
        # TODO: remove this for performance profiling
        ref_quant_config = torch.tensor(
            [kbits_high, vbits_high, kbits_low, vbits_low], 
            dtype=torch.int16, device=self.quant_config_tables.device)
        assert torch.equal(
            self.quant_config_tables[slot_ids],
            ref_quant_config.reshape((1, -1)).expand(len(slot_ids), 4))

        block_size_high = self.quant_to_block_token_num[(kbits_high, vbits_high)]
        block_size_low = self.quant_to_block_token_num[(kbits_low, vbits_low)]
        
        # number of used block per head
        block_nums = torch.zeros(
            (len(slot_ids) * self.num_layers * self.num_kv_heads * 2,),
            dtype=torch.int32, device='cuda')

        free_block_pos = torch.zeros(
            (len(slot_ids) * self.num_layers * self.num_kv_heads + 1,),
            dtype=torch.int32, device='cuda')
        # NOTE: we need to add 1 here because the end_block_pos is closed,
        # meaning that it already contains a usable block.
        # If we want to extend usable free blocks,
        # we need to start from the next poistion, otherwise we overwrite a usable block
        free_block_pos[0] = self.gpu_allocator.get_next_end_pos()

        # t1 = time.time()

        # prepare the memory layout
        mem_mgt_ops.prepare_free_prompt(
            torch.tensor(slot_ids, dtype=torch.int32, device='cuda'),
            self.kv_len_tables,
            self.block_num_tables,
            block_size_high,
            block_size_low,
            free_block_pos,
            block_nums)

        # prefix sum: compute offsets to write pruned blocks for each head
        free_block_pos.cumsum_(dim=0)
        # update gpu_allocator pointers
        # NOTE: the pointer might exceed free_blocks length
        # TODO: add a test for the cross-boundary corner case
        # NOTE: we need to subtract 1 here to compensate the +1 when initializing free_block_pos
        # so as to keep the end_block_pos a closed interval
        self.gpu_allocator.update_end_block_pos(free_block_pos[-1].item() - 1)

        # put pruned blocks to free_blocks and update num_blocks_tables
        mem_mgt_ops.free_prompt(
            torch.tensor(slot_ids, dtype=torch.int32, device='cuda'),
            self.block_tables,
            self.block_num_tables,
            self.gpu_allocator.free_blocks,
            free_block_pos,
            block_nums,
            self.gpu_allocator.block_refs)

        # tz = time.time()
        # print(f'free_prompt latency = {1000 * (tz - t0)} ms, '
        #       f'free_prompt kernel latency = {1000 * (tz - t1)} ms')

    def reset_pending_memory_ops(self) -> None:
        self.blocks_to_swap_in = {}
        self.blocks_to_swap_out = {}
        self.blocks_to_copy = {}

    '''
    def reset(self) -> None:
        for block_table in self.block_tables.values():
            self._free_block_table(block_table)
        self.block_tables.clear()
        self.reset_pending_memory_ops()

    def get_block_table(self, seq_id: int) -> np.ndarray:
        if seq_id not in self.block_tables:
            return None
        block_table = self.block_tables[seq_id]
        return block_table

    def get_kv_len_table(self, seq_id: int) -> np.ndarray:
        if seq_id not in self.kv_len_tables:
            return None
        return self.kv_len_tables[seq_id]

    def get_block_num_table(self, seq_id: int) -> np.ndarray:
        if seq_id not in self.block_num_tables:
            return None
        return self.block_num_tables[seq_id]

    '''

    def prepare_metadata(self, metadata: SequenceGroupMetadata) -> None:
        # update slot_ids if it is empty
        if not metadata.slot_ids:
            for seq_id in metadata.seq_data:
                assert seq_id in self.seq_to_slot
                assert self.seq_in_gpu[seq_id]
                slot_id = self.seq_to_slot[seq_id]
                metadata.slot_ids.append(slot_id)

    def get_num_free_gpu_blocks(self) -> int:
        return self.gpu_allocator.get_num_free_blocks()

    def get_num_free_cpu_blocks(self) -> int:
        return self.cpu_allocator.get_num_free_blocks()

    def log_stats(self, log_path: str, worker_id: int) -> None:
        with open(f'{log_path}/kv_len_{worker_id}.npy', 'wb+') as f:
            np.save(f, self.accum_kv_lens.cpu())
        with open(f'{log_path}/block_num_{worker_id}.npy', 'wb+') as f:
            np.save(f, self.accum_block_nums.cpu())
        with open(f'{log_path}/num_critical_tokens_{worker_id}.npy', 'wb+') as f:
            # print('mean num_critical_tokens = ', 
            #       torch.mean(self.accum_num_critical_tokens.float())
            #     )
            # np.save(f, self.accum_num_critical_tokens.cpu())
            np.save(f, self.accum_num_critical_tokens)


def _divide_round_up(x: int, y: int) -> int:
    return (x + y - 1) // y