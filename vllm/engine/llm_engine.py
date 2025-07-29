import time
from typing import TYPE_CHECKING, Any, Iterable, List, Optional, Tuple, Union

from vllm.config import (CacheConfig, ModelConfig, ParallelConfig,
                         SchedulerConfig)
from vllm.core.scheduler import Scheduler, SchedulerOutputs
from vllm.core.orchestrator import Orchestrator
from vllm.engine.arg_utils import EngineArgs
from vllm.engine.metrics import record_metrics
from vllm.logger import init_logger
from vllm.outputs import RequestOutput
from vllm.sampling_params import SamplingParams, SamplingType
from vllm.sequence import (SamplerOutput, Sequence, SequenceGroup,
                           SequenceGroupMetadata, SequenceGroupOutput,
                           SequenceOutput, SequenceStatus)
from vllm.transformers_utils.tokenizer import (detokenize_incrementally,
                                               get_tokenizer)
from vllm.utils import Counter

logger = init_logger(__name__)

_LOGGING_INTERVAL_SEC = 5

import torch

class LLMEngine:
    """An LLM engine that receives requests and generates texts.

    This is the main class for the vLLM engine. It receives requests
    from clients and generates texts from the LLM. It includes a tokenizer, a
    language model (possibly distributed across multiple GPUs), and GPU memory
    space allocated for intermediate states (aka KV cache). This class utilizes
    iteration-level scheduling and efficient memory management to maximize the
    serving throughput.

    The `LLM` class wraps this class for offline batched inference and the
    `AsyncLLMEngine` class wraps this class for online serving.

    NOTE: The config arguments are derived from the `EngineArgs` class. For the
    comprehensive list of arguments, see `EngineArgs`.

    Args:
        model_config: The configuration related to the LLM model.
        cache_config: The configuration related to the KV cache memory
            management.
        parallel_config: The configuration related to distributed execution.
        scheduler_config: The configuration related to the request scheduler.
        distributed_init_method: The initialization method for distributed
            execution. See `torch.distributed.init_process_group` for details.
        placement_group: Ray placement group for distributed execution.
            Required for distributed execution.
        log_stats: Whether to log statistics.
    """

    def __init__(
        self,
        model_config: ModelConfig,
        cache_config: CacheConfig,
        parallel_config: ParallelConfig,
        scheduler_config: SchedulerConfig,
        log_stats: bool,
    ) -> None:
        logger.info(
            "Initializing an LLM engine with config: "
            f"model={model_config.model!r}, "
            f"tokenizer={model_config.tokenizer!r}, "
            f"tokenizer_mode={model_config.tokenizer_mode}, "
            f"revision={model_config.revision}, "
            f"tokenizer_revision={model_config.tokenizer_revision}, "
            f"trust_remote_code={model_config.trust_remote_code}, "
            f"dtype={model_config.dtype}, "
            f"max_seq_len={model_config.max_model_len}, "
            f"download_dir={model_config.download_dir!r}, "
            f"load_format={model_config.load_format}, "
            f"tensor_parallel_size={parallel_config.tensor_parallel_size}, "
            f"quantization={model_config.quantization}, "
            f"enforce_eager={model_config.enforce_eager}, "
            f"seed={model_config.seed})")
        # TODO(woosuk): Print more configs in debug mode.

        self.model_config = model_config
        self.cache_config = cache_config
        self.parallel_config = parallel_config
        self.scheduler_config = scheduler_config
        self.log_stats = log_stats
        self._verify_args()

        self.tokenizer = get_tokenizer(
            model_config.tokenizer,
            tokenizer_mode=model_config.tokenizer_mode,
            trust_remote_code=model_config.trust_remote_code,
            tokenizer_revision=model_config.tokenizer_revision,
            revision=model_config.revision)

        if len(self.tokenizer) != self.model_config.get_vocab_size():
            logger.warning(
                f"The tokenizer's vocabulary size {len(self.tokenizer)}"
                f" does not match the model's vocabulary size "
                f"{self.model_config.get_vocab_size()}. This might "
                f"cause an error in decoding. Please change config.json "
                "to match the tokenizer's vocabulary size.")
        self.tokenizer_len = len(self.tokenizer)
        self.seq_counter = Counter()

        # GPU cluster orchestration
        self.orchestrator = Orchestrator(
            model_config=self.model_config,
            cache_config=self.cache_config,
            parallel_config=self.parallel_config,
            scheduler_config=self.scheduler_config,
        )
        # Create the parallel GPU workers.
        self.orchestrator.initialize_cluster()

        # Profile the memory usage and initialize the cache.
        self._init_cache()

        # Create the scheduler.
        self.scheduler = Scheduler(
            model_config=self.model_config,
            scheduler_config=self.scheduler_config,
            cache_config=self.cache_config,
            parallel_config=self.parallel_config,
            orchestrator=self.orchestrator)

        # Logging.
        self.last_logging_time = 0.0
        # List of (timestamp, num_tokens)
        self.num_prompt_tokens: List[Tuple[float, int]] = []
        # List of (timestamp, num_tokens)
        self.num_generation_tokens: List[Tuple[float, int]] = []

    def _verify_args(self) -> None:
        self.model_config.verify_with_parallel_config(self.parallel_config)
        self.cache_config.verify_with_parallel_config(self.parallel_config)

    def _init_cache(self) -> None:
        """Profiles the memory usage and initializes the KV cache."""
        # Get the maximum number of blocks that can be allocated on GPU and CPU.
        self.cache_config.compute_cache_block_size(self.model_config)

        num_blocks = self.orchestrator.run_workers(
            "profile_num_available_blocks",
            get_all_outputs=True,
            gpu_memory_utilization=self.cache_config.gpu_memory_utilization,
            cpu_swap_space=self.cache_config.swap_space_bytes,
            cache_block_bytes=self.cache_config.block_bytes,
        )

        # per device free gpu & cpu blocks
        num_gpu_blocks = [b[0] for b in num_blocks]
        num_cpu_blocks = [b[1] for b in num_blocks]
        min_gpu_blocks = min(num_gpu_blocks)
        min_cpu_blocks = min(num_cpu_blocks)
        # FIXME(woosuk): Change to debug log.
        logger.info(f"# GPU blocks: {num_gpu_blocks}, "
                    f"# CPU blocks: {num_cpu_blocks}")

        if min_gpu_blocks <= 0:
            raise ValueError("No available memory for the cache blocks. "
                             "Try increasing `gpu_memory_utilization` when "
                             "initializing the engine.")
        max_seq_len = self.cache_config.block_size * min_gpu_blocks
        if self.model_config.max_model_len > max_seq_len:
            raise ValueError(
                f"The model's max seq len ({self.model_config.max_model_len}) "
                "is larger than the maximum number of tokens that can be "
                f"stored in KV cache ({max_seq_len}). Try increasing "
                "`gpu_memory_utilization` or decreasing `max_model_len` when "
                "initializing the engine.")

        # list of blocks for each worker
        self.cache_config.num_gpu_blocks = num_gpu_blocks
        self.cache_config.num_cpu_blocks = num_cpu_blocks

        # Initialize the cache.
        self.orchestrator.run_workers("init_cache_engine", cache_config=self.cache_config)
        # Warm up the model. This includes capturing the model into CUDA graph
        # if enforce_eager is False.
        self.orchestrator.run_workers("warm_up_model")

    @classmethod
    def from_engine_args(cls, engine_args: EngineArgs) -> "LLMEngine":
        """Creates an LLM engine from the engine arguments."""
        # Create the engine configs.
        engine_configs = engine_args.create_engine_configs()

        # Initialize the cluster moved into orchestrator's init
        # distributed_init_method, placement_group = initialize_cluster(
        #     parallel_config)
        # Create the LLM engine.
        engine = cls(*engine_configs,
                    #  distributed_init_method,
                    #  placement_group,
                     log_stats=not engine_args.disable_log_stats)
        return engine

    def add_request(
        self,
        request_id: str,
        prompt: Optional[str],
        sampling_params: SamplingParams,
        quant_configs: List[int],
        quant_groups: List[int],
        compress_configs: List[float],
        prompt_token_ids: Optional[List[int]] = None,
        arrival_time: Optional[float] = None,
    ) -> SequenceGroup:
        """Add a request to the engine's request pool.

        The request is added to the request pool and will be processed by the
        scheduler as `engine.step()` is called. The exact scheduling policy is
        determined by the scheduler.

        Args:
            request_id: The unique ID of the request.
            prompt: The prompt string. Can be None if prompt_token_ids is
                provided.
            sampling_params: The sampling parameters for text generation.
            prompt_token_ids: The token IDs of the prompt. If None, we
                use the tokenizer to convert the prompts to token IDs.
            arrival_time: The arrival time of the request. If None, we use
                the current monotonic time.
            quant_configs: KV cache quantization configs
                [kbits_high_precision, vbits_high_precision, kbits_low_precision, vbits_low_precision]
            quant_groups: KV cache quantization configs
                [kgroups_high_precision, vgroups_high_precision, kgroups_low_precision, vgroups_low_precision]
            compress_configs: KV cache compression configs (thresholds for quant & prune)
                [prune_ratio, quant_ratio]
        """
        if arrival_time is None:
            arrival_time = time.monotonic()
        if prompt_token_ids is None:
            assert prompt is not None
            prompt_token_ids = self.tokenizer.encode(prompt)
        if sampling_params.model_seq or sampling_params.emulate_seq:
            if sampling_params.truth_token_ids is None:
                sampling_params.truth_token_ids = self.tokenizer.encode(
                    sampling_params.truth)
            assert sampling_params.truth_token_ids[:
                len(prompt_token_ids)] == prompt_token_ids

        # Create the sequences.
        block_size = self.cache_config.block_size
        seq_id = next(self.seq_counter)
        seq = Sequence(seq_id, prompt, prompt_token_ids, block_size)

        # Create the sequence group.
        seq_group = SequenceGroup(request_id, [seq], sampling_params,
                                  arrival_time,
                                  quant_configs, quant_groups, compress_configs)

        # Add the sequence group to the scheduler.
        self.scheduler.add_seq_group(seq_group)
        # NOTE: return seq_group to keep track of dataset stats
        return seq_group

    def abort_request(self, request_id: Union[str, Iterable[str]]) -> None:
        """Aborts a request(s) with the given ID.

        Args:
            request_id: The ID(s) of the request to abort.
        """
        self.scheduler.abort_seq_group(request_id)

    def get_model_config(self) -> ModelConfig:
        """Gets the model configuration."""
        return self.model_config

    def get_num_unfinished_requests(self) -> int:
        """Gets the number of unfinished requests."""
        return self.scheduler.get_num_unfinished_seq_groups()

    def has_unfinished_requests(self) -> bool:
        """Returns True if there are unfinished requests."""
        return self.scheduler.has_unfinished_seqs()

    def _schedule(
        self
    ) -> Tuple[List[SequenceGroupMetadata], SchedulerOutputs,
               List[RequestOutput]]:
        seq_group_metadata_list, scheduler_outputs = self.scheduler.schedule()
        return seq_group_metadata_list, scheduler_outputs, [
            RequestOutput.from_seq_group(seq_group)
            for seq_group in scheduler_outputs.ignored_seq_groups
        ]

    def _check_beam_search_early_stopping(
        self,
        early_stopping: Union[bool, str],
        sampling_params: SamplingParams,
        best_running_seq: Sequence,
        current_worst_seq: Sequence,
    ) -> bool:
        assert sampling_params.use_beam_search
        length_penalty = sampling_params.length_penalty
        if early_stopping is True:
            return True

        current_worst_score = (current_worst_seq.get_beam_search_score(
            length_penalty=length_penalty,
            eos_token_id=self.tokenizer.eos_token_id))
        if early_stopping is False:
            highest_attainable_score = (best_running_seq.get_beam_search_score(
                length_penalty=length_penalty,
                eos_token_id=self.tokenizer.eos_token_id))
        else:
            assert early_stopping == "never"
            if length_penalty > 0.0:
                # If length_penalty > 0.0, beam search will prefer longer
                # sequences. The highest attainable score calculation is
                # based on the longest possible sequence length in this case.
                max_possible_length = max(
                    best_running_seq.get_prompt_len() +
                    sampling_params.max_tokens,
                    self.scheduler_config.max_model_len)
                highest_attainable_score = (
                    best_running_seq.get_beam_search_score(
                        length_penalty=length_penalty,
                        eos_token_id=self.tokenizer.eos_token_id,
                        seq_len=max_possible_length))
            else:
                # Otherwise, beam search will prefer shorter sequences. The
                # highest attainable score calculation is based on the current
                # sequence length.
                highest_attainable_score = (
                    best_running_seq.get_beam_search_score(
                        length_penalty=length_penalty,
                        eos_token_id=self.tokenizer.eos_token_id))
        return current_worst_score >= highest_attainable_score

    def _process_sequence_group_outputs(self, seq_group: SequenceGroup,
                                        outputs: SequenceGroupOutput) -> None:
        # Process prompt logprobs
        prompt_logprobs = outputs.prompt_logprobs
        if prompt_logprobs is not None:
            seq_group.prompt_logprobs = prompt_logprobs

        # Process samples
        samples = outputs.samples
        parent_seqs = seq_group.get_seqs(status=SequenceStatus.RUNNING)
        existing_finished_seqs = seq_group.get_finished_seqs()
        parent_child_dict = {
            parent_seq.seq_id: []
            for parent_seq in parent_seqs
        }
        for sample in samples:
            parent_child_dict[sample.parent_seq_id].append(sample)
        # List of (child, parent)
        child_seqs: List[Tuple[Sequence, Sequence]] = []

        # Process the child samples for each parent sequence
        for parent in parent_seqs:
            child_samples: List[SequenceOutput] = parent_child_dict[
                parent.seq_id]
            if len(child_samples) == 0:
                # This parent sequence has no children samples. Remove
                # the parent sequence from the sequence group since it will
                # not be used in the future iterations.
                parent.status = SequenceStatus.FINISHED_ABORTED
                seq_group.remove(parent.seq_id)
                self.scheduler.free_seq(parent)
                continue
            # Fork the parent sequence if there are multiple child samples.
            for child_sample in child_samples[:-1]:
                new_child_seq_id = next(self.seq_counter)
                child = parent.fork(new_child_seq_id)
                child.append_token_id(child_sample.output_token,
                                      child_sample.logprobs)
                child_seqs.append((child, parent))
            # Continue the parent sequence for the last child sample.
            # We reuse the parent sequence here to reduce redundant memory
            # copies, especially when using non-beam search sampling methods.
            last_child_sample = child_samples[-1]
            parent.append_token_id(last_child_sample.output_token,
                                   last_child_sample.logprobs)
            child_seqs.append((parent, parent))

        for seq, _ in child_seqs:
            self._decode_sequence(seq, seq_group.sampling_params)
            self._check_stop(seq, seq_group.sampling_params)

        # Non-beam search case
        if not seq_group.sampling_params.use_beam_search:
            # For newly created child sequences, add them to the sequence group
            # and fork them in block manager if they are not finished.
            for seq, parent in child_seqs:
                if seq is not parent:
                    seq_group.add(seq)
                    if not seq.is_finished():
                        self.scheduler.fork_seq(parent, seq)

            # Free the finished and selected parent sequences' memory in block
            # manager. Keep them in the sequence group as candidate output.
            # NOTE: we need to fork the new sequences before freeing the
            # old sequences.
            for seq, parent in child_seqs:
                if seq is parent and seq.is_finished():
                    self.scheduler.free_seq(seq)
            return

        # Beam search case
        # Select the child sequences to keep in the sequence group.
        selected_child_seqs = []
        unselected_child_seqs = []
        beam_width = seq_group.sampling_params.best_of
        length_penalty = seq_group.sampling_params.length_penalty

        # Select the newly finished sequences with the highest scores
        # to replace existing finished sequences.
        # Tuple of (seq, parent, is_new)
        existing_finished_seqs = [(seq, None, False)
                                  for seq in existing_finished_seqs]
        new_finished_seqs = [(seq, parent, True) for seq, parent in child_seqs
                             if seq.is_finished()]
        all_finished_seqs = existing_finished_seqs + new_finished_seqs
        # Sort the finished sequences by their scores.
        all_finished_seqs.sort(key=lambda x: x[0].get_beam_search_score(
            length_penalty=length_penalty,
            eos_token_id=self.tokenizer.eos_token_id),
                               reverse=True)
        for seq, parent, is_new in all_finished_seqs[:beam_width]:
            if is_new:
                # A newly generated child sequence finishes and has a high
                # score, so we will add it into the sequence group.
                selected_child_seqs.append((seq, parent))
        for seq, parent, is_new in all_finished_seqs[beam_width:]:
            if is_new:
                # A newly generated child sequence finishes but has a low
                # score, so we will not add it into the sequence group.
                # Additionally, if this sequence is a continuation of a
                # parent sequence, we will need remove the parent sequence
                # from the sequence group.
                unselected_child_seqs.append((seq, parent))
            else:
                # An existing finished sequence has a low score, so we will
                # remove it from the sequence group.
                seq_group.remove(seq.seq_id)

        # select the top beam_width sequences from the running
        # sequences for the next iteration to continue the beam
        # search.
        running_child_seqs = [(seq, parent) for seq, parent in child_seqs
                              if not seq.is_finished()]
        # Sort the running sequences by their scores.
        running_child_seqs.sort(key=lambda x: x[0].get_beam_search_score(
            length_penalty=length_penalty,
            eos_token_id=self.tokenizer.eos_token_id),
                                reverse=True)

        # Check if we can stop the beam search.
        if len(running_child_seqs) == 0:
            # No running sequences, stop the beam search.
            stop_beam_search = True
        elif len(all_finished_seqs) < beam_width:
            # Not enough finished sequences, continue the beam search.
            stop_beam_search = False
        else:
            # Check the early stopping criteria
            best_running_seq = running_child_seqs[0][0]
            current_worst_seq = all_finished_seqs[beam_width - 1][0]
            stop_beam_search = self._check_beam_search_early_stopping(
                seq_group.sampling_params.early_stopping,
                seq_group.sampling_params, best_running_seq, current_worst_seq)

        if stop_beam_search:
            # Stop the beam search and remove all the running sequences from
            # the sequence group.
            unselected_child_seqs.extend(running_child_seqs)
        else:
            # Continue the beam search and select the top beam_width sequences
            # to continue the beam search.
            selected_child_seqs.extend(running_child_seqs[:beam_width])
            # The remaining running sequences will not be used in the next
            # iteration. Again, if these sequences are continuations of
            # parent sequences, we will need to remove the parent sequences
            # from the sequence group.
            unselected_child_seqs.extend(running_child_seqs[beam_width:])

        # For newly created child sequences, add them to the sequence group
        # and fork them in block manager if they are not finished.
        for seq, parent in selected_child_seqs:
            if seq is not parent:
                seq_group.add(seq)
                if not seq.is_finished():
                    self.scheduler.fork_seq(parent, seq)

        # Free the finished and selected parent sequences' memory in block
        # manager. Keep them in the sequence group as candidate output.
        for seq, parent in selected_child_seqs:
            if seq is parent and seq.is_finished():
                self.scheduler.free_seq(seq)

        # Remove the unselected parent sequences from the sequence group and
        # free their memory in block manager.
        for seq, parent in unselected_child_seqs:
            if seq is parent:
                # Remove the parent sequence if it is not selected for next
                # iteration
                seq_group.remove(seq.seq_id)
                self.scheduler.free_seq(seq)

    def _process_model_outputs(
            self, output: SamplerOutput,
            scheduler_outputs: SchedulerOutputs) -> List[RequestOutput]:
        # Update the scheduled sequence groups with the model outputs.
        scheduled_seq_groups = scheduler_outputs.scheduled_seq_groups
        for seq_group, outputs in zip(scheduled_seq_groups, output):
            self._process_sequence_group_outputs(seq_group, outputs)

        # Free the finished sequence groups.
        self.scheduler.free_finished_seq_groups()

        # Create the outputs.
        request_outputs: List[RequestOutput] = []
        for seq_group in (scheduled_seq_groups +
                          scheduler_outputs.ignored_seq_groups):
            request_output = RequestOutput.from_seq_group(seq_group)
            request_outputs.append(request_output)

        if self.log_stats:
            # Log the system stats.
            self._log_system_stats(scheduler_outputs.prompt_run,
                                   scheduler_outputs.num_batched_tokens)
        
        
        # print('[DEBUG] LLMEngine _process_model_outputs done')
        
        return request_outputs

    def step(self) -> List[RequestOutput]:
        """Performs one decoding iteration and returns newly generated results.

        This function performs one decoding iteration of the engine. It first
        schedules the sequences to be executed in the next iteration and the
        token blocks to be swapped in/out/copy. Then, it executes the model
        and updates the scheduler with the model outputs. Finally, it decodes
        the sequences and returns the newly generated results.
        """
        # print('\n*********** STEP ************')

        torch.cuda.synchronize()
        t0 = time.time()

        seq_group_metadata_list, scheduler_outputs, ignored = self._schedule()
        if scheduler_outputs.is_empty():
            return ignored

        torch.cuda.synchronize()
        t1 = time.time()
        # print(f'_schedule: {(1000 * (t1 - t0)):.2f} ms')

        # Execute the model.
        all_outputs = self.orchestrator.run_workers(
            "execute_model",
            get_all_outputs=True,
            seq_group_metadata_list=seq_group_metadata_list,
        )

        torch.cuda.synchronize()
        t2 = time.time()
        # print(f'execute_model: {(1000 * (t2 - t1)):.2f} ms')

        # update metadata of the centralized memory manager
        free_gpu_blocks = [x[1] for x in all_outputs]
        self.scheduler.memory_manager.update_free_gpu_blocks(free_gpu_blocks)
        free_cpu_blocks = [x[2] for x in all_outputs]
        self.scheduler.memory_manager.update_free_cpu_blocks(free_cpu_blocks)

        torch.cuda.synchronize()
        t3 = time.time()
        # print(f'update_free_blocks: {(1000 * (t3 - t2)):.2f} ms')

        model_output = all_outputs[0][0]
        # make sure all workers have the same model output
        for other_output in all_outputs[1:]:
            if model_output != other_output[0]:
                print(f'model_output = {model_output}')
                print(f'other_output = {other_output}')
                for seq_group in scheduler_outputs.scheduled_seq_groups:
                    for seq_id in seq_group.seqs_dict:
                        _seq = seq_group.seqs_dict[seq_id]
                        print(f'seq {seq_id} prompt = {_seq.prompt}, output = {_seq.output_text}, recent tokens = {_seq.data.output_token_ids[-50:]}')
            assert model_output == other_output[0]
        return self._process_model_outputs(model_output, scheduler_outputs)

    def log_sparse_kv_stats(self, log_path: str) -> None:
        self.scheduler.log_stats(log_path)

    def _log_system_stats(
        self,
        prompt_run: bool,
        num_batched_tokens: int,
    ) -> None:
        now = time.monotonic()
        # Log the number of batched input tokens.
        if prompt_run:
            self.num_prompt_tokens.append((now, num_batched_tokens))
        else:
            self.num_generation_tokens.append((now, num_batched_tokens))

        should_log = now - self.last_logging_time >= _LOGGING_INTERVAL_SEC
        if not should_log:
            return

        # Discard the old stats.
        self.num_prompt_tokens = [(t, n) for t, n in self.num_prompt_tokens
                                  if now - t < _LOGGING_INTERVAL_SEC]
        self.num_generation_tokens = [(t, n)
                                      for t, n in self.num_generation_tokens
                                      if now - t < _LOGGING_INTERVAL_SEC]

        if len(self.num_prompt_tokens) > 1:
            total_num_tokens = sum(n for _, n in self.num_prompt_tokens[:-1])
            window = now - self.num_prompt_tokens[0][0]
            avg_prompt_throughput = total_num_tokens / window
        else:
            avg_prompt_throughput = 0.0
        if len(self.num_generation_tokens) > 1:
            total_num_tokens = sum(n
                                   for _, n in self.num_generation_tokens[:-1])
            window = now - self.num_generation_tokens[0][0]
            avg_generation_throughput = total_num_tokens / window
        else:
            avg_generation_throughput = 0.0

        total_num_gpu_blocks = sum(self.cache_config.num_gpu_blocks)
        num_free_gpu_blocks = \
            self.scheduler.memory_manager.total_num_free_gpu_blocks()
        num_used_gpu_blocks = total_num_gpu_blocks - num_free_gpu_blocks
        gpu_cache_usage = num_used_gpu_blocks / total_num_gpu_blocks

        total_num_cpu_blocks = sum(self.cache_config.num_cpu_blocks)
        if total_num_cpu_blocks > 0:
            num_free_cpu_blocks = \
                self.scheduler.memory_manager.total_num_free_cpu_blocks()
            num_used_cpu_blocks = total_num_cpu_blocks - num_free_cpu_blocks
            cpu_cache_usage = num_used_cpu_blocks / total_num_cpu_blocks
        else:
            cpu_cache_usage = 0.0

        record_metrics(
            avg_prompt_throughput=avg_prompt_throughput,
            avg_generation_throughput=avg_generation_throughput,
            scheduler_running=len(self.scheduler.running),
            scheduler_swapped=len(self.scheduler.swapped),
            scheduler_waiting=len(self.scheduler.waiting),
            gpu_cache_usage=gpu_cache_usage,
            cpu_cache_usage=cpu_cache_usage,
        )

        logger.info("Avg prompt throughput: "
                    f"{avg_prompt_throughput:.1f} tokens/s, "
                    "Avg generation throughput: "
                    f"{avg_generation_throughput:.1f} tokens/s, "
                    f"Running: {len(self.scheduler.running)} reqs, "
                    f"Swapped: {len(self.scheduler.swapped)} reqs, "
                    f"Pending: {len(self.scheduler.waiting)} reqs, "
                    f"GPU KV cache usage: {gpu_cache_usage * 100:.1f}%, "
                    f"CPU KV cache usage: {cpu_cache_usage * 100:.1f}%")
        self.last_logging_time = now

    def _decode_sequence(self, seq: Sequence, prms: SamplingParams) -> None:
        """Decodes the new token for a sequence."""
        (new_tokens, new_output_text, prefix_offset,
         read_offset) = detokenize_incrementally(
             self.tokenizer,
             self.tokenizer_len,
             all_input_ids=seq.get_token_ids(),
             prev_tokens=seq.tokens,
             prefix_offset=seq.prefix_offset,
             read_offset=seq.read_offset,
             skip_special_tokens=prms.skip_special_tokens,
             spaces_between_special_tokens=prms.spaces_between_special_tokens,
         )
        if seq.tokens is None:
            seq.tokens = new_tokens
        else:
            seq.tokens.extend(new_tokens)
        seq.prefix_offset = prefix_offset
        seq.read_offset = read_offset
        seq.output_text += new_output_text
        
        if prms.sampling_type == SamplingType.EMULATE and \
            len(seq.tokens) >= len(prms.truth_token_ids):
            seq.emulated_output_text += new_output_text

        # # TODO(yanqi): for debug purposes. Remove later
        # print(f'seq_id: {seq.seq_id}, new_tokens: {new_tokens}, recent texts: {seq.output_text[-5:]}')

    def _check_stop(self, seq: Sequence,
                    sampling_params: SamplingParams) -> None:        
        """Stop the finished sequences."""
        if sampling_params.truth_token_ids is not None and \
            seq.data.get_len() < len(sampling_params.truth_token_ids):
            # print(f'Not stopped because length not met')
            return
        
        for stop_str in sampling_params.stop:
            if seq.output_text.endswith(stop_str):
                # print(f'!!!!!!!!! {stop_str} !!!!!!!!!!!!!')
                if not sampling_params.include_stop_str_in_output:
                    # Truncate the output text so that the stop string is
                    # not included in the output.
                    seq.output_text = seq.output_text[:-len(stop_str)]
                seq.status = SequenceStatus.FINISHED_STOPPED
                return
        if seq.get_last_token_id() in sampling_params.stop_token_ids:
            seq.status = SequenceStatus.FINISHED_STOPPED
            return

        # NOTE: Check if the sequence has reached identical length as the ground truth
        if sampling_params.model_seq and seq.get_len() == len(sampling_params.truth_token_ids):
            seq.status = SequenceStatus.FINISHED_STOPPED
            return

        # Check if the sequence has reached max_model_len.
        if seq.get_len() > self.scheduler_config.max_model_len:
            seq.status = SequenceStatus.FINISHED_LENGTH_CAPPED
            return

        # Check if the sequence has reached max_tokens.
        # print(f'output_len = {seq.get_output_len()}, max_tokens = {sampling_params.max_tokens}')
        
        if sampling_params.emulate_seq:
            if seq.get_len() - len(sampling_params.truth_token_ids) >= sampling_params.max_tokens:
                seq.status = SequenceStatus.FINISHED_LENGTH_CAPPED
                return
        elif seq.get_output_len() >= sampling_params.max_tokens:
            seq.status = SequenceStatus.FINISHED_LENGTH_CAPPED
            return

        # Check if the sequence has generated the EOS token.
        if ((not sampling_params.ignore_eos)
                and seq.get_last_token_id() == self.tokenizer.eos_token_id):
            seq.status = SequenceStatus.FINISHED_STOPPED
            return