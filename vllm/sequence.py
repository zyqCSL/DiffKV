"""Sequence and its related classes."""
import copy
import enum
import numpy as np
from typing import Dict, List, Optional, Union, Tuple

# from vllm.block import LogicalTokenBlock
from vllm.sampling_params import SamplingParams

PromptLogprobs = List[Optional[Dict[int, float]]]
SampleLogprobs = List[Dict[int, float]]

class SequenceStatus(enum.Enum):
    """Status of a sequence."""
    WAITING = enum.auto()
    RUNNING = enum.auto()
    SWAPPED = enum.auto()
    FINISHED_STOPPED = enum.auto()
    FINISHED_LENGTH_CAPPED = enum.auto()
    FINISHED_ABORTED = enum.auto()
    FINISHED_IGNORED = enum.auto()

    @staticmethod
    def is_finished(status: "SequenceStatus") -> bool:
        return status in [
            SequenceStatus.FINISHED_STOPPED,
            SequenceStatus.FINISHED_LENGTH_CAPPED,
            SequenceStatus.FINISHED_ABORTED,
            SequenceStatus.FINISHED_IGNORED,
        ]

    @staticmethod
    def get_finished_reason(status: "SequenceStatus") -> Union[str, None]:
        if status == SequenceStatus.FINISHED_STOPPED:
            finish_reason = "stop"
        elif status == SequenceStatus.FINISHED_LENGTH_CAPPED:
            finish_reason = "length"
        elif status == SequenceStatus.FINISHED_ABORTED:
            finish_reason = "abort"
        elif status == SequenceStatus.FINISHED_IGNORED:
            # The ignored sequences are the sequences whose prompt lengths
            # are longer than the model's length cap. Therefore, the stop
            # reason should also be "length" as in OpenAI API.
            finish_reason = "length"
        else:
            finish_reason = None
        return finish_reason


class SequenceData:
    """Data associated with a sequence.


    Args:
        prompt_token_ids: The token IDs of the prompt.

    Attributes:
        prompt_token_ids: The token IDs of the prompt.
        output_token_ids: The token IDs of the output.
        cumulative_logprob: The cumulative log probability of the output.
    """

    def __init__(
        self,
        prompt_token_ids: List[int],
    ) -> None:
        self.prompt_token_ids = prompt_token_ids
        self.output_token_ids: List[int] = []
        self.cumulative_logprob = 0.0

    def append_token_id(self, token_id: int, logprob: float) -> None:
        self.output_token_ids.append(token_id)
        self.cumulative_logprob += logprob

    def get_len(self) -> int:
        return len(self.output_token_ids) + len(self.prompt_token_ids)

    def get_prompt_len(self) -> int:
        return len(self.prompt_token_ids)

    def get_output_len(self) -> int:
        return len(self.output_token_ids)

    def get_token_ids(self) -> List[int]:
        return self.prompt_token_ids + self.output_token_ids

    def get_last_token_id(self) -> int:
        if not self.output_token_ids:
            return self.prompt_token_ids[-1]
        return self.output_token_ids[-1]

    def __repr__(self) -> str:
        return (f"SequenceData("
                f"prompt_token_ids={self.prompt_token_ids}, "
                f"output_token_ids={self.output_token_ids}, "
                f"cumulative_logprob={self.cumulative_logprob})")



class Sequence:
    """Stores the data, status, and block information of a sequence.

    Args:
        seq_id: The ID of the sequence.
        prompt: The prompt of the sequence.
        prompt_token_ids: The token IDs of the prompt.
        block_size: The block size of the sequence. Should be the same as the
            block size used by the block manager and cache engine.
    """

    def __init__(
        self,
        seq_id: int,
        prompt: str,
        prompt_token_ids: List[int],
        block_size: int,
    ) -> None:
        self.seq_id = seq_id
        self.prompt = prompt
        self.block_size = block_size

        self.data = SequenceData(prompt_token_ids)
        self.output_logprobs: SampleLogprobs = []
        self.output_text = ""
        
        # only for sequences emulating prompt sparsity
        self.emulated_output_text = ""

        # # logical_token_blocks need to be per-head and per-layer
        # self.logical_token_blocks: List[LogicalTokenBlock] = []
        # # Initialize the logical token blocks with the prompt token ids.
        # self._append_tokens_to_blocks(prompt_token_ids)
        self.status = SequenceStatus.WAITING

        # Used for incremental detokenization
        self.prefix_offset = 0
        self.read_offset = 0
        # Input + output tokens
        self.tokens: Optional[List[str]] = None

    # def _append_logical_block(self) -> None:
    #     block = LogicalTokenBlock(
    #         block_number=len(self.logical_token_blocks),
    #         block_size=self.block_size,
    #     )
    #     self.logical_token_blocks.append(block)

    # def _append_tokens_to_blocks(self, token_ids: List[int]) -> None:
    #     cursor = 0
    #     while cursor < len(token_ids):
    #         if not self.logical_token_blocks:
    #             self._append_logical_block()

    #         last_block = self.logical_token_blocks[-1]
    #         if last_block.is_full():
    #             self._append_logical_block()
    #             last_block = self.logical_token_blocks[-1]

    #         num_empty_slots = last_block.get_num_empty_slots()
    #         last_block.append_tokens(token_ids[cursor:cursor +
    #                                            num_empty_slots])
    #         cursor += num_empty_slots

    def append_token_id(
        self,
        token_id: int,
        logprobs: Dict[int, float],
    ) -> None:
        assert token_id in logprobs
        # self._append_tokens_to_blocks([token_id])
        self.output_logprobs.append(logprobs)
        self.data.append_token_id(token_id, logprobs[token_id])

    def get_len(self) -> int:
        return self.data.get_len()

    def get_prompt_len(self) -> int:
        return self.data.get_prompt_len()

    def get_output_len(self) -> int:
        return self.data.get_output_len()

    def get_token_ids(self) -> List[int]:
        return self.data.get_token_ids()

    def get_last_token_id(self) -> int:
        return self.data.get_last_token_id()

    def get_output_token_ids(self) -> List[int]:
        return self.data.output_token_ids

    def get_cumulative_logprob(self) -> float:
        return self.data.cumulative_logprob

    def get_beam_search_score(self,
                              length_penalty: float = 0.0,
                              seq_len: Optional[int] = None,
                              eos_token_id: Optional[int] = None) -> float:
        """Calculate the beam search score with length penalty.

        Adapted from

        https://github.com/huggingface/transformers/blob/ccb92be23def445f2afdea94c31286f84b89eb5b/src/transformers/generation/beam_search.py#L938
        """
        if seq_len is None:
            seq_len = self.get_len()
            # NOTE: HF implementation does not count the EOS token
            # towards the length, we align with that here for testing.
            if (eos_token_id is not None
                    and self.get_last_token_id() == eos_token_id):
                seq_len -= 1
        return self.get_cumulative_logprob() / (seq_len**length_penalty)

    def is_finished(self) -> bool:
        return SequenceStatus.is_finished(self.status)

    def fork(self, new_seq_id: int) -> "Sequence":
        new_seq = copy.deepcopy(self)
        new_seq.seq_id = new_seq_id
        return new_seq

    def __repr__(self) -> str:
        return (f"Sequence(seq_id={self.seq_id}, "
                f"status={self.status.name}")


class SequenceGroup:
    """A group of sequences that are generated from the same prompt.

    Args:
        request_id: The ID of the request.
        seqs: The list of sequences.
        sampling_params: The sampling parameters used to generate the outputs.
        arrival_time: The arrival time of the request.
    """

    def __init__(
        self,
        request_id: str,
        seqs: List[Sequence],
        sampling_params: SamplingParams,
        arrival_time: float,
        quant_configs: List[int],
        quant_groups: List[int],
        compress_configs: List[float],
    ) -> None:
        self.request_id = request_id
        self.seqs_dict = {seq.seq_id: seq for seq in seqs}
        self.sampling_params = sampling_params
        self.arrival_time = arrival_time
        self.prompt_logprobs: Optional[PromptLogprobs] = None
        
        # used in scheduler to decide how much memory is allocated to the group
        self.is_decode = False
        
        # kv pruning configs
        # assert len(compress_configs) == 4
        # _prune_thresh, _quant_thresh, _prune_ratio, _quant_ratio = compress_configs
        # assert _prune_thresh >= 0 and _prune_thresh <= 1
        # assert _quant_thresh >= 0 and _quant_thresh <= _prune_thresh
        # assert _prune_ratio >= 0 and _prune_ratio <= 1
        # assert _quant_ratio >= 0 and _quant_ratio <= _prune_ratio
        # self.compress_configs = compress_configs
        
        assert len(compress_configs) == 2
        _prune_ratio, _quant_ratio = compress_configs
        
        # print(_prune_ratio, _quant_ratio)
        
        assert _prune_ratio >= 0 and _prune_ratio <= 1
        assert _quant_ratio >= 0 and _quant_ratio >= _prune_ratio        
        self.compress_configs = compress_configs
        
        # kv quantization configs
        assert len(quant_configs) == 2 or len(quant_configs) == 4
        assert len(quant_configs) == len(quant_groups)
        self.quant_configs = tuple(quant_configs)
        self.quant_groups = tuple(quant_groups)
        # pad the config if only 1 quant config is used
        if len(self.quant_configs) == 2:
            # self.quant_configs += [(0, 0)]
            self.quant_configs += self.quant_configs
            self.quant_groups += self.quant_groups
            # assert _prune_thresh == _quant_thresh
            assert _prune_ratio == _quant_ratio
        else:
            assert self.quant_configs[0] != self.quant_configs[2] or \
                self.quant_configs[1] != self.quant_configs[3]
            assert self.quant_configs[0] >= self.quant_configs[2]     # key
            assert self.quant_configs[1] >= self.quant_configs[3]     # value
        

    @property
    def prompt(self) -> str:
        # All sequences in the group should have the same prompt.
        # We use the prompt of an arbitrary sequence.
        return next(iter(self.seqs_dict.values())).prompt

    @property
    def prompt_token_ids(self) -> List[int]:
        # All sequences in the group should have the same prompt.
        # We use the prompt of an arbitrary sequence.
        return next(iter(self.seqs_dict.values())).data.prompt_token_ids

    def get_max_num_running_seqs(self) -> int:
        """The maximum number of sequences running in parallel in the remaining
        lifetime of the request."""
        if self.sampling_params.use_beam_search:
            # For beam search, maximally there will always be `best_of` beam
            # candidates running in the future.
            return self.sampling_params.best_of
        else:
            if self.sampling_params.best_of > self.num_seqs():
                # At prompt stage, the sequence group is not yet filled up
                # and only have one sequence running. However, in the
                # generation stage, we will have `best_of` sequences running.
                return self.sampling_params.best_of
            # At sampling stages, return the number of actual sequences
            # that are not finished yet.
            return self.num_unfinished_seqs()

    def get_seqs(
        self,
        status: Optional[SequenceStatus] = None,
    ) -> List[Sequence]:
        if status is None:
            return list(self.seqs_dict.values())
        else:
            return [
                seq for seq in self.seqs_dict.values() if seq.status == status
            ]
    
    def get_seq_ids(
        self,
        status: Optional[SequenceStatus] = None,
    ) -> List[int]:
        if status is None:
            return list(self.seqs_dict.keys())
        else:
            return [
                seq_id for seq_id in self.seqs_dict.keys() if self.seqs_dict[seq_id].status == status
            ]

    def get_unfinished_seqs(self) -> List[Sequence]:
        return [
            seq for seq in self.seqs_dict.values() if not seq.is_finished()
        ]

    def get_finished_seqs(self) -> List[Sequence]:
        return [seq for seq in self.seqs_dict.values() if seq.is_finished()]

    def num_seqs(self, status: Optional[SequenceStatus] = None) -> int:
        return len(self.get_seqs(status))

    def num_unfinished_seqs(self) -> int:
        return len(self.get_unfinished_seqs())

    def num_finished_seqs(self) -> int:
        return len(self.get_finished_seqs())

    def find(self, seq_id: int) -> Sequence:
        if seq_id not in self.seqs_dict:
            raise ValueError(f"Sequence {seq_id} not found.")
        return self.seqs_dict[seq_id]

    def add(self, seq: Sequence) -> None:
        if seq.seq_id in self.seqs_dict:
            raise ValueError(f"Sequence {seq.seq_id} already exists.")
        self.seqs_dict[seq.seq_id] = seq

    def remove(self, seq_id: int) -> None:
        if seq_id not in self.seqs_dict:
            raise ValueError(f"Sequence {seq_id} not found.")
        del self.seqs_dict[seq_id]

    def is_finished(self) -> bool:
        return all(seq.is_finished() for seq in self.get_seqs())

    def __repr__(self) -> str:
        return (f"SequenceGroup(request_id={self.request_id}, "
                f"sampling_params={self.sampling_params}, "
                f"num_seqs={len(self.seqs_dict)})")


class SequenceGroupMetadata:
    """Metadata for a sequence group. Used to create `InputMetadata`.


    Args:
        request_id: The ID of the request.
        is_prompt: Whether the request is at prompt stage.
        seq_data: The sequence data. (Seq id -> sequence data)
        sampling_params: The sampling parameters used to generate the outputs.
    """

    def __init__(
        self,
        request_id: str,
        is_prompt: bool,
        seq_data: Dict[int, SequenceData],
        sampling_params: SamplingParams,
        num_bits_k_high: Optional[int],
        num_bits_v_high: Optional[int],
        num_bits_k_low: Optional[int],
        num_bits_v_low: Optional[int],
        num_chunks_k_high: Optional[int],
        num_chunks_v_high: Optional[int],
        num_chunks_k_low: Optional[int],
        num_chunks_v_low: Optional[int],
    ) -> None:
        '''
        NOTE: block_tables & kv_lens should be set on the worker side
        '''
        self.request_id = request_id
        self.is_prompt = is_prompt
        self.seq_data = seq_data
        self.sampling_params = sampling_params
        self.slot_ids: List[int] = []
        # quant config
        self.num_bits_k_high = num_bits_k_high
        self.num_bits_v_high = num_bits_v_high
        self.num_bits_k_low  = num_bits_k_low
        self.num_bits_v_low  = num_bits_v_low
        
        self.num_chunks_k_high = num_chunks_k_high
        self.num_chunks_v_high = num_chunks_v_high
        self.num_chunks_k_low  = num_chunks_k_low
        self.num_chunks_v_low  = num_chunks_v_low


class SequenceOutput:
    """The model output associated with a sequence.

    Args:
        parent_seq_id: The ID of the parent sequence (for forking in beam
            search).
        output_token: The output token ID.
        logprobs: The logprobs of the output token.
            (Token id -> logP(x_i+1 | x_0, ..., x_i))
    """

    def __init__(
        self,
        parent_seq_id: int,
        output_token: int,
        logprobs: Dict[int, float],
    ) -> None:
        self.parent_seq_id = parent_seq_id
        self.output_token = output_token
        self.logprobs = logprobs

    def __repr__(self) -> str:
        return (f"SequenceOutput(parent_seq_id={self.parent_seq_id}, "
                f"output_token={self.output_token}, "
                f"logprobs={self.logprobs})")

    # def __eq__(self, other: object) -> bool:
    #     if not isinstance(other, SequenceOutput):
    #         raise NotImplementedError()
    #     return (self.parent_seq_id == other.parent_seq_id
    #             and self.output_token == other.output_token
    #             and self.logprobs == other.logprobs)
    
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, SequenceOutput):
            raise NotImplementedError()
        
        # step 1. check parent seq and output tokens
        _eq_output = (self.parent_seq_id == other.parent_seq_id
                    and self.output_token == other.output_token)
        _eq_logprobs = True
        for token_id, logprob in self.logprobs.items():
            if token_id not in other.logprobs:
                _eq_logprobs = False
                break
            if np.isnan(logprob) and np.isnan(other.logprobs[token_id]):
                continue
            if not np.isclose(logprob, other.logprobs[token_id], atol=1e-5):
                _eq_logprobs = False
                break
        
        return _eq_output and _eq_logprobs


class SequenceGroupOutput:
    """The model output associated with a sequence group."""

    def __init__(
        self,
        samples: List[SequenceOutput],
        prompt_logprobs: Optional[PromptLogprobs],
    ) -> None:
        self.samples = samples
        self.prompt_logprobs = prompt_logprobs

    def __repr__(self) -> str:
        return (f"SequenceGroupOutput(samples={self.samples}, "
                f"prompt_logprobs={self.prompt_logprobs})")

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, SequenceGroupOutput):
            raise NotImplementedError()
        return self.samples == other.samples
        # return (self.samples == other.samples
        #         and self.prompt_logprobs == other.prompt_logprobs)


# For each sequence group, we generate a list of SequenceOutput object,
# each of which contains one possible candidate for the next token.
SamplerOutput = List[SequenceGroupOutput]
