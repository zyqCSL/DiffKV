from typing import List, Optional

import torch

class InputMetadata:
    """Metadata for input sequences. Used in SparsePagedAttention.
    Args:
        The context length varies across layers and heads
        prompt_lens: Lengths of prompts (full sequence).
        seq_start_loc: The cumulative sequence lengths of the sequences in the batch, used to index into sequence.
                       # E.g., if the sequence length is [4, 6], seq_start_loc is [0, 4, 10].
        max_context_len: The maximum context length of each layer.
        context_lens: Context lengths of each head in each layer (a list).
            Tensor shape = [batch_size, num_heads], the length of attention context for each sequence and each head.
        block_tables: pointer to gpu-native block tables
        kv_len_tables: pointer to gpu-native kv_len_tables
        block_num_tables: pointer to gpu-native block_num_tables
    """

    def __init__(
        self,
        slot_ids: List[int],
        prompt_lens: List[int],
        seq_start_loc: Optional[torch.Tensor],
        max_context_len: Optional[torch.Tensor],
        use_cuda_graph: bool,
        block_size: Optional[int],
        block_tables: Optional[torch.Tensor],
        kv_len_tables: Optional[torch.Tensor],
        # kv cache quant configs
        num_bits_k_high: int,
        num_bits_v_high: int,
        num_bits_k_low: int,
        num_bits_v_low: int,
        num_chunks_k_high: int,
        num_chunks_v_high: int,
        num_chunks_k_low: int,
        num_chunks_v_low: int,
        compress_config_tables: Optional[torch.Tensor],
        # cache block layout
        key_vec_size: int,
        val_vec_size: int,
        num_tokens_per_block_high: int,
        num_tokens_per_block_low: int,
    ) -> None:
        self.slot_ids = slot_ids
        self.prompt_lens = prompt_lens
        self.seq_start_loc = seq_start_loc
        self.max_context_len = max_context_len
        self.use_cuda_graph = use_cuda_graph

        # memory metadata
        self.block_size = block_size
        self.block_tables = block_tables
        self.kv_len_tables = kv_len_tables
        
        # kv cache quant & pruning configs
        self.num_bits_k_high = num_bits_k_high
        self.num_bits_v_high = num_bits_v_high
        self.num_bits_k_low  = num_bits_k_low
        self.num_bits_v_low  = num_bits_v_low
        self.num_chunks_k_high = num_chunks_k_high
        self.num_chunks_v_high = num_chunks_v_high
        self.num_chunks_k_low  = num_chunks_k_low
        self.num_chunks_v_low  = num_chunks_v_low
        
        self.compress_config_tables = compress_config_tables

        # cache block layout
        self.key_vec_size = key_vec_size
        self.val_vec_size = val_vec_size

        self.num_tokens_per_block_high = num_tokens_per_block_high
        self.num_tokens_per_block_low = num_tokens_per_block_low

        self.is_prompt = len(prompt_lens) > 0
        # Set during the execution of the first attention op.
        # FIXME(woosuk): This is a hack.
        self.attn_bias = None

    def __repr__(self) -> str:
        return ("InputMetadata("
                f"slot_ids={self.slot_ids}, "
                f"prompt_lens={self.prompt_lens}, "
                f"seq_start_loc={self.seq_start_loc}, "
                f"max_context_len={self.max_context_len}, "
                # f"context_lens={self.context_lens}, "
                f"block_size={self.block_size}, "
                f"num_bits_k_high={self.num_bits_k_high}, "
                f"num_bits_v_high={self.num_bits_v_high}, "
                f"num_bits_k_low={self.num_bits_k_low}, "
                f"num_bits_v_low={self.num_bits_v_low}, "
                f"num_chunks_k_high={self.num_chunks_k_high}, "
                f"num_chunks_v_high={self.num_chunks_v_high}, "
                f"num_chunks_k_low={self.num_chunks_k_low}, "
                f"num_chunks_v_low={self.num_chunks_v_low}, "
                f"num_tokens_per_block_high={self.num_tokens_per_block_high}, "
                f"num_tokens_per_block_low={self.num_tokens_per_block_low}, "
                f"use_cuda_graph={self.use_cuda_graph})")