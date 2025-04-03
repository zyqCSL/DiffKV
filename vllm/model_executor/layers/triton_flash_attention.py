#!/usr/bin/env python
"""
Fused Attention
===============

This is a Triton implementation of the Flash Attention v2 algorithm from Tri Dao
(https://tridao.me/publications/flash2/flash2.pdf)
Credits: OpenAI kernel team, AMD ML Frameworks Triton team

Features supported:

1) Fwd with causal masking
2) Any sequence lengths without padding (currently fwd kernel only)
3) Support for different sequence lengths for q and k
4) Nested tensor API currently does not support dropout or bias.

Not currently supported:

1) Non power of two head dims

"""

import torch
import triton
import triton.language as tl

import os
os.environ['TRITON_PRINT_AUTOTUNING'] = '1'  # Get the tuned optimal config

# torch_dtype: tl.constexpr = torch.float16

# Return the qk products computed between the last few tokens and all tokens
# in the prompt to obtain the compression metric.
# Writing all qk products to global memory is expensive, since it is proportional
# to the square of the sequence length.
# Note: BLOCK_Q should be equal to NUM_TOKENS_SCORE
NUM_TOKENS_SCORE: tl.constexpr = 64

@triton.jit
def cdiv_fn(x, y):
    return (x + y - 1) // y


@triton.jit
def max_fn(x, y):
    return tl.math.max(x, y)


@triton.jit
def load_fn(block_ptr, first, second, pad):
    if first and second:
        tensor = tl.load(block_ptr, boundary_check=(0, 1), padding_option=pad)
    elif first:
        tensor = tl.load(block_ptr, boundary_check=(0, ), padding_option=pad)
    elif second:
        tensor = tl.load(block_ptr, boundary_check=(1, ), padding_option=pad)
    else:
        tensor = tl.load(block_ptr)
    return tensor


@triton.jit
def _attn_fwd_inner(
    acc,
    l_i,
    m_i,
    q,
    K_block_ptr,
    V_block_ptr,
    is_last_q_block,
    is_second_last_q_block,
    num_queries_in_the_last_block,
    actual_seqlen_k,
    qk_products_block_ptr,
    block_min,
    block_max,
    offs_n_causal,
    masked_blocks,
    n_extra_tokens,
    bias_ptr,
    IS_CAUSAL: tl.constexpr,
    BLOCK_Q: tl.constexpr,
    HEAD_SIZE: tl.constexpr,
    BLOCK_KV: tl.constexpr,
    OFFS_M: tl.constexpr,
    OFFS_N: tl.constexpr,
    PRE_LOAD_V: tl.constexpr,
    MASK_STEPS: tl.constexpr,
    RETURN_QK_PRODUCTS: tl.constexpr,
):
    # loop over k, v, and update accumulator
    for start_n in range(block_min, block_max, BLOCK_KV):
        # For padded blocks, we will overrun the tensor size if
        # we load all BLOCK_KV. For others, the blocks are all within range.
        k = load_fn(
            K_block_ptr,
            False,
            MASK_STEPS and (n_extra_tokens != 0),
            "zero",
        )
        if PRE_LOAD_V:
            v = load_fn(
                V_block_ptr,
                MASK_STEPS and (n_extra_tokens != 0),
                False,
                "zero",
            )
        qk = tl.zeros([BLOCK_Q, BLOCK_KV], dtype=tl.float32)
        # We start from end of seqlen_k so only the first iteration would need
        # to be checked for padding if it is not a multiple of block_n
        # TODO: This can be optimized to only be true for the padded block.
        if MASK_STEPS:  # noqa: SIM102
            # If this is the last block / iteration, we want to
            # mask if the sequence length is not a multiple of block size
            # a solution is to always do BLOCK_Q // BLOCK_KV + 1 steps
            # if not is_modulo_mn. last step might get wasted but that is okay.
            # check if this masking works for that case.
            if (start_n + BLOCK_KV == block_max) and (n_extra_tokens != 0):
                boundary_m = tl.full([BLOCK_Q],
                                     actual_seqlen_k,
                                     dtype=tl.int32)
                size_n = start_n + OFFS_N[None, :]
                mask = size_n < boundary_m[:, None]
                qk = tl.where(mask, qk, float("-inf"))
        if IS_CAUSAL:
            causal_boundary = start_n + offs_n_causal
            causal_mask = OFFS_M[:, None] >= causal_boundary[None, :]
            qk = tl.where(causal_mask, qk, float("-inf"))
        # -- compute qk ----
        qk += tl.dot(q, k)
        if RETURN_QK_PRODUCTS and (is_last_q_block or is_second_last_q_block):
            tl.store(qk_products_block_ptr, qk, boundary_check=(0, 1))
            qk_products_block_ptr = tl.advance(qk_products_block_ptr, (0, BLOCK_KV))
        if bias_ptr is not None:
            bias = load_fn(bias_ptr, False, MASK_STEPS
                           and (n_extra_tokens != 0), "zero")
            # While bias is added after multiplying qk with scale, our
            # optimization to use 2^x instead of e^x results in an additional
            # scale factor of log2(e) which we must also multiply the bias with.
            qk += bias * 1.44269504089
        m_ij = tl.maximum(m_i, tl.max(qk, 1))
        qk = qk - m_ij[:, None]
        p = tl.math.exp2(qk)

        # CAVEAT: Must update l_ij before applying dropout
        l_ij = tl.sum(p, 1)
        # -- update output accumulator --
        alpha = tl.math.exp2(m_i - m_ij)
        acc = acc * alpha[:, None]
        if not PRE_LOAD_V:
            v = load_fn(
                V_block_ptr,
                MASK_STEPS and (n_extra_tokens != 0),
                False,
                "zero",
            )
        # -- update m_i and l_i
        l_i = l_i * alpha + l_ij
        # update m_i and l_i
        m_i = m_ij
        acc += tl.dot(p.to(V_block_ptr.type.element_ty), v)
        V_block_ptr = tl.advance(V_block_ptr, (BLOCK_KV, 0))
        K_block_ptr = tl.advance(K_block_ptr, (0, BLOCK_KV))
        if bias_ptr is not None:
            bias_ptr = tl.advance(bias_ptr, (0, BLOCK_KV))
    return acc, l_i, m_i

# Parameter tuning is slow.
configs = [
    triton.Config({'BLOCK_Q': BLOCK_Q, 'BLOCK_KV': BLOCK_KV, 'PRE_LOAD_V': PRE_LOAD_V},
                  num_stages=s, num_warps=w) \
    for BLOCK_Q in [64] \
    for BLOCK_KV in [16, 32, 64, 128, 256] \
    for PRE_LOAD_V in [True, False] \
    for s in [1, 2, 4, 8] \
    for w in [1, 2, 4, 8] \
]

L40_optimal_config = [
    triton.Config({'BLOCK_Q': 64, 'BLOCK_KV': 32, 'PRE_LOAD_V': False},
                    num_stages=1, num_warps=4)
]

H20_optimal_config = [
    triton.Config({'BLOCK_Q': 64, 'BLOCK_KV': 32, 'PRE_LOAD_V': True},
                    num_stages=1, num_warps=4)
]

@triton.autotune(configs=L40_optimal_config, key=['IS_CAUSAL', 'HEAD_SIZE'])
@triton.jit
def attn_fwd(
    Q,
    K,
    V,
    bias,
    scale,
    Out,
    stride_qz,
    stride_qh,
    stride_qm,
    stride_qk,
    stride_kz,
    stride_kh,
    stride_kn,
    stride_kk,
    stride_vz,
    stride_vh,
    stride_vk,
    stride_vn,
    stride_oz,
    stride_oh,
    stride_om,
    stride_on,
    stride_bz,
    stride_bh,
    stride_bm,
    stride_bn,
    cu_seqlens_q,
    cu_seqlens_k,
    qk_products,
    N_HEADS_Q: tl.constexpr,
    N_HEADS_KV: tl.constexpr,
    MAX_SEQLENS_Q: tl.constexpr,
    MAX_SEQLENS_K: tl.constexpr,
    VARLEN: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    BLOCK_Q: tl.constexpr,
    HEAD_SIZE: tl.constexpr,
    BLOCK_KV: tl.constexpr,
    PRE_LOAD_V: tl.constexpr,
    BIAS_TYPE: tl.constexpr,
    RETURN_QK_PRODUCTS: tl.constexpr,
):
    start_m = tl.program_id(0)
    q_head_idx = tl.program_id(1)
    batch_idx = tl.program_id(2)
    offs_m = start_m * BLOCK_Q + tl.arange(0, BLOCK_Q)
    offs_n = tl.arange(0, BLOCK_KV)
    if VARLEN:
        cu_seqlens_q_start = tl.load(cu_seqlens_q + batch_idx)
        cu_seqlens_q_end = tl.load(cu_seqlens_q + batch_idx + 1)
        seqlen_q = cu_seqlens_q_end - cu_seqlens_q_start
        # We have a one-size-fits-all grid in id(0). Some seqlens might be too
        # small for all start_m so for those we return early.
        if start_m * BLOCK_Q > seqlen_q:
            return
        cu_seqlens_k_start = tl.load(cu_seqlens_k + batch_idx)
        cu_seqlens_k_end = tl.load(cu_seqlens_k + batch_idx + 1)
        seqlen_k = cu_seqlens_k_end - cu_seqlens_k_start
    else:
        cu_seqlens_q_start = 0
        cu_seqlens_k_start = 0
        seqlen_q = MAX_SEQLENS_Q
        seqlen_k = MAX_SEQLENS_K

    # Now we compute whether we need to exit early due to causal masking.
    # This is because for seqlen_q > seqlen_k, M rows of the attn scores
    # are completely masked, resulting in 0s written to the output, and
    # inf written to LSE. We don't need to do any GEMMs in this case.
    # This block of code determines what N is, and if this WG is operating
    # on those M rows.
    n_blocks = cdiv_fn(seqlen_k, BLOCK_KV)
    if IS_CAUSAL:
        # If seqlen_q == seqlen_k, the attn scores are a square matrix.
        # If seqlen_q != seqlen_k, attn scores are rectangular which means
        # the causal mask boundary is bottom right aligned, and ends at either
        # the top edge (seqlen_q < seqlen_k) or left edge.
        # This captures the decrease in n_blocks if we have a rectangular attn
        # matrix
        n_blocks_seqlen = cdiv_fn(
            (start_m + 1) * BLOCK_Q + seqlen_k - seqlen_q, BLOCK_KV)
        # This is what adjusts the block_max for the current WG, only
        # if IS_CAUSAL. Otherwise we want to always iterate through all n_blocks
        n_blocks = min(n_blocks, n_blocks_seqlen)
        # If we have no blocks after adjusting for seqlen deltas, this WG is
        # part of the blocks that are all 0. We exit early.
        if n_blocks <= 0:
            o_offset = (batch_idx * stride_oz + cu_seqlens_q_start * stride_om +
                        q_head_idx * stride_oh)
            O_block_ptr = tl.make_block_ptr(
                base=Out + o_offset,
                shape=(seqlen_q, HEAD_SIZE),
                strides=(stride_om, stride_on),
                offsets=(start_m * BLOCK_Q, 0),
                block_shape=(BLOCK_Q, HEAD_SIZE),
                order=(1, 0),
            )
            acc = tl.zeros([BLOCK_Q, HEAD_SIZE], dtype=Out.type.element_ty)
            # We still need to write 0s to the result
            # tl.store(O_block_ptr,
            # acc.to(Out.type.element_ty), boundary_check=(0,1))
            # l_ptrs = L + batch_idx * N_HEADS_Q * MAX_SEQLENS_Q + q_head_idx * MAX_SEQLENS_Q
            #          + offs_m
            # We store inf to LSE, not -inf because in the bwd pass,
            # we subtract this
            # from qk which makes it -inf, such that exp(qk - inf) = 0
            # for these masked blocks.
            # l = tl.full([BLOCK_Q], value=float("inf"), dtype=tl.float32)
            # tl.store(l_ptrs, l)
            # TODO: Should dropout and return encoded softmax be handled here?
            return

    # If MQA / GQA, set the K and V head offsets appropriately.
    GROUP_SIZE: tl.constexpr = N_HEADS_Q // N_HEADS_KV
    kv_head_idx = q_head_idx // GROUP_SIZE if GROUP_SIZE != 1 else q_head_idx

    n_extra_tokens = 0
    if seqlen_k < BLOCK_KV:
        n_extra_tokens = BLOCK_KV - seqlen_k
    elif seqlen_k % BLOCK_KV:
        n_extra_tokens = seqlen_k % BLOCK_KV

    # Compute pointers for all the tensors used in this kernel.
    q_offset = (batch_idx * stride_qz + q_head_idx * stride_qh +
                cu_seqlens_q_start * stride_qm)
    Q_block_ptr = tl.make_block_ptr(
        base=Q + q_offset,
        shape=(seqlen_q, HEAD_SIZE),
        strides=(stride_qm, stride_qk),
        offsets=(start_m * BLOCK_Q, 0),
        block_shape=(BLOCK_Q, HEAD_SIZE),
        order=(1, 0),
    )
    k_offset = (batch_idx * stride_kz + kv_head_idx * stride_kh +
                cu_seqlens_k_start * stride_kn)
    K_block_ptr = tl.make_block_ptr(
        base=K + k_offset,
        shape=(HEAD_SIZE, seqlen_k),
        strides=(stride_kk, stride_kn),
        offsets=(0, 0),
        block_shape=(HEAD_SIZE, BLOCK_KV),
        order=(0, 1),
    )
    v_offset = (batch_idx * stride_vz + kv_head_idx * stride_vh +
                cu_seqlens_k_start * stride_vk)
    V_block_ptr = tl.make_block_ptr(
        base=V + v_offset,
        shape=(seqlen_k, HEAD_SIZE),
        strides=(stride_vk, stride_vn),
        offsets=(0, 0),
        block_shape=(BLOCK_KV, HEAD_SIZE),
        order=(1, 0),
    )
    if BIAS_TYPE != 0:
        bias_ptr = tl.make_block_ptr(
            base=bias + q_head_idx * stride_bh,
            shape=(seqlen_q, seqlen_k),
            strides=(stride_bm, stride_bn),
            offsets=(start_m * BLOCK_Q, 0),
            block_shape=(BLOCK_Q, BLOCK_KV),
            order=(1, 0),
        )
    else:
        bias_ptr = None
    if RETURN_QK_PRODUCTS:
        qk_products_offset = batch_idx * N_HEADS_Q * NUM_TOKENS_SCORE * MAX_SEQLENS_K \
                           + q_head_idx * NUM_TOKENS_SCORE * MAX_SEQLENS_K
        qk_products_block_ptr = tl.make_block_ptr(
            base=qk_products + qk_products_offset,
            shape=(NUM_TOKENS_SCORE, MAX_SEQLENS_K),
            strides=(MAX_SEQLENS_K, 1),
            offsets=(0, 0),
            block_shape=(NUM_TOKENS_SCORE, BLOCK_KV),
            order=(0, 1),
        )
    else:
        qk_products_block_ptr = None
    # initialize pointer to m and l
    m_i = tl.full([BLOCK_Q], float("-inf"), dtype=tl.float32)
    l_i = tl.full([BLOCK_Q], 1.0, dtype=tl.float32)
    acc = tl.zeros([BLOCK_Q, HEAD_SIZE], dtype=tl.float32)
    # scale scale by log_2(e) and use 2^x in the loop as we do not
    # have native e^x support in HW.
    qk_scale = scale * 1.44269504089
    # Q is loaded once at the beginning and shared by all N blocks.
    q = load_fn(Q_block_ptr, True, False, "zero")
    q = (q * qk_scale).to(Q_block_ptr.type.element_ty)

    # Here we compute how many full and masked blocks we have.
    padded_block_k = n_extra_tokens != 0
    is_modulo_mn = not padded_block_k and (seqlen_q % BLOCK_Q == 0)
    if IS_CAUSAL:
        # There are always at least BLOCK_Q // BLOCK_KV masked blocks.
        # Additionally there might be one more due to dissimilar seqlens.
        masked_blocks = BLOCK_Q // BLOCK_KV + (not is_modulo_mn)
    else:
        # Padding on Q does not need to be masked in the FA loop.
        masked_blocks = padded_block_k
    # if IS_CAUSAL, not is_modulo_mn does not always result in an additional
    # block. In this case we might exceed n_blocks so pick the min.
    masked_blocks = min(masked_blocks, n_blocks)
    n_full_blocks = n_blocks - masked_blocks
    block_min = 0
    block_max = n_blocks * BLOCK_KV

    is_last_q_block = (start_m + 1) * BLOCK_Q >= seqlen_q
    is_second_last_q_block = (not is_last_q_block) and (start_m + 2) * BLOCK_Q >= seqlen_q

    if seqlen_q % BLOCK_Q == 0:
        num_queries_in_the_last_block = BLOCK_Q
    else:
        num_queries_in_the_last_block = seqlen_q % BLOCK_Q

    if RETURN_QK_PRODUCTS:
        if is_last_q_block:
            qk_products_block_ptr = tl.advance(qk_products_block_ptr,
                                               (BLOCK_Q - num_queries_in_the_last_block, 0))
        elif is_second_last_q_block and num_queries_in_the_last_block != BLOCK_Q:
            qk_products_block_ptr = tl.advance(qk_products_block_ptr,
                                               (-num_queries_in_the_last_block, 0))

    # if q_head_idx == 0 and batch_idx == 0:
    #     tl.device_print("is_last_q_block", is_last_q_block)
    #     tl.device_print("is_second_last_q_block", is_second_last_q_block)
    #     tl.device_print("num_queries_in_the_last_block", num_queries_in_the_last_block)
    #     tl.device_print("start_m", start_m)
    #     tl.device_print("n_full_blocks", n_full_blocks)
    #     tl.device_print("masked_blocks", masked_blocks)

    # Compute for full blocks. Here we set causal to false regardless of its
    # value because there is no masking. Similarly we do not need padding.
    if n_full_blocks > 0:
        block_max = (n_blocks - masked_blocks) * BLOCK_KV
        acc, l_i, m_i = _attn_fwd_inner(
            acc,
            l_i,
            m_i,
            q,
            K_block_ptr,
            V_block_ptr,
            is_last_q_block,
            is_second_last_q_block,
            num_queries_in_the_last_block,
            seqlen_k,
            qk_products_block_ptr,
            # _, _, offs_n_causal, masked_blocks, n_extra_tokens, _
            block_min,
            block_max,
            0,
            0,
            0,
            bias_ptr,
            # IS_CAUSAL, ....
            False,
            BLOCK_Q,
            HEAD_SIZE,
            BLOCK_KV,
            offs_m,
            offs_n,
            # _, MASK_STEPS, ...
            PRE_LOAD_V,
            False,
            RETURN_QK_PRODUCTS,
        )
        block_min = block_max
        block_max = n_blocks * BLOCK_KV

    tl.debug_barrier()
    # Remaining blocks, if any, are masked.
    if masked_blocks > 0:
        offs_n_causal = offs_n + (seqlen_q - seqlen_k) if IS_CAUSAL else 0
        K_block_ptr = tl.advance(K_block_ptr, (0, n_full_blocks * BLOCK_KV))
        V_block_ptr = tl.advance(V_block_ptr, (n_full_blocks * BLOCK_KV, 0))
        if RETURN_QK_PRODUCTS:
            qk_products_block_ptr = tl.advance(qk_products_block_ptr,
                                               (0, n_full_blocks * BLOCK_KV))
        if bias_ptr is not None:
            bias_ptr = tl.advance(bias_ptr, (0, n_full_blocks * BLOCK_KV))
        acc, l_i, m_i = _attn_fwd_inner(
            acc,
            l_i,
            m_i,
            q,
            K_block_ptr,
            V_block_ptr,
            is_last_q_block,
            is_second_last_q_block,
            num_queries_in_the_last_block,
            seqlen_k,
            qk_products_block_ptr,
            block_min,
            block_max,
            offs_n_causal,
            masked_blocks,
            n_extra_tokens,
            bias_ptr,
            IS_CAUSAL,
            BLOCK_Q,
            HEAD_SIZE,
            BLOCK_KV,
            offs_m,
            offs_n,
            # _, MASK_STEPS, ...
            PRE_LOAD_V,
            True,
            RETURN_QK_PRODUCTS,
        )
    # epilogue
    acc = acc / l_i[:, None]
    # If seqlen_q > seqlen_k but the delta is not a multiple of BLOCK_Q,
    # then we have one block with a row of all NaNs which come from computing
    # softmax over a row of all -infs (-inf - inf = NaN). We check for that here
    # and store 0s where there are NaNs as these rows should've been zeroed out.
    end_m_idx = (start_m + 1) * BLOCK_Q
    start_m_idx = start_m * BLOCK_Q
    causal_start_idx = seqlen_q - seqlen_k
    acc = acc.to(Out.type.element_ty)
    if IS_CAUSAL:  # noqa: SIM102
        if causal_start_idx > start_m_idx and causal_start_idx < end_m_idx:
            out_mask_boundary = tl.full((HEAD_SIZE, ),
                                        causal_start_idx,
                                        dtype=tl.int32)
            mask_m_offsets = start_m_idx + tl.arange(0, BLOCK_Q)
            out_ptrs_mask = (mask_m_offsets[:, None] >=
                             out_mask_boundary[None, :])
            z = 0.0
            acc = tl.where(out_ptrs_mask, acc, z.to(acc.type.element_ty))
    # write back LSE
    # l_ptrs = L + batch_idx * N_HEADS_Q * MAX_SEQLENS_Q + q_head_idx * MAX_SEQLENS_Q + offs_m
    # If seqlen_q not multiple of BLOCK_Q, we need to mask out the last
    # few rows. This is only true for the last M block. For others,
    # overflow_size will be -ve
    # overflow_size = end_m_idx - seqlen_q
    # if overflow_size > 0:
    #    boundary = tl.full((BLOCK_Q,), BLOCK_Q - overflow_size, dtype=tl.int32)
    #    # This is a > check because mask being 0 blocks the store.
    #    l_ptrs_mask = boundary > tl.arange(0, BLOCK_Q)
    #    tl.store(l_ptrs, m_i + tl.math.log2(l_i), mask=l_ptrs_mask)
    # else:
    #    tl.store(l_ptrs, m_i + tl.math.log2(l_i))

    # write back O
    o_offset = (batch_idx * stride_oz + cu_seqlens_q_start * stride_om +
                q_head_idx * stride_oh)
    O_block_ptr = tl.make_block_ptr(
        base=Out + o_offset,
        shape=(seqlen_q, HEAD_SIZE),
        strides=(stride_om, stride_on),
        offsets=(start_m * BLOCK_Q, 0),
        block_shape=(BLOCK_Q, HEAD_SIZE),
        order=(1, 0),
    )
    # Need boundary check on this to make sure the padding from the
    # Q and KV tensors in both dims are not part of what we store back.
    # TODO: Do the boundary check optionally.
    tl.store(O_block_ptr, acc, boundary_check=(0, 1))


def check_args(
    q,
    k,
    v,
    varlen=True,
    max_seqlens=None,
    cu_seqlens_q=None,
    cu_seqlens_k=None,
):
    assert q.dim() == k.dim() and q.dim() == v.dim()
    if varlen:
        assert q.dim() == 3
        total_q, nheads_q, head_size = q.shape
        total_k, nheads_k, _ = k.shape
        assert cu_seqlens_q is not None
        assert cu_seqlens_k is not None
        assert len(cu_seqlens_q) == len(cu_seqlens_k)
    else:
        assert q.dim() == 4
        batch, nheads_q, seqlen_q, head_size = q.shape
        _, nheads_k, seqlen_k, _ = k.shape
        assert max_seqlens > 0
    assert k.shape == v.shape
    assert q.shape[-1] == k.shape[-1] and q.shape[-1] == v.shape[-1]
    # TODO: Change assert if we support qkl f8 and v f16
    assert q.dtype == k.dtype and q.dtype == v.dtype
    assert head_size <= 256
    assert (nheads_q % nheads_k) == 0


class _attention(torch.autograd.Function):

    @staticmethod
    def forward(
        ctx,
        q,
        k,
        v,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlens_q,
        max_seqlens_k,
        causal=False,
        scale=1.0,
        bias=None,
    ):
        check_args(
            q,
            k,
            v,
            varlen=True,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
        )

        out = torch.empty_like(q, dtype=v.dtype)

        if True:  # varlen
            total_q, nheads_q, head_size = q.shape
            total_k, nheads_k, _ = k.shape
            batch = len(cu_seqlens_q) - 1
            q_strides = (0, q.stride(1), q.stride(0), q.stride(2))
            k_strides = (0, k.stride(1), k.stride(0), k.stride(2))
            v_strides = (0, v.stride(1), v.stride(0), v.stride(2))
            o_strides = (0, out.stride(1), out.stride(0), out.stride(2))
        else:
            batch, seqlen_q, nheads_q, head_size = q.shape
            _, seqlen_k, nheads_k, _ = k.shape
            q_strides = (q.stride(0), q.stride(2), q.stride(1), q.stride(3))
            k_strides = (k.stride(0), k.stride(2), k.stride(1), k.stride(3))
            v_strides = (v.stride(0), v.stride(2), v.stride(1), v.stride(3))
            o_strides = (out.stride(0), out.stride(2), out.stride(1), out.stride(3))

        qk_products = torch.full((batch, nheads_q, NUM_TOKENS_SCORE, max_seqlens_k),
                                 float("-inf"), dtype=torch.float32, device=q.device)

        supported_head_sizes = {32, 64, 128, 256}
        if head_size not in supported_head_sizes:
            raise ValueError(f"Head size {head_size} not supported. Supported head sizes are {supported_head_sizes}")

        grid = lambda META: (
            triton.cdiv(max_seqlens_q, META["BLOCK_Q"]),
            nheads_q,
            batch,
        )

        if bias is not None:
            bias_strides = (
                bias.stride(0),
                bias.stride(1),
                bias.stride(2),
                bias.stride(3),
            )
        else:
            bias_strides = (0, 0, 0, 0)

        attn_fwd[grid](
            q,
            k,
            v,
            bias,
            scale,
            out,
            *q_strides,
            *k_strides,
            *v_strides,
            *o_strides,
            *bias_strides,
            cu_seqlens_q,
            cu_seqlens_k,
            qk_products=qk_products,
            N_HEADS_Q=nheads_q,
            N_HEADS_KV=nheads_k,
            MAX_SEQLENS_Q=max_seqlens_q,
            MAX_SEQLENS_K=max_seqlens_k,
            IS_CAUSAL=causal,
            VARLEN=True,
            HEAD_SIZE=head_size,
            BIAS_TYPE=0 if bias is None else 1,
            RETURN_QK_PRODUCTS=True,
        )

        ctx.grid = grid
        ctx.scale = scale
        ctx.HEAD_SIZE = head_size
        ctx.causal = causal
        ctx.qk_products = qk_products
        ctx.return_qk_products = True
        return out, qk_products


triton_attention = _attention.apply
