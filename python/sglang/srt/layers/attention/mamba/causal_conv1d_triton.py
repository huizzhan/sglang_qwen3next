# Copyright (c) 2024, Tri Dao.
# Adapted from https://github.com/Dao-AILab/causal-conv1d/blob/main/causal_conv1d/causal_conv1d_interface.py
# and https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/mamba/ops/causal_conv1d.py

from typing import List, Optional, Union, is_typeddict

import torch
import triton
from triton.experimental import gluon
import triton.experimental.gluon.language as gl
import triton.language as tl

PAD_SLOT_ID = -1

import os
os.environ["TRITON_PRINT_AUTOTUNING"] = "1"  # 显示 autotune 过程和结果
# os.environ["TRITON_CACHE_DIR"] = "/sgl-workspace/sglang/conv1d_gluon_cache_persistent_v1"
# os.environ["MLIR_ENABLE_DUMP"] = "1"
# os.environ["AMDGCN_ENABLE_DUMP"] = "1"


def make_block_layout_conv1d(w_ptr: torch.Tensor, block_size: int, num_warps: int):
    """
    为 conv1d kernel 动态计算 BlockedLayout
    
    Args:
        w_ptr: weight 张量（用于检查 dtype）
        block_size: BLOCK_N 大小
        num_warps: warp 数量
    
    Returns:
        gl.BlockedLayout: 优化后的数据布局
    """
    # 计算 size_per_thread
    # 对于 conv1d，我们使用简单的 1D 布局
    thread_nums = 64  # 每个 warp 的线程数
    vec = block_size // (thread_nums * num_warps)
    
    # # 根据 dtype 和 block_size 计算合适的 size_per_thread
    # bits = torch.finfo(w_ptr.dtype).bits if w_ptr.dtype.is_floating_point else 16
    # thread_load_bits = 128  # 每个线程加载的比特数
    
    # # 计算向量化大小
    # vec = min(thread_load_bits // bits, block_size)
    # vec = min(vec, block_size // (thread_nums * num_warps))
    # vec = max(vec, 1)  # 至少为 1
    
    return gl.BlockedLayout(
        size_per_thread=[vec],
        threads_per_warp=[thread_nums],
        warps_per_cta=[num_warps],
        order=[0],
    )

@triton.jit()
def _causal_conv1d_fwd_kernel(  # continuous batching
    # Pointers to matrices
    x_ptr,  # (dim, cu_seqlen) holding `batch` of actual sequences + padded sequences
    w_ptr,  # (dim, width)
    bias_ptr,
    initial_states_ptr,  # conv_states_ptr
    cache_indices_ptr,  # conv_state_indices_ptr
    has_initial_states_ptr,
    query_start_loc_ptr,
    o_ptr,  # (dim, seqlen) - actually pointing to x_ptr
    # Matrix dimensions
    dim: tl.constexpr,
    seqlen: tl.int32,  # cu_seqlen
    num_cache_lines: tl.constexpr,  # added to support vLLM larger cache lines
    # Strides
    stride_x_seq: tl.constexpr,  # stride to get to next sequence,
    stride_x_dim: tl.constexpr,  # stride to get to next feature-value,
    stride_x_token: tl.constexpr,  # stride to get to next token (same feature-index, same sequence-index)
    stride_w_dim: tl.constexpr,  # stride to get to next dim-axis value
    stride_w_width: tl.constexpr,  # stride to get to next width-axis value
    stride_istate_seq: tl.constexpr,
    stride_istate_dim: tl.constexpr,
    stride_istate_token: tl.constexpr,
    stride_o_seq: tl.constexpr,
    stride_o_dim: tl.constexpr,
    stride_o_token: tl.constexpr,
    # others
    pad_slot_id: tl.constexpr,
    # Meta-parameters
    HAS_BIAS: tl.constexpr,
    KERNEL_WIDTH: tl.constexpr,
    SILU_ACTIVATION: tl.constexpr,
    HAS_INITIAL_STATES: tl.constexpr,
    HAS_CACHE: tl.constexpr,
    IS_CONTINUOUS_BATCHING: tl.constexpr,
    USE_PAD_SLOT: tl.constexpr,
    NP2_STATELEN: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    conv_states_ptr = initial_states_ptr
    conv_state_indices_ptr = cache_indices_ptr
    stride_conv_state_seq = stride_istate_seq
    stride_conv_state_dim = stride_istate_dim
    stride_conv_state_tok = stride_istate_token
    state_len = (
        KERNEL_WIDTH - 1
    )  # can be passed via argument if it's not the same as this value

    # one program handles one chunk in a single sequence
    # rather than mixing sequences - to make updating initial_states across sequences efficiently

    # single-sequence id
    idx_seq = tl.program_id(0)
    chunk_offset = tl.program_id(1)

    # BLOCK_N elements along the feature-dimension (channel)
    idx_feats = tl.program_id(2) * BLOCK_N + tl.arange(0, BLOCK_N)

    if idx_seq == pad_slot_id:
        return

    sequence_start_index = tl.load(query_start_loc_ptr + idx_seq)
    sequence_end_index = tl.load(query_start_loc_ptr + idx_seq + 1)
    # find the actual sequence length
    seqlen = sequence_end_index - sequence_start_index

    token_offset = BLOCK_M * chunk_offset
    segment_len = min(BLOCK_M, seqlen - token_offset)

    if segment_len <= 0:
        return

    # base of the sequence
    x_base = (
        x_ptr + sequence_start_index * stride_x_token + idx_feats * stride_x_dim
    )  # [BLOCK_N,]

    if IS_CONTINUOUS_BATCHING:
        # cache_idx
        conv_state_batch_coord = tl.load(conv_state_indices_ptr + idx_seq).to(tl.int64)
    else:
        # cache_idx
        conv_state_batch_coord = idx_seq
    if USE_PAD_SLOT:  # noqa
        if conv_state_batch_coord == pad_slot_id:
            # not processing as this is not the actual sequence
            return
    conv_states_base = (
        conv_states_ptr
        + (conv_state_batch_coord * stride_conv_state_seq)
        + (idx_feats * stride_conv_state_dim)
    )  # [BLOCK_N,]

    w_base = w_ptr + (idx_feats * stride_w_dim)  # [BLOCK_N,]

    # Does 2 things:
    # 1. READ prior-block init-state data - [done by every Triton programs]
    # 2. update conv_state with new data [only by the Triton program handles chunk_offset=0]
    if chunk_offset == 0:
        # read from conv_states
        load_init_state = False
        if HAS_INITIAL_STATES:  # the new HAS_INITIAL_STATES
            load_init_state = tl.load(has_initial_states_ptr + idx_seq).to(tl.int1)
        if load_init_state:
            # load from conv_states
            prior_tokens = conv_states_base + (state_len - 1) * stride_conv_state_tok
            mask_w = idx_feats < dim
            if KERNEL_WIDTH == 2:
                conv_states_ptrs = prior_tokens  # [BLOCK_N]
                col0 = tl.load(conv_states_ptrs, mask_w, 0.0)
            if KERNEL_WIDTH == 3:
                conv_states_ptrs = prior_tokens  # [BLOCK_N]
                col1 = tl.load(conv_states_ptrs, mask_w, 0.0)
                conv_states_ptrs = prior_tokens - 1 * stride_conv_state_tok  # [BLOCK_N]
                col0 = tl.load(conv_states_ptrs, mask_w, 0.0)
            if KERNEL_WIDTH == 4:
                conv_states_ptrs = prior_tokens  # [BLOCK_N]
                col2 = tl.load(conv_states_ptrs, mask_w, 0.0)
                conv_states_ptrs = prior_tokens - 1 * stride_conv_state_tok  # [BLOCK_N]
                col1 = tl.load(conv_states_ptrs, mask_w, 0.0)
                conv_states_ptrs = prior_tokens - 2 * stride_conv_state_tok  # [BLOCK_N]
                col0 = tl.load(conv_states_ptrs, mask_w, 0.0)
            if KERNEL_WIDTH == 5:
                conv_states_ptrs = prior_tokens  # [BLOCK_N]
                col3 = tl.load(conv_states_ptrs, mask_w, 0.0)
                conv_states_ptrs = prior_tokens - 1 * stride_conv_state_tok  # [BLOCK_N]
                col2 = tl.load(conv_states_ptrs, mask_w, 0.0)
                conv_states_ptrs = prior_tokens - 2 * stride_conv_state_tok  # [BLOCK_N]
                col1 = tl.load(conv_states_ptrs, mask_w, 0.0)
                conv_states_ptrs = prior_tokens - 3 * stride_conv_state_tok  # [BLOCK_N]
                col0 = tl.load(conv_states_ptrs, mask_w, 0.0)
        else:
            # prior-tokens are zeros
            if KERNEL_WIDTH >= 2:  # STRATEGY1
                # first chunk and does not have prior-token, so just set to 0
                col0 = tl.zeros((BLOCK_N,), dtype=x_ptr.dtype.element_ty)
            if KERNEL_WIDTH >= 3:  # STRATEGY1
                col1 = tl.zeros((BLOCK_N,), dtype=x_ptr.dtype.element_ty)
            if KERNEL_WIDTH >= 4:  # STRATEGY1
                col2 = tl.zeros((BLOCK_N,), dtype=x_ptr.dtype.element_ty)
            if KERNEL_WIDTH >= 5:  # STRATEGY1
                col3 = tl.zeros((BLOCK_N,), dtype=x_ptr.dtype.element_ty)

        # STEP 2:
        # here prepare data for updating conv_state
        if (
            state_len <= seqlen
        ):  # SMALL_CACHE=True (only move part of 'x' into conv_state cache)
            # just read from 'x'
            # copy 'x' data to conv_state
            # load only 'x' data (and set 0 before 'x' if seqlen < state_len)
            idx_tokens_last = (seqlen - state_len) + tl.arange(
                0, NP2_STATELEN
            )  # [BLOCK_M]
            x_ptrs = (
                x_ptr
                + ((sequence_start_index + idx_tokens_last) * stride_x_token)[:, None]
                + (idx_feats * stride_x_dim)[None, :]
            )  # [BLOCK_M,BLOCK_N,]
            mask_x = (
                (idx_tokens_last >= 0)[:, None]
                & (idx_tokens_last < seqlen)[:, None]
                & (idx_feats < dim)[None, :]
            )  # token-index  # token-index  # feature-index
            loaded_x = tl.load(x_ptrs, mask_x, 0.0)
            new_conv_state = tl.load(x_ptrs, mask_x, 0.0)
            idx_tokens_conv = tl.arange(0, NP2_STATELEN)  # [BLOCK_M]
            conv_states_ptrs_target = (
                conv_states_base[None, :]
                + (idx_tokens_conv * stride_conv_state_tok)[:, None]
            )

            mask = (idx_tokens_conv < state_len)[:, None] & (idx_feats < dim)[None, :]
            tl.debug_barrier()  #  NOTE: use this due to bug in Triton compiler
            tl.store(conv_states_ptrs_target, new_conv_state, mask)

        else:
            if load_init_state:
                # update conv_state by shifting left, i.e. take last few cols from conv_state + cols from 'x'
                idx_tokens_conv = tl.arange(0, NP2_STATELEN)  # [BLOCK_M]

                conv_states_ptrs_source = (
                    conv_states_ptr
                    + (conv_state_batch_coord * stride_conv_state_seq)
                    + (idx_feats * stride_conv_state_dim)[None, :]
                    + ((idx_tokens_conv + seqlen) * stride_conv_state_tok)[:, None]
                )  # [BLOCK_M, BLOCK_N]
                mask = (
                    (conv_state_batch_coord < num_cache_lines)
                    & ((idx_tokens_conv + seqlen) < state_len)[:, None]
                    & (idx_feats < dim)[None, :]
                )
                conv_state = tl.load(conv_states_ptrs_source, mask, other=0.0)

                VAL = state_len - seqlen

                x_ptrs = (
                    x_base[None, :]
                    + ((idx_tokens_conv - VAL) * stride_x_token)[:, None]
                )  # [BLOCK_M, BLOCK_N]

                mask_x = (
                    (idx_tokens_conv - VAL >= 0)[:, None]
                    & (idx_tokens_conv - VAL < seqlen)[:, None]
                    & (idx_feats < dim)[None, :]
                )  # token-index  # token-index  # feature-index
                loaded_x = tl.load(x_ptrs, mask_x, 0.0)

                tl.debug_barrier()  # need this due to the bug in tl.where not enforcing this when data is the result of another tl.load
                new_conv_state = tl.where(
                    mask, conv_state, loaded_x
                )  # BUG in 'tl.where'  which requires a barrier before this
                conv_states_ptrs_target = (
                    conv_states_base
                    + (idx_tokens_conv * stride_conv_state_tok)[:, None]
                )  # [BLOCK_M, BLOCK_N]
                mask = (idx_tokens_conv < state_len)[:, None] & (idx_feats < dim)[
                    None, :
                ]
                tl.store(conv_states_ptrs_target, new_conv_state, mask)
            else:  # load_init_state == False
                # update conv_state by shifting left, BUT
                # set cols prior to 'x' as zeros + cols from 'x'
                idx_tokens_conv = tl.arange(0, NP2_STATELEN)  # [BLOCK_M]

                VAL = state_len - seqlen

                x_ptrs = (
                    x_base[None, :]
                    + ((idx_tokens_conv - VAL) * stride_x_token)[:, None]
                )  # [BLOCK_M, BLOCK_N]

                mask_x = (
                    (idx_tokens_conv - VAL >= 0)[:, None]
                    & (idx_tokens_conv - VAL < seqlen)[:, None]
                    & (idx_feats < dim)[None, :]
                )  # token-index  # token-index  # feature-index
                new_conv_state = tl.load(x_ptrs, mask_x, 0.0)

                conv_states_ptrs_target = (
                    conv_states_base
                    + (idx_tokens_conv * stride_conv_state_tok)[:, None]
                )  # [BLOCK_M, BLOCK_N]
                mask = (idx_tokens_conv < state_len)[:, None] & (idx_feats < dim)[
                    None, :
                ]
                tl.store(conv_states_ptrs_target, new_conv_state, mask)

    else:  # chunk_offset > 0
        # read prior-token data from `x`
        load_init_state = True
        prior_tokens = x_base + (token_offset - 1) * stride_x_token
        mask_w = idx_feats < dim
        if KERNEL_WIDTH == 2:
            conv_states_ptrs = prior_tokens  # [BLOCK_N]
            col0 = tl.load(conv_states_ptrs, mask_w, 0.0, cache_modifier=".ca")
        if KERNEL_WIDTH == 3:
            conv_states_ptrs = prior_tokens  # [BLOCK_N]
            col1 = tl.load(conv_states_ptrs, mask_w, 0.0, cache_modifier=".ca")
            conv_states_ptrs = prior_tokens - 1 * stride_x_token  # [BLOCK_N]
            col0 = tl.load(conv_states_ptrs, mask_w, 0.0, cache_modifier=".ca")
        if KERNEL_WIDTH == 4:
            conv_states_ptrs = prior_tokens  # [BLOCK_N]
            col2 = tl.load(conv_states_ptrs, mask_w, 0.0, cache_modifier=".ca")
            conv_states_ptrs = prior_tokens - 1 * stride_x_token  # [BLOCK_N]
            col1 = tl.load(conv_states_ptrs, mask_w, 0.0, cache_modifier=".ca")
            conv_states_ptrs = prior_tokens - 2 * stride_x_token  # [BLOCK_N]
            col0 = tl.load(conv_states_ptrs, mask_w, 0.0, cache_modifier=".ca")
        if KERNEL_WIDTH == 5:
            # ruff: noqa: F841
            conv_states_ptrs = prior_tokens  # [BLOCK_N]
            col3 = tl.load(conv_states_ptrs, mask_w, 0.0, cache_modifier=".ca")
            conv_states_ptrs = prior_tokens - 1 * stride_x_token  # [BLOCK_N]
            col2 = tl.load(conv_states_ptrs, mask_w, 0.0, cache_modifier=".ca")
            conv_states_ptrs = prior_tokens - 2 * stride_x_token  # [BLOCK_N]
            col1 = tl.load(conv_states_ptrs, mask_w, 0.0, cache_modifier=".ca")
            conv_states_ptrs = prior_tokens - 3 * stride_x_token  # [BLOCK_N]
            col0 = tl.load(conv_states_ptrs, mask_w, 0.0, cache_modifier=".ca")

    if HAS_BIAS:
        bias = bias_ptr + idx_feats
        mask_bias = idx_feats < dim
        acc_preload = tl.load(bias, mask=mask_bias, other=0.0).to(
            tl.float32
        )  # [BLOCK_N]
    else:
        acc_preload = tl.zeros((BLOCK_N,), dtype=tl.float32)

    x_base_1d = x_base + token_offset * stride_x_token  # starting of chunk

    # PRE-LOAD WEIGHTS
    mask_w = idx_feats < dim
    if KERNEL_WIDTH >= 2:
        w_ptrs = w_base + (0 * stride_w_width)  # [BLOCK_N] tensor
        w_col0 = tl.load(w_ptrs, mask_w, other=0.0)
        w_ptrs = w_base + (1 * stride_w_width)  # [BLOCK_N] tensor
        w_col1 = tl.load(w_ptrs, mask_w, other=0.0)
    if KERNEL_WIDTH >= 3:
        w_ptrs = w_base + (2 * stride_w_width)  # [BLOCK_N] tensor
        w_col2 = tl.load(w_ptrs, mask_w, other=0.0)
    if KERNEL_WIDTH >= 4:
        w_ptrs = w_base + (3 * stride_w_width)  # [BLOCK_N] tensor
        w_col3 = tl.load(w_ptrs, mask_w, other=0.0)
    mask_x_1d = idx_feats < dim
    for idx_token in range(segment_len):
        acc = acc_preload

        matrix_w = w_col0
        matrix_x = col0
        for j in tl.static_range(KERNEL_WIDTH):

            if KERNEL_WIDTH == 2:
                if j == 1:  # KERNEL_WIDTH-1:
                    matrix_w = w_col1
                    x_ptrs_1d = x_base_1d + idx_token * stride_x_token  # [BLOCK_N]
                    matrix_x = tl.load(x_ptrs_1d, mask=mask_x_1d)
            elif KERNEL_WIDTH == 3:
                if j == 1:
                    matrix_w = w_col1
                    matrix_x = col1
                elif j == 2:
                    matrix_w = w_col2
                    x_ptrs_1d = x_base_1d + idx_token * stride_x_token  # [BLOCK_N]
                    matrix_x = tl.load(x_ptrs_1d, mask=mask_x_1d)
            elif KERNEL_WIDTH == 4:
                if j == 1:
                    matrix_w = w_col1
                    matrix_x = col1
                elif j == 2:
                    matrix_w = w_col2
                    matrix_x = col2
                elif j == 3:
                    matrix_w = w_col3
                    x_ptrs_1d = x_base_1d + idx_token * stride_x_token  # [BLOCK_N]
                    matrix_x = tl.load(x_ptrs_1d, mask=mask_x_1d)

            acc += matrix_x * matrix_w  # [BLOCK_N]

        if KERNEL_WIDTH == 2:
            col0 = matrix_x
        elif KERNEL_WIDTH == 3:
            col0 = col1
            col1 = matrix_x
        elif KERNEL_WIDTH == 4:
            col0 = col1
            col1 = col2
            col2 = matrix_x

        if SILU_ACTIVATION:
            acc = acc / (1 + tl.exp(-acc))
        mask_1d = (idx_token < segment_len) & (
            idx_feats < dim
        )  # token-index  # feature-index
        o_ptrs = (
            o_ptr
            + (sequence_start_index + token_offset + idx_token) * stride_o_token
            + (idx_feats * stride_o_dim)
        )

        tl.store(o_ptrs, acc, mask=mask_1d)


def causal_conv1d_fn(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: Union[torch.Tensor, None],
    conv_states: torch.Tensor,
    query_start_loc: torch.Tensor,
    seq_lens_cpu: List[int],
    cache_indices: Optional[torch.Tensor] = None,
    has_initial_state: Optional[torch.Tensor] = None,
    activation: Optional[str] = "silu",
    pad_slot_id: int = PAD_SLOT_ID,
    validate_data=False,
    **kwargs,
):
    """support varlen + continuous batching when x is 2D tensor

    x: (dim,cu_seq_len)
        cu_seq_len = total tokens of all seqs in that batch
        sequences are concatenated from left to right for varlen
    weight: (dim, width)
    conv_states: (...,dim,width - 1) itype
        updated inplace if provided
        [it use `cache_indices` to get the index to the cache of conv_state for that sequence

        conv_state[cache_indices[i]] for seq-i - to be used as initial_state when has_initial_state[i] = True
             and after that conv_state[cache_indices[i]] need to be shift-left and updated with values from 'x'
        ]
    query_start_loc: (batch + 1) int32
        The cumulative sequence lengths of the sequences in
        the batch, used to index into sequence. prepended by 0.
        if
        x = [5, 1, 1, 1] <- continuous batching (batch=4)
        then
        query_start_loc = [0, 5, 6, 7, 8] <- the starting index of the next sequence; while the last value is
           the ending index of the last sequence
        [length(query_start_loc)-1 == batch]
        for example: query_start_loc = torch.Tensor([0,10,16,17]),
        x.shape=(dim,17)
    seq_lens_cpu: (batch) int32
        The sequence lengths of the sequences in the batch
    cache_indices: (batch)  int32
        indicates the corresponding state index,
        like so: conv_state = conv_states[cache_indices[batch_id]]
    has_initial_state: (batch) bool
        indicates whether should the kernel take the current state as initial
        state for the calculations
        [single boolean for each sequence in the batch: True or False]
    bias: (dim,)
    activation: either None or "silu" or "swish" or True
    pad_slot_id: int
        if cache_indices is passed, lets the kernel identify padded
        entries that will not be processed,
        for example: cache_indices = [pad_slot_id, 1, 20, pad_slot_id]
        in this case, the kernel will not process entries at
        indices 0 and 3

    out: same shape as `x`
    """
    if isinstance(activation, bool) and activation:
        activation = "silu"

    out = torch.empty_like(x)

    is_channel_last = (x.stride(0) == 1) & (x.stride(1) > 1)
    dim, cu_seqlen = x.shape
    _, width = weight.shape
    state_len = width - 1
    np2_statelen = triton.next_power_of_2(state_len)

    stride_x_seq = 0
    stride_x_dim = x.stride(0)
    stride_x_token = x.stride(1)
    stride_w_dim = weight.stride(0)
    stride_w_width = weight.stride(1)
    stride_istate_seq = 0
    stride_istate_dim = 0
    stride_istate_token = 0
    num_cache_lines = 0
    if conv_states is not None:
        # extensions to support vLLM:
        # 1. conv_states is used to replaced initial_states
        # 2. conv_states serve as a cache with num cache lines can be larger than batch size
        # 3. mapping from sequence x[idx] to a cache line at index as specified via cache_indices[idx]
        # 4. computation can be skipped if cache_indices[idx] == pad_slot_id
        num_cache_lines = conv_states.size(0)
        assert (
            num_cache_lines == conv_states.shape[0]
            and dim == conv_states.shape[1]
            and width - 1 <= conv_states.shape[2]
        )
        stride_istate_seq = conv_states.stride(0)
        stride_istate_dim = conv_states.stride(1)
        stride_istate_token = conv_states.stride(2)
        # assert stride_istate_dim == 1
    if out.dim() == 2:
        stride_o_seq = 0
        stride_o_dim = out.stride(0)
        stride_o_token = out.stride(1)
    else:
        stride_o_seq = out.stride(0)
        stride_o_dim = out.stride(1)
        stride_o_token = out.stride(2)

    if validate_data:
        assert x.dim() == 2
        assert query_start_loc is not None
        assert query_start_loc.dim() == 1
        assert x.stride(0) == 1 or x.stride(1) == 1
        padded_batch = query_start_loc.size(0) - 1
        if bias is not None:
            assert bias.dim() == 1
            assert dim == bias.size(0)
        if cache_indices is not None:
            assert cache_indices.dim() == 1
            assert padded_batch == cache_indices.size(0)
        if has_initial_state is not None:
            assert has_initial_state.size() == (padded_batch,)
            assert (
                conv_states is not None
            ), "ERROR: `has_initial_state` is used, which needs also `conv_states`"
        assert weight.stride(1) == 1
        assert (dim, width) == weight.shape
        assert is_channel_last, "Need to run in channel-last layout"

    def grid(META):
        max_seq_len = max(seq_lens_cpu)
        return (
            len(seq_lens_cpu),  # batch_size
            (max_seq_len + META["BLOCK_M"] - 1) // META["BLOCK_M"],
            triton.cdiv(dim, META["BLOCK_N"]),
        )

    _causal_conv1d_fwd_kernel[grid](
        # Pointers to matrices
        x,
        weight,
        bias,
        conv_states,
        cache_indices,
        has_initial_state,
        query_start_loc,
        out,
        # Matrix dimensions
        dim,
        cu_seqlen,
        num_cache_lines,
        # stride
        stride_x_seq,
        stride_x_dim,
        stride_x_token,
        stride_w_dim,
        stride_w_width,
        stride_istate_seq,
        stride_istate_dim,
        stride_istate_token,
        stride_o_seq,
        stride_o_dim,
        stride_o_token,
        # others
        pad_slot_id,
        # META
        HAS_BIAS=bias is not None,
        KERNEL_WIDTH=width,
        SILU_ACTIVATION=activation in ["silu", "swish"],
        HAS_INITIAL_STATES=has_initial_state is not None,
        HAS_CACHE=conv_states is not None,
        IS_CONTINUOUS_BATCHING=cache_indices is not None,
        USE_PAD_SLOT=pad_slot_id is not None,
        NP2_STATELEN=np2_statelen,
        # launch_cooperative_grid=True
        BLOCK_M=8,
        BLOCK_N=256,
        num_stages=2,
    )
    return out


@triton.jit()
def _causal_conv1d_update_kernel(
    # Pointers to matrices
    x_ptr,  # (batch, dim, seqlen)
    w_ptr,  # (dim, width)
    bias_ptr,
    conv_state_ptr,
    cache_seqlens_ptr,  # circular buffer
    conv_state_indices_ptr,
    num_accepted_tokens_ptr,
    intermediate_conv_window_ptr,
    o_ptr,  # (batch, dim, seqlen)
    # Matrix dimensions
    batch: int,
    dim: tl.constexpr,
    seqlen: tl.constexpr,
    state_len: tl.constexpr,
    num_cache_lines: tl.constexpr,  # added to support vLLM larger cache lines
    # Strides
    stride_x_seq: tl.constexpr,
    stride_x_dim: tl.constexpr,
    stride_x_token: tl.constexpr,
    stride_w_dim: tl.constexpr,
    stride_w_width: tl.constexpr,
    stride_conv_state_seq: tl.constexpr,
    stride_conv_state_dim: tl.constexpr,
    stride_conv_state_tok: tl.constexpr,
    stride_state_indices: tl.constexpr,
    stride_inter_seq: tl.constexpr,
    stride_inter_step: tl.constexpr,
    stride_inter_dim: tl.constexpr,
    stride_inter_win: tl.constexpr,
    stride_o_seq: tl.constexpr,
    stride_o_dim: tl.constexpr,
    stride_o_token: tl.constexpr,
    # others
    pad_slot_id: tl.constexpr,
    # Meta-parameters
    HAS_BIAS: tl.constexpr,
    KERNEL_WIDTH: tl.constexpr,
    SILU_ACTIVATION: tl.constexpr,
    IS_CONTINUOUS_BATCHING: tl.constexpr,
    IS_SPEC_DECODING: tl.constexpr,
    NP2_STATELEN: tl.constexpr,
    USE_PAD_SLOT: tl.constexpr,
    BLOCK_N: tl.constexpr,
    SAVE_INTERMEDIATE: tl.constexpr,
):
    # ruff: noqa: E501
    idx_seq = tl.program_id(0)
    if idx_seq >= batch:
        return

    # [BLOCK_N,] elements along the feature-dimension (channel)
    idx_feats = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)

    if IS_CONTINUOUS_BATCHING:
        # mask = idx_seq < batch
        conv_state_batch_coord = tl.load(
            conv_state_indices_ptr + idx_seq * stride_state_indices
        ).to(tl.int64)
    else:
        conv_state_batch_coord = idx_seq
    if USE_PAD_SLOT:  # noqa
        if conv_state_batch_coord == pad_slot_id:
            # not processing as this is not the actual sequence
            return

    if IS_SPEC_DECODING:
        # The rolling of conv state:
        #
        # Before forward, the conv_state is:
        # [history1, history2, ..., historyM].
        #
        # After forward, the conv_state becomes:
        # [history2, ..., historyM, draft1, draft2, ..., draftN].
        #
        # After acceptance, it becomes:
        #
        # - accept 1 tokens: [history2, ..., historyM, draft1]
        # - accept 2 tokens: [history3, ..., historyM, draft1, draft2]
        # - and so on.
        conv_state_token_offset = tl.load(num_accepted_tokens_ptr + idx_seq) - 1
    else:
        conv_state_token_offset = 0

    # STEP 1: READ init_state data
    conv_states_base = (
        conv_state_ptr
        + (conv_state_batch_coord * stride_conv_state_seq)
        + (idx_feats * stride_conv_state_dim)
    )
    mask_w = idx_feats < dim

    prior_tokens = conv_states_base + conv_state_token_offset * stride_conv_state_tok
    if KERNEL_WIDTH >= 2:
        conv_states_ptrs = prior_tokens  # [BLOCK_N]
        col0 = tl.load(conv_states_ptrs, mask_w, 0.0)
    if KERNEL_WIDTH >= 3:
        conv_states_ptrs = prior_tokens + 1 * stride_conv_state_tok  # [BLOCK_N]
        col1 = tl.load(conv_states_ptrs, mask_w, 0.0)
    if KERNEL_WIDTH >= 4:
        conv_states_ptrs = prior_tokens + 2 * stride_conv_state_tok  # [BLOCK_N]
        col2 = tl.load(conv_states_ptrs, mask_w, 0.0)
    if KERNEL_WIDTH == 5:
        conv_states_ptrs = prior_tokens + 3 * stride_conv_state_tok  # [BLOCK_N]
        col3 = tl.load(conv_states_ptrs, mask_w, 0.0)

    # STEP 2: assume state_len > seqlen
    idx_tokens = tl.arange(0, NP2_STATELEN)  # [BLOCK_M]

    # The conv_state updates works in a sliding window manner,
    # at each forward pass, the tokens are shift by 1, so we
    # load since idx_tokens + 1.
    conv_state_ptrs_source = (
        conv_state_ptr
        + (conv_state_batch_coord * stride_conv_state_seq)
        + conv_state_token_offset * stride_conv_state_tok
        + (idx_feats * stride_conv_state_dim)[None, :]
        + ((idx_tokens + (1 if IS_SPEC_DECODING else seqlen)) * stride_conv_state_tok)[
            :, None
        ]
    )  # [BLOCK_M, BLOCK_N]
    VAL = state_len - seqlen
    mask = (
        (conv_state_batch_coord < num_cache_lines)
        & (idx_tokens < VAL)[:, None]
        & (idx_feats < dim)[None, :]
    )
    conv_state = tl.load(conv_state_ptrs_source, mask, other=0.0)

    x_base = x_ptr + (idx_seq * stride_x_seq) + (idx_feats * stride_x_dim)  # [BLOCK_N]

    x_ptrs = (
        x_base[None, :] + ((idx_tokens - VAL) * stride_x_token)[:, None]
    )  # [BLOCK_M, BLOCK_N]

    mask_x = (
        (idx_tokens - VAL >= 0)[:, None]
        & (idx_tokens - VAL < seqlen)[:, None]
        & (idx_feats < dim)[None, :]
    )  # token-index  # token-index  # feature-index
    loaded_x = tl.load(x_ptrs, mask_x, 0.0)
    tl.debug_barrier()

    new_conv_state = tl.where(mask, conv_state, loaded_x)

    conv_state_base = (
        conv_state_ptr
        + (conv_state_batch_coord * stride_conv_state_seq)
        + (idx_feats * stride_conv_state_dim)
    )  # [BLOCK_N,]
    conv_state_ptrs_target = (
        conv_state_base + (idx_tokens * stride_conv_state_tok)[:, None]
    )  # [BLOCK_M, BLOCK_N]
    mask = (idx_tokens < state_len)[:, None] & (idx_feats < dim)[None, :]
    tl.store(conv_state_ptrs_target, new_conv_state, mask)

    # STEP 3: init accumulator
    if HAS_BIAS:
        bias = bias_ptr + idx_feats
        mask_bias = idx_feats < dim
        acc_preload = tl.load(bias, mask=mask_bias, other=0.0).to(
            tl.float32
        )  # [BLOCK_N]
    else:
        acc_preload = tl.zeros((BLOCK_N,), dtype=tl.float32)

    # STEP 4:
    # PRE-LOAD WEIGHTS
    # first kernel column, configured for weights to handle BLOCK_N features in range
    w_base = w_ptr + (idx_feats * stride_w_dim)  # [BLOCK_N,]
    mask_w = idx_feats < dim
    if KERNEL_WIDTH >= 2:
        w_ptrs = w_base + (0 * stride_w_width)  # [BLOCK_N] tensor
        w_col0 = tl.load(w_ptrs, mask_w, other=0.0)
        w_ptrs = w_base + (1 * stride_w_width)  # [BLOCK_N] tensor
        w_col1 = tl.load(w_ptrs, mask_w, other=0.0)
    if KERNEL_WIDTH >= 3:
        w_ptrs = w_base + (2 * stride_w_width)  # [BLOCK_N] tensor
        w_col2 = tl.load(w_ptrs, mask_w, other=0.0)
    if KERNEL_WIDTH >= 4:
        w_ptrs = w_base + (3 * stride_w_width)  # [BLOCK_N] tensor
        w_col3 = tl.load(w_ptrs, mask_w, other=0.0)

    x_base_1d = x_base  # starting of chunk [BLOCK_N]
    mask_x_1d = idx_feats < dim

    # STEP 5: compute each token
    for idx_token in tl.static_range(seqlen):
        acc = acc_preload

        matrix_w = w_col0
        matrix_x = col0
        for j in tl.static_range(KERNEL_WIDTH):
            if KERNEL_WIDTH == 2:
                if j == 1:  # KERNEL_WIDTH-1:
                    matrix_w = w_col1
                    x_ptrs_1d = x_base_1d + idx_token * stride_x_token  # [BLOCK_N]
                    matrix_x = tl.load(x_ptrs_1d, mask=mask_x_1d)
            elif KERNEL_WIDTH == 3:
                if j == 1:
                    matrix_w = w_col1
                    matrix_x = col1
                elif j == 2:
                    matrix_w = w_col2
                    x_ptrs_1d = x_base_1d + idx_token * stride_x_token  # [BLOCK_N]
                    matrix_x = tl.load(x_ptrs_1d, mask=mask_x_1d)
            elif KERNEL_WIDTH == 4:
                if j == 1:
                    matrix_w = w_col1
                    matrix_x = col1
                elif j == 2:
                    matrix_w = w_col2
                    matrix_x = col2
                elif j == 3:
                    matrix_w = w_col3
                    x_ptrs_1d = x_base_1d + idx_token * stride_x_token  # [BLOCK_N]
                    matrix_x = tl.load(x_ptrs_1d, mask=mask_x_1d)

            acc += matrix_x * matrix_w  # [BLOCK_N]

        if KERNEL_WIDTH == 2:
            col0 = matrix_x
        elif KERNEL_WIDTH == 3:
            col0 = col1
            col1 = matrix_x
        elif KERNEL_WIDTH == 4:
            col0 = col1
            col1 = col2
            col2 = matrix_x

        if SILU_ACTIVATION:
            acc = acc / (1 + tl.exp(-acc))
        mask_1d = (idx_token < seqlen) & (
            idx_feats < dim
        )  # token-index  # feature-index
        o_ptrs = (
            o_ptr
            + (idx_seq) * stride_o_seq
            + idx_token * stride_o_token
            + (idx_feats * stride_o_dim)
        )

        tl.store(o_ptrs, acc, mask=mask_1d)

        if SAVE_INTERMEDIATE:
            # Save the window state after consuming this token
            # Layout: [seq(cache line), step, dim, win(K-1)]
            base_ptr = (
                intermediate_conv_window_ptr
                + conv_state_batch_coord * stride_inter_seq
                + idx_token * stride_inter_step
                + idx_feats * stride_inter_dim
            )
            if KERNEL_WIDTH >= 2:
                tl.store(base_ptr + 0 * stride_inter_win, col0, mask=mask_w)
            if KERNEL_WIDTH >= 3:
                tl.store(base_ptr + 1 * stride_inter_win, col1, mask=mask_w)
            if KERNEL_WIDTH >= 4:
                tl.store(base_ptr + 2 * stride_inter_win, col2, mask=mask_w)

@gl._core.builtin
def tuple_combine(a: gl.tuple, b: gl.tensor, _semantic=None) -> gl.tuple:
    return gl.tuple([*a.values, b])

@gluon.jit()
def gluon_causal_conv1d_update_kernel(
    # Pointers to matrices
    x_ptr,  # (batch, dim, seqlen)
    w_ptr,  # (dim, width)
    bias_ptr,
    conv_state_ptr,
    cache_seqlens_ptr,  # circular buffer
    conv_state_indices_ptr,
    num_accepted_tokens_ptr,
    intermediate_conv_window_ptr,
    o_ptr,  # (batch, dim, seqlen)
    # Matrix dimensions
    batch: int,
    dim: gl.constexpr,
    seqlen: gl.constexpr,
    state_len: gl.constexpr,
    num_cache_lines: gl.constexpr,  # added to support vLLM larger cache lines
    # Strides
    stride_x_seq: gl.constexpr,
    stride_x_dim: gl.constexpr,
    stride_x_token: gl.constexpr,
    stride_w_dim: gl.constexpr,
    stride_w_width: gl.constexpr,
    stride_conv_state_seq: gl.constexpr,
    stride_conv_state_dim: gl.constexpr,
    stride_conv_state_tok: gl.constexpr,
    stride_state_indices: gl.constexpr,
    stride_inter_seq: gl.constexpr,
    stride_inter_step: gl.constexpr,
    stride_inter_dim: gl.constexpr,
    stride_inter_win: gl.constexpr,
    stride_o_seq: gl.constexpr,
    stride_o_dim: gl.constexpr,
    stride_o_token: gl.constexpr,
    # others
    pad_slot_id: gl.constexpr,
    # Meta-parameters
    HAS_BIAS: gl.constexpr,
    KERNEL_WIDTH: gl.constexpr,
    SILU_ACTIVATION: gl.constexpr,
    IS_CONTINUOUS_BATCHING: gl.constexpr,
    IS_SPEC_DECODING: gl.constexpr,
    NP2_STATELEN: gl.constexpr,
    USE_PAD_SLOT: gl.constexpr,
    BLOCK_N: gl.constexpr,
    SAVE_INTERMEDIATE: gl.constexpr,
):

    blocked: gl.constexpr = gl.BlockedLayout(
        size_per_thread=[2],
        threads_per_warp=[64],
        warps_per_cta=[2],
        order=[0],
    )

    blocked1: gl.constexpr = gl.BlockedLayout(
        size_per_thread=[1, 1],
        threads_per_warp=[4, 16],
        warps_per_cta=[1, 4],
        order=[0, 1],
    )
    blocked2: gl.constexpr = gl.BlockedLayout(
        size_per_thread=[1, 1],
        threads_per_warp=[1, 64],
        warps_per_cta=[1, 4],
        order=[1, 0],
    )

    shared_layout: gl.constexpr = gl.SwizzledSharedLayout(
        vec=1,
        per_phase=1,
        max_phase=1,
        order=[0]
    )

    # x_shared = gl.allocate_shared_memory(x_ptr.type.element_ty, [KERNEL_WIDTH - 1 + seqlen, BLOCK_N], shared_layout)  # [conv_state, x]

    update_index: gl.constexpr = KERNEL_WIDTH
    
    # ruff: noqa: E501
    idx_seq = gl.program_id(0)
    if idx_seq >= batch:
        return

    # [BLOCK_N,] elements along the feature-dimension (channel)
    idx_feats = gl.program_id(1) * BLOCK_N + gl.arange(0, BLOCK_N, layout=blocked)
    # idx_feats1 = gl.program_id(1) * BLOCK_N + gl.arange(0, BLOCK_N, layout=gl.SliceLayout(0, blocked1))
    # idx_feats2 = gl.program_id(1) * BLOCK_N + gl.arange(0, BLOCK_N, layout=gl.SliceLayout(0, blocked2))

    if IS_CONTINUOUS_BATCHING:
        # mask = idx_seq < batch
        conv_state_batch_coord = gl.load(
            conv_state_indices_ptr + idx_seq * stride_state_indices
        ).to(gl.int64)
    else:
        conv_state_batch_coord = idx_seq
    if USE_PAD_SLOT:  # noqa
        if conv_state_batch_coord == pad_slot_id:
            # not processing as this is not the actual sequence
            return

    if IS_SPEC_DECODING:
        # The rolling of conv state:
        #
        # Before forward, the conv_state is:
        # [history1, history2, ..., historyM].
        #
        # After forward, the conv_state becomes:
        # [history2, ..., historyM, draft1, draft2, ..., draftN].
        #
        # After acceptance, it becomes:
        #
        # - accept 1 tokens: [history2, ..., historyM, draft1]
        # - accept 2 tokens: [history3, ..., historyM, draft1, draft2]
        # - and so on.
        conv_state_token_offset = gl.load(num_accepted_tokens_ptr + idx_seq) - 1
    else:
        conv_state_token_offset = 0

    # STEP 1: READ init_state data
    conv_states_base = (
        conv_state_ptr
        + (conv_state_batch_coord * stride_conv_state_seq)
        + (idx_feats * stride_conv_state_dim)
    )
    mask_w = idx_feats < dim

    prior_tokens = conv_states_base + conv_state_token_offset * stride_conv_state_tok
    if KERNEL_WIDTH >= 2:
        conv_states_ptrs = prior_tokens  # [BLOCK_N]
        col0 = gl.load(conv_states_ptrs, mask_w, 0.0)
        conv_state_vecs = (col0,)
    if KERNEL_WIDTH >= 3:
        conv_states_ptrs = prior_tokens + 1 * stride_conv_state_tok  # [BLOCK_N]
        col1 = gl.load(conv_states_ptrs, mask_w, 0.0)
        conv_state_vecs = tuple_combine(conv_state_vecs, col1)
    if KERNEL_WIDTH >= 4:
        conv_states_ptrs = prior_tokens + 2 * stride_conv_state_tok  # [BLOCK_N]
        col2 = gl.load(conv_states_ptrs, mask_w, 0.0)
        conv_state_vecs = tuple_combine(conv_state_vecs, col2)
    if KERNEL_WIDTH == 5:
        conv_states_ptrs = prior_tokens + 3 * stride_conv_state_tok  # [BLOCK_N]
        col3 = gl.load(conv_states_ptrs, mask_w, 0.0)
        conv_state_vecs = tuple_combine(conv_state_vecs, col3)
    # # STEP 2: assume state_len > seqlen
    # idx_tokens = gl.arange(0, NP2_STATELEN, layout=gl.SliceLayout(1, blocked2))  # [BLOCK_M]

    # # The conv_state updates works in a sliding window manner,
    # # at each forward pass, the tokens are shift by 1, so we
    # # load since idx_tokens + 1.
    # conv_state_ptrs_source = (
    #     conv_state_ptr
    #     + (conv_state_batch_coord * stride_conv_state_seq)
    #     + conv_state_token_offset * stride_conv_state_tok
    #     + (idx_feats2 * stride_conv_state_dim)[None, :]
    #     + ((idx_tokens + (1 if IS_SPEC_DECODING else seqlen)) * stride_conv_state_tok)[
    #         :, None
    #     ]
    # )  # [BLOCK_M, BLOCK_N]
    # VAL = state_len - seqlen
    # mask = (
    #     (conv_state_batch_coord < num_cache_lines)
    #     & (idx_tokens < VAL)[:, None]
    #     & (idx_feats2 < dim)[None, :]
    # )
    # conv_state = gl.load(conv_state_ptrs_source, mask, other=0.0)


    # x_base = x_ptr + (idx_seq * stride_x_seq) + (idx_feats2 * stride_x_dim)  # [BLOCK_N]

    # x_ptrs = (
    #     x_base[None, :] + ((idx_tokens - VAL) * stride_x_token)[:, None]
    # )  # [BLOCK_M, BLOCK_N]

    # mask_x = (
    #     (idx_tokens - VAL >= 0)[:, None]
    #     & (idx_tokens - VAL < seqlen)[:, None]
    #     & (idx_feats2 < dim)[None, :]
    # )  # token-index  # token-index  # feature-index
    # loaded_x = gl.load(x_ptrs, mask_x, 0.0)
    # tl.debug_barrier()

    # new_conv_state = gl.where(mask, conv_state, loaded_x)

    conv_state_base = (
        conv_state_ptr
        + (conv_state_batch_coord * stride_conv_state_seq)
        + (idx_feats * stride_conv_state_dim)
    )  # [BLOCK_N,]
    # conv_state_ptrs_target = (
    #     conv_state_base + (idx_tokens * stride_conv_state_tok)[:, None]
    # )  # [BLOCK_M, BLOCK_N]
    # mask = (idx_tokens < state_len)[:, None] & (idx_feats2 < dim)[None, :]
    # gl.store(conv_state_ptrs_target, new_conv_state, mask)

    # STEP 3: init accumulator
    if HAS_BIAS:
        bias = bias_ptr + idx_feats
        mask_bias = idx_feats < dim
        acc_preload = gl.load(bias, mask=mask_bias, other=0.0).to(
            o_ptr.type.element_ty
        )  # [BLOCK_N]
    else:
        acc_preload = gl.zeros((BLOCK_N,), dtype=o_ptr.type.element_ty, layout=blocked)

    # STEP 4:
    # PRE-LOAD WEIGHTS
    # first kernel column, configured for weights to handle BLOCK_N features in range
    w_base = w_ptr + (idx_feats * stride_w_dim)  # [BLOCK_N,]
    mask_w = idx_feats < dim
    if KERNEL_WIDTH >= 2:
        w_ptrs = w_base + (0 * stride_w_width)  # [BLOCK_N] tensor
        w_col0 = gl.load(w_ptrs, mask_w, other=0.0)
        w_ptrs = w_base + (1 * stride_w_width)  # [BLOCK_N] tensor
        w_col1 = gl.load(w_ptrs, mask_w, other=0.0)
        w_vecs = (w_col0, w_col1)
    if KERNEL_WIDTH >= 3:
        w_ptrs = w_base + (2 * stride_w_width)  # [BLOCK_N] tensor
        w_col2 = gl.load(w_ptrs, mask_w, other=0.0)
        w_vecs = tuple_combine(w_vecs, w_col2)
    if KERNEL_WIDTH >= 4:
        w_ptrs = w_base + (3 * stride_w_width)  # [BLOCK_N] tensor
        w_col3 = gl.load(w_ptrs, mask_w, other=0.0)
        w_vecs = tuple_combine(w_vecs, w_col3)

    x_base_1d = x_ptr + (idx_seq * stride_x_seq) + (idx_feats * stride_x_dim)  # starting of chunk [BLOCK_N]
    mask_x_1d = idx_feats < dim

    # STEP 5: compute each token
    for idx_token in gl.static_range(seqlen):
        acc = acc_preload

        x_ptrs_1d = x_base_1d + idx_token * stride_x_token  # [BLOCK_N]
        x_vec = gl.load(x_ptrs_1d, mask=mask_x_1d)
        conv_state_vecs = tuple_combine(conv_state_vecs, x_vec)
        for j in gl.static_range(KERNEL_WIDTH):
            matrix_w = w_vecs[j]
            matrix_x = conv_state_vecs[j]

            acc += matrix_x * matrix_w  # [BLOCK_N]

        conv_state_vecs = conv_state_vecs[1:]

        if SILU_ACTIVATION:
            # Convert to fp32 for exp calculation, then convert back
            acc_fp32 = acc.to(gl.float32)
            acc = acc_fp32 / (1 + gl.exp(-acc_fp32))
            acc = acc.to(x_vec.dtype)
        mask_1d = (idx_token < seqlen) & (
            idx_feats < dim
        )  # token-index  # feature-index
        o_ptrs = (
            o_ptr
            + (idx_seq) * stride_o_seq
            + idx_token * stride_o_token
            + (idx_feats * stride_o_dim)
        )

        gl.store(o_ptrs, acc, mask=mask_1d)

        if SAVE_INTERMEDIATE:
            # Save the window state after consuming this token
            # Layout: [seq(cache line), step, dim, win(K-1)]
            base_ptr = (
                intermediate_conv_window_ptr
                + conv_state_batch_coord * stride_inter_seq
                + idx_token * stride_inter_step
                + idx_feats * stride_inter_dim
            )
            for l in gl.static_range(state_len):
                gl.store(base_ptr + l*stride_inter_win, conv_state_vecs[l], idx_feats < dim)

    for l in gl.static_range(state_len):
        gl.store(conv_state_base + l*stride_conv_state_tok, conv_state_vecs[l], idx_feats < dim)

@triton.autotune(
    configs=[
        # 调优 BLOCK_N、waves_per_eu 和 NUM_WARPS
        # 保持关系: BLOCK_N = size_per_thread × 64 × NUM_WARPS (size_per_thread 由 heuristics 自动计算)
        # 注意：autotune 同时测试多个配置时可能遇到 Gluon 编译器 bug
        # 建议：先单独测试每个配置确保正确性，然后再启用多个配置
        # triton.Config({'BLOCK_N': 2048}, num_warps=16, num_stages=1),
        # triton.Config({'BLOCK_N': 1024}, num_warps=8, num_stages=1),
        # triton.Config({'BLOCK_N': 512}, num_warps=4, num_stages=1),
        triton.Config({'BLOCK_N': 256}, num_warps=2, num_stages=1),
        # triton.Config({'BLOCK_N': 1024}, num_warps=16, num_stages=1),
        # triton.Config({'BLOCK_N': 512}, num_warps=8, num_stages=1),
        # triton.Config({'BLOCK_N': 256}, num_warps=4, num_stages=1),
        # triton.Config({'BLOCK_N': 128}, num_warps=2, num_stages=1),
        # triton.Config({'BLOCK_N': 2048, 'waves_per_eu': 4, 'NUM_WARPS': 16}, num_warps=16, num_stages=2),
    ],
    key=['dim'],  # key 参数决定何时重新 autotune
    reset_to_zero=['o_ptr'],  # 在测试不同配置时重置输出，防止结果累积
)
@triton.heuristics(values={
    'blocked': lambda args: make_block_layout_conv1d(
        args['x_ptr'],
        args['BLOCK_N'], 
        args['num_warps']
    )
})
@gluon.jit()
def gluon_causal_conv1d_update_kernel_v1(
    # Pointers to matrices
    x_ptr,  # (batch, dim, seqlen)
    w_ptr,  # (dim, width)
    bias_ptr,
    conv_state_ptr,
    cache_seqlens_ptr,  # circular buffer
    conv_state_indices_ptr,
    num_accepted_tokens_ptr,
    intermediate_conv_window_ptr,
    o_ptr,  # (batch, dim, seqlen)
    # Matrix dimensions
    batch: int,
    dim: gl.constexpr,
    seqlen: gl.constexpr,
    state_len: gl.constexpr,
    num_cache_lines: gl.constexpr,  # added to support vLLM larger cache lines
    # Strides
    stride_x_seq: gl.constexpr,
    stride_x_dim: gl.constexpr,
    stride_x_token: gl.constexpr,
    stride_w_dim: gl.constexpr,
    stride_w_width: gl.constexpr,
    stride_conv_state_seq: gl.constexpr,
    stride_conv_state_dim: gl.constexpr,
    stride_conv_state_tok: gl.constexpr,
    stride_state_indices: gl.constexpr,
    stride_inter_seq: gl.constexpr,
    stride_inter_step: gl.constexpr,
    stride_inter_dim: gl.constexpr,
    stride_inter_win: gl.constexpr,
    stride_o_seq: gl.constexpr,
    stride_o_dim: gl.constexpr,
    stride_o_token: gl.constexpr,
    # others
    pad_slot_id: gl.constexpr,
    # Meta-parameters
    HAS_BIAS: gl.constexpr,
    KERNEL_WIDTH: gl.constexpr,
    SILU_ACTIVATION: gl.constexpr,
    IS_CONTINUOUS_BATCHING: gl.constexpr,
    IS_SPEC_DECODING: gl.constexpr,
    NP2_STATELEN: gl.constexpr,
    USE_PAD_SLOT: gl.constexpr,
    BLOCK_N: gl.constexpr,
    SAVE_INTERMEDIATE: gl.constexpr,
    num_warps: gl.constexpr,
    blocked: gl.constexpr,
):

    # blocked: gl.constexpr = gl.BlockedLayout(
    #     size_per_thread=[2],
    #     threads_per_warp=[64],
    #     warps_per_cta=[2],
    #     order=[0],
    # )
    
    # ruff: noqa: E501
    idx_seq = gl.program_id(0)
    # if idx_seq >= batch:
    #     return

    # [BLOCK_N,] elements along the feature-dimension (channel)
    idx_feats = gl.program_id(1) * BLOCK_N + gl.arange(0, BLOCK_N, layout=blocked)

    conv_state_batch_coord = gl.load(
        conv_state_indices_ptr + idx_seq * stride_state_indices
    ).to(gl.int64)

    # if USE_PAD_SLOT:  # noqa
    #     if conv_state_batch_coord == pad_slot_id:
    #         # not processing as this is not the actual sequence
    #         return

    # STEP 1: READ init_state data
    conv_states_base = (
        conv_state_ptr
        + (conv_state_batch_coord * stride_conv_state_seq)
        + (idx_feats * stride_conv_state_dim)
    )
    mask = idx_feats < dim

    conv_states_ptrs = conv_states_base  # [BLOCK_N]
    col0 = gl.load(conv_states_ptrs, mask, 0.0)
    w_base = w_ptr + (idx_feats * stride_w_dim)  # [BLOCK_N,]
    w_ptrs = w_base + (0 * stride_w_width)  # [BLOCK_N] tensor
    w_col0 = gl.load(w_ptrs, mask, other=0.0)
    acc = gl.zeros((BLOCK_N,), dtype=o_ptr.type.element_ty, layout=blocked)
    acc += w_col0 * col0  # [BLOCK_N]

    conv_states_ptrs = conv_states_base + 1 * stride_conv_state_tok  # [BLOCK_N]
    col1 = gl.load(conv_states_ptrs, mask, 0.0)
    w_ptrs = w_base + (1 * stride_w_width)  # [BLOCK_N] tensor
    w_col1 = gl.load(w_ptrs, mask, other=0.0)
    acc += w_col1 * col1  # [BLOCK_N]

    conv_states_ptrs = conv_states_base + 2 * stride_conv_state_tok  # [BLOCK_N]
    col2 = gl.load(conv_states_ptrs, mask, 0.0)
    w_ptrs = w_base + (2 * stride_w_width)  # [BLOCK_N] tensor
    w_col2 = gl.load(w_ptrs, mask, other=0.0)
    acc += w_col2 * col2  # [BLOCK_N]

    w_ptrs = w_base + (3 * stride_w_width)  # [BLOCK_N] tensor
    w_col3 = gl.load(w_ptrs, mask, other=0.0)
    x_base_1d = x_ptr + (idx_seq * stride_x_seq) + (idx_feats * stride_x_dim)  # starting of chunk [BLOCK_N]
    x_vec = gl.load(x_base_1d, mask)
    acc += w_col3 * x_vec  # [BLOCK_N]

    # Convert to fp32 for exp calculation, then convert back
    acc_fp32 = acc.to(gl.float32)
    acc = acc_fp32 / (1 + gl.exp(-acc_fp32))
    acc = acc.to(x_vec.dtype)

    o_ptrs = (
        o_ptr
        + (idx_seq) * stride_o_seq
        + (idx_feats * stride_o_dim)
    )

    gl.store(o_ptrs, acc, mask)
    gl.store(conv_states_base, col1, mask)
    gl.store(conv_states_base + stride_conv_state_tok, col2, mask)
    gl.store(conv_states_base + 2 * stride_conv_state_tok, x_vec, mask)

@gluon.jit()
def gluon_causal_conv1d_update_kernel_v2(
    # Pointers to matrices
    x_ptr,  # (batch, dim, seqlen)
    w_ptr,  # (dim, width)
    bias_ptr,
    conv_state_ptr,
    cache_seqlens_ptr,  # circular buffer
    conv_state_indices_ptr,
    num_accepted_tokens_ptr,
    intermediate_conv_window_ptr,
    o_ptr,  # (batch, dim, seqlen)
    # Matrix dimensions
    batch: int,
    dim: gl.constexpr,
    seqlen: gl.constexpr,
    state_len: gl.constexpr,
    num_cache_lines: gl.constexpr,  # added to support vLLM larger cache lines
    # Strides
    stride_x_seq: gl.constexpr,
    stride_x_dim: gl.constexpr,
    stride_x_token: gl.constexpr,
    stride_w_dim: gl.constexpr,
    stride_w_width: gl.constexpr,
    stride_conv_state_seq: gl.constexpr,
    stride_conv_state_dim: gl.constexpr,
    stride_conv_state_tok: gl.constexpr,
    stride_state_indices: gl.constexpr,
    stride_inter_seq: gl.constexpr,
    stride_inter_step: gl.constexpr,
    stride_inter_dim: gl.constexpr,
    stride_inter_win: gl.constexpr,
    stride_o_seq: gl.constexpr,
    stride_o_dim: gl.constexpr,
    stride_o_token: gl.constexpr,
    # others
    pad_slot_id: gl.constexpr,
    # Meta-parameters
    HAS_BIAS: gl.constexpr,
    KERNEL_WIDTH: gl.constexpr,
    SILU_ACTIVATION: gl.constexpr,
    IS_CONTINUOUS_BATCHING: gl.constexpr,
    IS_SPEC_DECODING: gl.constexpr,
    NP2_STATELEN: gl.constexpr,
    USE_PAD_SLOT: gl.constexpr,
    BLOCK_N: gl.constexpr,
    SAVE_INTERMEDIATE: gl.constexpr,
):

    blocked: gl.constexpr = gl.BlockedLayout(
        size_per_thread=[2],
        threads_per_warp=[64],
        warps_per_cta=[2],
        order=[0],
    )

    blocked1: gl.constexpr = gl.BlockedLayout(
        size_per_thread=[1, 1],
        threads_per_warp=[4, 16],
        warps_per_cta=[1, 4],
        order=[0, 1],
    )
    blocked2: gl.constexpr = gl.BlockedLayout(
        size_per_thread=[1, 1],
        threads_per_warp=[1, 64],
        warps_per_cta=[1, 4],
        order=[1, 0],
    )

    shared_layout: gl.constexpr = gl.SwizzledSharedLayout(
        vec=1,
        per_phase=1,
        max_phase=1,
        order=[0]
    )

    # x_shared = gl.allocate_shared_memory(x_ptr.type.element_ty, [KERNEL_WIDTH - 1 + seqlen, BLOCK_N], shared_layout)  # [conv_state, x]

    update_index: gl.constexpr = KERNEL_WIDTH
    
    # ruff: noqa: E501
    idx_seq = gl.program_id(0)
    if idx_seq >= batch:
        return

    # [BLOCK_N,] elements along the feature-dimension (channel)
    idx_feats = gl.program_id(1) * BLOCK_N + gl.arange(0, BLOCK_N, layout=blocked)
    # idx_feats1 = gl.program_id(1) * BLOCK_N + gl.arange(0, BLOCK_N, layout=gl.SliceLayout(0, blocked1))
    # idx_feats2 = gl.program_id(1) * BLOCK_N + gl.arange(0, BLOCK_N, layout=gl.SliceLayout(0, blocked2))

    if IS_CONTINUOUS_BATCHING:
        # mask = idx_seq < batch
        conv_state_batch_coord = gl.load(
            conv_state_indices_ptr + idx_seq * stride_state_indices
        ).to(gl.int64)
    else:
        conv_state_batch_coord = idx_seq
    if USE_PAD_SLOT:  # noqa
        if conv_state_batch_coord == pad_slot_id:
            # not processing as this is not the actual sequence
            return

    if IS_SPEC_DECODING:
        # The rolling of conv state:
        #
        # Before forward, the conv_state is:
        # [history1, history2, ..., historyM].
        #
        # After forward, the conv_state becomes:
        # [history2, ..., historyM, draft1, draft2, ..., draftN].
        #
        # After acceptance, it becomes:
        #
        # - accept 1 tokens: [history2, ..., historyM, draft1]
        # - accept 2 tokens: [history3, ..., historyM, draft1, draft2]
        # - and so on.
        conv_state_token_offset = gl.load(num_accepted_tokens_ptr + idx_seq) - 1
    else:
        conv_state_token_offset = 0

    # STEP 1: READ init_state data
    conv_states_base = (
        conv_state_ptr
        + (conv_state_batch_coord * stride_conv_state_seq)
        + (idx_feats * stride_conv_state_dim)
    )
    mask_w = idx_feats < dim

    prior_tokens = conv_states_base + conv_state_token_offset * stride_conv_state_tok
    if KERNEL_WIDTH >= 2:
        conv_states_ptrs = prior_tokens  # [BLOCK_N]
        col0 = gl.load(conv_states_ptrs, mask_w, 0.0)
        conv_state_vecs = (col0,)
    if KERNEL_WIDTH >= 3:
        conv_states_ptrs = prior_tokens + 1 * stride_conv_state_tok  # [BLOCK_N]
        col1 = gl.load(conv_states_ptrs, mask_w, 0.0)
        conv_state_vecs = tuple_combine(conv_state_vecs, col1)
    if KERNEL_WIDTH >= 4:
        conv_states_ptrs = prior_tokens + 2 * stride_conv_state_tok  # [BLOCK_N]
        col2 = gl.load(conv_states_ptrs, mask_w, 0.0)
        conv_state_vecs = tuple_combine(conv_state_vecs, col2)
    if KERNEL_WIDTH == 5:
        conv_states_ptrs = prior_tokens + 3 * stride_conv_state_tok  # [BLOCK_N]
        col3 = gl.load(conv_states_ptrs, mask_w, 0.0)
        conv_state_vecs = tuple_combine(conv_state_vecs, col3)
    # # STEP 2: assume state_len > seqlen
    # idx_tokens = gl.arange(0, NP2_STATELEN, layout=gl.SliceLayout(1, blocked2))  # [BLOCK_M]

    # # The conv_state updates works in a sliding window manner,
    # # at each forward pass, the tokens are shift by 1, so we
    # # load since idx_tokens + 1.
    # conv_state_ptrs_source = (
    #     conv_state_ptr
    #     + (conv_state_batch_coord * stride_conv_state_seq)
    #     + conv_state_token_offset * stride_conv_state_tok
    #     + (idx_feats2 * stride_conv_state_dim)[None, :]
    #     + ((idx_tokens + (1 if IS_SPEC_DECODING else seqlen)) * stride_conv_state_tok)[
    #         :, None
    #     ]
    # )  # [BLOCK_M, BLOCK_N]
    # VAL = state_len - seqlen
    # mask = (
    #     (conv_state_batch_coord < num_cache_lines)
    #     & (idx_tokens < VAL)[:, None]
    #     & (idx_feats2 < dim)[None, :]
    # )
    # conv_state = gl.load(conv_state_ptrs_source, mask, other=0.0)


    # x_base = x_ptr + (idx_seq * stride_x_seq) + (idx_feats2 * stride_x_dim)  # [BLOCK_N]

    # x_ptrs = (
    #     x_base[None, :] + ((idx_tokens - VAL) * stride_x_token)[:, None]
    # )  # [BLOCK_M, BLOCK_N]

    # mask_x = (
    #     (idx_tokens - VAL >= 0)[:, None]
    #     & (idx_tokens - VAL < seqlen)[:, None]
    #     & (idx_feats2 < dim)[None, :]
    # )  # token-index  # token-index  # feature-index
    # loaded_x = gl.load(x_ptrs, mask_x, 0.0)
    # tl.debug_barrier()

    # new_conv_state = gl.where(mask, conv_state, loaded_x)

    conv_state_base = (
        conv_state_ptr
        + (conv_state_batch_coord * stride_conv_state_seq)
        + (idx_feats * stride_conv_state_dim)
    )  # [BLOCK_N,]
    # conv_state_ptrs_target = (
    #     conv_state_base + (idx_tokens * stride_conv_state_tok)[:, None]
    # )  # [BLOCK_M, BLOCK_N]
    # mask = (idx_tokens < state_len)[:, None] & (idx_feats2 < dim)[None, :]
    # gl.store(conv_state_ptrs_target, new_conv_state, mask)

    # STEP 3: init accumulator
    if HAS_BIAS:
        bias = bias_ptr + idx_feats
        mask_bias = idx_feats < dim
        acc_preload = gl.load(bias, mask=mask_bias, other=0.0).to(
            o_ptr.type.element_ty
        )  # [BLOCK_N]
    else:
        acc_preload = gl.zeros((BLOCK_N,), dtype=o_ptr.type.element_ty, layout=blocked)

    # STEP 4:
    # PRE-LOAD WEIGHTS
    # first kernel column, configured for weights to handle BLOCK_N features in range
    w_base = w_ptr + (idx_feats * stride_w_dim)  # [BLOCK_N,]
    mask_w = idx_feats < dim
    if KERNEL_WIDTH >= 2:
        w_ptrs = w_base + (0 * stride_w_width)  # [BLOCK_N] tensor
        w_col0 = gl.load(w_ptrs, mask_w, other=0.0)
        w_ptrs = w_base + (1 * stride_w_width)  # [BLOCK_N] tensor
        w_col1 = gl.load(w_ptrs, mask_w, other=0.0)
        w_vecs = (w_col0, w_col1)
    if KERNEL_WIDTH >= 3:
        w_ptrs = w_base + (2 * stride_w_width)  # [BLOCK_N] tensor
        w_col2 = gl.load(w_ptrs, mask_w, other=0.0)
        w_vecs = tuple_combine(w_vecs, w_col2)
    if KERNEL_WIDTH >= 4:
        w_ptrs = w_base + (3 * stride_w_width)  # [BLOCK_N] tensor
        w_col3 = gl.load(w_ptrs, mask_w, other=0.0)
        w_vecs = tuple_combine(w_vecs, w_col3)

    x_base_1d = x_ptr + (idx_seq * stride_x_seq) + (idx_feats * stride_x_dim)  # starting of chunk [BLOCK_N]
    mask_x_1d = idx_feats < dim

    # STEP 5: compute each token
    for idx_token in gl.static_range(seqlen):
        acc = acc_preload

        x_ptrs_1d = x_base_1d + idx_token * stride_x_token  # [BLOCK_N]
        x_vec = gl.load(x_ptrs_1d, mask=mask_x_1d)
        conv_state_vecs = tuple_combine(conv_state_vecs, x_vec)
        for j in gl.static_range(KERNEL_WIDTH):
            matrix_w = w_vecs[j]
            matrix_x = conv_state_vecs[j]

            acc += matrix_x * matrix_w  # [BLOCK_N]

        conv_state_vecs = conv_state_vecs[1:]

        if SILU_ACTIVATION:
            # Convert to fp32 for exp calculation, then convert back
            acc_fp32 = acc.to(gl.float32)
            acc = acc_fp32 / (1 + gl.exp(-acc_fp32))
            acc = acc.to(x_vec.dtype)
        mask_1d = (idx_token < seqlen) & (
            idx_feats < dim
        )  # token-index  # feature-index
        o_ptrs = (
            o_ptr
            + (idx_seq) * stride_o_seq
            + idx_token * stride_o_token
            + (idx_feats * stride_o_dim)
        )

        gl.store(o_ptrs, acc, mask=mask_1d)

        if SAVE_INTERMEDIATE:
            # Save the window state after consuming this token
            # Layout: [seq(cache line), step, dim, win(K-1)]
            base_ptr = (
                intermediate_conv_window_ptr
                + conv_state_batch_coord * stride_inter_seq
                + idx_token * stride_inter_step
                + idx_feats * stride_inter_dim
            )
            for l in gl.static_range(state_len):
                gl.store(base_ptr + l*stride_inter_win, conv_state_vecs[l], idx_feats < dim)

    for l in gl.static_range(state_len):
        gl.store(conv_state_base + l*stride_conv_state_tok, conv_state_vecs[l], idx_feats < dim)

@gluon.jit()
def gluon_causal_conv1d_update_persistent_kernel(
    # Pointers to matrices
    x_ptr,  # (batch, dim, seqlen)
    w_ptr,  # (dim, width)
    bias_ptr,
    conv_state_ptr,
    cache_seqlens_ptr,  # circular buffer
    conv_state_indices_ptr,
    num_accepted_tokens_ptr,
    intermediate_conv_window_ptr,
    o_ptr,  # (batch, dim, seqlen)
    # Matrix dimensions
    batch: int,
    dim: gl.constexpr,
    seqlen: gl.constexpr,
    state_len: gl.constexpr,
    num_cache_lines: gl.constexpr,  # added to support vLLM larger cache lines
    # Strides
    stride_x_seq: gl.constexpr,
    stride_x_dim: gl.constexpr,
    stride_x_token: gl.constexpr,
    stride_w_dim: gl.constexpr,
    stride_w_width: gl.constexpr,
    stride_conv_state_seq: gl.constexpr,
    stride_conv_state_dim: gl.constexpr,
    stride_conv_state_tok: gl.constexpr,
    stride_state_indices: gl.constexpr,
    stride_inter_seq: gl.constexpr,
    stride_inter_step: gl.constexpr,
    stride_inter_dim: gl.constexpr,
    stride_inter_win: gl.constexpr,
    stride_o_seq: gl.constexpr,
    stride_o_dim: gl.constexpr,
    stride_o_token: gl.constexpr,
    # others
    pad_slot_id: gl.constexpr,
    # Meta-parameters
    HAS_BIAS: gl.constexpr,
    KERNEL_WIDTH: gl.constexpr,
    SILU_ACTIVATION: gl.constexpr,
    IS_CONTINUOUS_BATCHING: gl.constexpr,
    IS_SPEC_DECODING: gl.constexpr,
    NP2_STATELEN: gl.constexpr,
    USE_PAD_SLOT: gl.constexpr,
    BLOCK_N: gl.constexpr,
    SAVE_INTERMEDIATE: gl.constexpr,
):

    blocked: gl.constexpr = gl.BlockedLayout(
        size_per_thread=[2],
        threads_per_warp=[64],
        warps_per_cta=[16],
        order=[0],
    )

    cu_idx = gl.program_id(0)  # Current CU ID (0-79)
    num_cus = 80                # Fixed number of CUs
    
    # Calculate total number of tasks
    num_dim_blocks = gl.cdiv(dim, BLOCK_N)  # Number of blocks needed in dim direction
    total_tasks = batch * num_dim_blocks     # Total number of tasks
    
    # Calculate number of tasks per CU (dynamic load balancing)
    per_cu_tasks = total_tasks // num_cus    # Base number of tasks per CU
    cu_mores = total_tasks % num_cus         # Remaining tasks

    # First cu_mores CUs handle 1 additional task
    cu_num_tasks = (per_cu_tasks + 1) if cu_idx < cu_mores else per_cu_tasks
    cu_task_offset = cu_idx * (per_cu_tasks + 1) if cu_idx < cu_mores else (cu_mores * (per_cu_tasks + 1) + (cu_idx - cu_mores) * per_cu_tasks)

    for task_idx in range(cu_num_tasks):
        global_task_id = cu_task_offset + task_idx
        idx_seq = global_task_id // num_dim_blocks
        dim_block_idx = global_task_id % num_dim_blocks
        idx_feats = dim_block_idx * BLOCK_N + gl.arange(0, BLOCK_N, layout=blocked)
        
        # Original processing logic remains unchanged
        if IS_CONTINUOUS_BATCHING:
            # mask = idx_seq < batch
            conv_state_batch_coord = gl.load(
                conv_state_indices_ptr + idx_seq * stride_state_indices
            ).to(gl.int64)
        else:
            conv_state_batch_coord = idx_seq
        
        # Check if this task should be processed
        should_process = True
        if USE_PAD_SLOT:  # noqa
            if conv_state_batch_coord == pad_slot_id:
                # not processing as this is not the actual sequence
                should_process = False
        
        # Only process if this is a valid task
        if should_process:
            if IS_SPEC_DECODING:
                # The rolling of conv state:
                #
                # Before forward, the conv_state is:
                # [history1, history2, ..., historyM].
                #
                # After forward, the conv_state becomes:
                # [history2, ..., historyM, draft1, draft2, ..., draftN].
                #
                # After acceptance, it becomes:
                #
                # - accept 1 tokens: [history2, ..., historyM, draft1]
                # - accept 2 tokens: [history3, ..., historyM, draft1, draft2]
                # - and so on.
                conv_state_token_offset = tl.load(num_accepted_tokens_ptr + idx_seq) - 1
            else:
                conv_state_token_offset = 0
            # STEP 1: READ init_state data
            conv_states_base = (
                conv_state_ptr
                + (conv_state_batch_coord * stride_conv_state_seq)
                + (idx_feats * stride_conv_state_dim)
            )
            mask_w = idx_feats < dim

            prior_tokens = conv_states_base + conv_state_token_offset * stride_conv_state_tok
            if KERNEL_WIDTH >= 2:
                conv_states_ptrs = prior_tokens  # [BLOCK_N]
                col0 = gl.load(conv_states_ptrs, mask_w, 0.0)
                conv_state_vecs = (col0,)
            if KERNEL_WIDTH >= 3:
                conv_states_ptrs = prior_tokens + 1 * stride_conv_state_tok  # [BLOCK_N]
                col1 = gl.load(conv_states_ptrs, mask_w, 0.0)
                conv_state_vecs = tuple_combine(conv_state_vecs, col1)
            if KERNEL_WIDTH >= 4:
                conv_states_ptrs = prior_tokens + 2 * stride_conv_state_tok  # [BLOCK_N]
                col2 = gl.load(conv_states_ptrs, mask_w, 0.0)
                conv_state_vecs = tuple_combine(conv_state_vecs, col2)
            if KERNEL_WIDTH == 5:
                conv_states_ptrs = prior_tokens + 3 * stride_conv_state_tok  # [BLOCK_N]
                col3 = gl.load(conv_states_ptrs, mask_w, 0.0)
                conv_state_vecs = tuple_combine(conv_state_vecs, col3)
            # # STEP 2: assume state_len > seqlen
            # idx_tokens = gl.arange(0, NP2_STATELEN, layout=gl.SliceLayout(1, blocked2))  # [BLOCK_M]

            # # The conv_state updates works in a sliding window manner,
            # # at each forward pass, the tokens are shift by 1, so we
            # # load since idx_tokens + 1.
            # conv_state_ptrs_source = (
            #     conv_state_ptr
            #     + (conv_state_batch_coord * stride_conv_state_seq)
            #     + conv_state_token_offset * stride_conv_state_tok
            #     + (idx_feats2 * stride_conv_state_dim)[None, :]
            #     + ((idx_tokens + (1 if IS_SPEC_DECODING else seqlen)) * stride_conv_state_tok)[
            #         :, None
            #     ]
            # )  # [BLOCK_M, BLOCK_N]
            # VAL = state_len - seqlen
            # mask = (
            #     (conv_state_batch_coord < num_cache_lines)
            #     & (idx_tokens < VAL)[:, None]
            #     & (idx_feats2 < dim)[None, :]
            # )
            # conv_state = gl.load(conv_state_ptrs_source, mask, other=0.0)


            # x_base = x_ptr + (idx_seq * stride_x_seq) + (idx_feats2 * stride_x_dim)  # [BLOCK_N]

            # x_ptrs = (
            #     x_base[None, :] + ((idx_tokens - VAL) * stride_x_token)[:, None]
            # )  # [BLOCK_M, BLOCK_N]

            # mask_x = (
            #     (idx_tokens - VAL >= 0)[:, None]
            #     & (idx_tokens - VAL < seqlen)[:, None]
            #     & (idx_feats2 < dim)[None, :]
            # )  # token-index  # token-index  # feature-index
            # loaded_x = gl.load(x_ptrs, mask_x, 0.0)
            # tl.debug_barrier()

            # new_conv_state = gl.where(mask, conv_state, loaded_x)

            conv_state_base = (
                conv_state_ptr
                + (conv_state_batch_coord * stride_conv_state_seq)
                + (idx_feats * stride_conv_state_dim)
            )  # [BLOCK_N,]
            # conv_state_ptrs_target = (
            #     conv_state_base + (idx_tokens * stride_conv_state_tok)[:, None]
            # )  # [BLOCK_M, BLOCK_N]
            # mask = (idx_tokens < state_len)[:, None] & (idx_feats2 < dim)[None, :]
            # gl.store(conv_state_ptrs_target, new_conv_state, mask)

            # STEP 3: init accumulator
            if HAS_BIAS:
                bias = bias_ptr + idx_feats
                mask_bias = idx_feats < dim
                acc_preload = gl.load(bias, mask=mask_bias, other=0.0).to(
                    o_ptr.type.element_ty
                )  # [BLOCK_N]
            else:
                acc_preload = gl.zeros((BLOCK_N,), dtype=o_ptr.type.element_ty, layout=blocked)

            # STEP 4:
            # PRE-LOAD WEIGHTS
            # first kernel column, configured for weights to handle BLOCK_N features in range
            w_base = w_ptr + (idx_feats * stride_w_dim)  # [BLOCK_N,]
            mask_w = idx_feats < dim
            if KERNEL_WIDTH >= 2:
                w_ptrs = w_base + (0 * stride_w_width)  # [BLOCK_N] tensor
                w_col0 = gl.load(w_ptrs, mask_w, other=0.0)
                w_ptrs = w_base + (1 * stride_w_width)  # [BLOCK_N] tensor
                w_col1 = gl.load(w_ptrs, mask_w, other=0.0)
                w_vecs = (w_col0, w_col1)
            if KERNEL_WIDTH >= 3:
                w_ptrs = w_base + (2 * stride_w_width)  # [BLOCK_N] tensor
                w_col2 = gl.load(w_ptrs, mask_w, other=0.0)
                w_vecs = tuple_combine(w_vecs, w_col2)
            if KERNEL_WIDTH >= 4:
                w_ptrs = w_base + (3 * stride_w_width)  # [BLOCK_N] tensor
                w_col3 = gl.load(w_ptrs, mask_w, other=0.0)
                w_vecs = tuple_combine(w_vecs, w_col3)

            x_base_1d = x_ptr + (idx_seq * stride_x_seq) + (idx_feats * stride_x_dim)  # starting of chunk [BLOCK_N]
            mask_x_1d = idx_feats < dim

            # STEP 5: compute each token
            for idx_token in gl.static_range(seqlen):
                acc = acc_preload

                x_ptrs_1d = x_base_1d + idx_token * stride_x_token  # [BLOCK_N]
                x_vec = gl.load(x_ptrs_1d, mask=mask_x_1d)
                conv_state_vecs = tuple_combine(conv_state_vecs, x_vec)
                for j in gl.static_range(KERNEL_WIDTH):
                    matrix_w = w_vecs[j]
                    matrix_x = conv_state_vecs[j]

                    acc += matrix_x * matrix_w  # [BLOCK_N]

                conv_state_vecs = conv_state_vecs[1:]

                if SILU_ACTIVATION:
                    # Convert to fp32 for exp calculation, then convert back
                    acc_fp32 = acc.to(gl.float32)
                    acc = acc_fp32 / (1 + gl.exp(-acc_fp32))
                    acc = acc.to(x_vec.dtype)
                mask_1d = (idx_token < seqlen) & (
                    idx_feats < dim
                )  # token-index  # feature-index
                o_ptrs = (
                    o_ptr
                    + (idx_seq) * stride_o_seq
                    + idx_token * stride_o_token
                    + (idx_feats * stride_o_dim)
                )

                gl.store(o_ptrs, acc, mask=mask_1d)

                if SAVE_INTERMEDIATE:
                    # Save the window state after consuming this token
                    # Layout: [seq(cache line), step, dim, win(K-1)]
                    base_ptr = (
                        intermediate_conv_window_ptr
                        + conv_state_batch_coord * stride_inter_seq
                        + idx_token * stride_inter_step
                        + idx_feats * stride_inter_dim
                    )
                    for l in gl.static_range(state_len):
                        gl.store(base_ptr + l*stride_inter_win, conv_state_vecs[l], idx_feats < dim)

            for l in gl.static_range(state_len):
                gl.store(conv_state_base + l*stride_conv_state_tok, conv_state_vecs[l], idx_feats < dim)

@gluon.jit()
def gluon_causal_conv1d_update_persistent_kernel_v1(
    # Pointers to matrices
    x_ptr,  # (batch, dim, seqlen)
    w_ptr,  # (dim, width)
    bias_ptr,
    conv_state_ptr,
    cache_seqlens_ptr,  # circular buffer
    conv_state_indices_ptr,
    num_accepted_tokens_ptr,
    intermediate_conv_window_ptr,
    o_ptr,  # (batch, dim, seqlen)
    # Matrix dimensions
    batch: int,
    dim: gl.constexpr,
    seqlen: gl.constexpr,
    state_len: gl.constexpr,
    num_cache_lines: gl.constexpr,  # added to support vLLM larger cache lines
    # Strides
    stride_x_seq: gl.constexpr,
    stride_x_dim: gl.constexpr,
    stride_x_token: gl.constexpr,
    stride_w_dim: gl.constexpr,
    stride_w_width: gl.constexpr,
    stride_conv_state_seq: gl.constexpr,
    stride_conv_state_dim: gl.constexpr,
    stride_conv_state_tok: gl.constexpr,
    stride_state_indices: gl.constexpr,
    stride_inter_seq: gl.constexpr,
    stride_inter_step: gl.constexpr,
    stride_inter_dim: gl.constexpr,
    stride_inter_win: gl.constexpr,
    stride_o_seq: gl.constexpr,
    stride_o_dim: gl.constexpr,
    stride_o_token: gl.constexpr,
    # others
    pad_slot_id: gl.constexpr,
    # Meta-parameters
    HAS_BIAS: gl.constexpr,
    KERNEL_WIDTH: gl.constexpr,
    SILU_ACTIVATION: gl.constexpr,
    IS_CONTINUOUS_BATCHING: gl.constexpr,
    IS_SPEC_DECODING: gl.constexpr,
    NP2_STATELEN: gl.constexpr,
    USE_PAD_SLOT: gl.constexpr,
    BLOCK_N: gl.constexpr,
    SAVE_INTERMEDIATE: gl.constexpr,
):
    # Note: Gluon does not support device_print, so parameter printing is done in Python wrapper
    # See the wrapper function for parameter logging
    
    blocked: gl.constexpr = gl.BlockedLayout(
        size_per_thread=[2],
        threads_per_warp=[64],
        warps_per_cta=[8],
        order=[0],
    )

    cu_idx = gl.program_id(0)  # Current CU ID (0-79)
    num_cus = 80                # Fixed number of CUs
    
    # Calculate total number of tasks
    num_dim_blocks = gl.cdiv(dim, BLOCK_N)  # Number of blocks needed in dim direction
    total_tasks = batch * num_dim_blocks     # Total number of tasks
    
    # New mapping: each CU processes tasks from the same dim_block across different batches
    # Determine dim_block_idx directly from cu_idx
    cus_per_dim_block = gl.cdiv(num_cus, num_dim_blocks)  # How many CUs allocated per dim_block
    dim_block_idx = cu_idx // cus_per_dim_block  # Which dim_block this CU handles
    cu_idx_in_dim_block = cu_idx % cus_per_dim_block  # Index of this CU within the dim_block group
    
    # Calculate number of batch tasks per CU for this dim_block
    per_cu_batches = batch // cus_per_dim_block  # Base number of batches per CU
    cu_mores = batch % cus_per_dim_block  # Remaining batches
    
    # First cu_mores CUs (within this dim_block group) handle 1 additional batch
    cu_num_tasks = (per_cu_batches + 1) if cu_idx_in_dim_block < cu_mores else per_cu_batches
    cu_task_offset = cu_idx_in_dim_block * (per_cu_batches + 1) if cu_idx_in_dim_block < cu_mores else (cu_mores * (per_cu_batches + 1) + (cu_idx_in_dim_block - cu_mores) * per_cu_batches)

    # idx_feats is constant for all tasks in this CU (same dim_block_idx)
    idx_feats = dim_block_idx * BLOCK_N + gl.arange(0, BLOCK_N, layout=blocked)

    # STEP 4:
    # PRE-LOAD WEIGHTS
    # first kernel column, configured for weights to handle BLOCK_N features in range
    w_base = w_ptr + (idx_feats * stride_w_dim)  # [BLOCK_N,]
    mask_w = idx_feats < dim
    w_ptrs = w_base + (0 * stride_w_width)  # [BLOCK_N] tensor
    w_col0 = gl.load(w_ptrs, mask_w, other=0.0)
    w_ptrs = w_base + (1 * stride_w_width)  # [BLOCK_N] tensor
    w_col1 = gl.load(w_ptrs, mask_w, other=0.0)
    w_vecs = (w_col0, w_col1)
    w_ptrs = w_base + (2 * stride_w_width)  # [BLOCK_N] tensor
    w_col2 = gl.load(w_ptrs, mask_w, other=0.0)
    w_vecs = tuple_combine(w_vecs, w_col2)
    w_ptrs = w_base + (3 * stride_w_width)  # [BLOCK_N] tensor
    w_col3 = gl.load(w_ptrs, mask_w, other=0.0)
    w_vecs = tuple_combine(w_vecs, w_col3)

    for task_idx in range(cu_num_tasks):
        # All tasks in this CU belong to the same dim_block_idx but different batches
        idx_seq = cu_task_offset + task_idx  # Batch index
        
        # # Original processing logic remains unchanged
        # if IS_CONTINUOUS_BATCHING:
        # mask = idx_seq < batch
        conv_state_batch_coord = gl.load(
            conv_state_indices_ptr + idx_seq * stride_state_indices
        ).to(gl.int64)
        # else:
        #     conv_state_batch_coord = idx_seq
        
        # Check if this task should be processed
        # should_process = True
        # if USE_PAD_SLOT:  # noqa
        #     if conv_state_batch_coord == pad_slot_id:
        #         # not processing as this is not the actual sequence
        #         should_process = False
        
        # Only process if this is a valid task
        # if should_process:
        conv_state_token_offset = 0
        # STEP 1: READ init_state data
        conv_states_base = (
            conv_state_ptr
            + (conv_state_batch_coord * stride_conv_state_seq)
            + (idx_feats * stride_conv_state_dim)
        )
        mask_w = idx_feats < dim

        prior_tokens = conv_states_base + conv_state_token_offset * stride_conv_state_tok
    # if KERNEL_WIDTH >= 2:
        conv_states_ptrs = prior_tokens  # [BLOCK_N]
        col0 = gl.load(conv_states_ptrs, mask_w, 0.0)
        conv_state_vecs = (col0,)
    # if KERNEL_WIDTH >= 3:
        conv_states_ptrs = prior_tokens + 1 * stride_conv_state_tok  # [BLOCK_N]
        col1 = gl.load(conv_states_ptrs, mask_w, 0.0)
        conv_state_vecs = tuple_combine(conv_state_vecs, col1)
    # if KERNEL_WIDTH >= 4:
        conv_states_ptrs = prior_tokens + 2 * stride_conv_state_tok  # [BLOCK_N]
        col2 = gl.load(conv_states_ptrs, mask_w, 0.0)
        conv_state_vecs = tuple_combine(conv_state_vecs, col2)

        conv_state_base = (
            conv_state_ptr
            + (conv_state_batch_coord * stride_conv_state_seq)
            + (idx_feats * stride_conv_state_dim)
        )  # [BLOCK_N,]
        # conv_state_ptrs_target = (
        #     conv_state_base + (idx_tokens * stride_conv_state_tok)[:, None]
        # )  # [BLOCK_M, BLOCK_N]
        # mask = (idx_tokens < state_len)[:, None] & (idx_feats2 < dim)[None, :]
        # gl.store(conv_state_ptrs_target, new_conv_state, mask)

        # STEP 3: init accumulator
        if HAS_BIAS:
            bias = bias_ptr + idx_feats
            mask_bias = idx_feats < dim
            acc_preload = gl.load(bias, mask=mask_bias, other=0.0).to(
                o_ptr.type.element_ty
            )  # [BLOCK_N]
        else:
            acc_preload = gl.zeros((BLOCK_N,), dtype=o_ptr.type.element_ty, layout=blocked)

        x_base_1d = x_ptr + (idx_seq * stride_x_seq) + (idx_feats * stride_x_dim)  # starting of chunk [BLOCK_N]
        mask_x_1d = idx_feats < dim

        # STEP 5: compute each token
        for idx_token in gl.static_range(seqlen):
            acc = acc_preload

            x_ptrs_1d = x_base_1d + idx_token * stride_x_token  # [BLOCK_N]
            x_vec = gl.load(x_ptrs_1d, mask=mask_x_1d)
            conv_state_vecs = tuple_combine(conv_state_vecs, x_vec)
            for j in gl.static_range(KERNEL_WIDTH):
                matrix_w = w_vecs[j]
                matrix_x = conv_state_vecs[j]

                acc += matrix_x * matrix_w  # [BLOCK_N]

            conv_state_vecs = conv_state_vecs[1:]

            # if SILU_ACTIVATION:
            # Convert to fp32 for exp calculation, then convert back
            acc_fp32 = acc.to(gl.float32)
            acc = acc_fp32 / (1 + gl.exp(-acc_fp32))
            acc = acc.to(x_vec.dtype)

            mask_1d = (idx_token < seqlen) & (
                idx_feats < dim
            )  # token-index  # feature-index
            o_ptrs = (
                o_ptr
                + (idx_seq) * stride_o_seq
                + idx_token * stride_o_token
                + (idx_feats * stride_o_dim)
            )

            gl.store(o_ptrs, acc, mask=mask_1d)

        for l in gl.static_range(state_len):
            gl.store(conv_state_base + l*stride_conv_state_tok, conv_state_vecs[l], idx_feats < dim)

@triton.autotune(
    configs=[
        # 调优 BLOCK_N、waves_per_eu 和 NUM_WARPS
        # 保持关系: BLOCK_N = size_per_thread × 64 × NUM_WARPS (size_per_thread 由 heuristics 自动计算)
        # 注意：autotune 同时测试多个配置时可能遇到 Gluon 编译器 bug
        # 建议：先单独测试每个配置确保正确性，然后再启用多个配置
        # triton.Config({'BLOCK_N': 2048}, num_warps=16, num_stages=1),
        # triton.Config({'BLOCK_N': 2048}, num_warps=16, num_stages=2),
        # triton.Config({'BLOCK_N': 2048}, num_warps=16, num_stages=3),
        # triton.Config({'BLOCK_N': 2048}, num_warps=16, num_stages=4),
        triton.Config({'BLOCK_N': 1024}, num_warps=8, num_stages=1),
        # triton.Config({'BLOCK_N': 256}, num_warps=4, num_stages=1),
        # triton.Config({'BLOCK_N': 512, 'waves_per_eu': 1, 'NUM_WARPS': 8}, num_warps=8, num_stages=1),
        # triton.Config({'BLOCK_N': 1024, 'waves_per_eu': 2, 'NUM_WARPS': 8}, num_warps=8, num_stages=2),
        # triton.Config({'BLOCK_N': 2048, 'waves_per_eu': 4, 'NUM_WARPS': 16}, num_warps=16, num_stages=1),
        # triton.Config({'BLOCK_N': 2048, 'waves_per_eu': 4, 'NUM_WARPS': 16}, num_warps=16, num_stages=2),
    ],
    key=['dim'],  # key 参数决定何时重新 autotune
    reset_to_zero=['o_ptr'],  # 在测试不同配置时重置输出，防止结果累积
)
@triton.heuristics(values={
    'blocked': lambda args: make_block_layout_conv1d(
        args['x_ptr'],  # 使用 w_ptr 获取 dtype（x_ptr 和 w_ptr 的 dtype 应该相同）
        args['BLOCK_N'], 
        args['num_warps']
    )
})
@gluon.jit()
def gluon_causal_conv1d_update_persistent_kernel_v2(
    # Pointers to matrices
    x_ptr,  # (batch, dim, seqlen)
    w_ptr,  # (dim, width)
    bias_ptr,
    conv_state_ptr,
    cache_seqlens_ptr,  # circular buffer
    conv_state_indices_ptr,
    num_accepted_tokens_ptr,
    intermediate_conv_window_ptr,
    o_ptr,  # (batch, dim, seqlen)
    # Matrix dimensions
    batch: int,
    dim: gl.constexpr,
    seqlen: gl.constexpr,
    state_len: gl.constexpr,
    num_cache_lines: gl.constexpr,  # added to support vLLM larger cache lines
    # Strides
    stride_x_seq: gl.constexpr,
    stride_x_dim: gl.constexpr,
    stride_x_token: gl.constexpr,
    stride_w_dim: gl.constexpr,
    stride_w_width: gl.constexpr,
    stride_conv_state_seq: gl.constexpr,
    stride_conv_state_dim: gl.constexpr,
    stride_conv_state_tok: gl.constexpr,
    stride_state_indices: gl.constexpr,
    stride_inter_seq: gl.constexpr,
    stride_inter_step: gl.constexpr,
    stride_inter_dim: gl.constexpr,
    stride_inter_win: gl.constexpr,
    stride_o_seq: gl.constexpr,
    stride_o_dim: gl.constexpr,
    stride_o_token: gl.constexpr,
    # others
    pad_slot_id: gl.constexpr,
    # Meta-parameters
    HAS_BIAS: gl.constexpr,
    KERNEL_WIDTH: gl.constexpr,
    SILU_ACTIVATION: gl.constexpr,
    IS_CONTINUOUS_BATCHING: gl.constexpr,
    IS_SPEC_DECODING: gl.constexpr,
    NP2_STATELEN: gl.constexpr,
    USE_PAD_SLOT: gl.constexpr,
    BLOCK_N: gl.constexpr,
    SAVE_INTERMEDIATE: gl.constexpr,
    num_warps: gl.constexpr,
    blocked: gl.constexpr,  # 通过 heuristics 动态计算
):

    cu_idx = gl.program_id(0)  # Current CU ID (0-79)
    num_cus = 80                # Fixed number of CUs
    
    # Calculate total number of tasks
    num_dim_blocks = gl.cdiv(dim, BLOCK_N)  # Number of blocks needed in dim direction
    total_tasks = batch * num_dim_blocks     # Total number of tasks
    
    # Calculate number of tasks per CU (dynamic load balancing)
    per_cu_tasks = total_tasks // num_cus    # Base number of tasks per CU
    cu_mores = total_tasks % num_cus         # Remaining tasks

    # First cu_mores CUs handle 1 additional task
    cu_num_tasks = (per_cu_tasks + 1) if cu_idx < cu_mores else per_cu_tasks
    cu_task_offset = cu_idx * (per_cu_tasks + 1) if cu_idx < cu_mores else (cu_mores * (per_cu_tasks + 1) + (cu_idx - cu_mores) * per_cu_tasks)

    for task_idx in range(cu_num_tasks):
        global_task_id = cu_task_offset + task_idx
        idx_seq = global_task_id // num_dim_blocks
        dim_block_idx = global_task_id % num_dim_blocks
        idx_feats = dim_block_idx * BLOCK_N + gl.arange(0, BLOCK_N, layout=blocked)
        
        # Original processing logic remains unchanged
        if IS_CONTINUOUS_BATCHING:
            # mask = idx_seq < batch
            conv_state_batch_coord = gl.load(
                conv_state_indices_ptr + idx_seq * stride_state_indices
            ).to(gl.int64)
        else:
            conv_state_batch_coord = idx_seq
        
        # Check if this task should be processed
        should_process = True
        if USE_PAD_SLOT:  # noqa
            if conv_state_batch_coord == pad_slot_id:
                # not processing as this is not the actual sequence
                should_process = False
        
        # Only process if this is a valid task
        if should_process:
            if IS_SPEC_DECODING:
                # The rolling of conv state:
                #
                # Before forward, the conv_state is:
                # [history1, history2, ..., historyM].
                #
                # After forward, the conv_state becomes:
                # [history2, ..., historyM, draft1, draft2, ..., draftN].
                #
                # After acceptance, it becomes:
                #
                # - accept 1 tokens: [history2, ..., historyM, draft1]
                # - accept 2 tokens: [history3, ..., historyM, draft1, draft2]
                # - and so on.
                conv_state_token_offset = tl.load(num_accepted_tokens_ptr + idx_seq) - 1
            else:
                conv_state_token_offset = 0
            # STEP 1: READ init_state data
            conv_states_base = (
                conv_state_ptr
                + (conv_state_batch_coord * stride_conv_state_seq)
                + (idx_feats * stride_conv_state_dim)
            )
            mask_w = idx_feats < dim

            prior_tokens = conv_states_base + conv_state_token_offset * stride_conv_state_tok
            if KERNEL_WIDTH >= 2:
                conv_states_ptrs = prior_tokens  # [BLOCK_N]
                col0 = gl.load(conv_states_ptrs, mask_w, 0.0)
                conv_state_vecs = (col0,)
            if KERNEL_WIDTH >= 3:
                conv_states_ptrs = prior_tokens + 1 * stride_conv_state_tok  # [BLOCK_N]
                col1 = gl.load(conv_states_ptrs, mask_w, 0.0)
                conv_state_vecs = tuple_combine(conv_state_vecs, col1)
            if KERNEL_WIDTH >= 4:
                conv_states_ptrs = prior_tokens + 2 * stride_conv_state_tok  # [BLOCK_N]
                col2 = gl.load(conv_states_ptrs, mask_w, 0.0)
                conv_state_vecs = tuple_combine(conv_state_vecs, col2)
            if KERNEL_WIDTH == 5:
                conv_states_ptrs = prior_tokens + 3 * stride_conv_state_tok  # [BLOCK_N]
                col3 = gl.load(conv_states_ptrs, mask_w, 0.0)
                conv_state_vecs = tuple_combine(conv_state_vecs, col3)
            # # STEP 2: assume state_len > seqlen
            # idx_tokens = gl.arange(0, NP2_STATELEN, layout=gl.SliceLayout(1, blocked2))  # [BLOCK_M]

            # # The conv_state updates works in a sliding window manner,
            # # at each forward pass, the tokens are shift by 1, so we
            # # load since idx_tokens + 1.
            # conv_state_ptrs_source = (
            #     conv_state_ptr
            #     + (conv_state_batch_coord * stride_conv_state_seq)
            #     + conv_state_token_offset * stride_conv_state_tok
            #     + (idx_feats2 * stride_conv_state_dim)[None, :]
            #     + ((idx_tokens + (1 if IS_SPEC_DECODING else seqlen)) * stride_conv_state_tok)[
            #         :, None
            #     ]
            # )  # [BLOCK_M, BLOCK_N]
            # VAL = state_len - seqlen
            # mask = (
            #     (conv_state_batch_coord < num_cache_lines)
            #     & (idx_tokens < VAL)[:, None]
            #     & (idx_feats2 < dim)[None, :]
            # )
            # conv_state = gl.load(conv_state_ptrs_source, mask, other=0.0)


            # x_base = x_ptr + (idx_seq * stride_x_seq) + (idx_feats2 * stride_x_dim)  # [BLOCK_N]

            # x_ptrs = (
            #     x_base[None, :] + ((idx_tokens - VAL) * stride_x_token)[:, None]
            # )  # [BLOCK_M, BLOCK_N]

            # mask_x = (
            #     (idx_tokens - VAL >= 0)[:, None]
            #     & (idx_tokens - VAL < seqlen)[:, None]
            #     & (idx_feats2 < dim)[None, :]
            # )  # token-index  # token-index  # feature-index
            # loaded_x = gl.load(x_ptrs, mask_x, 0.0)
            # tl.debug_barrier()

            # new_conv_state = gl.where(mask, conv_state, loaded_x)

            conv_state_base = (
                conv_state_ptr
                + (conv_state_batch_coord * stride_conv_state_seq)
                + (idx_feats * stride_conv_state_dim)
            )  # [BLOCK_N,]
            # conv_state_ptrs_target = (
            #     conv_state_base + (idx_tokens * stride_conv_state_tok)[:, None]
            # )  # [BLOCK_M, BLOCK_N]
            # mask = (idx_tokens < state_len)[:, None] & (idx_feats2 < dim)[None, :]
            # gl.store(conv_state_ptrs_target, new_conv_state, mask)

            # STEP 3: init accumulator
            if HAS_BIAS:
                bias = bias_ptr + idx_feats
                mask_bias = idx_feats < dim
                acc_preload = gl.load(bias, mask=mask_bias, other=0.0).to(
                    o_ptr.type.element_ty
                )  # [BLOCK_N]
            else:
                acc_preload = gl.zeros((BLOCK_N,), dtype=o_ptr.type.element_ty, layout=blocked)

            # STEP 4:
            # PRE-LOAD WEIGHTS
            # first kernel column, configured for weights to handle BLOCK_N features in range
            w_base = w_ptr + (idx_feats * stride_w_dim)  # [BLOCK_N,]
            mask_w = idx_feats < dim
            if KERNEL_WIDTH >= 2:
                w_ptrs = w_base + (0 * stride_w_width)  # [BLOCK_N] tensor
                w_col0 = gl.load(w_ptrs, mask_w, other=0.0)
                w_ptrs = w_base + (1 * stride_w_width)  # [BLOCK_N] tensor
                w_col1 = gl.load(w_ptrs, mask_w, other=0.0)
                w_vecs = (w_col0, w_col1)
            if KERNEL_WIDTH >= 3:
                w_ptrs = w_base + (2 * stride_w_width)  # [BLOCK_N] tensor
                w_col2 = gl.load(w_ptrs, mask_w, other=0.0)
                w_vecs = tuple_combine(w_vecs, w_col2)
            if KERNEL_WIDTH >= 4:
                w_ptrs = w_base + (3 * stride_w_width)  # [BLOCK_N] tensor
                w_col3 = gl.load(w_ptrs, mask_w, other=0.0)
                w_vecs = tuple_combine(w_vecs, w_col3)

            x_base_1d = x_ptr + (idx_seq * stride_x_seq) + (idx_feats * stride_x_dim)  # starting of chunk [BLOCK_N]
            mask_x_1d = idx_feats < dim

            # STEP 5: compute each token
            for idx_token in gl.static_range(seqlen):
                acc = acc_preload

                x_ptrs_1d = x_base_1d + idx_token * stride_x_token  # [BLOCK_N]
                x_vec = gl.load(x_ptrs_1d, mask=mask_x_1d)
                conv_state_vecs = tuple_combine(conv_state_vecs, x_vec)
                for j in gl.static_range(KERNEL_WIDTH):
                    matrix_w = w_vecs[j]
                    matrix_x = conv_state_vecs[j]

                    acc += matrix_x * matrix_w  # [BLOCK_N]

                conv_state_vecs = conv_state_vecs[1:]

                if SILU_ACTIVATION:
                    # Convert to fp32 for exp calculation, then convert back
                    acc_fp32 = acc.to(gl.float32)
                    acc = acc_fp32 / (1 + gl.exp(-acc_fp32))
                    acc = acc.to(x_vec.dtype)
                mask_1d = (idx_token < seqlen) & (
                    idx_feats < dim
                )  # token-index  # feature-index
                o_ptrs = (
                    o_ptr
                    + (idx_seq) * stride_o_seq
                    + idx_token * stride_o_token
                    + (idx_feats * stride_o_dim)
                )

                gl.store(o_ptrs, acc, mask=mask_1d)

                if SAVE_INTERMEDIATE:
                    # Save the window state after consuming this token
                    # Layout: [seq(cache line), step, dim, win(K-1)]
                    base_ptr = (
                        intermediate_conv_window_ptr
                        + conv_state_batch_coord * stride_inter_seq
                        + idx_token * stride_inter_step
                        + idx_feats * stride_inter_dim
                    )
                    for l in gl.static_range(state_len):
                        gl.store(base_ptr + l*stride_inter_win, conv_state_vecs[l], idx_feats < dim)

            for l in gl.static_range(state_len):
                gl.store(conv_state_base + l*stride_conv_state_tok, conv_state_vecs[l], idx_feats < dim)

@gluon.jit()
def gluon_causal_conv1d_update_persistent_kernel_v3(
    # Pointers to matrices
    x_ptr,  # (batch, dim, seqlen)
    w_ptr,  # (dim, width)
    bias_ptr,
    conv_state_ptr,
    cache_seqlens_ptr,  # circular buffer
    conv_state_indices_ptr,
    num_accepted_tokens_ptr,
    intermediate_conv_window_ptr,
    o_ptr,  # (batch, dim, seqlen)
    # Matrix dimensions
    batch: int,
    dim: gl.constexpr,
    seqlen: gl.constexpr,
    state_len: gl.constexpr,
    num_cache_lines: gl.constexpr,
    # Strides
    stride_x_seq: gl.constexpr,
    stride_x_dim: gl.constexpr,
    stride_x_token: gl.constexpr,
    stride_w_dim: gl.constexpr,
    stride_w_width: gl.constexpr,
    stride_conv_state_seq: gl.constexpr,
    stride_conv_state_dim: gl.constexpr,
    stride_conv_state_tok: gl.constexpr,
    stride_state_indices: gl.constexpr,
    stride_inter_seq: gl.constexpr,
    stride_inter_step: gl.constexpr,
    stride_inter_dim: gl.constexpr,
    stride_inter_win: gl.constexpr,
    stride_o_seq: gl.constexpr,
    stride_o_dim: gl.constexpr,
    stride_o_token: gl.constexpr,
    # others
    pad_slot_id: gl.constexpr,
    # Meta-parameters
    HAS_BIAS: gl.constexpr,
    KERNEL_WIDTH: gl.constexpr,
    SILU_ACTIVATION: gl.constexpr,
    IS_CONTINUOUS_BATCHING: gl.constexpr,
    IS_SPEC_DECODING: gl.constexpr,
    NP2_STATELEN: gl.constexpr,
    USE_PAD_SLOT: gl.constexpr,
    BLOCK_N: gl.constexpr,
    SAVE_INTERMEDIATE: gl.constexpr,
):
    """
    V3 kernel with double-buffering optimization for cross-task prefetching.
    Optimized for seqlen=1 (decode phase) scenario.
    
    Strategy:
      - Load next task's data while computing current task
      - Overlaps memory latency with computation
    Expected speedup: 1.18-1.22x by hiding 15-18% of memory access latency
    """
    
    blocked: gl.constexpr = gl.BlockedLayout(
        size_per_thread=[2],
        threads_per_warp=[64],
        warps_per_cta=[16],
        order=[0],
    )

    cu_idx = gl.program_id(0)
    num_cus = 80
    
    num_dim_blocks = gl.cdiv(dim, BLOCK_N)
    total_tasks = batch * num_dim_blocks
    
    cus_per_dim_block = gl.cdiv(num_cus, num_dim_blocks)
    dim_block_idx = cu_idx // cus_per_dim_block
    cu_idx_in_dim_block = cu_idx % cus_per_dim_block
    
    per_cu_batches = batch // cus_per_dim_block
    cu_mores = batch % cus_per_dim_block
    
    cu_num_tasks = (per_cu_batches + 1) if cu_idx_in_dim_block < cu_mores else per_cu_batches
    cu_task_offset = cu_idx_in_dim_block * (per_cu_batches + 1) if cu_idx_in_dim_block < cu_mores else (cu_mores * (per_cu_batches + 1) + (cu_idx_in_dim_block - cu_mores) * per_cu_batches)

    idx_feats = dim_block_idx * BLOCK_N + gl.arange(0, BLOCK_N, layout=blocked)

    # PRE-LOAD WEIGHTS
    w_base = w_ptr + (idx_feats * stride_w_dim)
    mask_w = idx_feats < dim
    w_ptrs = w_base + (0 * stride_w_width)
    w_col0 = gl.load(w_ptrs, mask_w, other=0.0)
    w_ptrs = w_base + (1 * stride_w_width)
    w_col1 = gl.load(w_ptrs, mask_w, other=0.0)
    w_vecs = (w_col0, w_col1)
    w_ptrs = w_base + (2 * stride_w_width)
    w_col2 = gl.load(w_ptrs, mask_w, other=0.0)
    w_vecs = tuple_combine(w_vecs, w_col2)
    w_ptrs = w_base + (3 * stride_w_width)
    w_col3 = gl.load(w_ptrs, mask_w, other=0.0)
    w_vecs = tuple_combine(w_vecs, w_col3)

    mask_feat = idx_feats < dim
    
    for task_idx in range(cu_num_tasks):
        idx_seq_current = cu_task_offset + task_idx
        
        # LOAD current task's data
        conv_state_batch_coord = gl.load(
            conv_state_indices_ptr + idx_seq_current * stride_state_indices
        ).to(gl.int64)
        
        conv_states_base = (
            conv_state_ptr
            + (conv_state_batch_coord * stride_conv_state_seq)
            + (idx_feats * stride_conv_state_dim)
        )
        prior_tokens = conv_states_base + 0 * stride_conv_state_tok
        col0 = gl.load(prior_tokens, mask_feat, 0.0)
        col1 = gl.load(prior_tokens + 1*stride_conv_state_tok, mask_feat, 0.0)
        col2 = gl.load(prior_tokens + 2*stride_conv_state_tok, mask_feat, 0.0)
        conv_state_vecs = (col0, col1, col2)
        
        x_base = x_ptr + (idx_seq_current * stride_x_seq) + (idx_feats * stride_x_dim)
        x_vec = gl.load(x_base + 0*stride_x_token, mask=mask_feat)
        
        # PREFETCH hint for next task
        if task_idx < cu_num_tasks - 1:
            idx_seq_next = cu_task_offset + task_idx + 1
            _ = conv_state_indices_ptr + idx_seq_next * stride_state_indices
        
        # COMPUTE current task
        if HAS_BIAS:
            bias_val = gl.load(bias_ptr + idx_feats, mask=mask_feat, other=0.0)
            acc = bias_val.to(o_ptr.type.element_ty)
        else:
            acc = gl.zeros((BLOCK_N,), dtype=o_ptr.type.element_ty, layout=blocked)
        
        conv_state_window = tuple_combine(conv_state_vecs, x_vec)
        
        for j in gl.static_range(KERNEL_WIDTH):
            weight_col = w_vecs[j]
            state_col = conv_state_window[j]
            acc += state_col * weight_col
        
        acc_fp32 = acc.to(gl.float32)
        acc = acc_fp32 / (1 + gl.exp(-acc_fp32))
        acc = acc.to(x_vec.dtype)
        
        # STORE current task results
        o_ptrs = (
            o_ptr
            + (idx_seq_current) * stride_o_seq
            + 0 * stride_o_token
            + (idx_feats * stride_o_dim)
        )
        gl.store(o_ptrs, acc, mask=mask_feat)
        
        conv_state_base_writeback = (
            conv_state_ptr
            + (conv_state_batch_coord * stride_conv_state_seq)
            + (idx_feats * stride_conv_state_dim)
        )
        updated_state = conv_state_window[1:]
        for l in gl.static_range(state_len):
            gl.store(
                conv_state_base_writeback + l*stride_conv_state_tok,
                updated_state[l],
                mask=mask_feat
            )

@gluon.jit()
def gluon_causal_conv1d_update_persistent_kernel_v4(
    # Pointers to matrices
    x_ptr,  # (batch, dim, seqlen)
    w_ptr,  # (dim, width)
    bias_ptr,
    conv_state_ptr,
    cache_seqlens_ptr,  # circular buffer
    conv_state_indices_ptr,
    num_accepted_tokens_ptr,
    intermediate_conv_window_ptr,
    o_ptr,  # (batch, dim, seqlen)
    # Matrix dimensions
    batch: int,
    dim: gl.constexpr,
    seqlen: gl.constexpr,
    state_len: gl.constexpr,
    num_cache_lines: gl.constexpr,  # added to support vLLM larger cache lines
    # Strides
    stride_x_seq: gl.constexpr,
    stride_x_dim: gl.constexpr,
    stride_x_token: gl.constexpr,
    stride_w_dim: gl.constexpr,
    stride_w_width: gl.constexpr,
    stride_conv_state_seq: gl.constexpr,
    stride_conv_state_dim: gl.constexpr,
    stride_conv_state_tok: gl.constexpr,
    stride_state_indices: gl.constexpr,
    stride_inter_seq: gl.constexpr,
    stride_inter_step: gl.constexpr,
    stride_inter_dim: gl.constexpr,
    stride_inter_win: gl.constexpr,
    stride_o_seq: gl.constexpr,
    stride_o_dim: gl.constexpr,
    stride_o_token: gl.constexpr,
    # others
    pad_slot_id: gl.constexpr,
    # Meta-parameters
    HAS_BIAS: gl.constexpr,
    KERNEL_WIDTH: gl.constexpr,
    SILU_ACTIVATION: gl.constexpr,
    IS_CONTINUOUS_BATCHING: gl.constexpr,
    IS_SPEC_DECODING: gl.constexpr,
    NP2_STATELEN: gl.constexpr,
    USE_PAD_SLOT: gl.constexpr,
    BLOCK_N: gl.constexpr,
    SAVE_INTERMEDIATE: gl.constexpr,
):
    # Note: Gluon does not support device_print, so parameter printing is done in Python wrapper
    # See the wrapper function for parameter logging
    
    blocked: gl.constexpr = gl.BlockedLayout(
        size_per_thread=[2],
        threads_per_warp=[64],
        warps_per_cta=[8],
        order=[0],
    )

    cu_idx = gl.program_id(0)  # Current CU ID (0-79)
    num_cus = 80                # Fixed number of CUs
    
    # Calculate total number of tasks
    num_dim_blocks = gl.cdiv(dim, BLOCK_N)  # Number of blocks needed in dim direction
    total_tasks = batch * num_dim_blocks     # Total number of tasks
    
    # New mapping: each CU processes tasks from the same dim_block across different batches
    # Determine dim_block_idx directly from cu_idx
    cus_per_dim_block = gl.cdiv(num_cus, num_dim_blocks)  # How many CUs allocated per dim_block
    dim_block_idx = cu_idx // cus_per_dim_block  # Which dim_block this CU handles
    cu_idx_in_dim_block = cu_idx % cus_per_dim_block  # Index of this CU within the dim_block group
    
    # Calculate number of batch tasks per CU for this dim_block
    per_cu_batches = batch // cus_per_dim_block  # Base number of batches per CU
    cu_mores = batch % cus_per_dim_block  # Remaining batches
    
    # First cu_mores CUs (within this dim_block group) handle 1 additional batch
    cu_num_tasks = (per_cu_batches + 1) if cu_idx_in_dim_block < cu_mores else per_cu_batches
    cu_task_offset = cu_idx_in_dim_block * (per_cu_batches + 1) if cu_idx_in_dim_block < cu_mores else (cu_mores * (per_cu_batches + 1) + (cu_idx_in_dim_block - cu_mores) * per_cu_batches)

    # idx_feats is constant for all tasks in this CU (same dim_block_idx)
    idx_feats = dim_block_idx * BLOCK_N + gl.arange(0, BLOCK_N, layout=blocked)
    mask = idx_feats < dim

    # STEP 1:
    # PRE-LOAD WEIGHTS
    # first kernel column, configured for weights to handle BLOCK_N features in range
    w_base = w_ptr + (idx_feats * stride_w_dim)  # [BLOCK_N,]

    w_ptrs = w_base + (0 * stride_w_width)  # [BLOCK_N] tensor
    w_col0 = gl.load(w_base, mask, other=0.0)


    w_ptrs = w_base + (1 * stride_w_width)  # [BLOCK_N] tensor
    w_col1 = gl.load(w_base + stride_w_width, mask, other=0.0)

    w_ptrs = w_base + (2 * stride_w_width)  # [BLOCK_N] tensor
    w_col2 = gl.load(w_ptrs, mask, other=0.0)

    w_ptrs = w_base + (3 * stride_w_width)  # [BLOCK_N] tensor
    w_col3 = gl.load(w_ptrs, mask, other=0.0)
    
    # Prologue: Pre-load col0 and col1 for the first iteration, compute partial sum
    idx_seq_next = cu_task_offset
    conv_state_batch_coord_next = gl.load(conv_state_indices_ptr + idx_seq_next * stride_state_indices).to(gl.int64)
    conv_states_base_next = (
        conv_state_ptr
        + (conv_state_batch_coord_next * stride_conv_state_seq)
        + (idx_feats * stride_conv_state_dim)
    )
    conv_states_ptrs = conv_states_base_next
    col0_next = gl.load(conv_states_ptrs, mask, 0.0)
    conv_states_ptrs = conv_states_ptrs + stride_conv_state_tok
    col1_next = gl.load(conv_states_ptrs, mask, 0.0)
    conv_states_ptrs = conv_states_ptrs + stride_conv_state_tok
    col2_next = gl.load(conv_states_ptrs, mask, 0.0)

    # Pre-compute the first two terms for the first iteration
    acc_next = gl.zeros((BLOCK_N,), dtype=o_ptr.type.element_ty, layout=blocked)
    acc_next += w_col0 * col0_next

    for task_idx in range(cu_num_tasks - 1):
        # All tasks in this CU belong to the same dim_block_idx but different batches
        idx_seq = idx_seq_next
        conv_states_base = conv_states_base_next
        col0 = col0_next
        col1 = col1_next
        col2 = col2_next
        acc = acc_next

        acc += w_col1 * col1
        acc += w_col2 * col2  # [BLOCK_N]

        x_base_1d = x_ptr + (idx_seq * stride_x_seq) + (idx_feats * stride_x_dim)  # starting of chunk [BLOCK_N]
        x_ptrs_1d = x_base_1d
        x_vec = gl.load(x_ptrs_1d, mask)
        acc += w_col3 * x_vec  # [BLOCK_N]

        # if SILU_ACTIVATION, Convert to fp32 for exp calculation, then convert back
        acc_fp32 = acc.to(gl.float32)
        acc = acc_fp32 / (1 + gl.exp(-acc_fp32))
        acc = acc.to(x_vec.dtype)

        o_ptrs = o_ptr + idx_seq * stride_o_seq + idx_feats * stride_o_dim
        gl.store(o_ptrs, acc, mask)
        
        conv_states_ptrs = conv_states_base
        gl.store(conv_states_ptrs, col1, mask)
        conv_states_ptrs = conv_states_ptrs + stride_conv_state_tok
        gl.store(conv_states_ptrs, col2, mask)
        conv_states_ptrs = conv_states_ptrs + stride_conv_state_tok
        gl.store(conv_states_ptrs, x_vec, mask)
        
        # Pre-load col0 and col1 for next iteration and compute partial sum
        idx_seq_next = idx_seq + 1
        conv_state_batch_coord_next = gl.load(conv_state_indices_ptr + idx_seq_next * stride_state_indices).to(gl.int64)
        conv_states_base_next = (
            conv_state_ptr
            + (conv_state_batch_coord_next * stride_conv_state_seq)
            + (idx_feats * stride_conv_state_dim)
        )
        
        conv_states_ptrs_next = conv_states_base_next
        col0_next = gl.load(conv_states_ptrs_next, mask, 0.0)
        conv_states_ptrs_next = conv_states_ptrs_next + stride_conv_state_tok
        col1_next = gl.load(conv_states_ptrs_next, mask, 0.0)
        conv_states_ptrs_next = conv_states_ptrs_next + stride_conv_state_tok
        col2_next = gl.load(conv_states_ptrs_next, mask, 0.0)
        
        # Pre-compute the first two terms for the first iteration
        acc_next = gl.zeros((BLOCK_N,), dtype=o_ptr.type.element_ty, layout=blocked)
        acc_next += w_col0 * col0_next

    
    # Epilogue: Process the last task
    idx_seq = idx_seq_next
    conv_states_base = conv_states_base_next
    col0 = col0_next
    col1 = col1_next
    col2 = col2_next
    acc = acc_next

    acc += w_col1 * col1
    acc += w_col2 * col2  # [BLOCK_N]

    x_base_1d = x_ptr + (idx_seq * stride_x_seq) + (idx_feats * stride_x_dim)
    x_vec = gl.load(x_base_1d, mask)
    acc += w_col3 * x_vec
    
    # SILU activation
    acc_fp32 = acc.to(gl.float32)
    acc = acc_fp32 / (1 + gl.exp(-acc_fp32))
    acc = acc.to(x_vec.dtype)
    
    # Store output
    o_ptrs = o_ptr + idx_seq * stride_o_seq + idx_feats * stride_o_dim
    gl.store(o_ptrs, acc, mask)
    
    # Write back updated conv state
    conv_states_ptrs = conv_states_base
    gl.store(conv_states_ptrs, col1, mask)
    conv_states_ptrs = conv_states_ptrs + stride_conv_state_tok
    gl.store(conv_states_ptrs, col2, mask)
    conv_states_ptrs = conv_states_ptrs + stride_conv_state_tok
    gl.store(conv_states_ptrs, x_vec, mask)
        
@gluon.jit()
def gluon_causal_conv1d_update_persistent_kernel_v5(
    # Pointers to matrices
    x_ptr,  # (batch, dim, seqlen)
    w_ptr,  # (dim, width)
    bias_ptr,
    conv_state_ptr,
    cache_seqlens_ptr,  # circular buffer
    conv_state_indices_ptr,
    num_accepted_tokens_ptr,
    intermediate_conv_window_ptr,
    o_ptr,  # (batch, dim, seqlen)
    # Matrix dimensions
    batch: int,
    dim: gl.constexpr,
    seqlen: gl.constexpr,
    state_len: gl.constexpr,
    num_cache_lines: gl.constexpr,  # added to support vLLM larger cache lines
    # Strides
    stride_x_seq: gl.constexpr,
    stride_x_dim: gl.constexpr,
    stride_x_token: gl.constexpr,
    stride_w_dim: gl.constexpr,
    stride_w_width: gl.constexpr,
    stride_conv_state_seq: gl.constexpr,
    stride_conv_state_dim: gl.constexpr,
    stride_conv_state_tok: gl.constexpr,
    stride_state_indices: gl.constexpr,
    stride_inter_seq: gl.constexpr,
    stride_inter_step: gl.constexpr,
    stride_inter_dim: gl.constexpr,
    stride_inter_win: gl.constexpr,
    stride_o_seq: gl.constexpr,
    stride_o_dim: gl.constexpr,
    stride_o_token: gl.constexpr,
    # others
    pad_slot_id: gl.constexpr,
    # Meta-parameters
    HAS_BIAS: gl.constexpr,
    KERNEL_WIDTH: gl.constexpr,
    SILU_ACTIVATION: gl.constexpr,
    IS_CONTINUOUS_BATCHING: gl.constexpr,
    IS_SPEC_DECODING: gl.constexpr,
    NP2_STATELEN: gl.constexpr,
    USE_PAD_SLOT: gl.constexpr,
    BLOCK_N: gl.constexpr,
    SAVE_INTERMEDIATE: gl.constexpr,
):
    # Note: Gluon does not support device_print, so parameter printing is done in Python wrapper
    # See the wrapper function for parameter logging
    
    blocked: gl.constexpr = gl.BlockedLayout(
        size_per_thread=[2],
        threads_per_warp=[64],
        warps_per_cta=[8],
        order=[0],
    )

    cu_idx = gl.program_id(0)  # Current CU ID (0-79)
    num_cus = 80                # Fixed number of CUs
    
    # Calculate total number of tasks
    num_dim_blocks = gl.cdiv(dim, BLOCK_N)  # Number of blocks needed in dim direction
    total_tasks = batch * num_dim_blocks     # Total number of tasks
    
    # New mapping: each CU processes tasks from the same dim_block across different batches
    # Determine dim_block_idx directly from cu_idx
    cus_per_dim_block = gl.cdiv(num_cus, num_dim_blocks)  # How many CUs allocated per dim_block
    dim_block_idx = cu_idx // cus_per_dim_block  # Which dim_block this CU handles
    cu_idx_in_dim_block = cu_idx % cus_per_dim_block  # Index of this CU within the dim_block group
    
    # Calculate number of batch tasks per CU for this dim_block
    per_cu_batches = batch // cus_per_dim_block  # Base number of batches per CU
    cu_mores = batch % cus_per_dim_block  # Remaining batches
    
    # First cu_mores CUs (within this dim_block group) handle 1 additional batch
    cu_num_tasks = (per_cu_batches + 1) if cu_idx_in_dim_block < cu_mores else per_cu_batches
    cu_task_offset = cu_idx_in_dim_block * (per_cu_batches + 1) if cu_idx_in_dim_block < cu_mores else (cu_mores * (per_cu_batches + 1) + (cu_idx_in_dim_block - cu_mores) * per_cu_batches)

    # idx_feats is constant for all tasks in this CU (same dim_block_idx)
    idx_feats = dim_block_idx * BLOCK_N + gl.arange(0, BLOCK_N, layout=blocked)
    mask = idx_feats < dim

    # STEP 1:
    # PRE-LOAD WEIGHTS
    # first kernel column, configured for weights to handle BLOCK_N features in range
    w_base = w_ptr + (idx_feats * stride_w_dim)  # [BLOCK_N,]

    w_ptrs = w_base + (0 * stride_w_width)  # [BLOCK_N] tensor
    w_col0 = gl.load(w_base, mask, other=0.0)


    w_ptrs = w_base + (1 * stride_w_width)  # [BLOCK_N] tensor
    w_col1 = gl.load(w_base + stride_w_width, mask, other=0.0)

    w_ptrs = w_base + (2 * stride_w_width)  # [BLOCK_N] tensor
    w_col2 = gl.load(w_ptrs, mask, other=0.0)

    w_ptrs = w_base + (3 * stride_w_width)  # [BLOCK_N] tensor
    w_col3 = gl.load(w_ptrs, mask, other=0.0)
    
    # Prologue: Pre-load col0 and col1 for the first iteration, compute partial sum
    idx_seq_next = cu_task_offset
    conv_state_batch_coord_ptr = (conv_state_indices_ptr + idx_seq_next * stride_state_indices)
    conv_state_batch_coord_next = gl.load(conv_state_batch_coord_ptr).to(gl.int64)
    conv_states_base_next = (
        conv_state_ptr
        + (conv_state_batch_coord_next * stride_conv_state_seq)
        + (idx_feats * stride_conv_state_dim)
    )
    conv_states_ptrs = conv_states_base_next
    col0_next = gl.load(conv_states_ptrs, mask, 0.0)
    conv_states_ptrs = conv_states_ptrs + stride_conv_state_tok
    col1_next = gl.load(conv_states_ptrs, mask, 0.0)
    conv_states_ptrs = conv_states_ptrs + stride_conv_state_tok
    col2_next = gl.load(conv_states_ptrs, mask, 0.0)

    for task_idx in range(cu_num_tasks - 1):
        # All tasks in this CU belong to the same dim_block_idx but different batches
        idx_seq = idx_seq_next
        conv_states_base = conv_states_base_next
        col0 = col0_next
        col1 = col1_next
        col2 = col2_next
        
        # Pre-compute the first two terms for the first iteration
        acc = gl.zeros((BLOCK_N,), dtype=o_ptr.type.element_ty, layout=blocked)
        acc += w_col0 * col0
        acc += w_col1 * col1
        acc += w_col2 * col2  # [BLOCK_N]

        x_base_1d = x_ptr + (idx_seq * stride_x_seq) + (idx_feats * stride_x_dim)  # starting of chunk [BLOCK_N]
        x_ptrs_1d = x_base_1d
        x_vec = gl.load(x_ptrs_1d, mask)
        acc += w_col3 * x_vec  # [BLOCK_N]

        # if SILU_ACTIVATION, Convert to fp32 for exp calculation, then convert back
        acc_fp32 = acc.to(gl.float32)
        acc = acc_fp32 / (1 + gl.exp(-acc_fp32))
        acc = acc.to(x_vec.dtype)

        o_ptrs = o_ptr + idx_seq * stride_o_seq + idx_feats * stride_o_dim
        gl.store(o_ptrs, acc, mask)
        
        conv_states_ptrs = conv_states_base
        gl.store(conv_states_ptrs, col1, mask)
        conv_states_ptrs = conv_states_ptrs + stride_conv_state_tok
        gl.store(conv_states_ptrs, col2, mask)
        conv_states_ptrs = conv_states_ptrs + stride_conv_state_tok
        gl.store(conv_states_ptrs, x_vec, mask)
        
        # Pre-load col0 and col1 for next iteration and compute partial sum
        idx_seq_next = idx_seq + 1
        conv_state_batch_coord_ptr = (conv_state_indices_ptr + idx_seq_next * stride_state_indices)
        conv_state_batch_coord_next = gl.load(conv_state_batch_coord_ptr).to(gl.int64)
        conv_states_base_next = (
            conv_state_ptr
            + (conv_state_batch_coord_next * stride_conv_state_seq)
            + (idx_feats * stride_conv_state_dim)
        )
        
        conv_states_ptrs_next = conv_states_base_next
        col0_next = gl.load(conv_states_ptrs_next, mask, 0.0)
        conv_states_ptrs_next = conv_states_ptrs_next + stride_conv_state_tok
        col1_next = gl.load(conv_states_ptrs_next, mask, 0.0)
        conv_states_ptrs_next = conv_states_ptrs_next + stride_conv_state_tok
        col2_next = gl.load(conv_states_ptrs_next, mask, 0.0)

    
    # Epilogue: Process the last task
    idx_seq = idx_seq_next
    conv_states_base = conv_states_base_next
    col0 = col0_next
    col1 = col1_next
    col2 = col2_next
        
    # Pre-compute the first two terms for the first iteration
    acc = gl.zeros((BLOCK_N,), dtype=o_ptr.type.element_ty, layout=blocked)
    acc += w_col0 * col0
    acc += w_col1 * col1
    acc += w_col2 * col2  # [BLOCK_N]

    x_base_1d = x_ptr + (idx_seq * stride_x_seq) + (idx_feats * stride_x_dim)
    x_vec = gl.load(x_base_1d, mask)
    acc += w_col3 * x_vec
    
    # SILU activation
    acc_fp32 = acc.to(gl.float32)
    acc = acc_fp32 / (1 + gl.exp(-acc_fp32))
    acc = acc.to(x_vec.dtype)
    
    # Store output
    o_ptrs = o_ptr + idx_seq * stride_o_seq + idx_feats * stride_o_dim
    gl.store(o_ptrs, acc, mask)
    
    # Write back updated conv state
    conv_states_ptrs = conv_states_base
    gl.store(conv_states_ptrs, col1, mask)
    conv_states_ptrs = conv_states_ptrs + stride_conv_state_tok
    gl.store(conv_states_ptrs, col2, mask)
    conv_states_ptrs = conv_states_ptrs + stride_conv_state_tok
    gl.store(conv_states_ptrs, x_vec, mask)

@gluon.jit()
def gluon_causal_conv1d_update_persistent_kernel_v6(
    # Pointers to matrices
    x_ptr,  # (batch, dim, seqlen)
    w_ptr,  # (dim, width)
    bias_ptr,
    conv_state_ptr,
    cache_seqlens_ptr,  # circular buffer
    conv_state_indices_ptr,
    num_accepted_tokens_ptr,
    intermediate_conv_window_ptr,
    o_ptr,  # (batch, dim, seqlen)
    # Matrix dimensions
    batch: int,
    dim: gl.constexpr,
    seqlen: gl.constexpr,
    state_len: gl.constexpr,
    num_cache_lines: gl.constexpr,  # added to support vLLM larger cache lines
    # Strides
    stride_x_seq: gl.constexpr,
    stride_x_dim: gl.constexpr,
    stride_x_token: gl.constexpr,
    stride_w_dim: gl.constexpr,
    stride_w_width: gl.constexpr,
    stride_conv_state_seq: gl.constexpr,
    stride_conv_state_dim: gl.constexpr,
    stride_conv_state_tok: gl.constexpr,
    stride_state_indices: gl.constexpr,
    stride_inter_seq: gl.constexpr,
    stride_inter_step: gl.constexpr,
    stride_inter_dim: gl.constexpr,
    stride_inter_win: gl.constexpr,
    stride_o_seq: gl.constexpr,
    stride_o_dim: gl.constexpr,
    stride_o_token: gl.constexpr,
    # others
    pad_slot_id: gl.constexpr,
    # Meta-parameters
    HAS_BIAS: gl.constexpr,
    KERNEL_WIDTH: gl.constexpr,
    SILU_ACTIVATION: gl.constexpr,
    IS_CONTINUOUS_BATCHING: gl.constexpr,
    IS_SPEC_DECODING: gl.constexpr,
    NP2_STATELEN: gl.constexpr,
    USE_PAD_SLOT: gl.constexpr,
    BLOCK_N: gl.constexpr,
    SAVE_INTERMEDIATE: gl.constexpr,
):
    # Note: Gluon does not support device_print, so parameter printing is done in Python wrapper
    # See the wrapper function for parameter logging
    
    blocked: gl.constexpr = gl.BlockedLayout(
        size_per_thread=[2],
        threads_per_warp=[64],
        warps_per_cta=[8],
        order=[0],
    )

    cu_idx = gl.program_id(0)  # Current CU ID (0-79)
    num_cus = 80                # Fixed number of CUs
    
    # Calculate total number of tasks
    num_dim_blocks = gl.cdiv(dim, BLOCK_N)  # Number of blocks needed in dim direction
    total_tasks = batch * num_dim_blocks     # Total number of tasks
    
    # New mapping: each CU processes tasks from the same dim_block across different batches
    # Determine dim_block_idx directly from cu_idx
    cus_per_dim_block = gl.cdiv(num_cus, num_dim_blocks)  # How many CUs allocated per dim_block
    dim_block_idx = cu_idx // cus_per_dim_block  # Which dim_block this CU handles
    cu_idx_in_dim_block = cu_idx % cus_per_dim_block  # Index of this CU within the dim_block group
    
    # Calculate number of batch tasks per CU for this dim_block
    per_cu_batches = batch // cus_per_dim_block  # Base number of batches per CU
    cu_mores = batch % cus_per_dim_block  # Remaining batches
    
    # First cu_mores CUs (within this dim_block group) handle 1 additional batch
    cu_num_tasks = (per_cu_batches + 1) if cu_idx_in_dim_block < cu_mores else per_cu_batches
    cu_task_offset = cu_idx_in_dim_block * (per_cu_batches + 1) if cu_idx_in_dim_block < cu_mores else (cu_mores * (per_cu_batches + 1) + (cu_idx_in_dim_block - cu_mores) * per_cu_batches)

    # idx_feats is constant for all tasks in this CU (same dim_block_idx)
    idx_feats = dim_block_idx * BLOCK_N + gl.arange(0, BLOCK_N, layout=blocked)
    mask = idx_feats < dim

    # STEP 1:
    # PRE-LOAD WEIGHTS
    # first kernel column, configured for weights to handle BLOCK_N features in range
    w_base = w_ptr + (idx_feats * stride_w_dim)  # [BLOCK_N,]

    w_ptrs = w_base + (0 * stride_w_width)  # [BLOCK_N] tensor
    w_col0 = gl.load(w_base, mask, other=0.0)


    w_ptrs = w_base + (1 * stride_w_width)  # [BLOCK_N] tensor
    w_col1 = gl.load(w_base + stride_w_width, mask, other=0.0)

    w_ptrs = w_base + (2 * stride_w_width)  # [BLOCK_N] tensor
    w_col2 = gl.load(w_ptrs, mask, other=0.0)

    w_ptrs = w_base + (3 * stride_w_width)  # [BLOCK_N] tensor
    w_col3 = gl.load(w_ptrs, mask, other=0.0)
    
    # Prologue: Pre-load col0 and col1 for the first iteration, compute partial sum
    idx_seq_next = cu_task_offset
    conv_state_batch_coord_ptr = (conv_state_indices_ptr + idx_seq_next * stride_state_indices)
    conv_state_batch_coord_next = gl.load(conv_state_batch_coord_ptr).to(gl.int64)
    conv_states_base_next = (
        conv_state_ptr
        + (conv_state_batch_coord_next * stride_conv_state_seq)
        + (idx_feats * stride_conv_state_dim)
    )
    conv_states_ptrs = conv_states_base_next
    col0_next = gl.load(conv_states_ptrs, mask, 0.0)
    conv_states_ptrs = conv_states_ptrs + stride_conv_state_tok
    col1_next = gl.load(conv_states_ptrs, mask, 0.0)
    conv_states_ptrs = conv_states_ptrs + stride_conv_state_tok
    col2_next = gl.load(conv_states_ptrs, mask, 0.0)
    
    x_base_1d = x_ptr + (idx_seq_next * stride_x_seq) + (idx_feats * stride_x_dim)  # starting of chunk [BLOCK_N]
    x_ptrs_1d = x_base_1d
    x_vec_next = gl.load(x_ptrs_1d, mask)

    for task_idx in range(cu_num_tasks - 1):
        # All tasks in this CU belong to the same dim_block_idx but different batches
        idx_seq = idx_seq_next
        conv_states_base = conv_states_base_next
        col0 = col0_next
        col1 = col1_next
        col2 = col2_next
        x_vec = x_vec_next
        
        # Pre-compute the first two terms for the first iteration
        acc = gl.zeros((BLOCK_N,), dtype=o_ptr.type.element_ty, layout=blocked)
        acc += w_col0 * col0
        acc += w_col1 * col1
        acc += w_col2 * col2  # [BLOCK_N]
        acc += w_col3 * x_vec  # [BLOCK_N]

        # if SILU_ACTIVATION, Convert to fp32 for exp calculation, then convert back
        acc_fp32 = acc.to(gl.float32)
        acc = acc_fp32 / (1 + gl.exp(-acc_fp32))
        acc = acc.to(x_vec.dtype)

        o_ptrs = o_ptr + idx_seq * stride_o_seq + idx_feats * stride_o_dim
        gl.store(o_ptrs, acc, mask)
        
        conv_states_ptrs = conv_states_base
        gl.store(conv_states_ptrs, col1, mask)
        conv_states_ptrs = conv_states_ptrs + stride_conv_state_tok
        gl.store(conv_states_ptrs, col2, mask)
        conv_states_ptrs = conv_states_ptrs + stride_conv_state_tok
        gl.store(conv_states_ptrs, x_vec, mask)
        
        # Pre-load col0 and col1 for next iteration and compute partial sum
        idx_seq_next = idx_seq + 1
        conv_state_batch_coord_ptr = (conv_state_indices_ptr + idx_seq_next * stride_state_indices)
        conv_state_batch_coord_next = gl.load(conv_state_batch_coord_ptr).to(gl.int64)
        conv_states_base_next = (
            conv_state_ptr
            + (conv_state_batch_coord_next * stride_conv_state_seq)
            + (idx_feats * stride_conv_state_dim)
        )
        
        conv_states_ptrs_next = conv_states_base_next
        col0_next = gl.load(conv_states_ptrs_next, mask, 0.0)
        conv_states_ptrs_next = conv_states_ptrs_next + stride_conv_state_tok
        col1_next = gl.load(conv_states_ptrs_next, mask, 0.0)
        conv_states_ptrs_next = conv_states_ptrs_next + stride_conv_state_tok
        col2_next = gl.load(conv_states_ptrs_next, mask, 0.0)

        x_base_1d = x_ptr + (idx_seq_next * stride_x_seq) + (idx_feats * stride_x_dim)
        x_ptrs_1d = x_base_1d
        x_vec_next = gl.load(x_ptrs_1d, mask)

    
    # Epilogue: Process the last task
    idx_seq = idx_seq_next
    conv_states_base = conv_states_base_next
    col0 = col0_next
    col1 = col1_next
    col2 = col2_next
    x_vec = x_vec_next
        
    # Pre-compute the first two terms for the first iteration
    acc = gl.zeros((BLOCK_N,), dtype=o_ptr.type.element_ty, layout=blocked)
    acc += w_col0 * col0
    acc += w_col1 * col1
    acc += w_col2 * col2
    acc += w_col3 * x_vec
    
    # SILU activation
    acc_fp32 = acc.to(gl.float32)
    acc = acc_fp32 / (1 + gl.exp(-acc_fp32))
    acc = acc.to(x_vec.dtype)
    
    # Store output
    o_ptrs = o_ptr + idx_seq * stride_o_seq + idx_feats * stride_o_dim
    gl.store(o_ptrs, acc, mask)
    
    # Write back updated conv state
    conv_states_ptrs = conv_states_base
    gl.store(conv_states_ptrs, col1, mask)
    conv_states_ptrs = conv_states_ptrs + stride_conv_state_tok
    gl.store(conv_states_ptrs, col2, mask)
    conv_states_ptrs = conv_states_ptrs + stride_conv_state_tok
    gl.store(conv_states_ptrs, x_vec, mask)

def causal_conv1d_update(
    x: torch.Tensor,
    conv_state: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    activation: Union[bool, str, None] = None,
    cache_seqlens: Optional[torch.Tensor] = None,
    conv_state_indices: Optional[torch.Tensor] = None,
    num_accepted_tokens: Optional[torch.Tensor] = None,
    intermediate_conv_window: Optional[torch.Tensor] = None,
    pad_slot_id: int = PAD_SLOT_ID,
    metadata=None,
    validate_data=False,
):
    """
    x: (batch, dim) or (batch, dim, seqlen)
        [shape=2: single token prediction]
        [shape=3: single or multiple tokens prediction]
    conv_state: (..., dim, state_len), where state_len >= width - 1
    weight: (dim, width)
    bias: (dim,)
    cache_seqlens: (batch,), dtype int32.
        If not None, the conv_state is treated as a circular buffer.
        The conv_state will be updated by copying x to the conv_state
        starting at the index
        @cache_seqlens % state_len.
    conv_state_indices: (batch,), dtype int32
        If not None, the conv_state is a larger tensor along the batch dim,
        and we are selecting the batch coords specified by conv_state_indices.
        Useful for a continuous batching scenario.
    pad_slot_id: int
            if cache_indices is passed, lets the kernel identify padded
            entries that will not be processed,
            for example: cache_indices = [pad_slot_id, 1 ,20 ,pad_slot_id]
            in this case, the kernel will not process entries at
            indices 0 and 3
    out: (batch, dim) or (batch, dim, seqlen)
    """
    if validate_data:
        assert cache_seqlens is None  # not implemented yet - ok for vLLM
        assert pad_slot_id is not None
        assert x.stride(1) == 1
    if isinstance(activation, bool):
        activation = "silu" if activation is True else None
    elif activation is not None:
        assert activation in ["silu", "swish"]
    unsqueeze = x.dim() == 2
    if unsqueeze:
        # make it (batch, dim, seqlen) with seqlen == 1
        x = x.unsqueeze(-1)
    batch, dim, seqlen = x.shape
    _, width = weight.shape
    # conv_state: (..., dim, state_len), where state_len >= width - 1
    num_cache_lines, _, state_len = conv_state.size()

    if validate_data:
        assert dim == weight.size(0)
        assert (
            conv_state.stride(-2) == 1
        ), f"ERROR: expect contiguous along feat-dim of conv_state (currently stride={conv_state.stride()})"
        assert state_len >= width - 1
        # when above happens, we don't shift-left to keep any records in conv_state
        assert dim == conv_state.size(1)
        if conv_state_indices is None:
            assert conv_state.size(0) >= batch
        else:
            assert (batch,) == conv_state_indices.shape

        assert num_cache_lines >= batch
        assert weight.stride(1) == 1  # Need this
        assert cache_seqlens is None  # not needed for vLLM - circular buffer

    # adopt the strategy in vLLM that overwrite on 'x' directly, rather than creating a new tensor 'o'
    out = x
    stride_w_dim, stride_w_width = weight.stride()

    stride_x_seq, stride_x_dim, stride_x_token = x.stride()  # X (batch, dim, seqlen)

    stride_o_seq, stride_o_dim, stride_o_token = out.stride()
    stride_istate_seq, stride_istate_dim, stride_istate_token = conv_state.stride()
    stride_state_indices = (
        conv_state_indices.stride(0) if conv_state_indices is not None else 0
    )
    if num_accepted_tokens is not None:
        state_len = width - 1 + (seqlen - 1)  # effective state_len needed
    else:
        state_len = width - 1
    np2_statelen = triton.next_power_of_2(state_len)

    def grid(META):
        return (
            batch,
            triton.cdiv(dim, META["BLOCK_N"]),
        )

    # prepare intermediate buffer strides if provided
    if intermediate_conv_window is not None:
        stride_inter_seq, stride_inter_step, stride_inter_dim, stride_inter_win = (
            intermediate_conv_window.stride(0),
            intermediate_conv_window.stride(1),
            intermediate_conv_window.stride(2),
            intermediate_conv_window.stride(3),
        )
    else:
        stride_inter_seq = stride_inter_step = stride_inter_dim = stride_inter_win = 0

    # _causal_conv1d_update_kernel[grid](
    gluon_causal_conv1d_update_kernel[grid](
        # Pointers to matrices
        x,
        weight,
        bias,
        conv_state,
        cache_seqlens,
        conv_state_indices,
        num_accepted_tokens,
        intermediate_conv_window if intermediate_conv_window is not None else x,
        out,
        # Matrix dimensions
        batch,
        dim,
        seqlen,
        state_len,
        num_cache_lines,
        # stride
        stride_x_seq,
        stride_x_dim,
        stride_x_token,
        stride_w_dim,
        stride_w_width,
        stride_istate_seq,
        stride_istate_dim,
        stride_istate_token,
        stride_state_indices,
        stride_inter_seq,
        stride_inter_step,
        stride_inter_dim,
        stride_inter_win,
        stride_o_seq,
        stride_o_dim,
        stride_o_token,
        # others
        pad_slot_id,
        # META
        HAS_BIAS=bias is not None,
        KERNEL_WIDTH=width,
        SILU_ACTIVATION=activation in ["silu", "swish"],
        IS_CONTINUOUS_BATCHING=conv_state_indices is not None,
        IS_SPEC_DECODING=num_accepted_tokens is not None,
        NP2_STATELEN=np2_statelen,
        USE_PAD_SLOT=pad_slot_id is not None,
        BLOCK_N=256,
        SAVE_INTERMEDIATE=intermediate_conv_window is not None,
        num_warps=2
    )
    if unsqueeze:
        out = out.squeeze(-1)
    return out

def causal_conv1d_update_v1(
    x: torch.Tensor,
    conv_state: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    activation: Union[bool, str, None] = None,
    cache_seqlens: Optional[torch.Tensor] = None,
    conv_state_indices: Optional[torch.Tensor] = None,
    num_accepted_tokens: Optional[torch.Tensor] = None,
    intermediate_conv_window: Optional[torch.Tensor] = None,
    pad_slot_id: int = PAD_SLOT_ID,
    metadata=None,
    validate_data=False,
):
    """
    x: (batch, dim) or (batch, dim, seqlen)
        [shape=2: single token prediction]
        [shape=3: single or multiple tokens prediction]
    conv_state: (..., dim, state_len), where state_len >= width - 1
    weight: (dim, width)
    bias: (dim,)
    cache_seqlens: (batch,), dtype int32.
        If not None, the conv_state is treated as a circular buffer.
        The conv_state will be updated by copying x to the conv_state
        starting at the index
        @cache_seqlens % state_len.
    conv_state_indices: (batch,), dtype int32
        If not None, the conv_state is a larger tensor along the batch dim,
        and we are selecting the batch coords specified by conv_state_indices.
        Useful for a continuous batching scenario.
    pad_slot_id: int
            if cache_indices is passed, lets the kernel identify padded
            entries that will not be processed,
            for example: cache_indices = [pad_slot_id, 1 ,20 ,pad_slot_id]
            in this case, the kernel will not process entries at
            indices 0 and 3
    out: (batch, dim) or (batch, dim, seqlen)
    """
    if validate_data:
        assert cache_seqlens is None  # not implemented yet - ok for vLLM
        assert pad_slot_id is not None
        assert x.stride(1) == 1
    if isinstance(activation, bool):
        activation = "silu" if activation is True else None
    elif activation is not None:
        assert activation in ["silu", "swish"]
    unsqueeze = x.dim() == 2
    if unsqueeze:
        # make it (batch, dim, seqlen) with seqlen == 1
        x = x.unsqueeze(-1)
    batch, dim, seqlen = x.shape
    _, width = weight.shape
    # conv_state: (..., dim, state_len), where state_len >= width - 1
    num_cache_lines, _, state_len = conv_state.size()

    if validate_data:
        assert dim == weight.size(0)
        assert (
            conv_state.stride(-2) == 1
        ), f"ERROR: expect contiguous along feat-dim of conv_state (currently stride={conv_state.stride()})"
        assert state_len >= width - 1
        # when above happens, we don't shift-left to keep any records in conv_state
        assert dim == conv_state.size(1)
        if conv_state_indices is None:
            assert conv_state.size(0) >= batch
        else:
            assert (batch,) == conv_state_indices.shape

        assert num_cache_lines >= batch
        assert weight.stride(1) == 1  # Need this
        assert cache_seqlens is None  # not needed for vLLM - circular buffer

    # adopt the strategy in vLLM that overwrite on 'x' directly, rather than creating a new tensor 'o'
    out = x
    stride_w_dim, stride_w_width = weight.stride()

    stride_x_seq, stride_x_dim, stride_x_token = x.stride()  # X (batch, dim, seqlen)

    stride_o_seq, stride_o_dim, stride_o_token = out.stride()
    stride_istate_seq, stride_istate_dim, stride_istate_token = conv_state.stride()
    stride_state_indices = (
        conv_state_indices.stride(0) if conv_state_indices is not None else 0
    )
    if num_accepted_tokens is not None:
        state_len = width - 1 + (seqlen - 1)  # effective state_len needed
    else:
        state_len = width - 1
    np2_statelen = triton.next_power_of_2(state_len)

    def grid(META):
        return (
            batch,
            triton.cdiv(dim, META["BLOCK_N"]),
        )

    # prepare intermediate buffer strides if provided
    if intermediate_conv_window is not None:
        stride_inter_seq, stride_inter_step, stride_inter_dim, stride_inter_win = (
            intermediate_conv_window.stride(0),
            intermediate_conv_window.stride(1),
            intermediate_conv_window.stride(2),
            intermediate_conv_window.stride(3),
        )
    else:
        stride_inter_seq = stride_inter_step = stride_inter_dim = stride_inter_win = 0

    # _causal_conv1d_update_kernel[grid](
    gluon_causal_conv1d_update_kernel_v1[grid](
        # Pointers to matrices
        x,
        weight,
        bias,
        conv_state,
        cache_seqlens,
        conv_state_indices,
        num_accepted_tokens,
        intermediate_conv_window if intermediate_conv_window is not None else x,
        out,
        # Matrix dimensions
        batch,
        dim,
        seqlen,
        state_len,
        num_cache_lines,
        # stride
        stride_x_seq,
        stride_x_dim,
        stride_x_token,
        stride_w_dim,
        stride_w_width,
        stride_istate_seq,
        stride_istate_dim,
        stride_istate_token,
        stride_state_indices,
        stride_inter_seq,
        stride_inter_step,
        stride_inter_dim,
        stride_inter_win,
        stride_o_seq,
        stride_o_dim,
        stride_o_token,
        # others
        pad_slot_id,
        # META
        HAS_BIAS=bias is not None,
        KERNEL_WIDTH=width,
        SILU_ACTIVATION=activation in ["silu", "swish"],
        IS_CONTINUOUS_BATCHING=conv_state_indices is not None,
        IS_SPEC_DECODING=num_accepted_tokens is not None,
        NP2_STATELEN=np2_statelen,
        USE_PAD_SLOT=pad_slot_id is not None,
        SAVE_INTERMEDIATE=intermediate_conv_window is not None,
    )
    if unsqueeze:
        out = out.squeeze(-1)
    return out

def causal_conv1d_update_v2(
    x: torch.Tensor,
    conv_state: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    activation: Union[bool, str, None] = None,
    cache_seqlens: Optional[torch.Tensor] = None,
    conv_state_indices: Optional[torch.Tensor] = None,
    num_accepted_tokens: Optional[torch.Tensor] = None,
    intermediate_conv_window: Optional[torch.Tensor] = None,
    pad_slot_id: int = PAD_SLOT_ID,
    metadata=None,
    validate_data=False,
):
    """
    x: (batch, dim) or (batch, dim, seqlen)
        [shape=2: single token prediction]
        [shape=3: single or multiple tokens prediction]
    conv_state: (..., dim, state_len), where state_len >= width - 1
    weight: (dim, width)
    bias: (dim,)
    cache_seqlens: (batch,), dtype int32.
        If not None, the conv_state is treated as a circular buffer.
        The conv_state will be updated by copying x to the conv_state
        starting at the index
        @cache_seqlens % state_len.
    conv_state_indices: (batch,), dtype int32
        If not None, the conv_state is a larger tensor along the batch dim,
        and we are selecting the batch coords specified by conv_state_indices.
        Useful for a continuous batching scenario.
    pad_slot_id: int
            if cache_indices is passed, lets the kernel identify padded
            entries that will not be processed,
            for example: cache_indices = [pad_slot_id, 1 ,20 ,pad_slot_id]
            in this case, the kernel will not process entries at
            indices 0 and 3
    out: (batch, dim) or (batch, dim, seqlen)
    """
    if validate_data:
        assert cache_seqlens is None  # not implemented yet - ok for vLLM
        assert pad_slot_id is not None
        assert x.stride(1) == 1
    if isinstance(activation, bool):
        activation = "silu" if activation is True else None
    elif activation is not None:
        assert activation in ["silu", "swish"]
    unsqueeze = x.dim() == 2
    if unsqueeze:
        # make it (batch, dim, seqlen) with seqlen == 1
        x = x.unsqueeze(-1)
    batch, dim, seqlen = x.shape
    _, width = weight.shape
    # conv_state: (..., dim, state_len), where state_len >= width - 1
    num_cache_lines, _, state_len = conv_state.size()

    if validate_data:
        assert dim == weight.size(0)
        assert (
            conv_state.stride(-2) == 1
        ), f"ERROR: expect contiguous along feat-dim of conv_state (currently stride={conv_state.stride()})"
        assert state_len >= width - 1
        # when above happens, we don't shift-left to keep any records in conv_state
        assert dim == conv_state.size(1)
        if conv_state_indices is None:
            assert conv_state.size(0) >= batch
        else:
            assert (batch,) == conv_state_indices.shape

        assert num_cache_lines >= batch
        assert weight.stride(1) == 1  # Need this
        assert cache_seqlens is None  # not needed for vLLM - circular buffer

    # adopt the strategy in vLLM that overwrite on 'x' directly, rather than creating a new tensor 'o'
    out = x
    stride_w_dim, stride_w_width = weight.stride()

    stride_x_seq, stride_x_dim, stride_x_token = x.stride()  # X (batch, dim, seqlen)

    stride_o_seq, stride_o_dim, stride_o_token = out.stride()
    stride_istate_seq, stride_istate_dim, stride_istate_token = conv_state.stride()
    stride_state_indices = (
        conv_state_indices.stride(0) if conv_state_indices is not None else 0
    )
    if num_accepted_tokens is not None:
        state_len = width - 1 + (seqlen - 1)  # effective state_len needed
    else:
        state_len = width - 1
    np2_statelen = triton.next_power_of_2(state_len)

    def grid(META):
        return (
            batch,
            triton.cdiv(dim, META["BLOCK_N"]),
        )

    # prepare intermediate buffer strides if provided
    if intermediate_conv_window is not None:
        stride_inter_seq, stride_inter_step, stride_inter_dim, stride_inter_win = (
            intermediate_conv_window.stride(0),
            intermediate_conv_window.stride(1),
            intermediate_conv_window.stride(2),
            intermediate_conv_window.stride(3),
        )
    else:
        stride_inter_seq = stride_inter_step = stride_inter_dim = stride_inter_win = 0

    # _causal_conv1d_update_kernel[grid](
    gluon_causal_conv1d_update_kernel_v2[grid](
        # Pointers to matrices
        x,
        weight,
        bias,
        conv_state,
        cache_seqlens,
        conv_state_indices,
        num_accepted_tokens,
        intermediate_conv_window if intermediate_conv_window is not None else x,
        out,
        # Matrix dimensions
        batch,
        dim,
        seqlen,
        state_len,
        num_cache_lines,
        # stride
        stride_x_seq,
        stride_x_dim,
        stride_x_token,
        stride_w_dim,
        stride_w_width,
        stride_istate_seq,
        stride_istate_dim,
        stride_istate_token,
        stride_state_indices,
        stride_inter_seq,
        stride_inter_step,
        stride_inter_dim,
        stride_inter_win,
        stride_o_seq,
        stride_o_dim,
        stride_o_token,
        # others
        pad_slot_id,
        # META
        HAS_BIAS=bias is not None,
        KERNEL_WIDTH=width,
        SILU_ACTIVATION=activation in ["silu", "swish"],
        IS_CONTINUOUS_BATCHING=conv_state_indices is not None,
        IS_SPEC_DECODING=num_accepted_tokens is not None,
        NP2_STATELEN=np2_statelen,
        USE_PAD_SLOT=pad_slot_id is not None,
        BLOCK_N=256,
        SAVE_INTERMEDIATE=intermediate_conv_window is not None,
        num_warps=2
    )
    if unsqueeze:
        out = out.squeeze(-1)
    return out

def causal_conv1d_update_persistent(
    x: torch.Tensor,
    conv_state: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    activation: Union[bool, str, None] = None,
    cache_seqlens: Optional[torch.Tensor] = None,
    conv_state_indices: Optional[torch.Tensor] = None,
    num_accepted_tokens: Optional[torch.Tensor] = None,
    intermediate_conv_window: Optional[torch.Tensor] = None,
    pad_slot_id: int = PAD_SLOT_ID,
    metadata=None,
    validate_data=False,
):
    """
    x: (batch, dim) or (batch, dim, seqlen)
        [shape=2: single token prediction]
        [shape=3: single or multiple tokens prediction]
    conv_state: (..., dim, state_len), where state_len >= width - 1
    weight: (dim, width)
    bias: (dim,)
    cache_seqlens: (batch,), dtype int32.
        If not None, the conv_state is treated as a circular buffer.
        The conv_state will be updated by copying x to the conv_state
        starting at the index
        @cache_seqlens % state_len.
    conv_state_indices: (batch,), dtype int32
        If not None, the conv_state is a larger tensor along the batch dim,
        and we are selecting the batch coords specified by conv_state_indices.
        Useful for a continuous batching scenario.
    pad_slot_id: int
            if cache_indices is passed, lets the kernel identify padded
            entries that will not be processed,
            for example: cache_indices = [pad_slot_id, 1 ,20 ,pad_slot_id]
            in this case, the kernel will not process entries at
            indices 0 and 3
    out: (batch, dim) or (batch, dim, seqlen)
    """
    if validate_data:
        assert cache_seqlens is None  # not implemented yet - ok for vLLM
        assert pad_slot_id is not None
        assert x.stride(1) == 1
    if isinstance(activation, bool):
        activation = "silu" if activation is True else None
    elif activation is not None:
        assert activation in ["silu", "swish"]
    unsqueeze = x.dim() == 2
    if unsqueeze:
        # make it (batch, dim, seqlen) with seqlen == 1
        x = x.unsqueeze(-1)
    batch, dim, seqlen = x.shape
    _, width = weight.shape
    # conv_state: (..., dim, state_len), where state_len >= width - 1
    num_cache_lines, _, state_len = conv_state.size()

    if validate_data:
        assert dim == weight.size(0)
        assert (
            conv_state.stride(-2) == 1
        ), f"ERROR: expect contiguous along feat-dim of conv_state (currently stride={conv_state.stride()})"
        assert state_len >= width - 1
        # when above happens, we don't shift-left to keep any records in conv_state
        assert dim == conv_state.size(1)
        if conv_state_indices is None:
            assert conv_state.size(0) >= batch
        else:
            assert (batch,) == conv_state_indices.shape

        assert num_cache_lines >= batch
        assert weight.stride(1) == 1  # Need this
        assert cache_seqlens is None  # not needed for vLLM - circular buffer

    # adopt the strategy in vLLM that overwrite on 'x' directly, rather than creating a new tensor 'o'
    out = x
    stride_w_dim, stride_w_width = weight.stride()

    stride_x_seq, stride_x_dim, stride_x_token = x.stride()  # X (batch, dim, seqlen)

    stride_o_seq, stride_o_dim, stride_o_token = out.stride()
    stride_istate_seq, stride_istate_dim, stride_istate_token = conv_state.stride()
    stride_state_indices = (
        conv_state_indices.stride(0) if conv_state_indices is not None else 0
    )
    if num_accepted_tokens is not None:
        state_len = width - 1 + (seqlen - 1)  # effective state_len needed
    else:
        state_len = width - 1
    np2_statelen = triton.next_power_of_2(state_len)

    grid = (80,)

    # prepare intermediate buffer strides if provided
    if intermediate_conv_window is not None:
        stride_inter_seq, stride_inter_step, stride_inter_dim, stride_inter_win = (
            intermediate_conv_window.stride(0),
            intermediate_conv_window.stride(1),
            intermediate_conv_window.stride(2),
            intermediate_conv_window.stride(3),
        )
    else:
        stride_inter_seq = stride_inter_step = stride_inter_dim = stride_inter_win = 0

    gluon_causal_conv1d_update_persistent_kernel[grid](
        # Pointers to matrices
        x,
        weight,
        bias,
        conv_state,
        cache_seqlens,
        conv_state_indices,
        num_accepted_tokens,
        intermediate_conv_window if intermediate_conv_window is not None else x,
        out,
        # Matrix dimensions
        batch,
        dim,
        seqlen,
        state_len,
        num_cache_lines,
        # stride
        stride_x_seq,
        stride_x_dim,
        stride_x_token,
        stride_w_dim,
        stride_w_width,
        stride_istate_seq,
        stride_istate_dim,
        stride_istate_token,
        stride_state_indices,
        stride_inter_seq,
        stride_inter_step,
        stride_inter_dim,
        stride_inter_win,
        stride_o_seq,
        stride_o_dim,
        stride_o_token,
        # others
        pad_slot_id,
        # META
        HAS_BIAS=bias is not None,
        KERNEL_WIDTH=width,
        SILU_ACTIVATION=activation in ["silu", "swish"],
        IS_CONTINUOUS_BATCHING=conv_state_indices is not None,
        IS_SPEC_DECODING=num_accepted_tokens is not None,
        NP2_STATELEN=np2_statelen,
        USE_PAD_SLOT=pad_slot_id is not None,
        BLOCK_N=2048,
        SAVE_INTERMEDIATE=intermediate_conv_window is not None,
        num_warps=16
    )
    if unsqueeze:
        out = out.squeeze(-1)
    return out

def causal_conv1d_update_persistent_v1(
    x: torch.Tensor,
    conv_state: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    activation: Union[bool, str, None] = None,
    cache_seqlens: Optional[torch.Tensor] = None,
    conv_state_indices: Optional[torch.Tensor] = None,
    num_accepted_tokens: Optional[torch.Tensor] = None,
    intermediate_conv_window: Optional[torch.Tensor] = None,
    pad_slot_id: int = PAD_SLOT_ID,
    metadata=None,
    validate_data=False,
):
    """
    x: (batch, dim) or (batch, dim, seqlen)
        [shape=2: single token prediction]
        [shape=3: single or multiple tokens prediction]
    conv_state: (..., dim, state_len), where state_len >= width - 1
    weight: (dim, width)
    bias: (dim,)
    cache_seqlens: (batch,), dtype int32.
        If not None, the conv_state is treated as a circular buffer.
        The conv_state will be updated by copying x to the conv_state
        starting at the index
        @cache_seqlens % state_len.
    conv_state_indices: (batch,), dtype int32
        If not None, the conv_state is a larger tensor along the batch dim,
        and we are selecting the batch coords specified by conv_state_indices.
        Useful for a continuous batching scenario.
    pad_slot_id: int
            if cache_indices is passed, lets the kernel identify padded
            entries that will not be processed,
            for example: cache_indices = [pad_slot_id, 1 ,20 ,pad_slot_id]
            in this case, the kernel will not process entries at
            indices 0 and 3
    out: (batch, dim) or (batch, dim, seqlen)
    """
    if validate_data:
        assert cache_seqlens is None  # not implemented yet - ok for vLLM
        assert pad_slot_id is not None
        assert x.stride(1) == 1
    if isinstance(activation, bool):
        activation = "silu" if activation is True else None
    elif activation is not None:
        assert activation in ["silu", "swish"]
    unsqueeze = x.dim() == 2
    if unsqueeze:
        # make it (batch, dim, seqlen) with seqlen == 1
        x = x.unsqueeze(-1)
    batch, dim, seqlen = x.shape
    _, width = weight.shape
    # conv_state: (..., dim, state_len), where state_len >= width - 1
    num_cache_lines, _, state_len = conv_state.size()

    if validate_data:
        assert dim == weight.size(0)
        assert (
            conv_state.stride(-2) == 1
        ), f"ERROR: expect contiguous along feat-dim of conv_state (currently stride={conv_state.stride()})"
        assert state_len >= width - 1
        # when above happens, we don't shift-left to keep any records in conv_state
        assert dim == conv_state.size(1)
        if conv_state_indices is None:
            assert conv_state.size(0) >= batch
        else:
            assert (batch,) == conv_state_indices.shape

        assert num_cache_lines >= batch
        assert weight.stride(1) == 1  # Need this
        assert cache_seqlens is None  # not needed for vLLM - circular buffer

    # adopt the strategy in vLLM that overwrite on 'x' directly, rather than creating a new tensor 'o'
    out = x
    stride_w_dim, stride_w_width = weight.stride()

    stride_x_seq, stride_x_dim, stride_x_token = x.stride()  # X (batch, dim, seqlen)

    stride_o_seq, stride_o_dim, stride_o_token = out.stride()
    stride_istate_seq, stride_istate_dim, stride_istate_token = conv_state.stride()
    stride_state_indices = (
        conv_state_indices.stride(0) if conv_state_indices is not None else 0
    )
    if num_accepted_tokens is not None:
        state_len = width - 1 + (seqlen - 1)  # effective state_len needed
    else:
        state_len = width - 1
    np2_statelen = triton.next_power_of_2(state_len)

    grid = (80,)

    # prepare intermediate buffer strides if provided
    if intermediate_conv_window is not None:
        stride_inter_seq, stride_inter_step, stride_inter_dim, stride_inter_win = (
            intermediate_conv_window.stride(0),
            intermediate_conv_window.stride(1),
            intermediate_conv_window.stride(2),
            intermediate_conv_window.stride(3),
        )
    else:
        stride_inter_seq = stride_inter_step = stride_inter_dim = stride_inter_win = 0

    # Print all kernel parameters
    print("=" * 80)
    print("gluon_causal_conv1d_update_persistent_kernel_v1 Parameters:")
    print("=" * 80)
    
    print("\n【输入张量详细信息】")
    print("-" * 80)
    
    print(f"\n1. x (输入激活):")
    print(f"   - shape: {x.shape} -> (batch={x.shape[0]}, dim={x.shape[1]}, seqlen={x.shape[2]})")
    print(f"   - dtype: {x.dtype}")
    print(f"   - stride: {x.stride()}")
    print(f"   - is_contiguous: {x.is_contiguous()}")
    print(f"   - data_ptr: {hex(x.data_ptr())}")
    print(f"   - device: {x.device}")
    
    print(f"\n2. weight (卷积权重):")
    print(f"   - shape: {weight.shape} -> (dim={weight.shape[0]}, width={weight.shape[1]})")
    print(f"   - dtype: {weight.dtype}")
    print(f"   - stride: {weight.stride()}")
    print(f"   - is_contiguous: {weight.is_contiguous()}")
    print(f"   - data_ptr: {hex(weight.data_ptr())}")
    print(f"   - device: {weight.device}")
    
    if bias is not None:
        print(f"\n3. bias (偏置):")
        print(f"   - shape: {bias.shape} -> (dim={bias.shape[0]},)")
        print(f"   - dtype: {bias.dtype}")
        print(f"   - stride: {bias.stride()}")
        print(f"   - is_contiguous: {bias.is_contiguous()}")
        print(f"   - data_ptr: {hex(bias.data_ptr())}")
        print(f"   - device: {bias.device}")
    else:
        print(f"\n3. bias: None")
    
    print(f"\n4. conv_state (卷积状态缓存):")
    print(f"   - shape: {conv_state.shape} -> (num_cache_lines={conv_state.shape[0]}, dim={conv_state.shape[1]}, state_len={conv_state.shape[2]})")
    print(f"   - dtype: {conv_state.dtype}")
    print(f"   - stride: {conv_state.stride()}")
    print(f"   - is_contiguous: {conv_state.is_contiguous()}")
    print(f"   - data_ptr: {hex(conv_state.data_ptr())}")
    print(f"   - device: {conv_state.device}")
    
    if conv_state_indices is not None:
        print(f"\n5. conv_state_indices (批次索引映射):")
        print(f"   - shape: {conv_state_indices.shape} -> (batch={conv_state_indices.shape[0]},)")
        print(f"   - dtype: {conv_state_indices.dtype}")
        print(f"   - stride: {conv_state_indices.stride()}")
        print(f"   - is_contiguous: {conv_state_indices.is_contiguous()}")
        print(f"   - data_ptr: {hex(conv_state_indices.data_ptr())}")
        print(f"   - device: {conv_state_indices.device}")
        print(f"   - values (前10个): {conv_state_indices[:min(10, len(conv_state_indices))].tolist()}")
        print(f"   - min/max: {conv_state_indices.min().item()}/{conv_state_indices.max().item()}")
    else:
        print(f"\n5. conv_state_indices: None")
    
    print(f"\n6. out (输出):")
    print(f"   - shape: {out.shape} -> (batch={out.shape[0]}, dim={out.shape[1]}, seqlen={out.shape[2]})")
    print(f"   - dtype: {out.dtype}")
    print(f"   - stride: {out.stride()}")
    print(f"   - is_contiguous: {out.is_contiguous()}")
    print(f"   - data_ptr: {hex(out.data_ptr())}")
    print(f"   - device: {out.device}")
    print(f"   - shares storage with x: {out.data_ptr() == x.data_ptr()}")
    
    print("\n【矩阵维度】")
    print("-" * 80)
    print(f"  batch:           {batch:6d}  (批次大小)")
    print(f"  dim:             {dim:6d}  (特征维度/通道数)")
    print(f"  seqlen:          {seqlen:6d}  (序列长度)")
    print(f"  width:           {width:6d}  (卷积核宽度)")
    print(f"  state_len:       {state_len:6d}  (状态长度)")
    print(f"  num_cache_lines: {num_cache_lines:6d}  (缓存行数)")
    print(f"  np2_statelen:    {np2_statelen:6d}  (2的幂次状态长度)")
    
    print("\n【步幅信息】")
    print("-" * 80)
    print(f"  x 张量步幅:")
    print(f"    stride_x_seq:   {stride_x_seq:8d}  (批次间步幅)")
    print(f"    stride_x_dim:   {stride_x_dim:8d}  (特征间步幅)")
    print(f"    stride_x_token: {stride_x_token:8d}  (token间步幅)")
    
    print(f"\n  weight 张量步幅:")
    print(f"    stride_w_dim:   {stride_w_dim:8d}  (特征间步幅)")
    print(f"    stride_w_width: {stride_w_width:8d}  (卷积宽度步幅)")
    
    print(f"\n  conv_state 张量步幅:")
    print(f"    stride_conv_state_seq: {stride_istate_seq:8d}  (序列/缓存行间步幅)")
    print(f"    stride_conv_state_dim: {stride_istate_dim:8d}  (特征间步幅)")
    print(f"    stride_conv_state_tok: {stride_istate_token:8d}  (token间步幅)")
    print(f"    stride_state_indices:  {stride_state_indices:8d}  (索引步幅)")
    
    print(f"\n  output 张量步幅:")
    print(f"    stride_o_seq:   {stride_o_seq:8d}  (批次间步幅)")
    print(f"    stride_o_dim:   {stride_o_dim:8d}  (特征间步幅)")
    print(f"    stride_o_token: {stride_o_token:8d}  (token间步幅)")
    
    if intermediate_conv_window is not None:
        print(f"\n  intermediate 张量步幅:")
        print(f"    stride_inter_seq:  {stride_inter_seq:8d}")
        print(f"    stride_inter_step: {stride_inter_step:8d}")
        print(f"    stride_inter_dim:  {stride_inter_dim:8d}")
        print(f"    stride_inter_win:  {stride_inter_win:8d}")
    
    print("\n【配置参数】")
    print("-" * 80)
    print(f"  pad_slot_id:           {pad_slot_id}")
    print(f"  HAS_BIAS:              {bias is not None}")
    print(f"  KERNEL_WIDTH:          {width}")
    print(f"  SILU_ACTIVATION:       {activation in ['silu', 'swish']} (activation={activation})")
    print(f"  IS_CONTINUOUS_BATCHING: {conv_state_indices is not None}")
    print(f"  IS_SPEC_DECODING:      {num_accepted_tokens is not None}")
    print(f"  USE_PAD_SLOT:          {pad_slot_id is not None}")
    print(f"  BLOCK_N:               1024")
    print(f"  SAVE_INTERMEDIATE:     {intermediate_conv_window is not None}")
    print(f"  num_warps:             8")
    
    print("\n【内核启动配置】")
    print("-" * 80)
    print(f"  grid:           {grid}")
    print(f"  预期CU数:       80")
    print(f"  dim_blocks:     {(dim + 1023) // 1024}")
    print(f"  total_tasks:    {batch * ((dim + 1023) // 1024)}")
    print("=" * 80)
    print()

    gluon_causal_conv1d_update_persistent_kernel_v1[grid](
        # Pointers to matrices
        x,
        weight,
        bias,
        conv_state,
        cache_seqlens,
        conv_state_indices,
        num_accepted_tokens,
        intermediate_conv_window if intermediate_conv_window is not None else x,
        out,
        # Matrix dimensions
        batch,
        dim,
        seqlen,
        state_len,
        num_cache_lines,
        # stride
        stride_x_seq,
        stride_x_dim,
        stride_x_token,
        stride_w_dim,
        stride_w_width,
        stride_istate_seq,
        stride_istate_dim,
        stride_istate_token,
        stride_state_indices,
        stride_inter_seq,
        stride_inter_step,
        stride_inter_dim,
        stride_inter_win,
        stride_o_seq,
        stride_o_dim,
        stride_o_token,
        # others
        pad_slot_id,
        # META
        HAS_BIAS=bias is not None,
        KERNEL_WIDTH=width,
        SILU_ACTIVATION=activation in ["silu", "swish"],
        IS_CONTINUOUS_BATCHING=conv_state_indices is not None,
        IS_SPEC_DECODING=num_accepted_tokens is not None,
        NP2_STATELEN=np2_statelen,
        USE_PAD_SLOT=pad_slot_id is not None,
        BLOCK_N=1024,
        SAVE_INTERMEDIATE=intermediate_conv_window is not None,
        num_warps=8
    )
    if unsqueeze:
        out = out.squeeze(-1)
    return out

def causal_conv1d_update_persistent_v2(
    x: torch.Tensor,
    conv_state: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    activation: Union[bool, str, None] = None,
    cache_seqlens: Optional[torch.Tensor] = None,
    conv_state_indices: Optional[torch.Tensor] = None,
    num_accepted_tokens: Optional[torch.Tensor] = None,
    intermediate_conv_window: Optional[torch.Tensor] = None,
    pad_slot_id: int = PAD_SLOT_ID,
    metadata=None,
    validate_data=False,
):
    """
    x: (batch, dim) or (batch, dim, seqlen)
        [shape=2: single token prediction]
        [shape=3: single or multiple tokens prediction]
    conv_state: (..., dim, state_len), where state_len >= width - 1
    weight: (dim, width)
    bias: (dim,)
    cache_seqlens: (batch,), dtype int32.
        If not None, the conv_state is treated as a circular buffer.
        The conv_state will be updated by copying x to the conv_state
        starting at the index
        @cache_seqlens % state_len.
    conv_state_indices: (batch,), dtype int32
        If not None, the conv_state is a larger tensor along the batch dim,
        and we are selecting the batch coords specified by conv_state_indices.
        Useful for a continuous batching scenario.
    pad_slot_id: int
            if cache_indices is passed, lets the kernel identify padded
            entries that will not be processed,
            for example: cache_indices = [pad_slot_id, 1 ,20 ,pad_slot_id]
            in this case, the kernel will not process entries at
            indices 0 and 3
    out: (batch, dim) or (batch, dim, seqlen)
    """
    if validate_data:
        assert cache_seqlens is None  # not implemented yet - ok for vLLM
        assert pad_slot_id is not None
        assert x.stride(1) == 1
    if isinstance(activation, bool):
        activation = "silu" if activation is True else None
    elif activation is not None:
        assert activation in ["silu", "swish"]
    unsqueeze = x.dim() == 2
    if unsqueeze:
        # make it (batch, dim, seqlen) with seqlen == 1
        x = x.unsqueeze(-1)
    batch, dim, seqlen = x.shape
    _, width = weight.shape
    # conv_state: (..., dim, state_len), where state_len >= width - 1
    num_cache_lines, _, state_len = conv_state.size()

    if validate_data:
        assert dim == weight.size(0)
        assert (
            conv_state.stride(-2) == 1
        ), f"ERROR: expect contiguous along feat-dim of conv_state (currently stride={conv_state.stride()})"
        assert state_len >= width - 1
        # when above happens, we don't shift-left to keep any records in conv_state
        assert dim == conv_state.size(1)
        if conv_state_indices is None:
            assert conv_state.size(0) >= batch
        else:
            assert (batch,) == conv_state_indices.shape

        assert num_cache_lines >= batch
        assert weight.stride(1) == 1  # Need this
        assert cache_seqlens is None  # not needed for vLLM - circular buffer

    # adopt the strategy in vLLM that overwrite on 'x' directly, rather than creating a new tensor 'o'
    out = x
    stride_w_dim, stride_w_width = weight.stride()

    stride_x_seq, stride_x_dim, stride_x_token = x.stride()  # X (batch, dim, seqlen)

    stride_o_seq, stride_o_dim, stride_o_token = out.stride()
    stride_istate_seq, stride_istate_dim, stride_istate_token = conv_state.stride()
    stride_state_indices = (
        conv_state_indices.stride(0) if conv_state_indices is not None else 0
    )
    if num_accepted_tokens is not None:
        state_len = width - 1 + (seqlen - 1)  # effective state_len needed
    else:
        state_len = width - 1
    np2_statelen = triton.next_power_of_2(state_len)

    grid = (80,)

    # prepare intermediate buffer strides if provided
    if intermediate_conv_window is not None:
        stride_inter_seq, stride_inter_step, stride_inter_dim, stride_inter_win = (
            intermediate_conv_window.stride(0),
            intermediate_conv_window.stride(1),
            intermediate_conv_window.stride(2),
            intermediate_conv_window.stride(3),
        )
    else:
        stride_inter_seq = stride_inter_step = stride_inter_dim = stride_inter_win = 0

    gluon_causal_conv1d_update_persistent_kernel_v2[grid](
        # Pointers to matrices
        x,
        weight,
        bias,
        conv_state,
        cache_seqlens,
        conv_state_indices,
        num_accepted_tokens,
        intermediate_conv_window if intermediate_conv_window is not None else x,
        out,
        # Matrix dimensions
        batch,
        dim,
        seqlen,
        state_len,
        num_cache_lines,
        # stride
        stride_x_seq,
        stride_x_dim,
        stride_x_token,
        stride_w_dim,
        stride_w_width,
        stride_istate_seq,
        stride_istate_dim,
        stride_istate_token,
        stride_state_indices,
        stride_inter_seq,
        stride_inter_step,
        stride_inter_dim,
        stride_inter_win,
        stride_o_seq,
        stride_o_dim,
        stride_o_token,
        # others
        pad_slot_id,
        # META
        HAS_BIAS=bias is not None,
        KERNEL_WIDTH=width,
        SILU_ACTIVATION=activation in ["silu", "swish"],
        IS_CONTINUOUS_BATCHING=conv_state_indices is not None,
        IS_SPEC_DECODING=num_accepted_tokens is not None,
        NP2_STATELEN=np2_statelen,
        USE_PAD_SLOT=pad_slot_id is not None,
        SAVE_INTERMEDIATE=intermediate_conv_window is not None,
    )
    if unsqueeze:
        out = out.squeeze(-1)
    return out

def causal_conv1d_update_persistent_v3(
    x: torch.Tensor,
    conv_state: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    activation: Union[bool, str, None] = None,
    cache_seqlens: Optional[torch.Tensor] = None,
    conv_state_indices: Optional[torch.Tensor] = None,
    num_accepted_tokens: Optional[torch.Tensor] = None,
    intermediate_conv_window: Optional[torch.Tensor] = None,
    pad_slot_id: int = PAD_SLOT_ID,
    metadata=None,
    validate_data=False,
):
    """
    x: (batch, dim) or (batch, dim, seqlen)
        [shape=2: single token prediction]
        [shape=3: single or multiple tokens prediction]
    conv_state: (..., dim, state_len), where state_len >= width - 1
    weight: (dim, width)
    bias: (dim,)
    cache_seqlens: (batch,), dtype int32.
        If not None, the conv_state is treated as a circular buffer.
        The conv_state will be updated by copying x to the conv_state
        starting at the index
        @cache_seqlens % state_len.
    conv_state_indices: (batch,), dtype int32
        If not None, the conv_state is a larger tensor along the batch dim,
        and we are selecting the batch coords specified by conv_state_indices.
        Useful for a continuous batching scenario.
    pad_slot_id: int
            if cache_indices is passed, lets the kernel identify padded
            entries that will not be processed,
            for example: cache_indices = [pad_slot_id, 1 ,20 ,pad_slot_id]
            in this case, the kernel will not process entries at
            indices 0 and 3
    out: (batch, dim) or (batch, dim, seqlen)
    """
    if validate_data:
        assert cache_seqlens is None  # not implemented yet - ok for vLLM
        assert pad_slot_id is not None
        assert x.stride(1) == 1
    if isinstance(activation, bool):
        activation = "silu" if activation is True else None
    elif activation is not None:
        assert activation in ["silu", "swish"]
    unsqueeze = x.dim() == 2
    if unsqueeze:
        # make it (batch, dim, seqlen) with seqlen == 1
        x = x.unsqueeze(-1)
    batch, dim, seqlen = x.shape
    _, width = weight.shape
    # conv_state: (..., dim, state_len), where state_len >= width - 1
    num_cache_lines, _, state_len = conv_state.size()

    if validate_data:
        assert dim == weight.size(0)
        assert (
            conv_state.stride(-2) == 1
        ), f"ERROR: expect contiguous along feat-dim of conv_state (currently stride={conv_state.stride()})"
        assert state_len >= width - 1
        # when above happens, we don't shift-left to keep any records in conv_state
        assert dim == conv_state.size(1)
        if conv_state_indices is None:
            assert conv_state.size(0) >= batch
        else:
            assert (batch,) == conv_state_indices.shape

        assert num_cache_lines >= batch
        assert weight.stride(1) == 1  # Need this
        assert cache_seqlens is None  # not needed for vLLM - circular buffer

    # adopt the strategy in vLLM that overwrite on 'x' directly, rather than creating a new tensor 'o'
    out = x
    stride_w_dim, stride_w_width = weight.stride()

    stride_x_seq, stride_x_dim, stride_x_token = x.stride()  # X (batch, dim, seqlen)

    stride_o_seq, stride_o_dim, stride_o_token = out.stride()
    stride_istate_seq, stride_istate_dim, stride_istate_token = conv_state.stride()
    stride_state_indices = (
        conv_state_indices.stride(0) if conv_state_indices is not None else 0
    )
    if num_accepted_tokens is not None:
        state_len = width - 1 + (seqlen - 1)  # effective state_len needed
    else:
        state_len = width - 1
    np2_statelen = triton.next_power_of_2(state_len)

    grid = (80,)

    # prepare intermediate buffer strides if provided
    if intermediate_conv_window is not None:
        stride_inter_seq, stride_inter_step, stride_inter_dim, stride_inter_win = (
            intermediate_conv_window.stride(0),
            intermediate_conv_window.stride(1),
            intermediate_conv_window.stride(2),
            intermediate_conv_window.stride(3),
        )
    else:
        stride_inter_seq = stride_inter_step = stride_inter_dim = stride_inter_win = 0

    gluon_causal_conv1d_update_persistent_kernel_v3[grid](
        # Pointers to matrices
        x,
        weight,
        bias,
        conv_state,
        cache_seqlens,
        conv_state_indices,
        num_accepted_tokens,
        intermediate_conv_window if intermediate_conv_window is not None else x,
        out,
        # Matrix dimensions
        batch,
        dim,
        seqlen,
        state_len,
        num_cache_lines,
        # stride
        stride_x_seq,
        stride_x_dim,
        stride_x_token,
        stride_w_dim,
        stride_w_width,
        stride_istate_seq,
        stride_istate_dim,
        stride_istate_token,
        stride_state_indices,
        stride_inter_seq,
        stride_inter_step,
        stride_inter_dim,
        stride_inter_win,
        stride_o_seq,
        stride_o_dim,
        stride_o_token,
        # others
        pad_slot_id,
        # META
        HAS_BIAS=bias is not None,
        KERNEL_WIDTH=width,
        SILU_ACTIVATION=activation in ["silu", "swish"],
        IS_CONTINUOUS_BATCHING=conv_state_indices is not None,
        IS_SPEC_DECODING=num_accepted_tokens is not None,
        NP2_STATELEN=np2_statelen,
        USE_PAD_SLOT=pad_slot_id is not None,
        BLOCK_N=2048,
        SAVE_INTERMEDIATE=intermediate_conv_window is not None,
        num_warps=16
    )
    if unsqueeze:
        out = out.squeeze(-1)
    return out

def causal_conv1d_update_persistent_v4(
    x: torch.Tensor,
    conv_state: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    activation: Union[bool, str, None] = None,
    cache_seqlens: Optional[torch.Tensor] = None,
    conv_state_indices: Optional[torch.Tensor] = None,
    num_accepted_tokens: Optional[torch.Tensor] = None,
    intermediate_conv_window: Optional[torch.Tensor] = None,
    pad_slot_id: int = PAD_SLOT_ID,
    metadata=None,
    validate_data=False,
):
    """
    x: (batch, dim) or (batch, dim, seqlen)
        [shape=2: single token prediction]
        [shape=3: single or multiple tokens prediction]
    conv_state: (..., dim, state_len), where state_len >= width - 1
    weight: (dim, width)
    bias: (dim,)
    cache_seqlens: (batch,), dtype int32.
        If not None, the conv_state is treated as a circular buffer.
        The conv_state will be updated by copying x to the conv_state
        starting at the index
        @cache_seqlens % state_len.
    conv_state_indices: (batch,), dtype int32
        If not None, the conv_state is a larger tensor along the batch dim,
        and we are selecting the batch coords specified by conv_state_indices.
        Useful for a continuous batching scenario.
    pad_slot_id: int
            if cache_indices is passed, lets the kernel identify padded
            entries that will not be processed,
            for example: cache_indices = [pad_slot_id, 1 ,20 ,pad_slot_id]
            in this case, the kernel will not process entries at
            indices 0 and 3
    out: (batch, dim) or (batch, dim, seqlen)
    """
    if validate_data:
        assert cache_seqlens is None  # not implemented yet - ok for vLLM
        assert pad_slot_id is not None
        assert x.stride(1) == 1
    if isinstance(activation, bool):
        activation = "silu" if activation is True else None
    elif activation is not None:
        assert activation in ["silu", "swish"]
    unsqueeze = x.dim() == 2
    if unsqueeze:
        # make it (batch, dim, seqlen) with seqlen == 1
        x = x.unsqueeze(-1)
    batch, dim, seqlen = x.shape
    _, width = weight.shape
    # conv_state: (..., dim, state_len), where state_len >= width - 1
    num_cache_lines, _, state_len = conv_state.size()

    if validate_data:
        assert dim == weight.size(0)
        assert (
            conv_state.stride(-2) == 1
        ), f"ERROR: expect contiguous along feat-dim of conv_state (currently stride={conv_state.stride()})"
        assert state_len >= width - 1
        # when above happens, we don't shift-left to keep any records in conv_state
        assert dim == conv_state.size(1)
        if conv_state_indices is None:
            assert conv_state.size(0) >= batch
        else:
            assert (batch,) == conv_state_indices.shape

        assert num_cache_lines >= batch
        assert weight.stride(1) == 1  # Need this
        assert cache_seqlens is None  # not needed for vLLM - circular buffer

    # adopt the strategy in vLLM that overwrite on 'x' directly, rather than creating a new tensor 'o'
    out = x
    stride_w_dim, stride_w_width = weight.stride()

    stride_x_seq, stride_x_dim, stride_x_token = x.stride()  # X (batch, dim, seqlen)

    stride_o_seq, stride_o_dim, stride_o_token = out.stride()
    stride_istate_seq, stride_istate_dim, stride_istate_token = conv_state.stride()
    stride_state_indices = (
        conv_state_indices.stride(0) if conv_state_indices is not None else 0
    )
    if num_accepted_tokens is not None:
        state_len = width - 1 + (seqlen - 1)  # effective state_len needed
    else:
        state_len = width - 1
    np2_statelen = triton.next_power_of_2(state_len)

    grid = (80,)

    # prepare intermediate buffer strides if provided
    if intermediate_conv_window is not None:
        stride_inter_seq, stride_inter_step, stride_inter_dim, stride_inter_win = (
            intermediate_conv_window.stride(0),
            intermediate_conv_window.stride(1),
            intermediate_conv_window.stride(2),
            intermediate_conv_window.stride(3),
        )
    else:
        stride_inter_seq = stride_inter_step = stride_inter_dim = stride_inter_win = 0

    gluon_causal_conv1d_update_persistent_kernel_v6[grid](
        # Pointers to matrices
        x,
        weight,
        bias,
        conv_state,
        cache_seqlens,
        conv_state_indices,
        num_accepted_tokens,
        intermediate_conv_window if intermediate_conv_window is not None else x,
        out,
        # Matrix dimensions
        batch,
        dim,
        seqlen,
        state_len,
        num_cache_lines,
        # stride
        stride_x_seq,
        stride_x_dim,
        stride_x_token,
        stride_w_dim,
        stride_w_width,
        stride_istate_seq,
        stride_istate_dim,
        stride_istate_token,
        stride_state_indices,
        stride_inter_seq,
        stride_inter_step,
        stride_inter_dim,
        stride_inter_win,
        stride_o_seq,
        stride_o_dim,
        stride_o_token,
        # others
        pad_slot_id,
        # META
        HAS_BIAS=bias is not None,
        KERNEL_WIDTH=width,
        SILU_ACTIVATION=activation in ["silu", "swish"],
        IS_CONTINUOUS_BATCHING=conv_state_indices is not None,
        IS_SPEC_DECODING=num_accepted_tokens is not None,
        NP2_STATELEN=np2_statelen,
        USE_PAD_SLOT=pad_slot_id is not None,
        BLOCK_N=1024,
        SAVE_INTERMEDIATE=intermediate_conv_window is not None,
        num_warps=8
    )
    if unsqueeze:
        out = out.squeeze(-1)
    return out