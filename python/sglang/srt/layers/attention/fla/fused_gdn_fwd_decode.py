"""
Fused Gating Delta Network (GDN) Forward Decode Kernel.

This module implements a fused Triton kernel that combines:
1. Causal Conv1D update with split Q/K/V
2. Sigmoid gating delta rule update

By fusing these operations, we avoid intermediate memory reads/writes
and achieve better performance for decode phase.
"""

from typing import Optional, Tuple

import torch
import triton
import triton.language as tl
from triton.experimental import gluon
import triton.experimental.gluon.language as gl

PAD_SLOT_ID = -1


def is_cuda():
    return triton.runtime.driver.active.get_current_target().backend == "cuda"


def get_cuda_autotune_config():
    return [
        triton.Config({'BV': 8}, num_stages=3, num_warps=1),
    ]


def get_hip_autotune_config():
    return [
        triton.Config({'BV': 16}, num_stages=1, num_warps=1),
    ]


def get_autotune_config():
    if is_cuda():
        return get_cuda_autotune_config()
    else:
        return get_hip_autotune_config()


@tl.core.builtin
def tuple_combine(a: tl.tuple, b: tl.tensor, _semantic=None) -> tl.tuple:
    """Helper function to combine a tuple with a new tensor element."""
    return tl.tuple([*a.values, b])

@triton.heuristics(
    {
        "USE_INITIAL_STATE": lambda args: args["h0_source"] is not None,
        "IS_VARLEN": lambda args: args["cu_seqlens"] is not None,
    }
)
@triton.autotune(
    configs=get_autotune_config(),
    key=['K', 'V'],
)
@triton.jit(do_not_specialize=["T"])
def fused_gdn_fwd_decode_kernel(
    # Conv1D inputs
    x_ptr,  # (batch, dim, seqlen) where dim = 2*key_dim + value_dim
    conv_w_ptr,  # (dim, conv_width)
    conv_bias_ptr,
    conv_state_ptr,
    conv_state_indices_ptr,
    # Gating inputs
    A_log,
    a,
    dt_bias,
    b,
    # SSM state
    h0_source,
    h0_indices,
    cu_seqlens,
    # Output
    o,
    # Dimensions
    key_dim: tl.constexpr,
    value_dim: tl.constexpr,
    batch: int,
    dim: tl.constexpr,
    seqlen: tl.constexpr,
    conv_state_len: tl.constexpr,
    num_cache_lines: tl.constexpr,
    T: int,
    # Gating parameters
    softplus_beta: float,
    softplus_threshold: float,
    scale: float,
    # Strides for conv
    stride_x_seq: tl.constexpr,
    stride_x_dim: tl.constexpr,
    stride_x_token: tl.constexpr,
    stride_conv_w_dim: tl.constexpr,
    stride_conv_w_width: tl.constexpr,
    stride_conv_state_seq: tl.constexpr,
    stride_conv_state_dim: tl.constexpr,
    stride_conv_state_tok: tl.constexpr,
    stride_state_indices: tl.constexpr,
    # Others
    pad_slot_id: tl.constexpr,
    # Meta-parameters
    B: tl.constexpr,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    HAS_CONV_BIAS: tl.constexpr,
    CONV_WIDTH: tl.constexpr,
    SILU_ACTIVATION: tl.constexpr,
    IS_CONTINUOUS_BATCHING: tl.constexpr,
    NP2_STATELEN: tl.constexpr,
    USE_PAD_SLOT: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    USE_QK_L2NORM_IN_KERNEL: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    """
    Optimized fused kernel with Q/K reuse across V heads in a group.
    
    Key optimization: Each block processes GROUP_SIZE (HV//H) value heads,
    sharing the same Q/K computation. Uses tl.tuple to store multiple hidden states.
    
    This kernel processes:
    1. Causal conv1d for Q and K (once per group, reused)
    2. For each V head in group: conv1d for V + delta rule update
    3. Output results for all V heads in the group
    """
    # Get program IDs - now indexed by Q/K heads (not V heads)
    i_k, i_v, i_nh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_n, i_h = i_nh // H, i_nh % H
    
    # Number of V heads per Q/K head (group size)
    GROUP_SIZE: tl.constexpr = HV // H
    
    # Handle variable length sequences
    if IS_VARLEN:
        bos, eos = (
            tl.load(cu_seqlens + i_n).to(tl.int64),
            tl.load(cu_seqlens + i_n + 1).to(tl.int64),
        )
        all = T
        T = eos - bos
        idx_seq = bos
    else:
        bos, eos = i_n * T, i_n * T + T
        all = B * T
        idx_seq = i_n
    
    if idx_seq >= batch:
        return
    
    # Get conv state batch coordinate
    if IS_CONTINUOUS_BATCHING:
        conv_state_batch_coord = tl.load(
            conv_state_indices_ptr + idx_seq * stride_state_indices
        ).to(tl.int64)
    else:
        conv_state_batch_coord = idx_seq
        
    if USE_PAD_SLOT:
        if conv_state_batch_coord == pad_slot_id:
            return
    
    # ============================================================================
    # Part 1: Causal Conv1D with Split Q/K/V
    # ============================================================================
    
    # We process K dimensions for Q/K, and V dimensions separately
    # For this fused kernel, we focus on one head dimension at a time
    o_k = i_k * BK + tl.arange(0, BK)
    o_v = i_v * BV + tl.arange(0, BV)

    i_hv_0 = i_h * GROUP_SIZE
    i_hv_1 = i_h * GROUP_SIZE + 1

    # Initialize SSM hidden states for all V heads in this group
    # Using separate variables instead of tuple for better Triton compatibility
    # Load initial states if needed
    b_A_logs = ()
    b_dt_biases = ()
    for i in tl.static_range(GROUP_SIZE):
        i_hv = i_h * GROUP_SIZE + i
        b_A_log = tl.load(A_log + i_hv).to(tl.float32)
        b_dt_bias = tl.load(dt_bias + i_hv).to(tl.float32)
        b_A_logs = tuple_combine(b_A_logs, b_A_log)
        b_dt_biases = tuple_combine(b_dt_biases, b_dt_bias)

    if USE_INITIAL_STATE:
        idx = tl.load(h0_indices + i_n)
        if idx >= 0:
            # ====================================================================
            # Load initial hidden states for all V heads in this Q/K head group
            # Each Q/K head group manages GROUP_SIZE (HV//H) value heads
            # ====================================================================
            b_hs = ()
            for i in tl.static_range(GROUP_SIZE):
                i_hv = i_h * GROUP_SIZE + i
                p_h = (
                    h0_source
                    + idx * HV * K * V
                    + i_hv * K * V
                    + o_k[:, None] * V
                    + o_v[None, :]
                )
                b_h = tl.load(p_h).to(tl.float32)
                b_hs = tuple_combine(b_hs, b_h)

            # ====================================================================
            # Pre-load conv_state sliding windows and weights for Q and K
            # Window size = CONV_WIDTH - 1 (history values)
            # This avoids repeated memory loads in the token loop
            # ====================================================================
            q_dim_start = i_h * K
            q_feats = q_dim_start + o_k
            k_dim_start = key_dim + i_h * K
            k_feats = k_dim_start + o_k
            
            # Pre-load Q conv_states and weights
            b_q_conv_states = ()
            q_weights = (tl.load(conv_w_ptr + q_feats * stride_conv_w_dim),)
            for i in tl.static_range(CONV_WIDTH-1):
                b_q_conv_state = tl.load(conv_state_ptr + (conv_state_batch_coord * stride_conv_state_seq) + (q_feats * stride_conv_state_dim) + i * stride_conv_state_tok)
                w_val = tl.load(conv_w_ptr + q_feats * stride_conv_w_dim + (i+1) * stride_conv_w_width)
                q_weights = tuple_combine(q_weights, w_val)
                b_q_conv_states = tuple_combine(b_q_conv_states, b_q_conv_state)
            
            # Pre-load K conv_states and weights
            b_k_conv_states = ()
            k_weights = (tl.load(conv_w_ptr + k_feats * stride_conv_w_dim),)
            for i in tl.static_range(CONV_WIDTH-1):
                b_k_conv_state = tl.load(conv_state_ptr + (conv_state_batch_coord * stride_conv_state_seq) + (k_feats * stride_conv_state_dim) + i * stride_conv_state_tok)
                w_val = tl.load(conv_w_ptr + k_feats * stride_conv_w_dim + (i+1) * stride_conv_w_width)
                k_weights = tuple_combine(k_weights, w_val)
                b_k_conv_states = tuple_combine(b_k_conv_states, b_k_conv_state)
            
            # Pre-load V conv_states and weights for all GROUP_SIZE V heads
            b_v_conv_states_all = ()  # Tuple of tuples: (v0_states, v1_states, ...)
            v_weights_all = ()  # Tuple of tuples: (v0_weights, v1_weights, ...)
            for i in tl.static_range(GROUP_SIZE):
                i_hv = i_h * GROUP_SIZE + i
                v_dim_start = 2 * key_dim + i_hv * V
                v_feats = v_dim_start + o_v
                
                b_v_conv_states = ()
                v_weights = (tl.load(conv_w_ptr + v_feats * stride_conv_w_dim),)
                for j in tl.static_range(CONV_WIDTH-1):
                    b_v_conv_state = tl.load(conv_state_ptr + (conv_state_batch_coord * stride_conv_state_seq) + (v_feats * stride_conv_state_dim) + j * stride_conv_state_tok)
                    w_val = tl.load(conv_w_ptr + v_feats * stride_conv_w_dim + (j+1) * stride_conv_w_width)
                    v_weights = tuple_combine(v_weights, w_val)
                    b_v_conv_states = tuple_combine(b_v_conv_states, b_v_conv_state)
                
                b_v_conv_states_all = tuple_combine(b_v_conv_states_all, b_v_conv_states)
                v_weights_all = tuple_combine(v_weights_all, v_weights)
            
            # ====================================================================
            # Main token processing loop
            # For each token: compute K → V0,V1,... → Q → Delta Rule Updates
            # Order optimized to hide memory latency (K and V computed before Q)
            # ====================================================================
            for idx_token in tl.static_range(seqlen):
                # ================================================================
                # Step 1: Conv1D for K (computed once, reused across all V heads)
                # ================================================================
                k_conv_acc = tl.load(conv_bias_ptr + k_feats).to(tl.float32) if HAS_CONV_BIAS else tl.zeros([BK], dtype=tl.float32)
                k_ptrs = (
                            x_ptr + idx_seq * stride_x_seq 
                            + k_feats * stride_x_dim 
                            + idx_token * stride_x_token
                        )
                b_k_conv_states = tuple_combine(b_k_conv_states, tl.load(k_ptrs))
                # Causal conv: use history from conv_state and current input
                for j in tl.static_range(CONV_WIDTH):
                    k_conv_acc += b_k_conv_states[j] * k_weights[j]

                b_k_conv_states = b_k_conv_states[1:]
                
                if SILU_ACTIVATION:
                    k_conv_acc = k_conv_acc / (1 + tl.exp(-k_conv_acc))
                
                b_k = k_conv_acc.to(tl.float32)
                
                # Apply L2 normalization to K
                if USE_QK_L2NORM_IN_KERNEL:
                    b_k = b_k / (tl.sqrt(tl.sum(b_k * b_k) + 1e-6))
                
                # ================================================================
                # Step 2: Pre-process first V head (V0) to enable loop fusion
                # Compute V0's Conv1D before Q, maintaining K→V→Q load order
                # ================================================================
                i_hv_0 = i_h * GROUP_SIZE + 0
                v_dim_start_0 = 2 * key_dim + i_hv_0 * V
                v_feats_0 = v_dim_start_0 + o_v
                
                v_conv_acc_0 = tl.load(conv_bias_ptr + v_feats_0).to(tl.float32) if HAS_CONV_BIAS else tl.zeros([BV], dtype=tl.float32)
                v_ptrs_0 = (
                    x_ptr + idx_seq * stride_x_seq 
                    + v_feats_0 * stride_x_dim 
                    + idx_token * stride_x_token
                )
                
                b_v_conv_states_0 = b_v_conv_states_all[0]
                v_weights_0 = v_weights_all[0]
                b_v_conv_states_0 = tuple_combine(b_v_conv_states_0, tl.load(v_ptrs_0))
                for j in tl.static_range(CONV_WIDTH):
                    v_conv_acc_0 += b_v_conv_states_0[j] * v_weights_0[j]
                b_v_conv_states_0 = b_v_conv_states_0[1:]
                
                if SILU_ACTIVATION:
                    v_conv_acc_0 = v_conv_acc_0 / (1 + tl.exp(-v_conv_acc_0))
                b_v_prev = v_conv_acc_0.to(tl.float32)
                
                new_b_v_conv_states_all = (b_v_conv_states_0,)
                
                # ================================================================
                # Step 3: Conv1D for Q (after first V, before loop)
                # This maintains K→V0→Q load order and avoids dynamic branching
                # ================================================================
                q_conv_acc = tl.load(conv_bias_ptr + q_feats).to(tl.float32) if HAS_CONV_BIAS else tl.zeros([BK], dtype=tl.float32)
                q_ptrs = (
                    x_ptr + idx_seq * stride_x_seq 
                    + q_feats * stride_x_dim 
                    + idx_token * stride_x_token
                )
                b_q_conv_states = tuple_combine(b_q_conv_states, tl.load(q_ptrs))
                for j in tl.static_range(CONV_WIDTH):
                    q_conv_acc += b_q_conv_states[j] * q_weights[j]
                b_q_conv_states = b_q_conv_states[1:]
                
                if SILU_ACTIVATION:
                    q_conv_acc = q_conv_acc / (1 + tl.exp(-q_conv_acc))
                
                b_q = q_conv_acc.to(tl.float32)
                if USE_QK_L2NORM_IN_KERNEL:
                    b_q_scale = scale / (tl.sqrt(tl.sum(b_q * b_q) + 1e-6))
                else:
                    b_q_scale = scale
                b_q = b_q * b_q_scale
                
                # ================================================================
                # Step 4: Fused loop for remaining V heads + Delta Rule updates
                # Loop structure: V_i Conv1D → V_{i-1} Delta Rule Update
                # This maintains K→V→Q load order while fusing computations
                # ================================================================
                for i in tl.static_range(1, GROUP_SIZE):
                    # ============================================================
                    # Part A: Compute V_i Conv1D (prefetch next V)
                    # ============================================================
                    i_hv = i_h * GROUP_SIZE + i
                    v_dim_start = 2 * key_dim + i_hv * V
                    v_feats = v_dim_start + o_v
                    
                    v_conv_acc = tl.load(conv_bias_ptr + v_feats).to(tl.float32) if HAS_CONV_BIAS else tl.zeros([BV], dtype=tl.float32)
                    v_ptrs = (
                        x_ptr + idx_seq * stride_x_seq 
                        + v_feats * stride_x_dim 
                        + idx_token * stride_x_token
                    )
                    
                    b_v_conv_states = b_v_conv_states_all[i]
                    v_weights = v_weights_all[i]
                    b_v_conv_states = tuple_combine(b_v_conv_states, tl.load(v_ptrs))
                    for j in tl.static_range(CONV_WIDTH):
                        v_conv_acc += b_v_conv_states[j] * v_weights[j]
                    b_v_conv_states = b_v_conv_states[1:]
                    new_b_v_conv_states_all = tuple_combine(new_b_v_conv_states_all, b_v_conv_states)
                    
                    if SILU_ACTIVATION:
                        v_conv_acc = v_conv_acc / (1 + tl.exp(-v_conv_acc))
                    b_v_curr = v_conv_acc.to(tl.float32)
                    
                    # ============================================================
                    # Part B: Delta Rule update for V_{i-1}
                    # ============================================================
                    i_hv_prev = i_h * GROUP_SIZE + (i - 1)
                    
                    # Load time-variant gating parameters
                    p_a = a + (bos + idx_token) * HV + i_hv_prev
                    p_b = b + (bos + idx_token) * HV + i_hv_prev
                    b_a = tl.load(p_a).to(tl.float32)
                    b_b = tl.load(p_b).to(tl.float32)
                    
                    # Compute gating factor
                    x = b_a + b_dt_biases[i - 1]
                    beta_x = softplus_beta * x
                    softplus_x = tl.where(
                        beta_x <= softplus_threshold,
                        (1.0 / softplus_beta) * tl.log(1.0 + tl.exp(beta_x)),
                        x,
                    )
                    b_g = -tl.exp(b_A_logs[i - 1]) * softplus_x
                    b_beta = 1.0 / (1.0 + tl.exp(-b_b))
                    
                    # Delta rule recurrent update
                    b_h = b_hs[i - 1]
                    b_h *= tl.exp(b_g)
                    b_v_prev -= tl.sum(b_h * b_k[:, None], 0)
                    b_v_prev *= b_beta
                    b_h += b_k[:, None] * b_v_prev[None, :]
                    
                    # Compute and store output
                    b_o = tl.sum(b_h * b_q[:, None], 0)
                    p_o = o + ((i_k * all + bos + idx_token) * HV + i_hv_prev) * V + o_v
                    tl.store(p_o, b_o.to(p_o.dtype.element_ty))
                    
                    # Store updated hidden state
                    p_h0 = (
                        h0_source
                        + idx * HV * K * V
                        + i_hv_prev * K * V
                        + o_k[:, None] * V
                        + o_v[None, :]
                    )
                    tl.store(p_h0, b_h.to(p_h0.dtype.element_ty))
                    
                    # Move to next iteration
                    b_v_prev = b_v_curr
                
                b_v_conv_states_all = new_b_v_conv_states_all
                
                # ================================================================
                # Step 4: Process last V head's Delta Rule update
                # ================================================================
                i_hv_last = i_h * GROUP_SIZE + (GROUP_SIZE - 1)
                
                p_a = a + (bos + idx_token) * HV + i_hv_last
                p_b = b + (bos + idx_token) * HV + i_hv_last
                b_a = tl.load(p_a).to(tl.float32)
                b_b = tl.load(p_b).to(tl.float32)
                
                x = b_a + b_dt_biases[GROUP_SIZE - 1]
                beta_x = softplus_beta * x
                softplus_x = tl.where(
                    beta_x <= softplus_threshold,
                    (1.0 / softplus_beta) * tl.log(1.0 + tl.exp(beta_x)),
                    x,
                )
                b_g = -tl.exp(b_A_logs[GROUP_SIZE - 1]) * softplus_x
                b_beta = 1.0 / (1.0 + tl.exp(-b_b))
                
                b_h = b_hs[GROUP_SIZE - 1]
                b_h *= tl.exp(b_g)
                b_v_prev -= tl.sum(b_h * b_k[:, None], 0)
                b_v_prev *= b_beta
                b_h += b_k[:, None] * b_v_prev[None, :]
                
                b_o = tl.sum(b_h * b_q[:, None], 0)
                p_o = o + ((i_k * all + bos + idx_token) * HV + i_hv_last) * V + o_v
                tl.store(p_o, b_o.to(p_o.dtype.element_ty))
                
                p_h0 = (
                    h0_source
                    + idx * HV * K * V
                    + i_hv_last * K * V
                    + o_k[:, None] * V
                    + o_v[None, :]
                )
                tl.store(p_h0, b_h.to(p_h0.dtype.element_ty))

            # ====================================================================
            # Write back final conv_state windows to memory
            # After processing all tokens, store the last CONV_WIDTH-1 states
            # ====================================================================
            # Write back Q conv_states
            for i in tl.static_range(CONV_WIDTH-1):
                tl.store(conv_state_ptr + (conv_state_batch_coord * stride_conv_state_seq) + (q_feats * stride_conv_state_dim) + i * stride_conv_state_tok, b_q_conv_states[i])
            
            # Write back K conv_states
            for i in tl.static_range(CONV_WIDTH-1):
                tl.store(conv_state_ptr + (conv_state_batch_coord * stride_conv_state_seq) + (k_feats * stride_conv_state_dim) + i * stride_conv_state_tok, b_k_conv_states[i])
            
            # Write back V conv_states for all V heads
            for i in tl.static_range(GROUP_SIZE):
                i_hv = i_h * GROUP_SIZE + i
                v_dim_start = 2 * key_dim + i_hv * V
                v_feats = v_dim_start + o_v
                
                b_v_conv_states = b_v_conv_states_all[i]
                for j in tl.static_range(CONV_WIDTH-1):
                    tl.store(conv_state_ptr + (conv_state_batch_coord * stride_conv_state_seq) + (v_feats * stride_conv_state_dim) + j * stride_conv_state_tok, b_v_conv_states[j])
            
            return

    # Initialize zero hidden states for all V heads in the group
    b_hs = ()
    for i in tl.static_range(GROUP_SIZE):
        b_h = tl.zeros([BK, BV], dtype=tl.float32)
        b_hs = tuple_combine(b_hs, b_h)

    # ====================================================================
    # Pre-load conv_state sliding windows and weights for Q and K
    # Window size = CONV_WIDTH - 1 (history values)
    # This avoids repeated memory loads in the token loop
    # ====================================================================
    q_dim_start = i_h * K
    q_feats = q_dim_start + o_k
    k_dim_start = key_dim + i_h * K
    k_feats = k_dim_start + o_k
    
    # Pre-load Q conv_states and weights
    b_q_conv_states = ()
    q_weights = (tl.load(conv_w_ptr + q_feats * stride_conv_w_dim),)
    for i in tl.static_range(CONV_WIDTH-1):
        b_q_conv_state = tl.load(conv_state_ptr + (conv_state_batch_coord * stride_conv_state_seq) + (q_feats * stride_conv_state_dim) + i * stride_conv_state_tok)
        w_val = tl.load(conv_w_ptr + q_feats * stride_conv_w_dim + (i+1) * stride_conv_w_width)
        q_weights = tuple_combine(q_weights, w_val)
        b_q_conv_states = tuple_combine(b_q_conv_states, b_q_conv_state)
    
    # Pre-load K conv_states and weights
    b_k_conv_states = ()
    k_weights = (tl.load(conv_w_ptr + k_feats * stride_conv_w_dim),)
    for i in tl.static_range(CONV_WIDTH-1):
        b_k_conv_state = tl.load(conv_state_ptr + (conv_state_batch_coord * stride_conv_state_seq) + (k_feats * stride_conv_state_dim) + i * stride_conv_state_tok)
        w_val = tl.load(conv_w_ptr + k_feats * stride_conv_w_dim + (i+1) * stride_conv_w_width)
        k_weights = tuple_combine(k_weights, w_val)
        b_k_conv_states = tuple_combine(b_k_conv_states, b_k_conv_state)
    
    # Pre-load V conv_states and weights for all GROUP_SIZE V heads
    b_v_conv_states_all = ()  # Tuple of tuples: (v0_states, v1_states, ...)
    v_weights_all = ()  # Tuple of tuples: (v0_weights, v1_weights, ...)
    for i in tl.static_range(GROUP_SIZE):
        i_hv = i_h * GROUP_SIZE + i
        v_dim_start = 2 * key_dim + i_hv * V
        v_feats = v_dim_start + o_v
        
        b_v_conv_states = ()
        v_weights = (tl.load(conv_w_ptr + v_feats * stride_conv_w_dim),)
        for j in tl.static_range(CONV_WIDTH-1):
            b_v_conv_state = tl.load(conv_state_ptr + (conv_state_batch_coord * stride_conv_state_seq) + (v_feats * stride_conv_state_dim) + j * stride_conv_state_tok)
            w_val = tl.load(conv_w_ptr + v_feats * stride_conv_w_dim + (j+1) * stride_conv_w_width)
            v_weights = tuple_combine(v_weights, w_val)
            b_v_conv_states = tuple_combine(b_v_conv_states, b_v_conv_state)
        
        b_v_conv_states_all = tuple_combine(b_v_conv_states_all, b_v_conv_states)
        v_weights_all = tuple_combine(v_weights_all, v_weights)

    # Process each token in the sequence
    for idx_token in tl.static_range(seqlen):
        # ====================================================================
        # Step 1: Conv1D for K (computed once, reused across all V heads)
        # ====================================================================
        k_conv_acc = tl.load(conv_bias_ptr + k_feats).to(tl.float32) if HAS_CONV_BIAS else tl.zeros([BK], dtype=tl.float32)
        k_ptrs = (
                    x_ptr + idx_seq * stride_x_seq 
                    + k_feats * stride_x_dim 
                    + idx_token * stride_x_token
                )
        b_k_conv_states = tuple_combine(b_k_conv_states, tl.load(k_ptrs))
        # Causal conv: use history from conv_state and current input
        for j in tl.static_range(CONV_WIDTH):
            k_conv_acc += b_k_conv_states[j] * k_weights[j]

        b_k_conv_states = b_k_conv_states[1:]
        
        if SILU_ACTIVATION:
            k_conv_acc = k_conv_acc / (1 + tl.exp(-k_conv_acc))
        
        b_k = k_conv_acc.to(tl.float32)
        
        # Apply L2 normalization to K
        if USE_QK_L2NORM_IN_KERNEL:
            b_k = b_k / (tl.sqrt(tl.sum(b_k * b_k) + 1e-6))
        
        # ================================================================
        # Step 2: Pre-process first V head (V0) to enable loop fusion
        # Compute V0's Conv1D before Q, maintaining K→V→Q load order
        # ================================================================
        i_hv_0 = i_h * GROUP_SIZE + 0
        v_dim_start_0 = 2 * key_dim + i_hv_0 * V
        v_feats_0 = v_dim_start_0 + o_v
        
        v_conv_acc_0 = tl.load(conv_bias_ptr + v_feats_0).to(tl.float32) if HAS_CONV_BIAS else tl.zeros([BV], dtype=tl.float32)
        v_ptrs_0 = (
            x_ptr + idx_seq * stride_x_seq 
            + v_feats_0 * stride_x_dim 
            + idx_token * stride_x_token
        )
        
        b_v_conv_states_0 = b_v_conv_states_all[0]
        v_weights_0 = v_weights_all[0]
        b_v_conv_states_0 = tuple_combine(b_v_conv_states_0, tl.load(v_ptrs_0))
        for j in tl.static_range(CONV_WIDTH):
            v_conv_acc_0 += b_v_conv_states_0[j] * v_weights_0[j]
        b_v_conv_states_0 = b_v_conv_states_0[1:]
        
        if SILU_ACTIVATION:
            v_conv_acc_0 = v_conv_acc_0 / (1 + tl.exp(-v_conv_acc_0))
        b_v_prev = v_conv_acc_0.to(tl.float32)
        
        new_b_v_conv_states_all = (b_v_conv_states_0,)
        
        # ================================================================
        # Step 3: Conv1D for Q (after first V, before loop)
        # This maintains K→V0→Q load order and avoids dynamic branching
        # ================================================================
        q_conv_acc = tl.load(conv_bias_ptr + q_feats).to(tl.float32) if HAS_CONV_BIAS else tl.zeros([BK], dtype=tl.float32)
        q_ptrs = (
            x_ptr + idx_seq * stride_x_seq 
            + q_feats * stride_x_dim 
            + idx_token * stride_x_token
        )
        b_q_conv_states = tuple_combine(b_q_conv_states, tl.load(q_ptrs))
        for j in tl.static_range(CONV_WIDTH):
            q_conv_acc += b_q_conv_states[j] * q_weights[j]
        b_q_conv_states = b_q_conv_states[1:]
        
        if SILU_ACTIVATION:
            q_conv_acc = q_conv_acc / (1 + tl.exp(-q_conv_acc))
        
        b_q = q_conv_acc.to(tl.float32)
        if USE_QK_L2NORM_IN_KERNEL:
            b_q_scale = scale / (tl.sqrt(tl.sum(b_q * b_q) + 1e-6))
        else:
            b_q_scale = scale
        b_q = b_q * b_q_scale
        
        # ================================================================
        # Step 4: Fused loop for remaining V heads + Delta Rule updates
        # Loop structure: V_i Conv1D → V_{i-1} Delta Rule Update
        # This maintains K→V→Q load order while fusing computations
        # ================================================================
        for i in tl.static_range(1, GROUP_SIZE):
            # ============================================================
            # Part A: Compute V_i Conv1D (prefetch next V)
            # ============================================================
            i_hv = i_h * GROUP_SIZE + i
            v_dim_start = 2 * key_dim + i_hv * V
            v_feats = v_dim_start + o_v
            
            v_conv_acc = tl.load(conv_bias_ptr + v_feats).to(tl.float32) if HAS_CONV_BIAS else tl.zeros([BV], dtype=tl.float32)
            v_ptrs = (
                x_ptr + idx_seq * stride_x_seq 
                + v_feats * stride_x_dim 
                + idx_token * stride_x_token
            )
            
            b_v_conv_states = b_v_conv_states_all[i]
            v_weights = v_weights_all[i]
            b_v_conv_states = tuple_combine(b_v_conv_states, tl.load(v_ptrs))
            for j in tl.static_range(CONV_WIDTH):
                v_conv_acc += b_v_conv_states[j] * v_weights[j]
            b_v_conv_states = b_v_conv_states[1:]
            new_b_v_conv_states_all = tuple_combine(new_b_v_conv_states_all, b_v_conv_states)
            
            if SILU_ACTIVATION:
                v_conv_acc = v_conv_acc / (1 + tl.exp(-v_conv_acc))
            b_v_curr = v_conv_acc.to(tl.float32)
            
            # ============================================================
            # Part B: Delta Rule update for V_{i-1}
            # ============================================================
            i_hv_prev = i_h * GROUP_SIZE + (i - 1)
            
            p_a = a + (bos + idx_token) * HV + i_hv_prev
            p_b = b + (bos + idx_token) * HV + i_hv_prev
            b_a = tl.load(p_a).to(tl.float32)
            b_b = tl.load(p_b).to(tl.float32)
            
            x = b_a + b_dt_biases[i - 1]
            beta_x = softplus_beta * x
            softplus_x = tl.where(
                beta_x <= softplus_threshold,
                (1.0 / softplus_beta) * tl.log(1.0 + tl.exp(beta_x)),
                x,
            )
            b_g = -tl.exp(b_A_logs[i - 1]) * softplus_x
            b_beta = 1.0 / (1.0 + tl.exp(-b_b))
            
            b_h = b_hs[i - 1]
            b_h *= tl.exp(b_g)
            b_v_prev -= tl.sum(b_h * b_k[:, None], 0)
            b_v_prev *= b_beta
            b_h += b_k[:, None] * b_v_prev[None, :]
            
            b_o = tl.sum(b_h * b_q[:, None], 0)
            p_o = o + ((i_k * all + bos + idx_token) * HV + i_hv_prev) * V + o_v
            tl.store(p_o, b_o.to(p_o.dtype.element_ty))
            
            # Move to next iteration
            b_v_prev = b_v_curr
        
        b_v_conv_states_all = new_b_v_conv_states_all
        
        # ================================================================
        # Step 5: Process last V head's Delta Rule update
        # ================================================================
        i_hv_last = i_h * GROUP_SIZE + (GROUP_SIZE - 1)
        
        p_a = a + (bos + idx_token) * HV + i_hv_last
        p_b = b + (bos + idx_token) * HV + i_hv_last
        b_a = tl.load(p_a).to(tl.float32)
        b_b = tl.load(p_b).to(tl.float32)
        
        x = b_a + b_dt_biases[GROUP_SIZE - 1]
        beta_x = softplus_beta * x
        softplus_x = tl.where(
            beta_x <= softplus_threshold,
            (1.0 / softplus_beta) * tl.log(1.0 + tl.exp(beta_x)),
            x,
        )
        b_g = -tl.exp(b_A_logs[GROUP_SIZE - 1]) * softplus_x
        b_beta = 1.0 / (1.0 + tl.exp(-b_b))
        
        b_h = b_hs[GROUP_SIZE - 1]
        b_h *= tl.exp(b_g)
        b_v_prev -= tl.sum(b_h * b_k[:, None], 0)
        b_v_prev *= b_beta
        b_h += b_k[:, None] * b_v_prev[None, :]
        
        b_o = tl.sum(b_h * b_q[:, None], 0)
        p_o = o + ((i_k * all + bos + idx_token) * HV + i_hv_last) * V + o_v
        tl.store(p_o, b_o.to(p_o.dtype.element_ty))
    
    # ====================================================================
    # Write back final conv_state windows to memory
    # After processing all tokens, store the last CONV_WIDTH-1 states
    # ====================================================================
    # Write back Q conv_states
    for i in tl.static_range(CONV_WIDTH-1):
        tl.store(conv_state_ptr + (conv_state_batch_coord * stride_conv_state_seq) + (q_feats * stride_conv_state_dim) + i * stride_conv_state_tok, b_q_conv_states[i])
    
    # Write back K conv_states
    for i in tl.static_range(CONV_WIDTH-1):
        tl.store(conv_state_ptr + (conv_state_batch_coord * stride_conv_state_seq) + (k_feats * stride_conv_state_dim) + i * stride_conv_state_tok, b_k_conv_states[i])
    
    # Write back V conv_states for all V heads
    for i in tl.static_range(GROUP_SIZE):
        i_hv = i_h * GROUP_SIZE + i
        v_dim_start = 2 * key_dim + i_hv * V
        v_feats = v_dim_start + o_v
        
        b_v_conv_states = b_v_conv_states_all[i]
        for j in tl.static_range(CONV_WIDTH-1):
            tl.store(conv_state_ptr + (conv_state_batch_coord * stride_conv_state_seq) + (v_feats * stride_conv_state_dim) + j * stride_conv_state_tok, b_v_conv_states[j])


def fused_gdn_fwd_decode(
    mixed_qkv: torch.Tensor,
    conv_state: torch.Tensor,
    conv_weight: torch.Tensor,
    A_log: torch.Tensor,
    a: torch.Tensor,
    dt_bias: torch.Tensor,
    b: torch.Tensor,
    ssm_state: torch.Tensor,
    key_dim: int,
    value_dim: int,
    num_heads_qk: int,
    num_heads_v: int,
    head_dim: int,
    conv_bias: Optional[torch.Tensor] = None,
    activation: Optional[str] = "silu",
    conv_state_indices: Optional[torch.Tensor] = None,
    ssm_state_indices: Optional[torch.Tensor] = None,
    pad_slot_id: int = PAD_SLOT_ID,
    scale: Optional[float] = None,
    use_qk_l2norm_in_kernel: bool = True,
    softplus_beta: float = 1.0,
    softplus_threshold: float = 20.0,
    cu_seqlens: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Fused Gating Delta Network forward decode operation.
    
    Args:
        mixed_qkv: Input tensor (batch, dim, seqlen) where dim = 2*key_dim + value_dim
        conv_state: Convolution state (num_cache_lines, dim, state_len)
        conv_weight: Convolution weights (dim, width)
        A_log: Gating parameter A (num_heads_v * head_dim,)
        a: Gating parameter a (batch, num_heads_v * head_dim)
        dt_bias: Gating parameter dt_bias (num_heads_v * head_dim,)
        b: Gating parameter b (batch, num_heads_v * head_dim)
        ssm_state: SSM state pool (num_cache_lines, num_heads_v * head_dim, head_dim, head_dim)
        key_dim: Dimension of query and key (= num_heads_qk * head_dim)
        value_dim: Dimension of value (= num_heads_v * head_dim)
        num_heads_qk: Number of query/key heads
        num_heads_v: Number of value heads
        head_dim: Dimension per head
        conv_bias: Optional convolution bias (dim,)
        activation: Activation function ("silu" or None)
        conv_state_indices: Optional batch indices for continuous batching
        ssm_state_indices: Optional batch indices for SSM state
        pad_slot_id: ID for padded slots
        scale: Query scaling factor
        use_qk_l2norm_in_kernel: Whether to apply L2 normalization to Q/K
        softplus_beta: Beta parameter for softplus
        softplus_threshold: Threshold for softplus
        cu_seqlens: Cumulative sequence lengths for variable length sequences
    
    Returns:
        Output tensor
    """
    batch, dim, seqlen = mixed_qkv.shape
    assert dim == 2 * key_dim + value_dim, f"dim {dim} != 2*{key_dim} + {value_dim}"
    assert key_dim == num_heads_qk * head_dim, f"key_dim {key_dim} != {num_heads_qk} * {head_dim}"
    assert value_dim == num_heads_v * head_dim, f"value_dim {value_dim} != {num_heads_v} * {head_dim}"
    
    _, conv_width = conv_weight.shape
    num_cache_lines, _, conv_state_len = conv_state.size()
    
    # Head dimensions
    # HV is the number of value heads (not total value dimension)
    HV = num_heads_v
    K = head_dim
    V = head_dim
    H = num_heads_qk
    
    if scale is None:
        scale = K ** -0.5
    
    T = seqlen
    N = batch if cu_seqlens is None else len(cu_seqlens) - 1
    B = batch
    
    BK = triton.next_power_of_2(K)
    NK = triton.cdiv(K, BK)
    assert NK == 1, "NK > 1 is not supported yet"
    
    # Create output tensor
    # Shape should match the gating delta rule output: (NK, B, T, HV, V)
    o = mixed_qkv.new_empty(NK, B, T, HV, V)
    
    # Grid configuration: launch N * H blocks (one per Q/K head, not per V head)
    # Each block processes GROUP_SIZE (HV//H) V heads
    grid = lambda META: (NK, triton.cdiv(V, META['BV']), N * H)
    
    # Launch kernel
    stride_state_indices = (
        conv_state_indices.stride(0) if conv_state_indices is not None else 0
    )
    np2_statelen = triton.next_power_of_2(conv_state_len)
    
    fused_gdn_fwd_decode_kernel[grid](
        x_ptr=mixed_qkv,
        conv_w_ptr=conv_weight,
        conv_bias_ptr=conv_bias,
        conv_state_ptr=conv_state,
        conv_state_indices_ptr=conv_state_indices,
        A_log=A_log,
        a=a,
        dt_bias=dt_bias,
        b=b,
        h0_source=ssm_state,
        h0_indices=ssm_state_indices,
        cu_seqlens=cu_seqlens,
        o=o,
        key_dim=key_dim,
        value_dim=value_dim,
        batch=batch,
        dim=dim,
        seqlen=seqlen,
        conv_state_len=conv_state_len,
        num_cache_lines=num_cache_lines,
        T=T,
        softplus_beta=softplus_beta,
        softplus_threshold=softplus_threshold,
        scale=scale,
        stride_x_seq=mixed_qkv.stride(0),
        stride_x_dim=mixed_qkv.stride(1),
        stride_x_token=mixed_qkv.stride(2),
        stride_conv_w_dim=conv_weight.stride(0),
        stride_conv_w_width=conv_weight.stride(1),
        stride_conv_state_seq=conv_state.stride(0),
        stride_conv_state_dim=conv_state.stride(1),
        stride_conv_state_tok=conv_state.stride(2),
        stride_state_indices=stride_state_indices,
        pad_slot_id=pad_slot_id,
        B=N,
        H=H,
        HV=HV,
        K=K,
        V=V,
        BK=BK,
        HAS_CONV_BIAS=conv_bias is not None,
        CONV_WIDTH=conv_width,
        SILU_ACTIVATION=activation in ["silu", "swish"],
        IS_CONTINUOUS_BATCHING=conv_state_indices is not None,
        NP2_STATELEN=np2_statelen,
        USE_PAD_SLOT=pad_slot_id is not None,
        USE_QK_L2NORM_IN_KERNEL=use_qk_l2norm_in_kernel,
    )
    
    # Squeeze first dimension (NK) to get (B, T, HV, V)
    o = o.squeeze(0)
    return o

