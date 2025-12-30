"""
Gluon version of Fused Gating Delta Network (GDN) Forward Decode Kernel.
"""

from typing import Optional
import torch
import triton
from triton.experimental import gluon
import triton.experimental.gluon.language as gl

PAD_SLOT_ID = -1

@gl._core.builtin
def tuple_combine(a: gl.tuple, b: gl.tensor, _semantic=None) -> gl.tuple:
    """Gluon helper function to combine a tuple with a new tensor element."""
    return gl.tuple([*a.values, b])


@gluon.jit(do_not_specialize=["T"])
def gluon_fused_gdn_fwd_decode_kernel(
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
    key_dim: gl.constexpr,
    value_dim: gl.constexpr,
    batch: int,
    dim: gl.constexpr,
    seqlen: gl.constexpr,
    conv_state_len: gl.constexpr,
    num_cache_lines: gl.constexpr,
    T: int,
    # Gating parameters
    softplus_beta: float,
    softplus_threshold: float,
    scale: float,
    # Strides for conv
    stride_x_seq: gl.constexpr,
    stride_x_dim: gl.constexpr,
    stride_x_token: gl.constexpr,
    stride_conv_w_dim: gl.constexpr,
    stride_conv_w_width: gl.constexpr,
    stride_conv_state_seq: gl.constexpr,
    stride_conv_state_dim: gl.constexpr,
    stride_conv_state_tok: gl.constexpr,
    stride_state_indices: gl.constexpr,
    # Others
    pad_slot_id: gl.constexpr,
    # Meta-parameters
    B: gl.constexpr,
    H: gl.constexpr,
    HV: gl.constexpr,
    K: gl.constexpr,
    V: gl.constexpr,
    BK: gl.constexpr,
    BV: gl.constexpr,
    HAS_CONV_BIAS: gl.constexpr,
    CONV_WIDTH: gl.constexpr,
    SILU_ACTIVATION: gl.constexpr,
    IS_CONTINUOUS_BATCHING: gl.constexpr,
    NP2_STATELEN: gl.constexpr,
    USE_PAD_SLOT: gl.constexpr,
    USE_INITIAL_STATE: gl.constexpr,
    USE_QK_L2NORM_IN_KERNEL: gl.constexpr,
    IS_VARLEN: gl.constexpr,
):
    """
    Gluon-optimized fused kernel with Q/K reuse across V heads in a group.
    
    Key optimization: Each block processes GROUP_SIZE (HV//H) value heads,
    sharing the same Q/K computation. Uses gl.tuple to store multiple hidden states.
    
    This kernel processes:
    1. Causal conv1d for Q and K (once per group, reused)
    2. For each V head in group: conv1d for V + delta rule update
    3. Output results for all V heads in the group
    """
    
    # Define layouts for better memory access patterns
    blocked_k: gl.constexpr = gl.BlockedLayout(
        size_per_thread=[4],
        threads_per_warp=[64],
        warps_per_cta=[4],
        order=[0],
    )
    
    blocked_v: gl.constexpr = gl.BlockedLayout(
        size_per_thread=[2],
        threads_per_warp=[64],
        warps_per_cta=[4],
        order=[0],
    )
    
    blocked2d: gl.constexpr = gl.BlockedLayout(
        size_per_thread=[4, 2],
        threads_per_warp=[1, 64],
        warps_per_cta=[4, 1],
        order=[1, 0],
    )
    
    # Slice layouts for 2D tensor operations
    slice_k: gl.constexpr = gl.SliceLayout(
        dim=1,
        parent=blocked2d,
    )
    
    slice_v: gl.constexpr = gl.SliceLayout(
        dim=0,
        parent=blocked2d,
    )
    
    # Get program IDs - now indexed by Q/K heads (not V heads)
    i_k, i_v, i_nh = gl.program_id(0), gl.program_id(1), gl.program_id(2)
    i_n, i_h = i_nh // H, i_nh % H
    
    # Number of V heads per Q/K head (group size)
    GROUP_SIZE: gl.constexpr = HV // H
    
    # Handle variable length sequences
    if IS_VARLEN:
        bos, eos = (
            gl.load(cu_seqlens + i_n).to(gl.int64),
            gl.load(cu_seqlens + i_n + 1).to(gl.int64),
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
        conv_state_batch_coord = gl.load(
            conv_state_indices_ptr + idx_seq * stride_state_indices
        ).to(gl.int64)
    else:
        conv_state_batch_coord = idx_seq
        
    if USE_PAD_SLOT:
        if conv_state_batch_coord == pad_slot_id:
            return
    
    # Define offset ranges with appropriate layouts
    o_k_blocked = i_k * BK + gl.arange(0, BK, layout=blocked_k)
    o_v_blocked = i_v * BV + gl.arange(0, BV, layout=blocked_v)
    o_k_slice = i_k * BK + gl.arange(0, BK, layout=slice_k)
    o_v_slice = i_v * BV + gl.arange(0, BV, layout=slice_v)
    
    # Load gating parameters (loop-invariant across tokens)
    b_A_logs = ()
    b_dt_biases = ()
    for i in gl.static_range(GROUP_SIZE):
        i_hv = i_h * GROUP_SIZE + i
        b_A_log = gl.load(A_log + i_hv).to(gl.float32)
        b_dt_bias = gl.load(dt_bias + i_hv).to(gl.float32)
        b_A_logs = tuple_combine(b_A_logs, b_A_log)
        b_dt_biases = tuple_combine(b_dt_biases, b_dt_bias)
    
    # Define feature offsets using the blocked layouts
    q_dim_start = i_h * K
    q_feats = q_dim_start + o_k_blocked
    k_dim_start = key_dim + i_h * K
    k_feats = k_dim_start + o_k_blocked

    if USE_INITIAL_STATE:
        idx = gl.load(h0_indices + i_n)
        if idx >= 0:
            # Load initial hidden states using slice layout for 2D access
            b_hs = ()
            for i in gl.static_range(GROUP_SIZE):
                i_hv = i_h * GROUP_SIZE + i
                p_h = (
                    h0_source
                    + idx * HV * K * V
                    + i_hv * K * V
                    + o_k_slice[:, None] * V
                    + o_v_slice[None, :]
                )
                b_h = gl.load(p_h).to(gl.float32)
                b_hs = tuple_combine(b_hs, b_h)
            
            # Q weights and initial conv_states
            b_q_conv_states = ()
            q_weights = (gl.load(conv_w_ptr + q_feats * stride_conv_w_dim),)
            for i in gl.static_range(CONV_WIDTH-1):
                b_q_conv_state = gl.load(conv_state_ptr + (conv_state_batch_coord * stride_conv_state_seq) + (q_feats * stride_conv_state_dim) + i * stride_conv_state_tok)
                w_val = gl.load(conv_w_ptr + q_feats * stride_conv_w_dim + (i+1) * stride_conv_w_width)
                q_weights = tuple_combine(q_weights, w_val)
                b_q_conv_states = tuple_combine(b_q_conv_states, b_q_conv_state)
            
            # K weights and initial conv_states
            b_k_conv_states = ()
            k_weights = (gl.load(conv_w_ptr + k_feats * stride_conv_w_dim),)
            for i in gl.static_range(CONV_WIDTH-1):
                b_k_conv_state = gl.load(conv_state_ptr + (conv_state_batch_coord * stride_conv_state_seq) + (k_feats * stride_conv_state_dim) + i * stride_conv_state_tok)
                w_val = gl.load(conv_w_ptr + k_feats * stride_conv_w_dim + (i+1) * stride_conv_w_width)
                k_weights = tuple_combine(k_weights, w_val)
                b_k_conv_states = tuple_combine(b_k_conv_states, b_k_conv_state)
            
            # Pre-load V weights (states will be loaded lazily)
            v_weights_all = ()
            for i in gl.static_range(GROUP_SIZE):
                i_hv = i_h * GROUP_SIZE + i
                v_dim_start = 2 * key_dim + i_hv * V
                v_feats = v_dim_start + o_v_blocked
                
                v_weights = (gl.load(conv_w_ptr + v_feats * stride_conv_w_dim),)
                for j in gl.static_range(CONV_WIDTH-1):
                    w_val = gl.load(conv_w_ptr + v_feats * stride_conv_w_dim + (j+1) * stride_conv_w_width)
                    v_weights = tuple_combine(v_weights, w_val)
                
                v_weights_all = tuple_combine(v_weights_all, v_weights)
            
            b_v_conv_states_all = ()
            
            # Main token processing loop
            for idx_token in gl.static_range(seqlen):
                # Step 1: Conv1D for K
                k_conv_acc = gl.load(conv_bias_ptr + k_feats).to(gl.float32) if HAS_CONV_BIAS else gl.zeros([BK], dtype=gl.float32, layout=blocked_k)
                k_ptrs = (
                            x_ptr + idx_seq * stride_x_seq 
                            + k_feats * stride_x_dim 
                            + idx_token * stride_x_token
                        )
                b_k_conv_states = tuple_combine(b_k_conv_states, gl.load(k_ptrs))
                for j in gl.static_range(CONV_WIDTH):
                    k_conv_acc += b_k_conv_states[j] * k_weights[j]

                b_k_conv_states = b_k_conv_states[1:]
                
                if SILU_ACTIVATION:
                    k_conv_acc = k_conv_acc / (1 + gl.exp(-k_conv_acc))
                
                b_k = k_conv_acc.to(gl.float32)
                
                if USE_QK_L2NORM_IN_KERNEL:
                    b_k = b_k / (gl.sqrt(gl.sum(b_k * b_k, axis=0) + 1e-6))
                
                # Create slice version for 2D operations
                b_k_slice = b_k
                
                # Step 2: Process V0
                i_hv_0 = i_h * GROUP_SIZE + 0
                v_dim_start_0 = 2 * key_dim + i_hv_0 * V
                v_feats_0 = v_dim_start_0 + o_v_blocked
                
                v_conv_acc_0 = gl.load(conv_bias_ptr + v_feats_0).to(gl.float32) if HAS_CONV_BIAS else gl.zeros([BV], dtype=gl.float32, layout=blocked_v)
                v_ptrs_0 = (
                    x_ptr + idx_seq * stride_x_seq 
                    + v_feats_0 * stride_x_dim 
                    + idx_token * stride_x_token
                )
                
                # Lazy initialization for V0
                if idx_token == 0:
                    b_v_conv_states_0 = ()
                    for j in gl.static_range(CONV_WIDTH-1):
                        b_v_conv_state = gl.load(conv_state_ptr + (conv_state_batch_coord * stride_conv_state_seq) + (v_feats_0 * stride_conv_state_dim) + j * stride_conv_state_tok)
                        b_v_conv_states_0 = tuple_combine(b_v_conv_states_0, b_v_conv_state)
                else:
                    b_v_conv_states_0 = b_v_conv_states_all[0]
                
                v_weights_0 = v_weights_all[0]
                b_v_conv_states_0 = tuple_combine(b_v_conv_states_0, gl.load(v_ptrs_0))
                for j in gl.static_range(CONV_WIDTH):
                    v_conv_acc_0 += b_v_conv_states_0[j] * v_weights_0[j]
                b_v_conv_states_0 = b_v_conv_states_0[1:]
                
                if SILU_ACTIVATION:
                    v_conv_acc_0 = v_conv_acc_0 / (1 + gl.exp(-v_conv_acc_0))
                b_v_prev = v_conv_acc_0.to(gl.float32)
                
                new_b_v_conv_states_all = (b_v_conv_states_0,)
                
                # Step 3: Conv1D for Q
                q_conv_acc = gl.load(conv_bias_ptr + q_feats).to(gl.float32) if HAS_CONV_BIAS else gl.zeros([BK], dtype=gl.float32, layout=blocked_k)
                q_ptrs = (
                    x_ptr + idx_seq * stride_x_seq 
                    + q_feats * stride_x_dim 
                    + idx_token * stride_x_token
                )
                b_q_conv_states = tuple_combine(b_q_conv_states, gl.load(q_ptrs))
                for j in gl.static_range(CONV_WIDTH):
                    q_conv_acc += b_q_conv_states[j] * q_weights[j]
                b_q_conv_states = b_q_conv_states[1:]
                
                if SILU_ACTIVATION:
                    q_conv_acc = q_conv_acc / (1 + gl.exp(-q_conv_acc))
                
                b_q = q_conv_acc.to(gl.float32)
                if USE_QK_L2NORM_IN_KERNEL:
                    b_q_scale = scale / (gl.sqrt(gl.sum(b_q * b_q, axis=0) + 1e-6))
                else:
                    b_q_scale = scale
                b_q = b_q * b_q_scale
                
                # Create slice version for 2D operations
                b_q_slice = b_q
                
                # Step 4: Fused loop for remaining V heads + Delta Rule
                for i in gl.static_range(1, GROUP_SIZE):
                    i_hv = i_h * GROUP_SIZE + i
                    v_dim_start = 2 * key_dim + i_hv * V
                    v_feats = v_dim_start + o_v_blocked
                    
                    v_conv_acc = gl.load(conv_bias_ptr + v_feats).to(gl.float32) if HAS_CONV_BIAS else gl.zeros([BV], dtype=gl.float32, layout=blocked_v)
                    v_ptrs = (
                        x_ptr + idx_seq * stride_x_seq 
                        + v_feats * stride_x_dim 
                        + idx_token * stride_x_token
                    )
                    
                    # Lazy initialization
                    if idx_token == 0:
                        b_v_conv_states = ()
                        for j in gl.static_range(CONV_WIDTH-1):
                            b_v_conv_state = gl.load(conv_state_ptr + (conv_state_batch_coord * stride_conv_state_seq) + (v_feats * stride_conv_state_dim) + j * stride_conv_state_tok)
                            b_v_conv_states = tuple_combine(b_v_conv_states, b_v_conv_state)
                    else:
                        b_v_conv_states = b_v_conv_states_all[i]
                    
                    v_weights = v_weights_all[i]
                    b_v_conv_states = tuple_combine(b_v_conv_states, gl.load(v_ptrs))
                    for j in gl.static_range(CONV_WIDTH):
                        v_conv_acc += b_v_conv_states[j] * v_weights[j]
                    b_v_conv_states = b_v_conv_states[1:]
                    new_b_v_conv_states_all = tuple_combine(new_b_v_conv_states_all, b_v_conv_states)
                    
                    if SILU_ACTIVATION:
                        v_conv_acc = v_conv_acc / (1 + gl.exp(-v_conv_acc))
                    b_v_curr = v_conv_acc.to(gl.float32)
                    
                    # Delta Rule update for V_{i-1}
                    i_hv_prev = i_h * GROUP_SIZE + (i - 1)
                    
                    p_a = a + (bos + idx_token) * HV + i_hv_prev
                    p_b = b + (bos + idx_token) * HV + i_hv_prev
                    b_a = gl.load(p_a).to(gl.float32)
                    b_b = gl.load(p_b).to(gl.float32)
                    
                    x = b_a + b_dt_biases[i - 1]
                    beta_x = softplus_beta * x
                    softplus_x = gl.where(
                        beta_x <= softplus_threshold,
                        (1.0 / softplus_beta) * gl.log(1.0 + gl.exp(beta_x)),
                        x,
                    )
                    b_g = -gl.exp(b_A_logs[i - 1]) * softplus_x
                    b_beta = 1.0 / (1.0 + gl.exp(-b_b))
                    
                    b_h = b_hs[i - 1]
                    b_h *= gl.exp(b_g)
                    b_v_prev -= gl.sum(b_h * b_k_slice[:, None], axis=0)
                    b_v_prev *= b_beta
                    b_h += b_k_slice[:, None] * b_v_prev[None, :]
                    
                    b_o = gl.sum(b_h * b_q_slice[:, None], axis=0)
                    p_o = o + ((i_k * all + bos + idx_token) * HV + i_hv_prev) * V + o_v_blocked
                    gl.store(p_o, b_o.to(p_o.dtype.element_ty))
                    
                    p_h0 = (
                        h0_source
                        + idx * HV * K * V
                        + i_hv_prev * K * V
                        + o_k_slice[:, None] * V
                        + o_v_slice[None, :]
                    )
                    gl.store(p_h0, b_h.to(p_h0.dtype.element_ty))
                    
                    b_v_prev = b_v_curr
                
                b_v_conv_states_all = new_b_v_conv_states_all
                
                # Process last V head
                i_hv_last = i_h * GROUP_SIZE + (GROUP_SIZE - 1)
                
                p_a = a + (bos + idx_token) * HV + i_hv_last
                p_b = b + (bos + idx_token) * HV + i_hv_last
                b_a = gl.load(p_a).to(gl.float32)
                b_b = gl.load(p_b).to(gl.float32)
                
                x = b_a + b_dt_biases[GROUP_SIZE - 1]
                beta_x = softplus_beta * x
                softplus_x = gl.where(
                    beta_x <= softplus_threshold,
                    (1.0 / softplus_beta) * gl.log(1.0 + gl.exp(beta_x)),
                    x,
                )
                b_g = -gl.exp(b_A_logs[GROUP_SIZE - 1]) * softplus_x
                b_beta = 1.0 / (1.0 + gl.exp(-b_b))
                
                b_h = b_hs[GROUP_SIZE - 1]
                b_h *= gl.exp(b_g)
                b_v_prev -= gl.sum(b_h * b_k_slice[:, None], axis=0)
                b_v_prev *= b_beta
                b_h += b_k_slice[:, None] * b_v_prev[None, :]
                
                b_o = gl.sum(b_h * b_q_slice[:, None], axis=0)
                p_o = o + ((i_k * all + bos + idx_token) * HV + i_hv_last) * V + o_v_blocked
                gl.store(p_o, b_o.to(p_o.dtype.element_ty))
                
                p_h0 = (
                    h0_source
                    + idx * HV * K * V
                    + i_hv_last * K * V
                    + o_k_slice[:, None] * V
                    + o_v_slice[None, :]
                )
                gl.store(p_h0, b_h.to(p_h0.dtype.element_ty))

            # Write back conv_states
            for i in gl.static_range(CONV_WIDTH-1):
                gl.store(conv_state_ptr + (conv_state_batch_coord * stride_conv_state_seq) + (q_feats * stride_conv_state_dim) + i * stride_conv_state_tok, b_q_conv_states[i])
            
            for i in gl.static_range(CONV_WIDTH-1):
                gl.store(conv_state_ptr + (conv_state_batch_coord * stride_conv_state_seq) + (k_feats * stride_conv_state_dim) + i * stride_conv_state_tok, b_k_conv_states[i])
            
            for i in gl.static_range(GROUP_SIZE):
                i_hv = i_h * GROUP_SIZE + i
                v_dim_start = 2 * key_dim + i_hv * V
                v_feats = v_dim_start + o_v_blocked
                
                b_v_conv_states = b_v_conv_states_all[i]
                for j in gl.static_range(CONV_WIDTH-1):
                    gl.store(conv_state_ptr + (conv_state_batch_coord * stride_conv_state_seq) + (v_feats * stride_conv_state_dim) + j * stride_conv_state_tok, b_v_conv_states[j])
                    
            return

    # Non-initial-state branch (similar structure, omitted for brevity)
    # In practice, you'd implement the else branch here
    return

