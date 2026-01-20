"""
Gluon version of Fused Gating Delta Network (GDN) Forward Decode Kernel.
"""

from typing import Optional
import torch
import triton
from triton.experimental import gluon
import triton.experimental.gluon.language as gl
import triton.language as tl

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
    Gluon-optimized fused kernel with Q/K reuse and batched V head processing.
    
    Key optimizations:
    1. Each block processes GROUP_SIZE (HV//H) value heads simultaneously
    2. Q/K are computed once per group and broadcast across all V heads
    3. V heads are represented as tensor dimensions for efficient batching
    4. Delta Rule updates use broadcasting instead of loops
    """
    
    # ============================================================================
    # Layout Definitions
    # ============================================================================
    blocked_k: gl.constexpr = gl.BlockedLayout(
        size_per_thread=[2],
        threads_per_warp=[64],
        warps_per_cta=[1],
        order=[0],
    )
    
    blocked_v: gl.constexpr = gl.BlockedLayout(
        size_per_thread=[1],
        threads_per_warp=[64],
        warps_per_cta=[1],
        order=[0],
    )
    
    blocked2d: gl.constexpr = gl.BlockedLayout(
        size_per_thread=[1, 1],
        threads_per_warp=[4, 16],
        warps_per_cta=[1, 1],
        order=[1, 0],
    )

    blocked3d: gl.constexpr = gl.BlockedLayout(
        size_per_thread=[1, 1, 4],
        threads_per_warp=[1, 16, 4],
        warps_per_cta=[1, 1, 1],
        order=[2, 1, 0],
    )

    blocked3d1: gl.constexpr = gl.BlockedLayout(
        size_per_thread=[1, 1, 1],
        threads_per_warp=[1, 4, 16],
        warps_per_cta=[1, 1, 1],
        order=[2, 1, 0],
    )
    
    # Slice layouts for 2D tensor operations
    slice_bk: gl.constexpr = gl.SliceLayout(dim=0,
        parent=gl.SliceLayout(
            dim=2,
            parent=blocked3d,
        ),
    )
    
    slice_v: gl.constexpr = gl.SliceLayout(dim=1, parent=blocked3d)
    slice_bv: gl.constexpr = gl.SliceLayout(dim=0, parent=slice_v)
    slice_group: gl.constexpr = gl.SliceLayout(dim=1,
        parent=gl.SliceLayout(
            dim=2,
            parent=blocked3d,
        ),
    )
    slice_group_11: gl.constexpr = gl.SliceLayout(dim=1,
        parent=gl.SliceLayout(
            dim=1,
            parent=blocked3d,
        ),
    )
    
    # ============================================================================
    # Program ID and Dimension Setup
    # ============================================================================
    # Get program IDs - indexed by Q/K heads (not V heads)
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
    
    # ============================================================================
    # Offset Initialization
    # ============================================================================
    # Define offset ranges for tensor blocks
    o_k = i_k * BK + gl.arange(0, BK, layout=blocked_k)
    o_v = i_v * BV + gl.arange(0, BV, layout=blocked_v)

    o_k_slice = i_k * BK + gl.arange(0, BK, layout=slice_bk)
    o_v_slice = i_v * BV + gl.arange(0, BV, layout=slice_bv)

    # Define V head indices for this Q/K head group
    # i_hv: [GROUP_SIZE] - Absolute indices of V heads in this group
    i_hv = i_h * GROUP_SIZE + gl.arange(0, GROUP_SIZE, layout=slice_group)
    i_hv_11 = i_h * GROUP_SIZE + gl.arange(0, GROUP_SIZE, layout=slice_group_11)

    # Load time-invariant gating parameters for all V heads in this group
    # b_A_log: [GROUP_SIZE] - Log of recurrent matrix eigenvalues
    # b_dt_bias: [GROUP_SIZE] - Time step bias parameters
    b_A_log = gl.load(A_log + i_hv_11).to(gl.float32)
    b_dt_bias = gl.load(dt_bias + i_hv_11).to(gl.float32)

    if USE_INITIAL_STATE:
        idx = gl.load(h0_indices + i_n)
        if idx >= 0:
            # ================================================================
            # Load initial hidden states for all V heads
            # Shape: [GROUP_SIZE, BK, BV]
            # ================================================================
            p_h = (
                h0_source
                + idx * HV * K * V
                + i_hv[:, None, None] * K * V
                + o_k_slice[None, :, None] * V
                + o_v_slice[None, None, :]
            )
            b_h = gl.load(p_h).to(gl.float32)  # [GROUP_SIZE, BK, BV]

            # ================================================================
            # Pre-load conv_state sliding windows and weights for K, V, Q
            # ================================================================
            
            # K conv setup (shared across all V heads)
            k_dim_start = key_dim + i_h * K
            k_feats = k_dim_start + o_k
            
            b_k_conv_states = ()
            k_weights = (gl.load(conv_w_ptr + k_feats * stride_conv_w_dim),)
            for i in gl.static_range(CONV_WIDTH-1):
                b_k_conv_state = gl.load(conv_state_ptr + (conv_state_batch_coord * stride_conv_state_seq) + (k_feats * stride_conv_state_dim) + i * stride_conv_state_tok)
                w_val = gl.load(conv_w_ptr + k_feats * stride_conv_w_dim + (i+1) * stride_conv_w_width)
                k_weights = tuple_combine(k_weights, w_val)
                b_k_conv_states = tuple_combine(b_k_conv_states, b_k_conv_state)
            
            # V conv setup (batched for all GROUP_SIZE V heads)
            v_dim_start = 2 * key_dim + i_hv_11 * V
            v_feats = v_dim_start[:, None] + o_v_slice[None, :]  # [GROUP_SIZE, BV]
            
            b_v_conv_states = ()
            v_weights = (gl.load(conv_w_ptr + v_feats * stride_conv_w_dim),)
            for j in gl.static_range(CONV_WIDTH-1):
                b_v_conv_state = gl.load(conv_state_ptr + (conv_state_batch_coord * stride_conv_state_seq) + (v_feats * stride_conv_state_dim) + j * stride_conv_state_tok)
                w_val = gl.load(conv_w_ptr + v_feats * stride_conv_w_dim + (j+1) * stride_conv_w_width)
                v_weights = tuple_combine(v_weights, w_val)
                b_v_conv_states = tuple_combine(b_v_conv_states, b_v_conv_state)

            # Q conv setup (shared across all V heads)
            q_dim_start = i_h * K
            q_feats = q_dim_start + o_k
            
            b_q_conv_states = ()
            q_weights = (gl.load(conv_w_ptr + q_feats * stride_conv_w_dim),)
            for i in gl.static_range(CONV_WIDTH-1):
                b_q_conv_state = gl.load(conv_state_ptr + (conv_state_batch_coord * stride_conv_state_seq) + (q_feats * stride_conv_state_dim) + i * stride_conv_state_tok)
                w_val = gl.load(conv_w_ptr + q_feats * stride_conv_w_dim + (i+1) * stride_conv_w_width)
                q_weights = tuple_combine(q_weights, w_val)
                b_q_conv_states = tuple_combine(b_q_conv_states, b_q_conv_state)
            
            # ================================================================
            # Main token processing loop
            # Processing order: K → V (all heads) → Q → Delta Rule (all heads)
            # ================================================================
            for idx_token in gl.static_range(seqlen):
                # ============================================================
                # Step 1: Conv1D for K
                # Shape: [BK]
                # ============================================================
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
                
                b_k = k_conv_acc.to(gl.float32)  # [BK]
                
                if USE_QK_L2NORM_IN_KERNEL:
                    b_k = b_k / (gl.sqrt(gl.sum(b_k * b_k, axis=0) + 1e-6))
                
                # ============================================================
                # Step 2: Conv1D for all V heads
                # Shape: [GROUP_SIZE, BV]
                # ============================================================
                v_conv_acc = gl.load(conv_bias_ptr + v_feats).to(gl.float32) if HAS_CONV_BIAS else gl.zeros([GROUP_SIZE, BV], dtype=gl.float32, layout=blocked2d)
                
                v_ptrs = (
                    x_ptr + idx_seq * stride_x_seq 
                    + v_feats * stride_x_dim 
                    + idx_token * stride_x_token
                )
                b_v_conv_states = tuple_combine(b_v_conv_states, gl.load(v_ptrs))
                
                for j in gl.static_range(CONV_WIDTH):
                    v_conv_acc += b_v_conv_states[j] * v_weights[j]
                
                b_v_conv_states = b_v_conv_states[1:]
                
                if SILU_ACTIVATION:
                    v_conv_acc = v_conv_acc / (1 + gl.exp(-v_conv_acc))
                
                b_v = v_conv_acc.to(gl.float32)  # [GROUP_SIZE, BV]
                
                # ============================================================
                # Step 3: Conv1D for Q
                # Shape: [BK]
                # ============================================================
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
                
                b_q = q_conv_acc.to(gl.float32)  # [BK]
                
                if USE_QK_L2NORM_IN_KERNEL:
                    b_q_scale = scale / (gl.sqrt(gl.sum(b_q * b_q, axis=0) + 1e-6))
                else:
                    b_q_scale = scale
                b_q = b_q * b_q_scale  # [BK]
                
                # ============================================================
                # Step 4: Batched Delta Rule updates for all V heads
                # Using broadcasting for efficient parallel processing
                # ============================================================
                
                # Load time-variant gating parameters
                p_a = a + (bos + idx_token) * HV + i_hv_11
                p_b = b + (bos + idx_token) * HV + i_hv_11
                b_a = gl.load(p_a).to(gl.float32)  # [GROUP_SIZE]
                b_b = gl.load(p_b).to(gl.float32)  # [GROUP_SIZE]
                
                # Compute gating factors
                x = b_a + b_dt_bias  # [GROUP_SIZE]
                beta_x = softplus_beta * x
                softplus_x = gl.where(
                    beta_x <= softplus_threshold,
                    (1.0 / softplus_beta) * gl.log(1.0 + gl.exp(beta_x)),
                    x,
                )
                b_g = -gl.exp(b_A_log) * softplus_x  # [GROUP_SIZE]
                b_beta = 1.0 / (1.0 + gl.exp(-b_b))  # [GROUP_SIZE]

                b_k = gl.convert_layout(b_k, layout=slice_bk)
                b_v = gl.convert_layout(b_v, layout=slice_v)
                b_q = gl.convert_layout(b_q, layout=slice_bk)
                
                # Batched Delta Rule recurrent update using broadcasting
                # Step 4a: Apply exponential decay to hidden states
                b_g = gl.convert_layout(b_g, layout=slice_group)
                b_h *= gl.exp(b_g[:, None, None])  # [GROUP_SIZE, BK, BV]
                
                # Step 4b: Delta rule correction
                b_v -= gl.sum(b_h * b_k[None, :, None], axis=1)  # [GROUP_SIZE, BV]
                
                # Step 4c: Apply beta gating
                b_v *= b_beta[:, None]  # [GROUP_SIZE, BV]
                
                # Step 4d: Update hidden states with outer product
                b_h += b_k[None, :, None] * b_v[:, None, :]  # [GROUP_SIZE, BK, BV]
                
                # Step 4e: Compute outputs for all V heads
                b_o = gl.sum(b_h * b_q[None, :, None], axis=1)  # [GROUP_SIZE, BV]
                
                # Step 4f: Store outputs for all V heads
                p_o = o + ((i_k * all + bos + idx_token) * HV + i_hv_11[:, None]) * V + o_v_slice[None, :]
                gl.store(p_o, b_o.to(p_o.dtype.element_ty))
                
                # Step 4g: Store updated hidden states for all V heads
                p_h0 = (
                    h0_source
                    + idx * HV * K * V
                    + i_hv[:, None, None] * K * V
                    + o_k_slice[None, :, None] * V
                    + o_v_slice[None, None, :]
                )
                gl.store(p_h0, b_h.to(p_h0.dtype.element_ty))

            # ================================================================
            # Write back final conv_state sliding windows to memory in
            # ================================================================
            q_feats_slice = i_h * K + o_k
            k_feats_slice = key_dim + i_h * K + o_k
            v_feats_slice = 2 * key_dim + i_hv_11[:, None] * V + o_v_slice[None, :]
            # Write back Q conv_states
            if i_v == V//BV-1:
                for i in gl.static_range(CONV_WIDTH-1):
                    gl.store(conv_state_ptr + (conv_state_batch_coord * stride_conv_state_seq) + (q_feats_slice * stride_conv_state_dim) + i * stride_conv_state_tok, b_q_conv_states[i])
                
                # Write back K conv_states
                for i in gl.static_range(CONV_WIDTH-1):
                    gl.store(conv_state_ptr + (conv_state_batch_coord * stride_conv_state_seq) + (k_feats_slice * stride_conv_state_dim) + i * stride_conv_state_tok, b_k_conv_states[i])
            
            # Write back V conv_states for all V heads
            for i in gl.static_range(CONV_WIDTH-1): 
                gl.store(conv_state_ptr + (conv_state_batch_coord * stride_conv_state_seq) + (v_feats_slice * stride_conv_state_dim) + i * stride_conv_state_tok, b_v_conv_states[i])
            
            return

    # ========================================================================
    # Non-initial-state branch: Zero initialization
    # ========================================================================
    
    # Initialize zero hidden states for all V heads in the group
    # Shape: [GROUP_SIZE, BK, BV]
    b_h = gl.zeros([GROUP_SIZE, BK, BV], dtype=gl.float32, layout=blocked3d)

    # K conv setup (shared across all V heads)
    k_dim_start = key_dim + i_h * K
    k_feats = k_dim_start + o_k
    
    b_k_conv_states = ()
    k_weights = (gl.load(conv_w_ptr + k_feats * stride_conv_w_dim),)
    for i in gl.static_range(CONV_WIDTH-1):
        b_k_conv_state = gl.load(conv_state_ptr + (conv_state_batch_coord * stride_conv_state_seq) + (k_feats * stride_conv_state_dim) + i * stride_conv_state_tok)
        w_val = gl.load(conv_w_ptr + k_feats * stride_conv_w_dim + (i+1) * stride_conv_w_width)
        k_weights = tuple_combine(k_weights, w_val)
        b_k_conv_states = tuple_combine(b_k_conv_states, b_k_conv_state)
    
    # V conv setup (batched for all GROUP_SIZE V heads)
    v_dim_start = 2 * key_dim + i_hv_11[:, None] * V
    v_feats = v_dim_start + o_v_slice[None, :]
    
    b_v_conv_states = ()
    v_weights = (gl.load(conv_w_ptr + v_feats * stride_conv_w_dim),)
    for j in gl.static_range(CONV_WIDTH-1):
        b_v_conv_state = gl.load(conv_state_ptr + (conv_state_batch_coord * stride_conv_state_seq) + (v_feats * stride_conv_state_dim) + j * stride_conv_state_tok)
        w_val = gl.load(conv_w_ptr + v_feats * stride_conv_w_dim + (j+1) * stride_conv_w_width)
        v_weights = tuple_combine(v_weights, w_val)
        b_v_conv_states = tuple_combine(b_v_conv_states, b_v_conv_state)

    # Q conv setup (shared across all V heads)
    q_dim_start = i_h * K
    q_feats = q_dim_start + o_k
    
    b_q_conv_states = ()
    q_weights = (gl.load(conv_w_ptr + q_feats * stride_conv_w_dim),)
    for i in gl.static_range(CONV_WIDTH-1):
        b_q_conv_state = gl.load(conv_state_ptr + (conv_state_batch_coord * stride_conv_state_seq) + (q_feats * stride_conv_state_dim) + i * stride_conv_state_tok)
        w_val = gl.load(conv_w_ptr + q_feats * stride_conv_w_dim + (i+1) * stride_conv_w_width)
        q_weights = tuple_combine(q_weights, w_val)
        b_q_conv_states = tuple_combine(b_q_conv_states, b_q_conv_state)
    
    # ========================================================================
    # Main token processing loop (identical to initial-state branch)
    # ========================================================================
    for idx_token in gl.static_range(seqlen):
        # ====================================================================
        # Step 1: Conv1D for K (shared across all V heads)
        # ====================================================================
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
        
        # ====================================================================
        # Step 2: Conv1D for all V heads (batched processing)
        # ====================================================================
        v_conv_acc = gl.load(conv_bias_ptr + v_feats).to(gl.float32) if HAS_CONV_BIAS else gl.zeros([GROUP_SIZE, BV], dtype=gl.float32, layout=blocked2d)
        v_ptrs = (
            x_ptr + idx_seq * stride_x_seq 
            + v_feats * stride_x_dim 
            + idx_token * stride_x_token
        )
        
        b_v_conv_states = tuple_combine(b_v_conv_states, gl.load(v_ptrs))
        for j in gl.static_range(CONV_WIDTH):
            v_conv_acc += b_v_conv_states[j] * v_weights[j]
        b_v_conv_states = b_v_conv_states[1:]
        
        if SILU_ACTIVATION:
            v_conv_acc = v_conv_acc / (1 + gl.exp(-v_conv_acc))
        b_v = v_conv_acc.to(gl.float32)
        
        # ====================================================================
        # Step 3: Conv1D for Q (shared across all V heads)
        # ====================================================================
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
        
        # ====================================================================
        # Step 4: Batched Delta Rule updates for all V heads
        # ====================================================================
        p_a = a + (bos + idx_token) * HV + i_hv_11
        p_b = b + (bos + idx_token) * HV + i_hv_11
        b_a = gl.load(p_a).to(gl.float32)
        b_b = gl.load(p_b).to(gl.float32)
        x = b_a + b_dt_bias
        beta_x = softplus_beta * x
        softplus_x = gl.where(
            beta_x <= softplus_threshold,
            (1.0 / softplus_beta) * gl.log(1.0 + gl.exp(beta_x)),
            x,
        )
        b_g = -gl.exp(b_A_log) * softplus_x
        b_beta = 1.0 / (1.0 + gl.exp(-b_b))

        b_k = gl.convert_layout(b_k, layout=slice_bk)
        b_v = gl.convert_layout(b_v, layout=slice_v)
        b_q = gl.convert_layout(b_q, layout=slice_bk)
        
        # Batched Delta Rule update with broadcasting
        b_g = gl.convert_layout(b_g, layout=slice_group)
        b_h *= gl.exp(b_g[:, None, None])
        b_v -= gl.sum(b_h * b_k[None, :, None], axis=1)
        b_v *= b_beta[:, None]
        b_h += b_k[None, :, None] * b_v[:, None, :]
        
        b_o = gl.sum(b_h * b_q[None, :, None], axis=1)
        p_o = o + ((i_k * all + bos + idx_token) * HV + i_hv_11[:, None]) * V + o_v_slice[None, :]
        gl.store(p_o, b_o.to(p_o.dtype.element_ty))
        
    # ========================================================================
    # Write back final conv_state sliding windows to memory out
    # ========================================================================
    q_feats_slice = i_h * K + o_k
    k_feats_slice = key_dim + i_h * K + o_k
    v_feats_slice = 2 * key_dim + i_hv_11[:, None] * V + o_v_slice[None, :]
    if i_v == V//BV-1:
        for i in gl.static_range(CONV_WIDTH-1):
            gl.store(conv_state_ptr + (conv_state_batch_coord * stride_conv_state_seq) + (q_feats_slice * stride_conv_state_dim) + i * stride_conv_state_tok, b_q_conv_states[i])
        
        for i in gl.static_range(CONV_WIDTH-1):
            gl.store(conv_state_ptr + (conv_state_batch_coord * stride_conv_state_seq) + (k_feats_slice * stride_conv_state_dim) + i * stride_conv_state_tok, b_k_conv_states[i])
    
    for j in gl.static_range(CONV_WIDTH-1):
        gl.store(conv_state_ptr + (conv_state_batch_coord * stride_conv_state_seq) + (v_feats_slice * stride_conv_state_dim) + j * stride_conv_state_tok, b_v_conv_states[j])


@gluon.jit(do_not_specialize=["T"])
def gluon_fused_gdn_fwd_decode_kernel_v2(
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
    Simplified Gluon fused kernel where each block processes one V head.
    
    Grid layout: (num_k_blocks, num_v_blocks, batch * num_heads_v)
    
    Key differences from v1:
    - Grid indexed by V heads instead of Q/K heads
    - No GROUP_SIZE loop - each block handles one V head
    - Simpler logic, potentially better parallelism
    - Hidden state shape: [BK, BV] instead of [GROUP_SIZE, BK, BV]
    """
    # ============================================================================
    # Program ID and dimension setup
    # Grid is indexed by V heads, not Q/K heads
    # ============================================================================
    i_k, i_v, i_nhv = gl.program_id(0), gl.program_id(1), gl.program_id(2)
    i_n, i_hv = i_nhv // HV, i_nhv % HV
    
    # Compute corresponding Q/K head for this V head
    GROUP_SIZE: gl.constexpr = HV // H
    i_h = i_hv // GROUP_SIZE

    q_dim_start = i_h * K
    k_dim_start = key_dim + i_h * K
    v_dim_start = 2 * key_dim + i_hv * V
    
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
    
    # ============================================================================
    # Define layouts and offset ranges
    # ============================================================================
    # BlockedLayout for K and V dimensions
    blocked2d: gl.constexpr = gl.BlockedLayout(
        size_per_thread=[1, 4],
        threads_per_warp=[8, 8],
        warps_per_cta=[1, 1],
        order=[1, 0],
    )
    blocked2d1: gl.constexpr = gl.BlockedLayout(
        size_per_thread=[1, 1],
        threads_per_warp=[2, 32],
        warps_per_cta=[1, 1],
        order=[1, 0],
    )
    blocked2d2: gl.constexpr = gl.BlockedLayout(
        size_per_thread=[1, 1],
        threads_per_warp=[1, 64],
        warps_per_cta=[1, 1],
        order=[0, 1],
    )
    blocked1: gl.constexpr = gl.BlockedLayout(
        size_per_thread=[1],
        threads_per_warp=[64],
        warps_per_cta=[1],
        order=[0],
    )
    blocked2: gl.constexpr = gl.BlockedLayout(
        size_per_thread=[2],
        threads_per_warp=[64],
        warps_per_cta=[1],
        order=[0],
    )
    slice1: gl.constexpr = gl.SliceLayout(
        dim=1,
        parent=blocked2d,
    )
    slice4: gl.constexpr = gl.SliceLayout(
        dim=0,
        parent=blocked2d,
    )
    
    # Define offset ranges
    # o_k: [BK] - Offsets for K dimension
    # o_v: [BV] - Offsets for V dimension
    o_k = i_k * BK + gl.arange(0, BK, layout=blocked2)
    o_v = i_v * BV + gl.arange(0, BV, layout=blocked1)

    o_k_slice = i_k * BK + gl.arange(0, BK, layout=slice1)
    o_v_slice = i_v * BV + gl.arange(0, BV, layout=slice4)

    # Load gating parameters for this single V head (scalar values)
    b_A_log = gl.load(A_log + i_hv).to(gl.float32)
    b_dt_bias = gl.load(dt_bias + i_hv).to(gl.float32)
    
    # ============================================================================
    # Define feature offsets for Q, K, V
    # ============================================================================

    q_feats = q_dim_start + o_k  # [BK]
    k_feats = k_dim_start + o_k  # [BK]
    v_feats = v_dim_start + o_v  # [BV]

    k_conv_w_ptr = conv_w_ptr + k_feats * stride_conv_w_dim
    v_conv_w_ptr = conv_w_ptr + v_feats * stride_conv_w_dim
    q_conv_w_ptr = conv_w_ptr + q_feats * stride_conv_w_dim

    k_conv_state_ptr = conv_state_ptr + conv_state_batch_coord * stride_conv_state_seq + k_feats * stride_conv_state_dim
    v_conv_state_ptr = conv_state_ptr + conv_state_batch_coord * stride_conv_state_seq + v_feats * stride_conv_state_dim
    q_conv_state_ptr = conv_state_ptr + conv_state_batch_coord * stride_conv_state_seq + q_feats * stride_conv_state_dim

    k_ptrs = x_ptr + idx_seq * stride_x_seq + k_feats * stride_x_dim
    v_ptrs = x_ptr + idx_seq * stride_x_seq + v_feats * stride_x_dim
    q_ptrs = x_ptr + idx_seq * stride_x_seq + q_feats * stride_x_dim

    # ============================================================================
    # Branch: With initial state
    # ============================================================================
    if USE_INITIAL_STATE:
        idx = gl.load(h0_indices + i_n)
        b_h = gl.zeros([BK, BV], dtype=gl.float32, layout=blocked2d)
        if idx >= 0:
            # ====================================================================
            # Load initial hidden state for this V head
            # Shape: [BK, BV] (2D, not 3D like v1)
            # ====================================================================
            offsets = idx * HV * K * V + i_hv * K * V + o_k_slice[:, None] * V + o_v_slice[None, :]
            p_h = h0_source + offsets
            # b_h = gl.amd.cdna3.buffer_load(h0_source, offsets).to(gl.float32)  # [BK, BV]
            b_h = gl.load(p_h).to(gl.float32)
            
    # ====================================================================
    # Pre-load conv_state sliding windows and weights for Q, K, V
    # ====================================================================            
    k_conv_acc = gl.load(conv_bias_ptr + k_feats).to(gl.float32) if HAS_CONV_BIAS else gl.zeros([BK], dtype=gl.float32, layout=blocked2)
    # K conv setup
    b_k_conv_states = ()
    k_weights = (gl.load(k_conv_w_ptr),)
    for i in gl.static_range(CONV_WIDTH-1):
        b_k_conv_state = gl.load(k_conv_state_ptr + i * stride_conv_state_tok)
        w_val = gl.load(k_conv_w_ptr + (i+1) * stride_conv_w_width)
        k_weights = tuple_combine(k_weights, w_val)
        b_k_conv_states = tuple_combine(b_k_conv_states, b_k_conv_state)

    b_k_conv_states = tuple_combine(b_k_conv_states, gl.load(k_ptrs))
    for j in gl.static_range(CONV_WIDTH):
        k_conv_acc += b_k_conv_states[j] * k_weights[j]
    gl.amd.cdna3.sched_barrier(0)
    v_conv_acc = gl.load(conv_bias_ptr + v_feats).to(gl.float32) if HAS_CONV_BIAS else gl.zeros([BV], dtype=gl.float32, layout=blocked1)
    # V conv setup
    b_v_conv_states = ()
    v_weights = (gl.load(v_conv_w_ptr),)
    for i in gl.static_range(CONV_WIDTH-1):
        b_v_conv_state = gl.load(v_conv_state_ptr + i * stride_conv_state_tok)
        w_val = gl.load(v_conv_w_ptr + (i+1) * stride_conv_w_width)
        v_weights = tuple_combine(v_weights, w_val)
        b_v_conv_states = tuple_combine(b_v_conv_states, b_v_conv_state)

    b_v_conv_states = tuple_combine(b_v_conv_states, gl.load(v_ptrs))
    for j in gl.static_range(CONV_WIDTH):
        v_conv_acc += b_v_conv_states[j] * v_weights[j]
    gl.amd.cdna3.sched_barrier(0)
    q_conv_acc = gl.load(conv_bias_ptr + q_feats).to(gl.float32) if HAS_CONV_BIAS else gl.zeros([BK], dtype=gl.float32, layout=blocked2)
    # Q conv setup
    b_q_conv_states = ()
    q_weights = (gl.load(q_conv_w_ptr),)
    for i in gl.static_range(CONV_WIDTH-1):
        b_q_conv_state = gl.load(q_conv_state_ptr + i * stride_conv_state_tok)
        w_val = gl.load(q_conv_w_ptr + (i+1) * stride_conv_w_width)
        q_weights = tuple_combine(q_weights, w_val)
        b_q_conv_states = tuple_combine(b_q_conv_states, b_q_conv_state)

    b_q_conv_states = tuple_combine(b_q_conv_states, gl.load(q_ptrs))
    for j in gl.static_range(CONV_WIDTH):
        q_conv_acc += b_q_conv_states[j] * q_weights[j]
    gl.amd.cdna3.sched_barrier(0)
    # Load time-variant gating parameters (scalars)
    p_a = a + (bos + 0) * HV + i_hv
    p_b = b + (bos + 0) * HV + i_hv
    b_a = gl.load(p_a).to(gl.float32)
    b_b = gl.load(p_b).to(gl.float32)

    p_o = o + ((i_k * all + bos + 0) * HV + i_hv) * V + o_v_slice

    b_k_conv_states = b_k_conv_states[1:]
    b_v_conv_states = b_v_conv_states[1:]
    b_q_conv_states = b_q_conv_states[1:]

    # ====================================================================
    # Main token processing loop
    # For each token: compute K → V → Q → Delta Rule Update
    # ====================================================================
    for idx_token in gl.static_range(1, seqlen):
        # ================================================================
        # Step 1: Conv1D for K
        # Shape: [BK]
        # ================================================================
        k_conv_acc0 = gl.load(conv_bias_ptr + k_feats).to(gl.float32) if HAS_CONV_BIAS else gl.zeros([BK], dtype=gl.float32, layout=blocked2)
        k_ptrs0 = x_ptr + idx_seq * stride_x_seq + k_feats * stride_x_dim + idx_token * stride_x_token
        b_k_conv_states = tuple_combine(b_k_conv_states, gl.load(k_ptrs0))
        for j in gl.static_range(CONV_WIDTH):
            k_conv_acc0 += b_k_conv_states[j] * k_weights[j]

        # ================================================================
        # Step 2: Conv1D for V
        # Shape: [BV]
        # ================================================================
        v_conv_acc0 = gl.load(conv_bias_ptr + v_feats).to(gl.float32) if HAS_CONV_BIAS else gl.zeros([BV], dtype=gl.float32, layout=blocked1)
        v_ptrs0 = x_ptr + idx_seq * stride_x_seq + v_feats * stride_x_dim + idx_token * stride_x_token
        b_v_conv_states = tuple_combine(b_v_conv_states, gl.load(v_ptrs0))
        for j in gl.static_range(CONV_WIDTH):
            v_conv_acc0 += b_v_conv_states[j] * v_weights[j]

        # ================================================================
        # Step 3: Conv1D for Q
        # Shape: [BK]
        # ================================================================
        q_conv_acc0 = gl.load(conv_bias_ptr + q_feats).to(gl.float32) if HAS_CONV_BIAS else gl.zeros([BK], dtype=gl.float32, layout=blocked2)
        q_ptrs0 = x_ptr + idx_seq * stride_x_seq + q_feats * stride_x_dim + idx_token * stride_x_token
        b_q_conv_states = tuple_combine(b_q_conv_states, gl.load(q_ptrs0))
        for j in gl.static_range(CONV_WIDTH):
            q_conv_acc0 += b_q_conv_states[j] * q_weights[j]

        if SILU_ACTIVATION:
            k_conv_acc = k_conv_acc / (1 + gl.exp(-k_conv_acc))
        b_k = k_conv_acc.to(gl.float32)  # [BK]
        
        if USE_QK_L2NORM_IN_KERNEL:
            b_k = b_k / (gl.sqrt(gl.sum(b_k * b_k, axis=0) + 1e-6))

        if SILU_ACTIVATION:
            v_conv_acc = v_conv_acc / (1 + gl.exp(-v_conv_acc))
        b_v = v_conv_acc.to(gl.float32)  # [BV]

        if SILU_ACTIVATION:
            q_conv_acc = q_conv_acc / (1 + gl.exp(-q_conv_acc))
        b_q = q_conv_acc.to(gl.float32)  # [BK]
        
        if USE_QK_L2NORM_IN_KERNEL:
            b_q_scale = scale / (gl.sqrt(gl.sum(b_q * b_q, axis=0) + 1e-6))
        else:
            b_q_scale = scale
        b_q = b_q * b_q_scale  # [BK]
        
        # ================================================================
        # Step 4: Delta Rule update for single V head
        # No broadcasting needed - all operations are on scalars and 2D tensors
        # ================================================================
        
        # Load time-variant gating parameters (scalars)
        p_a0 = a + (bos + idx_token) * HV + i_hv
        p_b0 = b + (bos + idx_token) * HV + i_hv
        b_a0 = gl.load(p_a0).to(gl.float32)
        b_b0 = gl.load(p_b0).to(gl.float32)
        
        # Compute gating factors (scalars)
        x = b_a + b_dt_bias
        beta_x = softplus_beta * x
        softplus_x = gl.where(
            beta_x <= softplus_threshold,
            (1.0 / softplus_beta) * gl.log(1.0 + gl.exp(beta_x)),
            x,
        )
        b_g = -gl.exp(b_A_log) * softplus_x  # scalar
        b_beta = 1.0 / (1.0 + gl.exp(-b_b))  # scalar
        
        # Delta Rule recurrent update
        # b_h: [BK, BV]
        # b_k: [BK] -> broadcast to [:, None] for [BK, 1]
        # b_v: [BV] -> broadcast to [None, :] for [1, BV]
        # b_q: [BK] -> broadcast to [:, None] for [BK, 1]
        # b_g, b_beta: scalars -> broadcast naturally
        b_k = gl.convert_layout(b_k, layout=slice1)
        b_v = gl.convert_layout(b_v, layout=slice4)
        b_q = gl.convert_layout(b_q, layout=slice1)
        
        b_h *= gl.exp(b_g)  # [BK, BV] * scalar
        b_v -= gl.sum(b_h * b_k[:, None], axis=0)  # [BV] -= sum([BK, BV] * [BK, 1], axis=0)
        b_v *= b_beta  # [BV] * scalar
        b_h += b_k[:, None] * b_v[None, :]  # [BK, BV] += [BK, 1] * [1, BV]
        
        # Compute and store output
        b_o = gl.sum(b_h * b_q[:, None], axis=0)  # [BV] = sum([BK, BV] * [BK, 1], axis=0)

        gl.store(p_o, b_o.to(p_o.dtype.element_ty))

        p_o = o + ((i_k * all + bos + idx_token) * HV + i_hv) * V + o_v_slice

        k_conv_acc = k_conv_acc0
        v_conv_acc = v_conv_acc0
        q_conv_acc = q_conv_acc0
        b_a = b_a0
        b_b = b_b0

        b_k_conv_states = b_k_conv_states[1:]
        b_v_conv_states = b_v_conv_states[1:]
        b_q_conv_states = b_q_conv_states[1:]

    if SILU_ACTIVATION:
        k_conv_acc = k_conv_acc / (1 + gl.exp(-k_conv_acc))
    b_k = k_conv_acc.to(gl.float32)  # [BK]
    
    if USE_QK_L2NORM_IN_KERNEL:
        b_k = b_k / (gl.sqrt(gl.sum(b_k * b_k, axis=0) + 1e-6))

    if SILU_ACTIVATION:
        v_conv_acc = v_conv_acc / (1 + gl.exp(-v_conv_acc))
    b_v = v_conv_acc.to(gl.float32)  # [BV]

    if SILU_ACTIVATION:
        q_conv_acc = q_conv_acc / (1 + gl.exp(-q_conv_acc))
    b_q = q_conv_acc.to(gl.float32)  # [BK]
    
    if USE_QK_L2NORM_IN_KERNEL:
        b_q_scale = scale / (gl.sqrt(gl.sum(b_q * b_q, axis=0) + 1e-6))
    else:
        b_q_scale = scale
    b_q = b_q * b_q_scale  # [BK]
    
    # Compute gating factors (scalars)
    x = b_a + b_dt_bias
    beta_x = softplus_beta * x
    softplus_x = gl.where(
        beta_x <= softplus_threshold,
        (1.0 / softplus_beta) * gl.log(1.0 + gl.exp(beta_x)),
        x,
    )
    b_g = -gl.exp(b_A_log) * softplus_x  # scalar
    b_beta = 1.0 / (1.0 + gl.exp(-b_b))  # scalar
    
    # Delta Rule recurrent update
    # b_h: [BK, BV]
    # b_k: [BK] -> broadcast to [:, None] for [BK, 1]
    # b_v: [BV] -> broadcast to [None, :] for [1, BV]
    # b_q: [BK] -> broadcast to [:, None] for [BK, 1]
    # b_g, b_beta: scalars -> broadcast naturally
    b_k = gl.convert_layout(b_k, layout=slice1)
    b_v = gl.convert_layout(b_v, layout=slice4)
    b_q = gl.convert_layout(b_q, layout=slice1)
    
    b_h *= gl.exp(b_g)  # [BK, BV] * scalar
    b_v -= gl.sum(b_h * b_k[:, None], axis=0)  # [BV] -= sum([BK, BV] * [BK, 1], axis=0)
    b_v *= b_beta  # [BV] * scalar
    b_h += b_k[:, None] * b_v[None, :]  # [BK, BV] += [BK, 1] * [1, BV]
    
    # Compute and store output
    b_o = gl.sum(b_h * b_q[:, None], axis=0)  # [BV] = sum([BK, BV] * [BK, 1], axis=0)
    gl.store(p_o, b_o.to(p_o.dtype.element_ty))

    if USE_INITIAL_STATE:
        if idx >= 0:
            offsets = idx * HV * K * V + i_hv * K * V + o_k_slice[:, None] * V + o_v_slice[None, :]
            p_h = h0_source + offsets
            gl.store(p_h, b_h.to(p_h.dtype.element_ty))
    
    # ====================================================================
    # Write back conv_states
    # ====================================================================
    for i in gl.static_range(CONV_WIDTH-1):
        gl.store(conv_state_ptr + (conv_state_batch_coord * stride_conv_state_seq) + (q_feats * stride_conv_state_dim) + i * stride_conv_state_tok, b_q_conv_states[i])
        gl.store(conv_state_ptr + (conv_state_batch_coord * stride_conv_state_seq) + (k_feats * stride_conv_state_dim) + i * stride_conv_state_tok, b_k_conv_states[i])
        gl.store(conv_state_ptr + (conv_state_batch_coord * stride_conv_state_seq) + (v_feats * stride_conv_state_dim) + i * stride_conv_state_tok, b_v_conv_states[i])
            
@gluon.jit(do_not_specialize=["T"])
def gluon_fused_gdn_fwd_decode_kernel_v3(
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
    Simplified Gluon fused kernel where each block processes one V head.
    
    Grid layout: (num_k_blocks, num_v_blocks, batch * num_heads_v)
    
    Key differences from v1:
    - Grid indexed by V heads instead of Q/K heads
    - No GROUP_SIZE loop - each block handles one V head
    - Simpler logic, potentially better parallelism
    - Hidden state shape: [BK, BV] instead of [GROUP_SIZE, BK, BV]
    """
    # ============================================================================
    # Program ID and dimension setup
    # Grid is indexed by V heads, not Q/K heads
    # ============================================================================
    i_k, i_hv, i_n = gl.program_id(0), gl.program_id(1), gl.program_id(2)
    
    # Compute corresponding Q/K head for this V head
    GROUP_SIZE: gl.constexpr = HV // H
    i_h = i_hv // GROUP_SIZE

    q_dim_start = i_h * K
    k_dim_start = key_dim + i_h * K
    v_dim_start = 2 * key_dim + i_hv * V

    
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
    
    # ============================================================================
    # Define layouts and offset ranges
    # ============================================================================
    # BlockedLayout for K and V dimensions
    blocked2d: gl.constexpr = gl.BlockedLayout(
        size_per_thread=[1, 4],
        threads_per_warp=[8, 8],
        warps_per_cta=[1, 4],
        order=[1, 0],
    )
    blocked2: gl.constexpr = gl.BlockedLayout(
        size_per_thread=[2],
        threads_per_warp=[64],
        warps_per_cta=[4],
        order=[0],
    )
    slice1: gl.constexpr = gl.SliceLayout(
        dim=1,
        parent=blocked2d,
    )
    slice4: gl.constexpr = gl.SliceLayout(
        dim=0,
        parent=blocked2d,
    )


    # Define offset ranges
    # o_k: [BK] - Offsets for K dimension
    # o_v: [BV] - Offsets for V dimension
    o_k = i_k * BK + gl.arange(0, BK, layout=blocked2)
    o_k_slice = i_k * BK + gl.arange(0, BK, layout=slice1)
    q_feats = q_dim_start + o_k  # [BK]
    k_feats = k_dim_start + o_k  # [BK]


    # Load gating parameters for this single V head (scalar values)
    b_A_log = gl.load(A_log + i_hv).to(gl.float32)
    b_dt_bias = gl.load(dt_bias + i_hv).to(gl.float32)
    
    # ============================================================================
    # Define feature offsets for Q, K, V
    # ============================================================================

    i_v = 0

    o_v = i_v * BV + gl.arange(0, BV, layout=blocked2)
    o_v_slice = i_v * BV + gl.arange(0, BV, layout=slice4)
    v_feats = v_dim_start + o_v  # [BV]
    
    # ============================================================================
    # Branch: With initial state
    # ============================================================================
    if USE_INITIAL_STATE:
        idx = gl.load(h0_indices + i_n)
        b_h = gl.zeros([BK, BV], dtype=gl.float32, layout=blocked2d)
        if idx >= 0:
            # ====================================================================
            # Load initial hidden state for this V head
            # Shape: [BK, BV] (2D, not 3D like v1)
            # ====================================================================
            offsets = idx * HV * K * V + i_hv * K * V + o_k_slice[:, None] * V + o_v_slice[None, :]
            p_h = h0_source + offsets
            b_h = gl.load(p_h).to(gl.float32)
            # b_h = gl.amd.cdna3.buffer_load(h0_source, offsets).to(gl.float32)  # [BK, BV]

            
    # ====================================================================
    # Pre-load conv_state sliding windows and weights for Q, K, V
    # ====================================================================            
    k_conv_acc = gl.load(conv_bias_ptr + k_feats).to(gl.float32) if HAS_CONV_BIAS else gl.zeros([BK], dtype=gl.float32, layout=blocked2)
    # K conv setup
    b_k_conv_states = ()
    k_weights = (gl.load(conv_w_ptr + k_feats * stride_conv_w_dim),)
    for i in gl.static_range(CONV_WIDTH-1):
        b_k_conv_state = gl.load(conv_state_ptr + (conv_state_batch_coord * stride_conv_state_seq) + (k_feats * stride_conv_state_dim) + i * stride_conv_state_tok)
        w_val = gl.load(conv_w_ptr + k_feats * stride_conv_w_dim + (i+1) * stride_conv_w_width)
        k_weights = tuple_combine(k_weights, w_val)
        b_k_conv_states = tuple_combine(b_k_conv_states, b_k_conv_state)

    k_ptrs = x_ptr + idx_seq * stride_x_seq + k_feats * stride_x_dim + 0 * stride_x_token
    b_k_conv_states = tuple_combine(b_k_conv_states, gl.load(k_ptrs))
    for j in gl.static_range(CONV_WIDTH):
        k_conv_acc += b_k_conv_states[j] * k_weights[j]
    gl.amd.cdna3.sched_barrier(0)
    v_conv_acc = gl.load(conv_bias_ptr + v_feats).to(gl.float32) if HAS_CONV_BIAS else gl.zeros([BV], dtype=gl.float32, layout=blocked2)
    # V conv setup
    b_v_conv_states = ()
    v_weights = (gl.load(conv_w_ptr + v_feats * stride_conv_w_dim),)
    for i in gl.static_range(CONV_WIDTH-1):
        b_v_conv_state = gl.load(conv_state_ptr + (conv_state_batch_coord * stride_conv_state_seq) + (v_feats * stride_conv_state_dim) + i * stride_conv_state_tok)
        w_val = gl.load(conv_w_ptr + v_feats * stride_conv_w_dim + (i+1) * stride_conv_w_width)
        v_weights = tuple_combine(v_weights, w_val)
        b_v_conv_states = tuple_combine(b_v_conv_states, b_v_conv_state)

    v_ptrs = x_ptr + idx_seq * stride_x_seq + v_feats * stride_x_dim + 0 * stride_x_token
    b_v_conv_states = tuple_combine(b_v_conv_states, gl.load(v_ptrs))
    for j in gl.static_range(CONV_WIDTH):
        v_conv_acc += b_v_conv_states[j] * v_weights[j]
    gl.amd.cdna3.sched_barrier(0)
    q_conv_acc = gl.load(conv_bias_ptr + q_feats).to(gl.float32) if HAS_CONV_BIAS else gl.zeros([BK], dtype=gl.float32, layout=blocked2)
    # Q conv setup
    b_q_conv_states = ()
    q_weights = (gl.load(conv_w_ptr + q_feats * stride_conv_w_dim),)
    for i in gl.static_range(CONV_WIDTH-1):
        b_q_conv_state = gl.load(conv_state_ptr + (conv_state_batch_coord * stride_conv_state_seq) + (q_feats * stride_conv_state_dim) + i * stride_conv_state_tok)
        w_val = gl.load(conv_w_ptr + q_feats * stride_conv_w_dim + (i+1) * stride_conv_w_width)
        q_weights = tuple_combine(q_weights, w_val)
        b_q_conv_states = tuple_combine(b_q_conv_states, b_q_conv_state)

    q_ptrs = x_ptr + idx_seq * stride_x_seq + q_feats * stride_x_dim + 0 * stride_x_token
    b_q_conv_states = tuple_combine(b_q_conv_states, gl.load(q_ptrs))
    for j in gl.static_range(CONV_WIDTH):
        q_conv_acc += b_q_conv_states[j] * q_weights[j]
    gl.amd.cdna3.sched_barrier(0)
    # Load time-variant gating parameters (scalars)
    p_a = a + (bos + 0) * HV + i_hv
    p_b = b + (bos + 0) * HV + i_hv
    b_a = gl.load(p_a).to(gl.float32)
    b_b = gl.load(p_b).to(gl.float32)

    p_o = o + ((i_k * all + bos + 0) * HV + i_hv) * V + o_v_slice

    b_k_conv_states = b_k_conv_states[1:]
    b_v_conv_states = b_v_conv_states[1:]
    b_q_conv_states = b_q_conv_states[1:]

    # ====================================================================
    # Main token processing loop
    # For each token: compute K → V → Q → Delta Rule Update
    # ====================================================================
    for idx_token in gl.static_range(1, seqlen):
        # ================================================================
        # Step 1: Conv1D for K
        # Shape: [BK]
        # ================================================================
        k_conv_acc0 = gl.load(conv_bias_ptr + k_feats).to(gl.float32) if HAS_CONV_BIAS else gl.zeros([BK], dtype=gl.float32, layout=blocked2)
        k_ptrs0 = x_ptr + idx_seq * stride_x_seq + k_feats * stride_x_dim + idx_token * stride_x_token
        b_k_conv_states = tuple_combine(b_k_conv_states, gl.load(k_ptrs0))
        for j in gl.static_range(CONV_WIDTH):
            k_conv_acc0 += b_k_conv_states[j] * k_weights[j]

        # ================================================================
        # Step 2: Conv1D for V
        # Shape: [BV]
        # ================================================================
        v_conv_acc0 = gl.load(conv_bias_ptr + v_feats).to(gl.float32) if HAS_CONV_BIAS else gl.zeros([BV], dtype=gl.float32, layout=blocked2)
        v_ptrs0 = x_ptr + idx_seq * stride_x_seq + v_feats * stride_x_dim + idx_token * stride_x_token
        b_v_conv_states = tuple_combine(b_v_conv_states, gl.load(v_ptrs0))
        for j in gl.static_range(CONV_WIDTH):
            v_conv_acc0 += b_v_conv_states[j] * v_weights[j]

        # ================================================================
        # Step 3: Conv1D for Q
        # Shape: [BK]
        # ================================================================
        q_conv_acc0 = gl.load(conv_bias_ptr + q_feats).to(gl.float32) if HAS_CONV_BIAS else gl.zeros([BK], dtype=gl.float32, layout=blocked2)
        q_ptrs0 = x_ptr + idx_seq * stride_x_seq + q_feats * stride_x_dim + idx_token * stride_x_token
        b_q_conv_states = tuple_combine(b_q_conv_states, gl.load(q_ptrs0))
        for j in gl.static_range(CONV_WIDTH):
            q_conv_acc0 += b_q_conv_states[j] * q_weights[j]

        if SILU_ACTIVATION:
            k_conv_acc = k_conv_acc / (1 + gl.exp(-k_conv_acc))
        b_k = k_conv_acc.to(gl.float32)  # [BK]
        
        if USE_QK_L2NORM_IN_KERNEL:
            b_k = b_k / (gl.sqrt(gl.sum(b_k * b_k, axis=0) + 1e-6))

        if SILU_ACTIVATION:
            v_conv_acc = v_conv_acc / (1 + gl.exp(-v_conv_acc))
        b_v = v_conv_acc.to(gl.float32)  # [BV]

        if SILU_ACTIVATION:
            q_conv_acc = q_conv_acc / (1 + gl.exp(-q_conv_acc))
        b_q = q_conv_acc.to(gl.float32)  # [BK]
        
        if USE_QK_L2NORM_IN_KERNEL:
            b_q_scale = scale / (gl.sqrt(gl.sum(b_q * b_q, axis=0) + 1e-6))
        else:
            b_q_scale = scale
        b_q = b_q * b_q_scale  # [BK]
        
        # ================================================================
        # Step 4: Delta Rule update for single V head
        # No broadcasting needed - all operations are on scalars and 2D tensors
        # ================================================================
        
        # Load time-variant gating parameters (scalars)
        p_a0 = a + (bos + idx_token) * HV + i_hv
        p_b0 = b + (bos + idx_token) * HV + i_hv
        b_a0 = gl.load(p_a0).to(gl.float32)
        b_b0 = gl.load(p_b0).to(gl.float32)
        
        # Compute gating factors (scalars)
        x = b_a + b_dt_bias
        beta_x = softplus_beta * x
        softplus_x = gl.where(
            beta_x <= softplus_threshold,
            (1.0 / softplus_beta) * gl.log(1.0 + gl.exp(beta_x)),
            x,
        )
        b_g = -gl.exp(b_A_log) * softplus_x  # scalar
        b_beta = 1.0 / (1.0 + gl.exp(-b_b))  # scalar
        
        # Delta Rule recurrent update
        # b_h: [BK, BV]
        # b_k: [BK] -> broadcast to [:, None] for [BK, 1]
        # b_v: [BV] -> broadcast to [None, :] for [1, BV]
        # b_q: [BK] -> broadcast to [:, None] for [BK, 1]
        # b_g, b_beta: scalars -> broadcast naturally
        b_k = gl.convert_layout(b_k, layout=slice1)
        b_v = gl.convert_layout(b_v, layout=slice4)
        b_q = gl.convert_layout(b_q, layout=slice1)
        
        b_h *= gl.exp(b_g)  # [BK, BV] * scalar
        b_v -= gl.sum(b_h * b_k[:, None], axis=0)  # [BV] -= sum([BK, BV] * [BK, 1], axis=0)
        b_v *= b_beta  # [BV] * scalar
        b_h += b_k[:, None] * b_v[None, :]  # [BK, BV] += [BK, 1] * [1, BV]
        
        # Compute and store output
        b_o = gl.sum(b_h * b_q[:, None], axis=0)  # [BV] = sum([BK, BV] * [BK, 1], axis=0)

        gl.store(p_o, b_o.to(p_o.dtype.element_ty))

        p_o = o + ((i_k * all + bos + idx_token) * HV + i_hv) * V + o_v_slice

        k_conv_acc = k_conv_acc0
        v_conv_acc = v_conv_acc0
        q_conv_acc = q_conv_acc0
        b_a = b_a0
        b_b = b_b0

        b_k_conv_states = b_k_conv_states[1:]
        b_v_conv_states = b_v_conv_states[1:]
        b_q_conv_states = b_q_conv_states[1:]

    if SILU_ACTIVATION:
        k_conv_acc = k_conv_acc / (1 + gl.exp(-k_conv_acc))
    b_k = k_conv_acc.to(gl.float32)  # [BK]
    
    if USE_QK_L2NORM_IN_KERNEL:
        b_k = b_k / (gl.sqrt(gl.sum(b_k * b_k, axis=0) + 1e-6))

    if SILU_ACTIVATION:
        v_conv_acc = v_conv_acc / (1 + gl.exp(-v_conv_acc))
    b_v = v_conv_acc.to(gl.float32)  # [BV]

    if SILU_ACTIVATION:
        q_conv_acc = q_conv_acc / (1 + gl.exp(-q_conv_acc))
    b_q = q_conv_acc.to(gl.float32)  # [BK]
    
    if USE_QK_L2NORM_IN_KERNEL:
        b_q_scale = scale / (gl.sqrt(gl.sum(b_q * b_q, axis=0) + 1e-6))
    else:
        b_q_scale = scale
    b_q = b_q * b_q_scale  # [BK]
    
    # Compute gating factors (scalars)
    x = b_a + b_dt_bias
    beta_x = softplus_beta * x
    softplus_x = gl.where(
        beta_x <= softplus_threshold,
        (1.0 / softplus_beta) * gl.log(1.0 + gl.exp(beta_x)),
        x,
    )
    b_g = -gl.exp(b_A_log) * softplus_x  # scalar
    b_beta = 1.0 / (1.0 + gl.exp(-b_b))  # scalar
    
    # Delta Rule recurrent update
    # b_h: [BK, BV]
    # b_k: [BK] -> broadcast to [:, None] for [BK, 1]
    # b_v: [BV] -> broadcast to [None, :] for [1, BV]
    # b_q: [BK] -> broadcast to [:, None] for [BK, 1]
    # b_g, b_beta: scalars -> broadcast naturally
    b_k = gl.convert_layout(b_k, layout=slice1)
    b_v = gl.convert_layout(b_v, layout=slice4)
    b_q = gl.convert_layout(b_q, layout=slice1)
    
    b_h *= gl.exp(b_g)  # [BK, BV] * scalar
    b_v -= gl.sum(b_h * b_k[:, None], axis=0)  # [BV] -= sum([BK, BV] * [BK, 1], axis=0)
    b_v *= b_beta  # [BV] * scalar
    b_h += b_k[:, None] * b_v[None, :]  # [BK, BV] += [BK, 1] * [1, BV]
    
    # Compute and store output
    b_o = gl.sum(b_h * b_q[:, None], axis=0)  # [BV] = sum([BK, BV] * [BK, 1], axis=0)
    gl.store(p_o, b_o.to(p_o.dtype.element_ty))
    if USE_INITIAL_STATE:
        if idx >= 0:
            offsets = idx * HV * K * V + i_hv * K * V + o_k_slice[:, None] * V + o_v_slice[None, :]
            p_h = h0_source + offsets
            # gl.amd.cdna3.buffer_store(b_h.to(p_h.dtype.element_ty), h0_source, offsets)
            gl.store(p_h, b_h.to(p_h.dtype.element_ty))
    
    # ====================================================================
    # Write back conv_states
    # ====================================================================
    for i in gl.static_range(CONV_WIDTH-1):
        gl.store(conv_state_ptr + (conv_state_batch_coord * stride_conv_state_seq) + (q_feats * stride_conv_state_dim) + i * stride_conv_state_tok, b_q_conv_states[i])
        gl.store(conv_state_ptr + (conv_state_batch_coord * stride_conv_state_seq) + (k_feats * stride_conv_state_dim) + i * stride_conv_state_tok, b_k_conv_states[i])
        gl.store(conv_state_ptr + (conv_state_batch_coord * stride_conv_state_seq) + (v_feats * stride_conv_state_dim) + i * stride_conv_state_tok, b_v_conv_states[i])


@gluon.jit(do_not_specialize=["T"])
def gluon_fused_gdn_fwd_decode_kernel_v4(
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
    num_warps: gl.constexpr,
):
    """
    Simplified Gluon fused kernel where each block processes one V head.
    
    Grid layout: (num_k_blocks, num_v_blocks, batch * num_heads_v)
    
    Key differences from v1:
    - Grid indexed by V heads instead of Q/K heads
    - No GROUP_SIZE loop - each block handles one V head
    - Simpler logic, potentially better parallelism
    - Hidden state shape: [BK, BV] instead of [GROUP_SIZE, BK, BV]
    """
    # ============================================================================
    # Program ID and dimension setup
    # Grid is indexed by V heads, not Q/K heads
    # ============================================================================
    gl.static_assert(seqlen == 1, "seqlen must be 1")
    i_k_, i_hv, i_n = gl.program_id(0), gl.program_id(1), gl.program_id(2)
    i_k: gl.constexpr = 0
    
    
    # Compute corresponding Q/K head for this V head
    GROUP_SIZE: gl.constexpr = HV // H
    i_h = i_hv // GROUP_SIZE

    q_dim_start = i_h * K
    k_dim_start = key_dim + i_h * K
    v_dim_start = 2 * key_dim + i_hv * V

    
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
    
    # ============================================================================
    # Define layouts and offset ranges
    # ============================================================================
    # BlockedLayout for K and V dimensions
    blocked2d: gl.constexpr = gl.BlockedLayout(
        size_per_thread=[8, 4],
        threads_per_warp=[4, 16],
        warps_per_cta=[2, 2],
        order=[1, 0],
    )
    blocked_k: gl.constexpr = gl.BlockedLayout(
        size_per_thread=[1],
        threads_per_warp=[64],
        warps_per_cta=[num_warps],
        order=[0],
    )
    blocked_v: gl.constexpr = gl.BlockedLayout(
        size_per_thread=[1],
        threads_per_warp=[64],
        warps_per_cta=[num_warps],
        order=[0],
    )
    slice_k: gl.constexpr = gl.SliceLayout(
        dim=1,
        parent=blocked2d,
    )
    slice_v: gl.constexpr = gl.SliceLayout(
        dim=0,
        parent=blocked2d,
    )
    shared_mem_layout: gl.constexpr = gl.SwizzledSharedLayout(
        vec=1,
        per_phase=1,
        max_phase=1,
        order=[0]
    )

    shared_q = gl.allocate_shared_memory(gl.float32, [K//BK, BK], shared_mem_layout)
    shared_k = gl.allocate_shared_memory(gl.float32, [K//BK, BK], shared_mem_layout)
    shared_v = gl.allocate_shared_memory(gl.float32, [BV], shared_mem_layout)

    # Define offset ranges
    # o_k: [BK] - Offsets for K dimension
    # o_v: [BV] - Offsets for V dimension
    o_k = i_k * BK + gl.arange(0, BK, layout=blocked_k)
    o_k_slice = i_k * BK + gl.arange(0, BK, layout=slice_k)
    q_feats0 = q_dim_start + o_k  # [BK]
    k_feats0 = k_dim_start + o_k  # [BK]

    q_feats1 = q_dim_start + BK + o_k  # [BK]
    k_feats1 = k_dim_start + BK + o_k  # [BK]

    # Load gating parameters for this single V head (scalar values)
    b_A_log = gl.load(A_log + i_hv).to(gl.float32)
    b_dt_bias = gl.load(dt_bias + i_hv).to(gl.float32)
    
    # Load time-variant gating parameters (scalars)
    p_a = a + (bos + 0) * HV + i_hv
    p_b = b + (bos + 0) * HV + i_hv
    b_a = gl.load(p_a).to(gl.float32)
    b_b = gl.load(p_b).to(gl.float32)

    # Compute gating factors (scalars)
    x = b_a + b_dt_bias
    beta_x = softplus_beta * x
    softplus_x = gl.where(
        beta_x <= softplus_threshold,
        (1.0 / softplus_beta) * gl.log(1.0 + gl.exp(beta_x)),
        x,
    )
    b_g = -gl.exp(b_A_log) * softplus_x  # scalar
    b_beta = 1.0 / (1.0 + gl.exp(-b_b))  # scalar

    # ============================================================================
    # Define feature offsets for Q, K, V
    # ============================================================================

    i_v = 0
    o_v = i_v * BV + gl.arange(0, BV, layout=blocked_v)
    o_v_slice = i_v * BV + gl.arange(0, BV, layout=slice_v)
    v_feats = v_dim_start + o_v  # [BV]

    p_h0 = h0_source + i_hv * K * V + o_k_slice[:, None] * V + o_v_slice[None, :]
    p_h1 = p_h0 + BK * V
    p_o = o + ((i_k * all + bos + 0) * HV + i_hv) * V + o_v_slice
    
    # ====================================================================
    # Pre-load conv_state sliding windows and weights for Q, K, V
    # ====================================================================            
    k_conv_acc0 = gl.load(conv_bias_ptr + k_feats0).to(gl.float32) if HAS_CONV_BIAS else gl.zeros([BK], dtype=gl.float32, layout=blocked_k)
    k_conv_acc1 = gl.load(conv_bias_ptr + k_feats1).to(gl.float32) if HAS_CONV_BIAS else gl.zeros([BK], dtype=gl.float32, layout=blocked_k)
    # K conv setup
    b_k_conv_states0, b_k_conv_states1 = (), ()
    k_weights0, k_weights1 = (gl.load(conv_w_ptr + k_feats0 * stride_conv_w_dim),), (gl.load(conv_w_ptr + k_feats1 * stride_conv_w_dim),)
    for i in gl.static_range(CONV_WIDTH-1):
        b_k_conv_state0 = gl.load(conv_state_ptr + (conv_state_batch_coord * stride_conv_state_seq) + (k_feats0 * stride_conv_state_dim) + i * stride_conv_state_tok)
        b_k_conv_state1 = gl.load(conv_state_ptr + (conv_state_batch_coord * stride_conv_state_seq) + (k_feats1 * stride_conv_state_dim) + i * stride_conv_state_tok)
        w_val0 = gl.load(conv_w_ptr + k_feats0 * stride_conv_w_dim + (i+1) * stride_conv_w_width)
        w_val1 = gl.load(conv_w_ptr + k_feats1 * stride_conv_w_dim + (i+1) * stride_conv_w_width)
        k_weights0 = tuple_combine(k_weights0, w_val0)
        k_weights1 = tuple_combine(k_weights1, w_val1)
        b_k_conv_states0 = tuple_combine(b_k_conv_states0, b_k_conv_state0)
        b_k_conv_states1 = tuple_combine(b_k_conv_states1, b_k_conv_state1)

    k_ptrs0 = x_ptr + idx_seq * stride_x_seq + k_feats0 * stride_x_dim + 0 * stride_x_token
    k_ptrs1 = x_ptr + idx_seq * stride_x_seq + k_feats1 * stride_x_dim + 0 * stride_x_token
    b_k_conv_states0 = tuple_combine(b_k_conv_states0, gl.load(k_ptrs0))
    b_k_conv_states1 = tuple_combine(b_k_conv_states1, gl.load(k_ptrs1))
    for j in gl.static_range(CONV_WIDTH):
        k_conv_acc0 += b_k_conv_states0[j] * k_weights0[j]
        k_conv_acc1 += b_k_conv_states1[j] * k_weights1[j]
    gl.amd.cdna3.sched_barrier(0)
    v_conv_acc = gl.load(conv_bias_ptr + v_feats).to(gl.float32) if HAS_CONV_BIAS else gl.zeros([BV], dtype=gl.float32, layout=blocked_v)
    # V conv setup
    b_v_conv_states = ()
    v_weights = (gl.load(conv_w_ptr + v_feats * stride_conv_w_dim),)
    for i in gl.static_range(CONV_WIDTH-1):
        b_v_conv_state = gl.load(conv_state_ptr + (conv_state_batch_coord * stride_conv_state_seq) + (v_feats * stride_conv_state_dim) + i * stride_conv_state_tok)
        w_val = gl.load(conv_w_ptr + v_feats * stride_conv_w_dim + (i+1) * stride_conv_w_width)
        v_weights = tuple_combine(v_weights, w_val)
        b_v_conv_states = tuple_combine(b_v_conv_states, b_v_conv_state)

    v_ptrs = x_ptr + idx_seq * stride_x_seq + v_feats * stride_x_dim + 0 * stride_x_token
    b_v_conv_states = tuple_combine(b_v_conv_states, gl.load(v_ptrs))
    for j in gl.static_range(CONV_WIDTH):
        v_conv_acc += b_v_conv_states[j] * v_weights[j]
    gl.amd.cdna3.sched_barrier(0)
    q_conv_acc0 = gl.load(conv_bias_ptr + q_feats0).to(gl.float32) if HAS_CONV_BIAS else gl.zeros([BK], dtype=gl.float32, layout=blocked_k)
    q_conv_acc1 = gl.load(conv_bias_ptr + q_feats1).to(gl.float32) if HAS_CONV_BIAS else gl.zeros([BK], dtype=gl.float32, layout=blocked_k)
    # Q conv setup
    b_q_conv_states0, b_q_conv_states1 = (), ()
    q_weights0, q_weights1 = (gl.load(conv_w_ptr + q_feats0 * stride_conv_w_dim),), (gl.load(conv_w_ptr + q_feats1 * stride_conv_w_dim),)
    for i in gl.static_range(CONV_WIDTH-1):
        b_q_conv_state0 = gl.load(conv_state_ptr + (conv_state_batch_coord * stride_conv_state_seq) + (q_feats0 * stride_conv_state_dim) + i * stride_conv_state_tok)
        b_q_conv_state1 = gl.load(conv_state_ptr + (conv_state_batch_coord * stride_conv_state_seq) + (q_feats1 * stride_conv_state_dim) + i * stride_conv_state_tok)
        w_val0 = gl.load(conv_w_ptr + q_feats0 * stride_conv_w_dim + (i+1) * stride_conv_w_width)
        w_val1 = gl.load(conv_w_ptr + q_feats1 * stride_conv_w_dim + (i+1) * stride_conv_w_width)
        q_weights0 = tuple_combine(q_weights0, w_val0)
        q_weights1 = tuple_combine(q_weights1, w_val1)
        b_q_conv_states0 = tuple_combine(b_q_conv_states0, b_q_conv_state0)
        b_q_conv_states1 = tuple_combine(b_q_conv_states1, b_q_conv_state1)

    q_ptrs0 = x_ptr + idx_seq * stride_x_seq + q_feats0 * stride_x_dim + 0 * stride_x_token
    q_ptrs1 = x_ptr + idx_seq * stride_x_seq + q_feats1 * stride_x_dim + 0 * stride_x_token
    b_q_conv_states0 = tuple_combine(b_q_conv_states0, gl.load(q_ptrs0))
    b_q_conv_states1 = tuple_combine(b_q_conv_states1, gl.load(q_ptrs1))
    for j in gl.static_range(CONV_WIDTH):
        q_conv_acc0 += b_q_conv_states0[j] * q_weights0[j]
        q_conv_acc1 += b_q_conv_states1[j] * q_weights1[j]
    gl.amd.cdna3.sched_barrier(0)

    b_h0 = gl.zeros([BK, BV], dtype=gl.float32, layout=blocked2d)
    b_h1 = gl.zeros([BK, BV], dtype=gl.float32, layout=blocked2d)
    if USE_INITIAL_STATE:
        idx = gl.load(h0_indices + i_n)
        if idx >= 0:
            b_h0 = gl.load(p_h0 + idx * HV * K * V).to(gl.float32)
            b_h1 = gl.load(p_h1 + idx * HV * K * V).to(gl.float32)

    if SILU_ACTIVATION:
        k_conv_acc0 = k_conv_acc0 / (1 + gl.exp(-k_conv_acc0))
        k_conv_acc1 = k_conv_acc1 / (1 + gl.exp(-k_conv_acc1))
    b_k0 = k_conv_acc0.to(gl.float32)  # [BK]
    b_k1 = k_conv_acc1.to(gl.float32)  # [BK]
    
    if USE_QK_L2NORM_IN_KERNEL:
        b_k_rcp = 1.0 / (gl.sqrt(gl.sum(b_k0 * b_k0 + b_k1 * b_k1, axis=0) + 1e-6))
        b_k0 = b_k0 * b_k_rcp
        b_k1 = b_k1 * b_k_rcp

    if SILU_ACTIVATION:
        v_conv_acc = v_conv_acc / (1 + gl.exp(-v_conv_acc))
    b_v = v_conv_acc.to(gl.float32)  # [BV]

    if SILU_ACTIVATION:
        q_conv_acc0 = q_conv_acc0 / (1 + gl.exp(-q_conv_acc0))
        q_conv_acc1 = q_conv_acc1 / (1 + gl.exp(-q_conv_acc1))
    b_q0 = q_conv_acc0.to(gl.float32)  # [BK]
    b_q1 = q_conv_acc1.to(gl.float32)  # [BK]
    
    if USE_QK_L2NORM_IN_KERNEL:
        b_q_rcp = scale / (gl.sqrt(gl.sum(b_q0 * b_q0 + b_q1 * b_q1, axis=0) + 1e-6))
    else:
        b_q_rcp = scale
    b_q0 = b_q0 * b_q_rcp  # [BK]
    b_q1 = b_q1 * b_q_rcp  # [BK]

    for i in gl.static_range(CONV_WIDTH-1):
        gl.store(conv_state_ptr + (conv_state_batch_coord * stride_conv_state_seq) + (k_feats0 * stride_conv_state_dim) + i * stride_conv_state_tok, b_k_conv_states0[i+1])
        gl.store(conv_state_ptr + (conv_state_batch_coord * stride_conv_state_seq) + (k_feats1 * stride_conv_state_dim) + i * stride_conv_state_tok, b_k_conv_states1[i+1])
        gl.store(conv_state_ptr + (conv_state_batch_coord * stride_conv_state_seq) + (v_feats * stride_conv_state_dim) + i * stride_conv_state_tok, b_v_conv_states[i+1])
        gl.store(conv_state_ptr + (conv_state_batch_coord * stride_conv_state_seq) + (q_feats0 * stride_conv_state_dim) + i * stride_conv_state_tok, b_q_conv_states0[i+1])
        gl.store(conv_state_ptr + (conv_state_batch_coord * stride_conv_state_seq) + (q_feats1 * stride_conv_state_dim) + i * stride_conv_state_tok, b_q_conv_states1[i+1])


    # b_k0 = gl.convert_layout(b_k0, layout=slice1)
    # b_k1 = gl.convert_layout(b_k1, layout=slice1)
    # b_v = gl.convert_layout(b_v, layout=slice4)
    # b_q0 = gl.convert_layout(b_q0, layout=slice1)
    # b_q1 = gl.convert_layout(b_q1, layout=slice1)
    shared_k.index(0).store(b_k0)
    shared_k.index(1).store(b_k1)
    shared_q.index(0).store(b_q0)
    shared_q.index(1).store(b_q1)
    shared_v.store(b_v)

    b_k0 = shared_k.index(0).load(layout=slice_k)
    b_k1 = shared_k.index(1).load(layout=slice_k)
    b_q0 = shared_q.index(0).load(layout=slice_k)
    b_q1 = shared_q.index(1).load(layout=slice_k)

    for i_v in gl.static_range(1, V//BV):
        o_v0 = o_v + BV
        o_v_slice0 = o_v_slice + BV
        p_o0 = p_o + BV
        p_h00 = p_h0 + BV
        p_h10 = p_h1 + BV
        v_feats0 = v_dim_start + o_v0  # [BV]

        v_conv_acc0 = gl.load(conv_bias_ptr + v_feats0).to(gl.float32) if HAS_CONV_BIAS else gl.zeros([BV], dtype=gl.float32, layout=blocked_v)
        # V conv setup
        b_v_conv_states0 = ()
        v_weights0 = (gl.load(conv_w_ptr + v_feats0 * stride_conv_w_dim),)
        for i in gl.static_range(CONV_WIDTH-1):
            b_v_conv_state0 = gl.load(conv_state_ptr + (conv_state_batch_coord * stride_conv_state_seq) + (v_feats0 * stride_conv_state_dim) + i * stride_conv_state_tok)
            w_val0 = gl.load(conv_w_ptr + v_feats0 * stride_conv_w_dim + (i+1) * stride_conv_w_width)
            v_weights0 = tuple_combine(v_weights0, w_val0)
            b_v_conv_states0 = tuple_combine(b_v_conv_states0, b_v_conv_state0)

        v_ptrs0 = x_ptr + idx_seq * stride_x_seq + v_feats0 * stride_x_dim + 0 * stride_x_token
        b_v_conv_states0 = tuple_combine(b_v_conv_states0, gl.load(v_ptrs0))
        for j in gl.static_range(CONV_WIDTH):
            v_conv_acc0 += b_v_conv_states0[j] * v_weights0[j]
        gl.amd.cdna3.sched_barrier(0)

        b_h00 = gl.zeros([BK, BV], dtype=gl.float32, layout=blocked2d)
        b_h10 = gl.zeros([BK, BV], dtype=gl.float32, layout=blocked2d)
        if USE_INITIAL_STATE:
            if idx >= 0:
                b_h00 = gl.load(p_h00 + idx * HV * K * V).to(gl.float32)
                b_h10 = gl.load(p_h10 + idx * HV * K * V).to(gl.float32)
                # b_h = gl.amd.cdna3.buffer_load(h0_source, offsets).to(gl.float32)  # [BK, BV]
        
        b_v = shared_v.load(layout=slice_v)

        for i in gl.static_range(CONV_WIDTH-1):
            gl.store(conv_state_ptr + (conv_state_batch_coord * stride_conv_state_seq) + (v_feats0 * stride_conv_state_dim) + i * stride_conv_state_tok, b_v_conv_states0[i+1])

        if SILU_ACTIVATION:
            v_conv_acc0 = v_conv_acc0 / (1 + gl.exp(-v_conv_acc0))
        b_v0 = v_conv_acc0.to(gl.float32)  # [BV]

        # Delta Rule recurrent update
        # b_v0 = gl.convert_layout(b_v0, layout=slice4)
        shared_v.store(b_v0)
        
        b_h0 *= gl.exp(b_g)  # [BK, BV] * scalar
        b_h1 *= gl.exp(b_g)  # [BK, BV] * scalar
        b_v -= gl.sum(b_h0 * b_k0[:, None] + b_h1 * b_k1[:, None], axis=0)  # [BV] -= sum([BK, BV] * [BK, 1], axis=0)
        b_v *= b_beta  # [BV] * scalar
        b_h0 += b_k0[:, None] * b_v[None, :]  # [BK, BV] += [BK, 1] * [1, BV]
        b_h1 += b_k1[:, None] * b_v[None, :]  # [BK, BV] += [BK, 1] * [1, BV]
        
        # Compute and store output
        b_o = gl.sum(b_h0 * b_q0[:, None] + b_h1 * b_q1[:, None], axis=0)  # [BV] = sum([BK, BV] * [BK, 1], axis=0)

        gl.store(p_o, b_o.to(p_o.dtype.element_ty))


        if USE_INITIAL_STATE:
            if idx >= 0:
                # gl.amd.cdna3.buffer_store(b_h.to(p_h.dtype.element_ty), h0_source, offsets)
                gl.store(p_h0 + idx * HV * K * V, b_h0.to(p_h0.dtype.element_ty))
                gl.store(p_h1 + idx * HV * K * V, b_h1.to(p_h1.dtype.element_ty))
      
        o_v = o_v0
        o_v_slice = o_v_slice0
        p_o = p_o0
        b_h0 = b_h00
        b_h1 = b_h10
        b_v = b_v0
        p_h0 = p_h00
        p_h1 = p_h10
    
    # Delta Rule recurrent update
    b_v = shared_v.load(layout=slice_v)

    b_h0 *= gl.exp(b_g)  # [BK, BV] * scalar
    b_h1 *= gl.exp(b_g)  # [BK, BV] * scalar
    b_v -= gl.sum(b_h0 * b_k0[:, None] + b_h1 * b_k1[:, None], axis=0)  # [BV] -= sum([BK, BV] * [BK, 1], axis=0)
    b_v *= b_beta  # [BV] * scalar
    b_h0 += b_k0[:, None] * b_v[None, :]  # [BK, BV] += [BK, 1] * [1, BV]
    b_h1 += b_k1[:, None] * b_v[None, :]  # [BK, BV] += [BK, 1] * [1, BV]
    
    # Compute and store output
    b_o = gl.sum(b_h0 * b_q0[:, None] + b_h1 * b_q1[:, None], axis=0)  # [BV] = sum([BK, BV] * [BK, 1], axis=0)
    gl.store(p_o, b_o.to(p_o.dtype.element_ty))

    if USE_INITIAL_STATE:
        if idx >= 0:
            # gl.amd.cdna3.buffer_store(b_h.to(p_h.dtype.element_ty), h0_source, offsets)
            gl.store(p_h0 + idx * HV * K * V, b_h0.to(p_h0.dtype.element_ty))
            gl.store(p_h1 + idx * HV * K * V, b_h1.to(p_h1.dtype.element_ty))

@gluon.jit(do_not_specialize=["T"])
def gluon_fused_gdn_fwd_decode_kernel_v5(
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
    num_warps: gl.constexpr,
):
    """
    Simplified Gluon fused kernel where each block processes one V head.
    
    Grid layout: (num_k_blocks, num_v_blocks, batch * num_heads_v)
    
    Key differences from v1:
    - Grid indexed by V heads instead of Q/K heads
    - No GROUP_SIZE loop - each block handles one V head
    - Simpler logic, potentially better parallelism
    - Hidden state shape: [BK, BV] instead of [GROUP_SIZE, BK, BV]
    """
    # ============================================================================
    # Program ID and dimension setup
    # Grid is indexed by V heads, not Q/K heads
    # ============================================================================
    gl.static_assert(seqlen == 1, "seqlen must be 1")
    cu_idx = gl.program_id(0)
    hv_idx = cu_idx % HV
    batch_idx = cu_idx // HV
    num_cus = 80
    batch_cus = num_cus // HV
    per_cu_batchs = batch // batch_cus
    cu_mores = batch % batch_cus

    cu_tasks = (per_cu_batchs+1) if batch_idx < cu_mores else per_cu_batchs
    cu_offs = 0 if batch_idx < cu_mores else cu_mores

    # tl.device_print("", hv_idx)
    
    for task_idx in range(cu_tasks):
        i_n = batch_idx * cu_tasks + task_idx + cu_offs

        # tl.device_print("", i_n)

        i_k: gl.constexpr = 0
        i_hv = hv_idx

        # Compute corresponding Q/K head for this V head
        GROUP_SIZE: gl.constexpr = HV // H
        i_h = i_hv // GROUP_SIZE

        q_dim_start = i_h * K
        k_dim_start = key_dim + i_h * K
        v_dim_start = 2 * key_dim + i_hv * V

        
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
        
        if idx_seq < batch:
            # Get conv state batch coordinate
            if IS_CONTINUOUS_BATCHING:
                conv_state_batch_coord = gl.load(
                    conv_state_indices_ptr + idx_seq * stride_state_indices
                ).to(gl.int64)
            else:
                conv_state_batch_coord = idx_seq
                
            if USE_PAD_SLOT:
                if conv_state_batch_coord != pad_slot_id:
            
                    # ============================================================================
                    # Define layouts and offset ranges
                    # ============================================================================
                    # BlockedLayout for K and V dimensions
                    blocked2d: gl.constexpr = gl.BlockedLayout(
                        size_per_thread=[1, 4],
                        threads_per_warp=[8, 8],
                        warps_per_cta=[1, 4],
                        order=[1, 0],
                    )
                    blocked2: gl.constexpr = gl.BlockedLayout(
                        size_per_thread=[2],
                        threads_per_warp=[64],
                        warps_per_cta=[4],
                        order=[0],
                    )
                    slice1: gl.constexpr = gl.SliceLayout(
                        dim=1,
                        parent=blocked2d,
                    )
                    slice4: gl.constexpr = gl.SliceLayout(
                        dim=0,
                        parent=blocked2d,
                    )


                    # Define offset ranges
                    # o_k: [BK] - Offsets for K dimension
                    # o_v: [BV] - Offsets for V dimension
                    o_k = i_k * BK + gl.arange(0, BK, layout=blocked2)
                    o_k_slice = i_k * BK + gl.arange(0, BK, layout=slice1)
                    q_feats = q_dim_start + o_k  # [BK]
                    k_feats = k_dim_start + o_k  # [BK]


                    # Load gating parameters for this single V head (scalar values)
                    b_A_log = gl.load(A_log + i_hv).to(gl.float32)
                    b_dt_bias = gl.load(dt_bias + i_hv).to(gl.float32)
                    
                    # ============================================================================
                    # Define feature offsets for Q, K, V
                    # ============================================================================

                    i_v = 0

                    o_v = i_v * BV + gl.arange(0, BV, layout=blocked2)
                    o_v_slice = i_v * BV + gl.arange(0, BV, layout=slice4)
                    v_feats = v_dim_start + o_v  # [BV]
                    
                    # ============================================================================
                    # Branch: With initial state
                    # ============================================================================
                    if USE_INITIAL_STATE:
                        idx = gl.load(h0_indices + i_n)
                        b_h = gl.zeros([BK, BV], dtype=gl.float32, layout=blocked2d)
                        if idx >= 0:
                            # ====================================================================
                            # Load initial hidden state for this V head
                            # Shape: [BK, BV] (2D, not 3D like v1)
                            # ====================================================================
                            offsets = idx * HV * K * V + i_hv * K * V + o_k_slice[:, None] * V + o_v_slice[None, :]
                            p_h = h0_source + offsets
                            b_h = gl.load(p_h).to(gl.float32)
                            # b_h = gl.amd.cdna3.buffer_load(h0_source, offsets).to(gl.float32)  # [BK, BV]

                            
                    # ====================================================================
                    # Pre-load conv_state sliding windows and weights for Q, K, V
                    # ====================================================================            
                    k_conv_acc = gl.load(conv_bias_ptr + k_feats).to(gl.float32) if HAS_CONV_BIAS else gl.zeros([BK], dtype=gl.float32, layout=blocked2)
                    # K conv setup
                    b_k_conv_states = ()
                    k_weights = (gl.load(conv_w_ptr + k_feats * stride_conv_w_dim),)
                    for i in gl.static_range(CONV_WIDTH-1):
                        b_k_conv_state = gl.load(conv_state_ptr + (conv_state_batch_coord * stride_conv_state_seq) + (k_feats * stride_conv_state_dim) + i * stride_conv_state_tok)
                        w_val = gl.load(conv_w_ptr + k_feats * stride_conv_w_dim + (i+1) * stride_conv_w_width)
                        k_weights = tuple_combine(k_weights, w_val)
                        b_k_conv_states = tuple_combine(b_k_conv_states, b_k_conv_state)

                    k_ptrs = x_ptr + idx_seq * stride_x_seq + k_feats * stride_x_dim + 0 * stride_x_token
                    b_k_conv_states = tuple_combine(b_k_conv_states, gl.load(k_ptrs))
                    for j in gl.static_range(CONV_WIDTH):
                        k_conv_acc += b_k_conv_states[j] * k_weights[j]
                    gl.amd.cdna3.sched_barrier(0)
                    v_conv_acc = gl.load(conv_bias_ptr + v_feats).to(gl.float32) if HAS_CONV_BIAS else gl.zeros([BV], dtype=gl.float32, layout=blocked2)
                    # V conv setup
                    b_v_conv_states = ()
                    v_weights = (gl.load(conv_w_ptr + v_feats * stride_conv_w_dim),)
                    for i in gl.static_range(CONV_WIDTH-1):
                        b_v_conv_state = gl.load(conv_state_ptr + (conv_state_batch_coord * stride_conv_state_seq) + (v_feats * stride_conv_state_dim) + i * stride_conv_state_tok)
                        w_val = gl.load(conv_w_ptr + v_feats * stride_conv_w_dim + (i+1) * stride_conv_w_width)
                        v_weights = tuple_combine(v_weights, w_val)
                        b_v_conv_states = tuple_combine(b_v_conv_states, b_v_conv_state)

                    v_ptrs = x_ptr + idx_seq * stride_x_seq + v_feats * stride_x_dim + 0 * stride_x_token
                    b_v_conv_states = tuple_combine(b_v_conv_states, gl.load(v_ptrs))
                    for j in gl.static_range(CONV_WIDTH):
                        v_conv_acc += b_v_conv_states[j] * v_weights[j]
                    gl.amd.cdna3.sched_barrier(0)
                    q_conv_acc = gl.load(conv_bias_ptr + q_feats).to(gl.float32) if HAS_CONV_BIAS else gl.zeros([BK], dtype=gl.float32, layout=blocked2)
                    # Q conv setup
                    b_q_conv_states = ()
                    q_weights = (gl.load(conv_w_ptr + q_feats * stride_conv_w_dim),)
                    for i in gl.static_range(CONV_WIDTH-1):
                        b_q_conv_state = gl.load(conv_state_ptr + (conv_state_batch_coord * stride_conv_state_seq) + (q_feats * stride_conv_state_dim) + i * stride_conv_state_tok)
                        w_val = gl.load(conv_w_ptr + q_feats * stride_conv_w_dim + (i+1) * stride_conv_w_width)
                        q_weights = tuple_combine(q_weights, w_val)
                        b_q_conv_states = tuple_combine(b_q_conv_states, b_q_conv_state)

                    q_ptrs = x_ptr + idx_seq * stride_x_seq + q_feats * stride_x_dim + 0 * stride_x_token
                    b_q_conv_states = tuple_combine(b_q_conv_states, gl.load(q_ptrs))
                    for j in gl.static_range(CONV_WIDTH):
                        q_conv_acc += b_q_conv_states[j] * q_weights[j]
                    gl.amd.cdna3.sched_barrier(0)
                    # Load time-variant gating parameters (scalars)
                    p_a = a + (bos + 0) * HV + i_hv
                    p_b = b + (bos + 0) * HV + i_hv
                    b_a = gl.load(p_a).to(gl.float32)
                    b_b = gl.load(p_b).to(gl.float32)

                    p_o = o + ((i_k * all + bos + 0) * HV + i_hv) * V + o_v_slice

                    b_k_conv_states = b_k_conv_states[1:]
                    b_v_conv_states = b_v_conv_states[1:]
                    b_q_conv_states = b_q_conv_states[1:]

                    # ====================================================================
                    # Main token processing loop
                    # For each token: compute K → V → Q → Delta Rule Update
                    # ====================================================================
                    for idx_token in gl.static_range(1, seqlen):
                        # ================================================================
                        # Step 1: Conv1D for K
                        # Shape: [BK]
                        # ================================================================
                        k_conv_acc0 = gl.load(conv_bias_ptr + k_feats).to(gl.float32) if HAS_CONV_BIAS else gl.zeros([BK], dtype=gl.float32, layout=blocked2)
                        k_ptrs0 = x_ptr + idx_seq * stride_x_seq + k_feats * stride_x_dim + idx_token * stride_x_token
                        b_k_conv_states = tuple_combine(b_k_conv_states, gl.load(k_ptrs0))
                        for j in gl.static_range(CONV_WIDTH):
                            k_conv_acc0 += b_k_conv_states[j] * k_weights[j]

                        # ================================================================
                        # Step 2: Conv1D for V
                        # Shape: [BV]
                        # ================================================================
                        v_conv_acc0 = gl.load(conv_bias_ptr + v_feats).to(gl.float32) if HAS_CONV_BIAS else gl.zeros([BV], dtype=gl.float32, layout=blocked2)
                        v_ptrs0 = x_ptr + idx_seq * stride_x_seq + v_feats * stride_x_dim + idx_token * stride_x_token
                        b_v_conv_states = tuple_combine(b_v_conv_states, gl.load(v_ptrs0))
                        for j in gl.static_range(CONV_WIDTH):
                            v_conv_acc0 += b_v_conv_states[j] * v_weights[j]

                        # ================================================================
                        # Step 3: Conv1D for Q
                        # Shape: [BK]
                        # ================================================================
                        q_conv_acc0 = gl.load(conv_bias_ptr + q_feats).to(gl.float32) if HAS_CONV_BIAS else gl.zeros([BK], dtype=gl.float32, layout=blocked2)
                        q_ptrs0 = x_ptr + idx_seq * stride_x_seq + q_feats * stride_x_dim + idx_token * stride_x_token
                        b_q_conv_states = tuple_combine(b_q_conv_states, gl.load(q_ptrs0))
                        for j in gl.static_range(CONV_WIDTH):
                            q_conv_acc0 += b_q_conv_states[j] * q_weights[j]

                        if SILU_ACTIVATION:
                            k_conv_acc = k_conv_acc / (1 + gl.exp(-k_conv_acc))
                        b_k = k_conv_acc.to(gl.float32)  # [BK]
                        
                        if USE_QK_L2NORM_IN_KERNEL:
                            b_k = b_k / (gl.sqrt(gl.sum(b_k * b_k, axis=0) + 1e-6))

                        if SILU_ACTIVATION:
                            v_conv_acc = v_conv_acc / (1 + gl.exp(-v_conv_acc))
                        b_v = v_conv_acc.to(gl.float32)  # [BV]

                        if SILU_ACTIVATION:
                            q_conv_acc = q_conv_acc / (1 + gl.exp(-q_conv_acc))
                        b_q = q_conv_acc.to(gl.float32)  # [BK]
                        
                        if USE_QK_L2NORM_IN_KERNEL:
                            b_q_scale = scale / (gl.sqrt(gl.sum(b_q * b_q, axis=0) + 1e-6))
                        else:
                            b_q_scale = scale
                        b_q = b_q * b_q_scale  # [BK]
                        
                        # ================================================================
                        # Step 4: Delta Rule update for single V head
                        # No broadcasting needed - all operations are on scalars and 2D tensors
                        # ================================================================
                        
                        # Load time-variant gating parameters (scalars)
                        p_a0 = a + (bos + idx_token) * HV + i_hv
                        p_b0 = b + (bos + idx_token) * HV + i_hv
                        b_a0 = gl.load(p_a0).to(gl.float32)
                        b_b0 = gl.load(p_b0).to(gl.float32)
                        
                        # Compute gating factors (scalars)
                        x = b_a + b_dt_bias
                        beta_x = softplus_beta * x
                        softplus_x = gl.where(
                            beta_x <= softplus_threshold,
                            (1.0 / softplus_beta) * gl.log(1.0 + gl.exp(beta_x)),
                            x,
                        )
                        b_g = -gl.exp(b_A_log) * softplus_x  # scalar
                        b_beta = 1.0 / (1.0 + gl.exp(-b_b))  # scalar
                        
                        # Delta Rule recurrent update
                        # b_h: [BK, BV]
                        # b_k: [BK] -> broadcast to [:, None] for [BK, 1]
                        # b_v: [BV] -> broadcast to [None, :] for [1, BV]
                        # b_q: [BK] -> broadcast to [:, None] for [BK, 1]
                        # b_g, b_beta: scalars -> broadcast naturally
                        b_k = gl.convert_layout(b_k, layout=slice1)
                        b_v = gl.convert_layout(b_v, layout=slice4)
                        b_q = gl.convert_layout(b_q, layout=slice1)
                        
                        b_h *= gl.exp(b_g)  # [BK, BV] * scalar
                        b_v -= gl.sum(b_h * b_k[:, None], axis=0)  # [BV] -= sum([BK, BV] * [BK, 1], axis=0)
                        b_v *= b_beta  # [BV] * scalar
                        b_h += b_k[:, None] * b_v[None, :]  # [BK, BV] += [BK, 1] * [1, BV]
                        
                        # Compute and store output
                        b_o = gl.sum(b_h * b_q[:, None], axis=0)  # [BV] = sum([BK, BV] * [BK, 1], axis=0)

                        gl.store(p_o, b_o.to(p_o.dtype.element_ty))

                        p_o = o + ((i_k * all + bos + idx_token) * HV + i_hv) * V + o_v_slice

                        k_conv_acc = k_conv_acc0
                        v_conv_acc = v_conv_acc0
                        q_conv_acc = q_conv_acc0
                        b_a = b_a0
                        b_b = b_b0

                        b_k_conv_states = b_k_conv_states[1:]
                        b_v_conv_states = b_v_conv_states[1:]
                        b_q_conv_states = b_q_conv_states[1:]

                    if SILU_ACTIVATION:
                        k_conv_acc = k_conv_acc / (1 + gl.exp(-k_conv_acc))
                    b_k = k_conv_acc.to(gl.float32)  # [BK]
                    
                    if USE_QK_L2NORM_IN_KERNEL:
                        b_k = b_k / (gl.sqrt(gl.sum(b_k * b_k, axis=0) + 1e-6))

                    if SILU_ACTIVATION:
                        v_conv_acc = v_conv_acc / (1 + gl.exp(-v_conv_acc))
                    b_v = v_conv_acc.to(gl.float32)  # [BV]

                    if SILU_ACTIVATION:
                        q_conv_acc = q_conv_acc / (1 + gl.exp(-q_conv_acc))
                    b_q = q_conv_acc.to(gl.float32)  # [BK]
                    
                    if USE_QK_L2NORM_IN_KERNEL:
                        b_q_scale = scale / (gl.sqrt(gl.sum(b_q * b_q, axis=0) + 1e-6))
                    else:
                        b_q_scale = scale
                    b_q = b_q * b_q_scale  # [BK]
                    
                    # Compute gating factors (scalars)
                    x = b_a + b_dt_bias
                    beta_x = softplus_beta * x
                    softplus_x = gl.where(
                        beta_x <= softplus_threshold,
                        (1.0 / softplus_beta) * gl.log(1.0 + gl.exp(beta_x)),
                        x,
                    )
                    b_g = -gl.exp(b_A_log) * softplus_x  # scalar
                    b_beta = 1.0 / (1.0 + gl.exp(-b_b))  # scalar
                    
                    # Delta Rule recurrent update
                    # b_h: [BK, BV]
                    # b_k: [BK] -> broadcast to [:, None] for [BK, 1]
                    # b_v: [BV] -> broadcast to [None, :] for [1, BV]
                    # b_q: [BK] -> broadcast to [:, None] for [BK, 1]
                    # b_g, b_beta: scalars -> broadcast naturally
                    b_k = gl.convert_layout(b_k, layout=slice1)
                    b_v = gl.convert_layout(b_v, layout=slice4)
                    b_q = gl.convert_layout(b_q, layout=slice1)
                    
                    b_h *= gl.exp(b_g)  # [BK, BV] * scalar
                    b_v -= gl.sum(b_h * b_k[:, None], axis=0)  # [BV] -= sum([BK, BV] * [BK, 1], axis=0)
                    b_v *= b_beta  # [BV] * scalar
                    b_h += b_k[:, None] * b_v[None, :]  # [BK, BV] += [BK, 1] * [1, BV]
                    
                    # Compute and store output
                    b_o = gl.sum(b_h * b_q[:, None], axis=0)  # [BV] = sum([BK, BV] * [BK, 1], axis=0)
                    gl.store(p_o, b_o.to(p_o.dtype.element_ty))
                    if USE_INITIAL_STATE:
                        if idx >= 0:
                            offsets = idx * HV * K * V + i_hv * K * V + o_k_slice[:, None] * V + o_v_slice[None, :]
                            p_h = h0_source + offsets
                            # gl.amd.cdna3.buffer_store(b_h.to(p_h.dtype.element_ty), h0_source, offsets)
                            gl.store(p_h, b_h.to(p_h.dtype.element_ty))
                    
                    # ====================================================================
                    # Write back conv_states
                    # ====================================================================
                    for i in gl.static_range(CONV_WIDTH-1):
                        if i_h == GROUP_SIZE - 1:
                            gl.store(conv_state_ptr + (conv_state_batch_coord * stride_conv_state_seq) + (q_feats * stride_conv_state_dim) + i * stride_conv_state_tok, b_q_conv_states[i])
                            gl.store(conv_state_ptr + (conv_state_batch_coord * stride_conv_state_seq) + (k_feats * stride_conv_state_dim) + i * stride_conv_state_tok, b_k_conv_states[i])
                        gl.store(conv_state_ptr + (conv_state_batch_coord * stride_conv_state_seq) + (v_feats * stride_conv_state_dim) + i * stride_conv_state_tok, b_v_conv_states[i])

@gluon.jit(do_not_specialize=["T"])
def gluon_fused_gdn_fwd_decode_kernel_v6(
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
    Gluon-optimized fused kernel with Q/K reuse and batched V head processing.
    
    Key optimizations:
    1. Each block processes GROUP_SIZE (HV//H) value heads simultaneously
    2. Q/K are computed once per group and broadcast across all V heads
    3. V heads are represented as tensor dimensions for efficient batching
    4. Delta Rule updates use broadcasting instead of loops
    """
    
    # ============================================================================
    # Layout Definitions
    # ============================================================================
    blocked_k: gl.constexpr = gl.BlockedLayout(
        size_per_thread=[2],
        threads_per_warp=[64],
        warps_per_cta=[4],
        order=[0],
    )
    
    blocked_v: gl.constexpr = gl.BlockedLayout(
        size_per_thread=[1],
        threads_per_warp=[64],
        warps_per_cta=[4],
        order=[0],
    )
    
    blocked2d: gl.constexpr = gl.BlockedLayout(
        size_per_thread=[1, 1],
        threads_per_warp=[4, 16],
        warps_per_cta=[1, 4],
        order=[1, 0],
    )

    blocked3d: gl.constexpr = gl.BlockedLayout(
        size_per_thread=[1, 1, 4],
        threads_per_warp=[1, 16, 4],
        warps_per_cta=[1, 1, 4],
        order=[2, 1, 0],
    )

    blocked3d1: gl.constexpr = gl.BlockedLayout(
        size_per_thread=[1, 1, 1],
        threads_per_warp=[1, 4, 16],
        warps_per_cta=[1, 1, 4],
        order=[2, 1, 0],
    )
    
    # Slice layouts for 2D tensor operations
    slice_bk: gl.constexpr = gl.SliceLayout(dim=0,
        parent=gl.SliceLayout(
            dim=2,
            parent=blocked3d,
        ),
    )
    
    slice_v: gl.constexpr = gl.SliceLayout(dim=1, parent=blocked3d)
    slice_bv: gl.constexpr = gl.SliceLayout(dim=0, parent=slice_v)
    slice_group: gl.constexpr = gl.SliceLayout(dim=1,
        parent=gl.SliceLayout(
            dim=2,
            parent=blocked3d,
        ),
    )
    slice_group_11: gl.constexpr = gl.SliceLayout(dim=1,
        parent=gl.SliceLayout(
            dim=1,
            parent=blocked3d,
        ),
    )
    
    gl.static_assert(seqlen == 1, "seqlen must be 1")
    cu_idx = gl.program_id(0)
    i_h = cu_idx % H
    batch_idx = cu_idx // H
    num_cus = 80
    batch_cus = num_cus // H
    per_cu_batchs = batch // batch_cus
    cu_mores = batch % batch_cus

    cu_tasks = (per_cu_batchs+1) if batch_idx < cu_mores else per_cu_batchs
    cu_offs = 0 if batch_idx < cu_mores else cu_mores

    i_k: gl.constexpr = 0
    i_v: gl.constexpr = 0

    # Number of V heads per Q/K head (group size) - MUST be defined before use
    GROUP_SIZE: gl.constexpr = HV // H

    o_k = i_k * BK + gl.arange(0, BK, layout=blocked_k)
    o_v = i_v * BV + gl.arange(0, BV, layout=blocked_v)

    o_k_slice = i_k * BK + gl.arange(0, BK, layout=slice_bk)
    o_v_slice = i_v * BV + gl.arange(0, BV, layout=slice_bv)

    i_hv = i_h * GROUP_SIZE + gl.arange(0, GROUP_SIZE, layout=slice_group)
    i_hv_11 = i_h * GROUP_SIZE + gl.arange(0, GROUP_SIZE, layout=slice_group_11)

    b_A_log = gl.load(A_log + i_hv_11).to(gl.float32)
    b_dt_bias = gl.load(dt_bias + i_hv_11).to(gl.float32)

    k_dim_start = key_dim + i_h * K
    k_feats = k_dim_start + o_k
    
    for task_idx in range(cu_tasks):
        i_n = batch_idx * cu_tasks + task_idx + cu_offs

        # tl.device_print("", i_n)
        
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

        if idx_seq < batch:

            # Get conv state batch coordinate
            if IS_CONTINUOUS_BATCHING:
                conv_state_batch_coord = gl.load(
                    conv_state_indices_ptr + idx_seq * stride_state_indices
                ).to(gl.int64)
            else:
                conv_state_batch_coord = idx_seq
                
            if USE_PAD_SLOT:
                if conv_state_batch_coord != pad_slot_id:
                    b_h = gl.zeros([GROUP_SIZE, BK, BV], dtype=gl.float32, layout=blocked3d)
                    if USE_INITIAL_STATE:
                        idx = gl.load(h0_indices + i_n)
                        if idx >= 0:
                            # ================================================================
                            # Load initial hidden states for all V heads
                            # Shape: [GROUP_SIZE, BK, BV]
                            # ================================================================
                            p_h = (
                                h0_source
                                + idx * HV * K * V
                                + i_hv[:, None, None] * K * V
                                + o_k_slice[None, :, None] * V
                                + o_v_slice[None, None, :]
                            )
                            b_h = gl.load(p_h).to(gl.float32)  # [GROUP_SIZE, BK, BV]

                    # K conv setup (shared across all V heads)

                    
                    b_k_conv_states = ()
                    k_weights = (gl.load(conv_w_ptr + k_feats * stride_conv_w_dim),)
                    for i in gl.static_range(CONV_WIDTH-1):
                        b_k_conv_state = gl.load(conv_state_ptr + (conv_state_batch_coord * stride_conv_state_seq) + (k_feats * stride_conv_state_dim) + i * stride_conv_state_tok)
                        w_val = gl.load(conv_w_ptr + k_feats * stride_conv_w_dim + (i+1) * stride_conv_w_width)
                        k_weights = tuple_combine(k_weights, w_val)
                        b_k_conv_states = tuple_combine(b_k_conv_states, b_k_conv_state)
                    
                    # V conv setup (batched for all GROUP_SIZE V heads)
                    v_dim_start = 2 * key_dim + i_hv_11 * V
                    v_feats = v_dim_start[:, None] + o_v_slice[None, :]  # [GROUP_SIZE, BV]
                    
                    b_v_conv_states = ()
                    v_weights = (gl.load(conv_w_ptr + v_feats * stride_conv_w_dim),)
                    for j in gl.static_range(CONV_WIDTH-1):
                        b_v_conv_state = gl.load(conv_state_ptr + (conv_state_batch_coord * stride_conv_state_seq) + (v_feats * stride_conv_state_dim) + j * stride_conv_state_tok)
                        w_val = gl.load(conv_w_ptr + v_feats * stride_conv_w_dim + (j+1) * stride_conv_w_width)
                        v_weights = tuple_combine(v_weights, w_val)
                        b_v_conv_states = tuple_combine(b_v_conv_states, b_v_conv_state)

                    # Q conv setup (shared across all V heads)
                    q_dim_start = i_h * K
                    q_feats = q_dim_start + o_k
                    
                    b_q_conv_states = ()
                    q_weights = (gl.load(conv_w_ptr + q_feats * stride_conv_w_dim),)
                    for i in gl.static_range(CONV_WIDTH-1):
                        b_q_conv_state = gl.load(conv_state_ptr + (conv_state_batch_coord * stride_conv_state_seq) + (q_feats * stride_conv_state_dim) + i * stride_conv_state_tok)
                        w_val = gl.load(conv_w_ptr + q_feats * stride_conv_w_dim + (i+1) * stride_conv_w_width)
                        q_weights = tuple_combine(q_weights, w_val)
                        b_q_conv_states = tuple_combine(b_q_conv_states, b_q_conv_state)
                    
                    # ================================================================
                    # Main token processing loop
                    # Processing order: K → V (all heads) → Q → Delta Rule (all heads)
                    # ================================================================
                    for idx_token in gl.static_range(seqlen):
                        # ============================================================
                        # Step 1: Conv1D for K
                        # Shape: [BK]
                        # ============================================================
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
                        
                        b_k = k_conv_acc.to(gl.float32)  # [BK]
                        
                        if USE_QK_L2NORM_IN_KERNEL:
                            b_k = b_k / (gl.sqrt(gl.sum(b_k * b_k, axis=0) + 1e-6))
                        
                        # ============================================================
                        # Step 2: Conv1D for all V heads
                        # Shape: [GROUP_SIZE, BV]
                        # ============================================================
                        v_conv_acc = gl.load(conv_bias_ptr + v_feats).to(gl.float32) if HAS_CONV_BIAS else gl.zeros([GROUP_SIZE, BV], dtype=gl.float32, layout=blocked2d)
                        
                        v_ptrs = (
                            x_ptr + idx_seq * stride_x_seq 
                            + v_feats * stride_x_dim 
                            + idx_token * stride_x_token
                        )
                        b_v_conv_states = tuple_combine(b_v_conv_states, gl.load(v_ptrs))
                        
                        for j in gl.static_range(CONV_WIDTH):
                            v_conv_acc += b_v_conv_states[j] * v_weights[j]
                        
                        b_v_conv_states = b_v_conv_states[1:]
                        
                        if SILU_ACTIVATION:
                            v_conv_acc = v_conv_acc / (1 + gl.exp(-v_conv_acc))
                        
                        b_v = v_conv_acc.to(gl.float32)  # [GROUP_SIZE, BV]
                        
                        # ============================================================
                        # Step 3: Conv1D for Q
                        # Shape: [BK]
                        # ============================================================
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
                        
                        b_q = q_conv_acc.to(gl.float32)  # [BK]
                        
                        if USE_QK_L2NORM_IN_KERNEL:
                            b_q_scale = scale / (gl.sqrt(gl.sum(b_q * b_q, axis=0) + 1e-6))
                        else:
                            b_q_scale = scale
                        b_q = b_q * b_q_scale  # [BK]
                        
                        # ============================================================
                        # Step 4: Batched Delta Rule updates for all V heads
                        # Using broadcasting for efficient parallel processing
                        # ============================================================
                        
                        # Load time-variant gating parameters
                        p_a = a + (bos + idx_token) * HV + i_hv_11
                        p_b = b + (bos + idx_token) * HV + i_hv_11
                        b_a = gl.load(p_a).to(gl.float32)  # [GROUP_SIZE]
                        b_b = gl.load(p_b).to(gl.float32)  # [GROUP_SIZE]
                        
                        # Compute gating factors
                        x = b_a + b_dt_bias  # [GROUP_SIZE]
                        beta_x = softplus_beta * x
                        softplus_x = gl.where(
                            beta_x <= softplus_threshold,
                            (1.0 / softplus_beta) * gl.log(1.0 + gl.exp(beta_x)),
                            x,
                        )
                        b_g = -gl.exp(b_A_log) * softplus_x  # [GROUP_SIZE]
                        b_beta = 1.0 / (1.0 + gl.exp(-b_b))  # [GROUP_SIZE]

                        b_k = gl.convert_layout(b_k, layout=slice_bk)
                        b_v = gl.convert_layout(b_v, layout=slice_v)
                        b_q = gl.convert_layout(b_q, layout=slice_bk)
                        
                        # Batched Delta Rule recurrent update using broadcasting
                        # Step 4a: Apply exponential decay to hidden states
                        b_g = gl.convert_layout(b_g, layout=slice_group)
                        b_h *= gl.exp(b_g[:, None, None])  # [GROUP_SIZE, BK, BV]
                        
                        # Step 4b: Delta rule correction
                        b_v -= gl.sum(b_h * b_k[None, :, None], axis=1)  # [GROUP_SIZE, BV]
                        
                        # Step 4c: Apply beta gating
                        b_v *= b_beta[:, None]  # [GROUP_SIZE, BV]
                        
                        # Step 4d: Update hidden states with outer product
                        b_h += b_k[None, :, None] * b_v[:, None, :]  # [GROUP_SIZE, BK, BV]
                        
                        # Step 4e: Compute outputs for all V heads
                        b_o = gl.sum(b_h * b_q[None, :, None], axis=1)  # [GROUP_SIZE, BV]
                        
                        # Step 4f: Store outputs for all V heads
                        p_o = o + ((i_k * all + bos + idx_token) * HV + i_hv_11[:, None]) * V + o_v_slice[None, :]
                        gl.store(p_o, b_o.to(p_o.dtype.element_ty))
                        
                        # Step 4g: Store updated hidden states for all V heads
                        p_h0 = (
                            h0_source
                            + idx * HV * K * V
                            + i_hv[:, None, None] * K * V
                            + o_k_slice[None, :, None] * V
                            + o_v_slice[None, None, :]
                        )
                        gl.store(p_h0, b_h.to(p_h0.dtype.element_ty))

                    # ================================================================
                    # Write back final conv_state sliding windows to memory in
                    # ================================================================
                    q_feats_slice = i_h * K + o_k
                    k_feats_slice = key_dim + i_h * K + o_k
                    v_feats_slice = 2 * key_dim + i_hv_11[:, None] * V + o_v_slice[None, :]
                
                    # Write back V conv_states for all V heads
                    for i in gl.static_range(CONV_WIDTH-1):
                        gl.store(conv_state_ptr + (conv_state_batch_coord * stride_conv_state_seq) + (k_feats_slice * stride_conv_state_dim) + i * stride_conv_state_tok, b_k_conv_states[i])
                        gl.store(conv_state_ptr + (conv_state_batch_coord * stride_conv_state_seq) + (v_feats_slice * stride_conv_state_dim) + i * stride_conv_state_tok, b_v_conv_states[i])
                        gl.store(conv_state_ptr + (conv_state_batch_coord * stride_conv_state_seq) + (q_feats_slice * stride_conv_state_dim) + i * stride_conv_state_tok, b_q_conv_states[i])

        
def fused_gdn_fwd_decode_gluon(
    mixed_qkv,
    conv_state,
    conv_weight,
    A_log,
    a,
    dt_bias,
    b,
    ssm_state,
    key_dim,
    value_dim,
    num_heads_qk,
    num_heads_v,
    head_dim,
    conv_bias=None,
    activation="silu",
    conv_state_indices=None,
    ssm_state_indices=None,
    pad_slot_id=PAD_SLOT_ID,
    scale=None,
    use_qk_l2norm_in_kernel=True,
    softplus_beta=1.0,
    softplus_threshold=20.0,
    cu_seqlens=None,
):
    """Wrapper function for Gluon kernel."""
    batch, dim, seqlen = mixed_qkv.shape
    assert dim == 2 * key_dim + value_dim
    assert key_dim == num_heads_qk * head_dim
    assert value_dim == num_heads_v * head_dim
    
    _, conv_width = conv_weight.shape
    num_cache_lines, _, conv_state_len = conv_state.size()
    
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
    assert NK == 1
    
    o = mixed_qkv.new_empty(NK, B, T, HV, V)
    
    grid = lambda META: (NK, triton.cdiv(V, META['BV']), N * H)
    
    stride_state_indices = (
        conv_state_indices.stride(0) if conv_state_indices is not None else 0
    )
    np2_statelen = triton.next_power_of_2(conv_state_len)
    
    # Determine BV (can be tuned)
    BV = 16
    
    gluon_fused_gdn_fwd_decode_kernel[grid](
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
        BV=BV,
        HAS_CONV_BIAS=conv_bias is not None,
        CONV_WIDTH=conv_width,
        SILU_ACTIVATION=activation in ["silu", "swish"],
        IS_CONTINUOUS_BATCHING=conv_state_indices is not None,
        NP2_STATELEN=np2_statelen,
        USE_PAD_SLOT=pad_slot_id is not None,
        USE_INITIAL_STATE=ssm_state is not None,
        USE_QK_L2NORM_IN_KERNEL=use_qk_l2norm_in_kernel,
        IS_VARLEN=cu_seqlens is not None,
        num_warps=1
    )
    
    o = o.squeeze(0)
    return o


def fused_gdn_fwd_decode_gluon_v2(
    mixed_qkv,
    conv_state,
    conv_weight,
    A_log,
    a,
    dt_bias,
    b,
    ssm_state,
    key_dim,
    value_dim,
    num_heads_qk,
    num_heads_v,
    head_dim,
    conv_bias=None,
    activation="silu",
    conv_state_indices=None,
    ssm_state_indices=None,
    pad_slot_id=PAD_SLOT_ID,
    scale=None,
    use_qk_l2norm_in_kernel=True,
    softplus_beta=1.0,
    softplus_threshold=20.0,
    cu_seqlens=None,
):
    """
    Wrapper function for Gluon v2 kernel (V-indexed grid).
    
    Key difference from v1: Grid is indexed by V heads instead of Q/K heads.
    """
    
    batch, dim, seqlen = mixed_qkv.shape
    assert dim == 2 * key_dim + value_dim
    assert key_dim == num_heads_qk * head_dim
    assert value_dim == num_heads_v * head_dim
    
    _, conv_width = conv_weight.shape
    num_cache_lines, _, conv_state_len = conv_state.size()
    
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
    assert NK == 1
    
    o = mixed_qkv.new_empty(NK, B, T, HV, V)
    
    # v2 uses V-indexed grid: (num_k_blocks, num_v_blocks, batch * num_heads_v)
    grid = lambda META: (NK, triton.cdiv(V, META['BV']), N * HV)
    
    stride_state_indices = (
        conv_state_indices.stride(0) if conv_state_indices is not None else 0
    )
    np2_statelen = triton.next_power_of_2(conv_state_len)
    
    # Determine BV (can be tuned)
    BV = 32  # v2 might benefit from larger BV
    
    gluon_fused_gdn_fwd_decode_kernel_v2[grid](
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
        BV=BV,
        HAS_CONV_BIAS=conv_bias is not None,
        CONV_WIDTH=conv_width,
        SILU_ACTIVATION=activation in ["silu", "swish"],
        IS_CONTINUOUS_BATCHING=conv_state_indices is not None,
        NP2_STATELEN=np2_statelen,
        USE_PAD_SLOT=pad_slot_id is not None,
        USE_INITIAL_STATE=ssm_state is not None,
        USE_QK_L2NORM_IN_KERNEL=use_qk_l2norm_in_kernel,
        IS_VARLEN=cu_seqlens is not None,
        num_warps=1
    )
    
    o = o.squeeze(0)
    return o

def fused_gdn_fwd_decode_gluon_v3(
    mixed_qkv,
    conv_state,
    conv_weight,
    A_log,
    a,
    dt_bias,
    b,
    ssm_state,
    key_dim,
    value_dim,
    num_heads_qk,
    num_heads_v,
    head_dim,
    conv_bias=None,
    activation="silu",
    conv_state_indices=None,
    ssm_state_indices=None,
    pad_slot_id=PAD_SLOT_ID,
    scale=None,
    use_qk_l2norm_in_kernel=True,
    softplus_beta=1.0,
    softplus_threshold=20.0,
    cu_seqlens=None,
):
    """
    Wrapper function for Gluon v2 kernel (V-indexed grid).
    
    Key difference from v1: Grid is indexed by V heads instead of Q/K heads.
    """
    
    batch, dim, seqlen = mixed_qkv.shape
    assert dim == 2 * key_dim + value_dim
    assert key_dim == num_heads_qk * head_dim
    assert value_dim == num_heads_v * head_dim
    
    _, conv_width = conv_weight.shape
    num_cache_lines, _, conv_state_len = conv_state.size()
    
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
    assert NK == 1
    
    o = mixed_qkv.new_empty(NK, B, T, HV, V)
    
    # v2 uses V-indexed grid: (num_k_blocks, num_v_blocks, batch * num_heads_v)
    grid = lambda META: (NK, HV, N)
    
    stride_state_indices = (
        conv_state_indices.stride(0) if conv_state_indices is not None else 0
    )
    np2_statelen = triton.next_power_of_2(conv_state_len)
    
    # Determine BV (can be tuned)
    BV = 128  # v2 might benefit from larger BV
    
    gluon_fused_gdn_fwd_decode_kernel_v3[grid](
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
        BV=BV,
        HAS_CONV_BIAS=conv_bias is not None,
        CONV_WIDTH=conv_width,
        SILU_ACTIVATION=activation in ["silu", "swish"],
        IS_CONTINUOUS_BATCHING=conv_state_indices is not None,
        NP2_STATELEN=np2_statelen,
        USE_PAD_SLOT=pad_slot_id is not None,
        USE_INITIAL_STATE=ssm_state is not None,
        USE_QK_L2NORM_IN_KERNEL=use_qk_l2norm_in_kernel,
        IS_VARLEN=cu_seqlens is not None,
        num_warps=4
    )
    
    o = o.squeeze(0)
    return o

def fused_gdn_fwd_decode_gluon_v4(
    mixed_qkv,
    conv_state,
    conv_weight,
    A_log,
    a,
    dt_bias,
    b,
    ssm_state,
    key_dim,
    value_dim,
    num_heads_qk,
    num_heads_v,
    head_dim,
    conv_bias=None,
    activation="silu",
    conv_state_indices=None,
    ssm_state_indices=None,
    pad_slot_id=PAD_SLOT_ID,
    scale=None,
    use_qk_l2norm_in_kernel=True,
    softplus_beta=1.0,
    softplus_threshold=20.0,
    cu_seqlens=None,
):
    """
    Wrapper function for Gluon v2 kernel (V-indexed grid).
    
    Key difference from v1: Grid is indexed by V heads instead of Q/K heads.
    """
    
    batch, dim, seqlen = mixed_qkv.shape
    assert dim == 2 * key_dim + value_dim
    assert key_dim == num_heads_qk * head_dim
    assert value_dim == num_heads_v * head_dim
    
    _, conv_width = conv_weight.shape
    num_cache_lines, _, conv_state_len = conv_state.size()
    
    HV = num_heads_v
    K = head_dim
    V = head_dim
    H = num_heads_qk
    
    if scale is None:
        scale = K ** -0.5
    
    T = seqlen
    N = batch if cu_seqlens is None else len(cu_seqlens) - 1
    B = batch

    o = mixed_qkv.new_empty(B, T, HV, V)
    
    # v2 uses V-indexed grid: (num_k_blocks, num_v_blocks, batch * num_heads_v)
    grid = lambda META: (1, HV, N)
    
    stride_state_indices = (
        conv_state_indices.stride(0) if conv_state_indices is not None else 0
    )
    np2_statelen = triton.next_power_of_2(conv_state_len)
    
    # Determine BV (can be tuned)
    BK = triton.next_power_of_2(K) // 2
    BV = 128  # v2 might benefit from larger BV
    
    gluon_fused_gdn_fwd_decode_kernel_v4[grid](
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
        BV=BV,
        HAS_CONV_BIAS=conv_bias is not None,
        CONV_WIDTH=conv_width,
        SILU_ACTIVATION=activation in ["silu", "swish"],
        IS_CONTINUOUS_BATCHING=conv_state_indices is not None,
        NP2_STATELEN=np2_statelen,
        USE_PAD_SLOT=pad_slot_id is not None,
        USE_INITIAL_STATE=ssm_state is not None,
        USE_QK_L2NORM_IN_KERNEL=use_qk_l2norm_in_kernel,
        IS_VARLEN=cu_seqlens is not None,
        num_warps=4
    )
    
    return o

def fused_gdn_fwd_decode_gluon_v5(
    mixed_qkv,
    conv_state,
    conv_weight,
    A_log,
    a,
    dt_bias,
    b,
    ssm_state,
    key_dim,
    value_dim,
    num_heads_qk,
    num_heads_v,
    head_dim,
    conv_bias=None,
    activation="silu",
    conv_state_indices=None,
    ssm_state_indices=None,
    pad_slot_id=PAD_SLOT_ID,
    scale=None,
    use_qk_l2norm_in_kernel=True,
    softplus_beta=1.0,
    softplus_threshold=20.0,
    cu_seqlens=None,
):
    """
    Wrapper function for Gluon v2 kernel (V-indexed grid).
    
    Key difference from v1: Grid is indexed by V heads instead of Q/K heads.
    """
    
    batch, dim, seqlen = mixed_qkv.shape
    assert dim == 2 * key_dim + value_dim
    assert key_dim == num_heads_qk * head_dim
    assert value_dim == num_heads_v * head_dim
    
    _, conv_width = conv_weight.shape
    num_cache_lines, _, conv_state_len = conv_state.size()
    
    HV = num_heads_v
    K = head_dim
    V = head_dim
    H = num_heads_qk
    
    if scale is None:
        scale = K ** -0.5
    
    T = seqlen
    N = batch if cu_seqlens is None else len(cu_seqlens) - 1
    B = batch

    o = mixed_qkv.new_empty(B, T, HV, V)
    
    # v2 uses V-indexed grid: (num_k_blocks, num_v_blocks, batch * num_heads_v)
    grid = lambda META: (80,1,1)
    
    stride_state_indices = (
        conv_state_indices.stride(0) if conv_state_indices is not None else 0
    )
    np2_statelen = triton.next_power_of_2(conv_state_len)
    
    # Determine BV (can be tuned)
    BK = triton.next_power_of_2(K) // 2
    BV = 128  # v2 might benefit from larger BV

    print(f"@@@@@ {N=}, {B=}")
    
    gluon_fused_gdn_fwd_decode_kernel_v5[grid](
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
        BV=BV,
        HAS_CONV_BIAS=conv_bias is not None,
        CONV_WIDTH=conv_width,
        SILU_ACTIVATION=activation in ["silu", "swish"],
        IS_CONTINUOUS_BATCHING=conv_state_indices is not None,
        NP2_STATELEN=np2_statelen,
        USE_PAD_SLOT=pad_slot_id is not None,
        USE_INITIAL_STATE=ssm_state is not None,
        USE_QK_L2NORM_IN_KERNEL=use_qk_l2norm_in_kernel,
        IS_VARLEN=cu_seqlens is not None,
        num_warps=4
    )
    
    return o

def fused_gdn_fwd_decode_gluon_v6(
    mixed_qkv,
    conv_state,
    conv_weight,
    A_log,
    a,
    dt_bias,
    b,
    ssm_state,
    key_dim,
    value_dim,
    num_heads_qk,
    num_heads_v,
    head_dim,
    conv_bias=None,
    activation="silu",
    conv_state_indices=None,
    ssm_state_indices=None,
    pad_slot_id=PAD_SLOT_ID,
    scale=None,
    use_qk_l2norm_in_kernel=True,
    softplus_beta=1.0,
    softplus_threshold=20.0,
    cu_seqlens=None,
):
    """Wrapper function for Gluon kernel."""
    batch, dim, seqlen = mixed_qkv.shape
    assert dim == 2 * key_dim + value_dim
    assert key_dim == num_heads_qk * head_dim
    assert value_dim == num_heads_v * head_dim
    
    _, conv_width = conv_weight.shape
    num_cache_lines, _, conv_state_len = conv_state.size()
    
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
    assert NK == 1
    
    o = mixed_qkv.new_empty(NK, B, T, HV, V)
    
    grid = (80,)
    
    stride_state_indices = (
        conv_state_indices.stride(0) if conv_state_indices is not None else 0
    )
    np2_statelen = triton.next_power_of_2(conv_state_len)
    
    # Determine BV (can be tuned)
    BV = 128
    
    gluon_fused_gdn_fwd_decode_kernel_v6[grid](
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
        BV=BV,
        HAS_CONV_BIAS=conv_bias is not None,
        CONV_WIDTH=conv_width,
        SILU_ACTIVATION=activation in ["silu", "swish"],
        IS_CONTINUOUS_BATCHING=conv_state_indices is not None,
        NP2_STATELEN=np2_statelen,
        USE_PAD_SLOT=pad_slot_id is not None,
        USE_INITIAL_STATE=ssm_state is not None,
        USE_QK_L2NORM_IN_KERNEL=use_qk_l2norm_in_kernel,
        IS_VARLEN=cu_seqlens is not None,
        num_warps=4
    )
    
    o = o.squeeze(0)
    return o
