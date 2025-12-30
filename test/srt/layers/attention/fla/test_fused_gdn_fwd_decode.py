"""
Unit tests for Fused Gating Delta Network (GDN) Forward Decode Kernel.

Tests correctness and performance of the fused kernel against the reference
implementation that runs causal_conv1d_update_split_qkv and 
fused_sigmoid_gating_delta_rule_update separately.
"""

import pytest
import torch
import time

from sglang.srt.layers.attention.mamba.causal_conv1d_split_qkv import (
    causal_conv1d_update_split_qkv,
)
from sglang.srt.layers.attention.fla.fused_sigmoid_gating_recurrent import (
    fused_sigmoid_gating_delta_rule_update,
)
from sglang.srt.layers.attention.fla.fused_gdn_fwd_decode import (
    fused_gdn_fwd_decode,
    PAD_SLOT_ID,
)
from sglang.srt.layers.attention.fla.fused_gdn_fwd_decode_gluon import (
    gluon_fused_gdn_fwd_decode_kernel,
)
import triton


def gdn_fwd_decode_ref(
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
    conv_bias=None,
    activation="silu",
    conv_state_indices=None,
    ssm_state_indices=None,
    scale=None,
    use_qk_l2norm_in_kernel=True,
    softplus_beta=1.0,
    softplus_threshold=20.0,
    cu_seqlens=None,
):
    """
    Reference implementation using separate kernels.
    
    This mimics the behavior in hybrid_linear_attn_backend.py lines 278-326.
    """
    # Step 1: Causal Conv1D with split Q/K/V
    query, key, value = causal_conv1d_update_split_qkv(
        mixed_qkv,
        conv_state,
        conv_weight,
        key_dim=key_dim,
        value_dim=value_dim,
        bias=conv_bias,
        activation=activation,
        conv_state_indices=conv_state_indices,
        use_gluon=True,
    )
    
    # Reshape to match expected input format for gating delta rule
    batch, _, seqlen = query.shape
    
    query = query.view(batch, seqlen, num_heads_qk, head_dim)
    key = key.view(batch, seqlen, num_heads_qk, head_dim)
    value = value.view(batch, seqlen, num_heads_v, head_dim)
    
    # Step 2: Sigmoid gating delta rule update
    output = fused_sigmoid_gating_delta_rule_update(
        A_log=A_log,
        a=a,
        dt_bias=dt_bias,
        softplus_beta=softplus_beta,
        softplus_threshold=softplus_threshold,
        q=query,
        k=key,
        v=value,
        b=b,
        initial_state_source=ssm_state,
        initial_state_indices=ssm_state_indices,
        scale=scale,
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
        cu_seqlens=cu_seqlens,
    )
    
    return output


class TestFusedGDNFwdDecode:
    """Test suite for fused GDN forward decode kernel."""
    
    @pytest.fixture
    def device(self):
        if not torch.cuda.is_available():
            pytest.skip("CUDA device not available")
        return "cuda"
    
    @pytest.fixture
    def dtype(self):
        return torch.bfloat16
    
    def create_test_inputs(
        self,
        batch_size,
        key_dim,
        value_dim,
        num_heads_qk,
        num_heads_v,
        head_dim,
        seqlen,
        conv_width,
        device,
        dtype,
        has_bias=True,
    ):
        """Create test inputs for the GDN forward decode operation."""
        assert key_dim == num_heads_qk * head_dim
        assert value_dim == num_heads_v * head_dim
        
        dim = 2 * key_dim + value_dim
        HV = num_heads_v * head_dim
        
        # Conv1D inputs
        mixed_qkv = torch.randn(batch_size, dim, seqlen, device=device, dtype=dtype)
        conv_state = torch.randn(
            batch_size + 10, dim, conv_width - 1, device=device, dtype=dtype
        )
        conv_weight = torch.randn(dim, conv_width, device=device, dtype=dtype)
        conv_bias = torch.randn(dim, device=device, dtype=dtype) if has_bias else None
        
        # Gating inputs
        A_log = torch.randn(HV, device=device, dtype=torch.float32)
        a = torch.randn(batch_size, HV, device=device, dtype=dtype)
        dt_bias = torch.randn(HV, device=device, dtype=dtype)
        b = torch.randn(batch_size, HV, device=device, dtype=dtype)
        
        # SSM state
        ssm_state = torch.randn(
            batch_size + 10, HV, head_dim, head_dim,
            device=device, dtype=torch.float32
        )
        
        # Indices for continuous batching
        conv_state_indices = torch.arange(batch_size, device=device, dtype=torch.int32)
        ssm_state_indices = torch.arange(batch_size, device=device, dtype=torch.int32)
        
        return {
            "mixed_qkv": mixed_qkv,
            "conv_state": conv_state,
            "conv_weight": conv_weight,
            "conv_bias": conv_bias,
            "A_log": A_log,
            "a": a,
            "dt_bias": dt_bias,
            "b": b,
            "ssm_state": ssm_state,
            "conv_state_indices": conv_state_indices,
            "ssm_state_indices": ssm_state_indices,
            "key_dim": key_dim,
            "value_dim": value_dim,
            "num_heads_qk": num_heads_qk,
            "num_heads_v": num_heads_v,
            "head_dim": head_dim,
        }
    
    @pytest.mark.parametrize("batch_size", [1])
    @pytest.mark.parametrize("num_heads_qk", [4])
    @pytest.mark.parametrize("num_heads_v", [8])
    @pytest.mark.parametrize("head_dim", [128])
    @pytest.mark.parametrize("seqlen", [1])
    @pytest.mark.parametrize("conv_width", [4])
    @pytest.mark.parametrize("has_bias", [True])
    def test_correctness(
        self,
        batch_size,
        num_heads_qk,
        num_heads_v,
        head_dim,
        seqlen,
        conv_width,
        has_bias,
        device,
        dtype,
    ):
        """Test that fused kernel produces the same results as reference."""
        key_dim = num_heads_qk * head_dim
        value_dim = num_heads_v * head_dim
        
        # Create inputs
        inputs = self.create_test_inputs(
            batch_size, key_dim, value_dim, num_heads_qk, num_heads_v, head_dim,
            seqlen, conv_width, device, dtype, has_bias
        )
        
        # Clone states for separate runs
        conv_state_ref = inputs["conv_state"].clone()
        conv_state_fused = inputs["conv_state"].clone()
        ssm_state_ref = inputs["ssm_state"].clone()
        ssm_state_fused = inputs["ssm_state"].clone()
        
        # Run reference
        output_ref = gdn_fwd_decode_ref(
            mixed_qkv=inputs["mixed_qkv"].clone(),
            conv_state=conv_state_ref,
            conv_weight=inputs["conv_weight"],
            A_log=inputs["A_log"],
            a=inputs["a"],
            dt_bias=inputs["dt_bias"],
            b=inputs["b"],
            ssm_state=ssm_state_ref,
            key_dim=key_dim,
            value_dim=value_dim,
            num_heads_qk=num_heads_qk,
            num_heads_v=num_heads_v,
            head_dim=head_dim,
            conv_bias=inputs["conv_bias"],
            activation="silu",
            conv_state_indices=inputs["conv_state_indices"],
            ssm_state_indices=inputs["ssm_state_indices"],
            use_qk_l2norm_in_kernel=True,
            softplus_beta=1.0,
            softplus_threshold=20.0,
        )
        
        # Run fused kernel
        output_fused = fused_gdn_fwd_decode(
            mixed_qkv=inputs["mixed_qkv"].clone(),
            conv_state=conv_state_fused,
            conv_weight=inputs["conv_weight"],
            A_log=inputs["A_log"],
            a=inputs["a"],
            dt_bias=inputs["dt_bias"],
            b=inputs["b"],
            ssm_state=ssm_state_fused,
            key_dim=key_dim,
            value_dim=value_dim,
            num_heads_qk=num_heads_qk,
            num_heads_v=num_heads_v,
            head_dim=head_dim,
            conv_bias=inputs["conv_bias"],
            activation="silu",
            conv_state_indices=inputs["conv_state_indices"],
            ssm_state_indices=inputs["ssm_state_indices"],
            use_qk_l2norm_in_kernel=True,
            softplus_beta=1.0,
            softplus_threshold=20.0,
        )
        
        # Compare outputs
        rtol, atol = 1e-2, 5e-2 if dtype == torch.bfloat16 else (3e-3, 5e-3)
        
        # Check output match
        output_diff = (output_fused - output_ref).abs().max().item()
        print(f"\n[B={batch_size}, H_qk={num_heads_qk}, H_v={num_heads_v}, D={head_dim}]")
        print(f"  Output max diff: {output_diff:.6e}")
        
        assert torch.allclose(output_fused, output_ref, rtol=rtol, atol=atol), \
            f"Output mismatch: max diff = {output_diff}"
        
        # Check SSM state match
        ssm_state_diff = (ssm_state_fused - ssm_state_ref).abs().max().item()
        print(f"  SSM state max diff: {ssm_state_diff:.6e}")
        
        assert torch.allclose(ssm_state_fused, ssm_state_ref, rtol=rtol, atol=atol), \
            f"SSM state mismatch: max diff = {ssm_state_diff}"
        
        # Check conv_state match
        conv_state_diff = (conv_state_fused - conv_state_ref).abs().max().item()
        print(f"  Conv state max diff: {conv_state_diff:.6e}")
        
        assert torch.allclose(conv_state_fused, conv_state_ref, rtol=rtol, atol=atol), \
            f"Conv state mismatch: max diff = {conv_state_diff}"
        
        print(f"  ✓ Correctness test passed!")
    
    @pytest.mark.parametrize("batch_size", [1, 8, 32, 64])
    def test_decode_throughput(self, batch_size, device, dtype):
        """Test decode throughput with various batch sizes."""
        num_heads_qk = 4
        num_heads_v = 8
        head_dim = 128
        key_dim = num_heads_qk * head_dim
        value_dim = num_heads_v * head_dim
        seqlen = 1
        conv_width = 4
        
        # Create inputs
        inputs = self.create_test_inputs(
            batch_size, key_dim, value_dim, num_heads_qk, num_heads_v, head_dim,
            seqlen, conv_width, device, dtype, has_bias=True
        )
        
        # ====================================================================
        # Benchmark Reference (Separate Kernels)
        # ====================================================================
        
        # Warmup
        for _ in range(3):
            conv_state_tmp = inputs["conv_state"].clone()
            ssm_state_tmp = inputs["ssm_state"].clone()
            _ = gdn_fwd_decode_ref(
                mixed_qkv=inputs["mixed_qkv"].clone(),
                conv_state=conv_state_tmp,
                conv_weight=inputs["conv_weight"],
                A_log=inputs["A_log"],
                a=inputs["a"],
                dt_bias=inputs["dt_bias"],
                b=inputs["b"],
                ssm_state=ssm_state_tmp,
                key_dim=key_dim,
                value_dim=value_dim,
                num_heads_qk=num_heads_qk,
                num_heads_v=num_heads_v,
                head_dim=head_dim,
                conv_bias=inputs["conv_bias"],
                activation="silu",
                conv_state_indices=inputs["conv_state_indices"],
                ssm_state_indices=inputs["ssm_state_indices"],
                use_qk_l2norm_in_kernel=True,
            )
        torch.cuda.synchronize()
        
        # Benchmark
        num_iters = 100
        start = time.time()
        for _ in range(num_iters):
            conv_state_tmp = inputs["conv_state"].clone()
            ssm_state_tmp = inputs["ssm_state"].clone()
            _ = gdn_fwd_decode_ref(
                mixed_qkv=inputs["mixed_qkv"].clone(),
                conv_state=conv_state_tmp,
                conv_weight=inputs["conv_weight"],
                A_log=inputs["A_log"],
                a=inputs["a"],
                dt_bias=inputs["dt_bias"],
                b=inputs["b"],
                ssm_state=ssm_state_tmp,
                key_dim=key_dim,
                value_dim=value_dim,
                num_heads_qk=num_heads_qk,
                num_heads_v=num_heads_v,
                head_dim=head_dim,
                conv_bias=inputs["conv_bias"],
                activation="silu",
                conv_state_indices=inputs["conv_state_indices"],
                ssm_state_indices=inputs["ssm_state_indices"],
                use_qk_l2norm_in_kernel=True,
            )
        torch.cuda.synchronize()
        ref_time = (time.time() - start) / num_iters * 1000  # ms
        
        # ====================================================================
        # Benchmark Fused Kernel
        # ====================================================================
        
        # Warmup
        for _ in range(3):
            conv_state_tmp = inputs["conv_state"].clone()
            ssm_state_tmp = inputs["ssm_state"].clone()
            _ = fused_gdn_fwd_decode(
                mixed_qkv=inputs["mixed_qkv"].clone(),
                conv_state=conv_state_tmp,
                conv_weight=inputs["conv_weight"],
                A_log=inputs["A_log"],
                a=inputs["a"],
                dt_bias=inputs["dt_bias"],
                b=inputs["b"],
                ssm_state=ssm_state_tmp,
                key_dim=key_dim,
                value_dim=value_dim,
                num_heads_qk=num_heads_qk,
                num_heads_v=num_heads_v,
                head_dim=head_dim,
                conv_bias=inputs["conv_bias"],
                activation="silu",
                conv_state_indices=inputs["conv_state_indices"],
                ssm_state_indices=inputs["ssm_state_indices"],
                use_qk_l2norm_in_kernel=True,
            )
        torch.cuda.synchronize()
        
        # Benchmark
        start = time.time()
        for _ in range(num_iters):
            conv_state_tmp = inputs["conv_state"].clone()
            ssm_state_tmp = inputs["ssm_state"].clone()
            _ = fused_gdn_fwd_decode(
                mixed_qkv=inputs["mixed_qkv"].clone(),
                conv_state=conv_state_tmp,
                conv_weight=inputs["conv_weight"],
                A_log=inputs["A_log"],
                a=inputs["a"],
                dt_bias=inputs["dt_bias"],
                b=inputs["b"],
                ssm_state=ssm_state_tmp,
                key_dim=key_dim,
                value_dim=value_dim,
                num_heads_qk=num_heads_qk,
                num_heads_v=num_heads_v,
                head_dim=head_dim,
                conv_bias=inputs["conv_bias"],
                activation="silu",
                conv_state_indices=inputs["conv_state_indices"],
                ssm_state_indices=inputs["ssm_state_indices"],
                use_qk_l2norm_in_kernel=True,
            )
        torch.cuda.synchronize()
        fused_time = (time.time() - start) / num_iters * 1000  # ms
        
        # Calculate metrics
        speedup = ref_time / fused_time
        throughput_ref = (num_iters * batch_size) / (ref_time * num_iters / 1000)
        throughput_fused = (num_iters * batch_size) / (fused_time * num_iters / 1000)
        
        print(f"\n{'='*70}")
        print(f"Decode Throughput Test: batch_size={batch_size}")
        print(f"{'='*70}")
        print(f"Configuration:")
        print(f"  - num_heads_qk: {num_heads_qk}")
        print(f"  - num_heads_v:  {num_heads_v}")
        print(f"  - head_dim:     {head_dim}")
        print(f"  - seqlen:       {seqlen}")
        print(f"  - dtype:        {dtype}")
        print(f"\nPerformance Results (averaged over {num_iters} iterations):")
        print(f"\n  Reference (Separate Kernels):")
        print(f"    - Time per iteration:  {ref_time:.4f} ms")
        print(f"    - Throughput:          {throughput_ref:.2f} tokens/s")
        print(f"\n  Fused Kernel:")
        print(f"    - Time per iteration:  {fused_time:.4f} ms")
        print(f"    - Throughput:          {throughput_fused:.2f} tokens/s")
        print(f"\n  Performance Comparison:")
        print(f"    - Speedup (Fused/Reference): {speedup:.2f}x")
        print(f"    - Time saved:                {ref_time - fused_time:.4f} ms")
        
        if speedup > 1.05:
            print(f"    - Status:                    ✓ Fused kernel is {speedup:.2f}x FASTER")
        elif speedup < 0.95:
            print(f"    - Status:                    ⚠ Reference is {1/speedup:.2f}x FASTER")
        else:
            print(f"    - Status:                    ≈ Performance is similar")
        
        print(f"{'='*70}\n")
    
    @pytest.mark.parametrize("activation", ["silu", None])
    @pytest.mark.parametrize("use_qk_l2norm", [True, False])
    def test_different_configs(self, activation, use_qk_l2norm, device, dtype):
        """Test different activation and normalization configurations."""
        batch_size = 16
        num_heads_qk = 4
        num_heads_v = 8
        head_dim = 128
        key_dim = num_heads_qk * head_dim
        value_dim = num_heads_v * head_dim
        seqlen = 1
        conv_width = 4
        
        # Create inputs
        inputs = self.create_test_inputs(
            batch_size, key_dim, value_dim, num_heads_qk, num_heads_v, head_dim,
            seqlen, conv_width, device, dtype, has_bias=True
        )
        
        # Clone states
        conv_state_ref = inputs["conv_state"].clone()
        conv_state_fused = inputs["conv_state"].clone()
        ssm_state_ref = inputs["ssm_state"].clone()
        ssm_state_fused = inputs["ssm_state"].clone()
        
        # Run reference
        output_ref = gdn_fwd_decode_ref(
            mixed_qkv=inputs["mixed_qkv"].clone(),
            conv_state=conv_state_ref,
            conv_weight=inputs["conv_weight"],
            A_log=inputs["A_log"],
            a=inputs["a"],
            dt_bias=inputs["dt_bias"],
            b=inputs["b"],
            ssm_state=ssm_state_ref,
            key_dim=key_dim,
            value_dim=value_dim,
            num_heads_qk=num_heads_qk,
            num_heads_v=num_heads_v,
            head_dim=head_dim,
            conv_bias=inputs["conv_bias"],
            activation=activation,
            conv_state_indices=inputs["conv_state_indices"],
            ssm_state_indices=inputs["ssm_state_indices"],
            use_qk_l2norm_in_kernel=use_qk_l2norm,
        )
        
        # Run fused
        output_fused = fused_gdn_fwd_decode(
            mixed_qkv=inputs["mixed_qkv"].clone(),
            conv_state=conv_state_fused,
            conv_weight=inputs["conv_weight"],
            A_log=inputs["A_log"],
            a=inputs["a"],
            dt_bias=inputs["dt_bias"],
            b=inputs["b"],
            ssm_state=ssm_state_fused,
            key_dim=key_dim,
            value_dim=value_dim,
            num_heads_qk=num_heads_qk,
            num_heads_v=num_heads_v,
            head_dim=head_dim,
            conv_bias=inputs["conv_bias"],
            activation=activation,
            conv_state_indices=inputs["conv_state_indices"],
            ssm_state_indices=inputs["ssm_state_indices"],
            use_qk_l2norm_in_kernel=use_qk_l2norm,
        )
        
        # Compare
        rtol, atol = 1e-2, 5e-2
        assert torch.allclose(output_fused, output_ref, rtol=rtol, atol=atol)
        
        print(f"✓ Config test passed: activation={activation}, l2norm={use_qk_l2norm}")


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
    )
    
    o = o.squeeze(0)
    return o


class TestGluonFusedGDNFwdDecode:
    """Test suite for Gluon version of Fused GDN Forward Decode kernel."""
    
    @pytest.fixture
    def device(self):
        if not torch.cuda.is_available():
            pytest.skip("CUDA device not available")
        return "cuda"
    
    @pytest.fixture
    def dtype(self):
        return torch.bfloat16
    
    def create_test_inputs(
        self,
        batch_size,
        key_dim,
        value_dim,
        num_heads_qk,
        num_heads_v,
        head_dim,
        seqlen,
        conv_width,
        device,
        dtype,
        has_bias=True,
    ):
        """Create test inputs for the GDN forward decode operation."""
        assert key_dim == num_heads_qk * head_dim
        assert value_dim == num_heads_v * head_dim
        
        dim = 2 * key_dim + value_dim
        HV = num_heads_v * head_dim
        
        mixed_qkv = torch.randn(batch_size, dim, seqlen, device=device, dtype=dtype)
        conv_state = torch.randn(
            batch_size + 10, dim, conv_width - 1, device=device, dtype=dtype
        )
        conv_weight = torch.randn(dim, conv_width, device=device, dtype=dtype)
        conv_bias = torch.randn(dim, device=device, dtype=dtype) if has_bias else None
        
        A_log = torch.randn(HV, device=device, dtype=torch.float32)
        a = torch.randn(batch_size, HV, device=device, dtype=dtype)
        dt_bias = torch.randn(HV, device=device, dtype=dtype)
        b = torch.randn(batch_size, HV, device=device, dtype=dtype)
        
        ssm_state = torch.randn(
            batch_size + 10, HV, head_dim, head_dim,
            device=device, dtype=torch.float32
        )
        
        conv_state_indices = torch.arange(batch_size, device=device, dtype=torch.int32)
        ssm_state_indices = torch.arange(batch_size, device=device, dtype=torch.int32)
        
        return {
            "mixed_qkv": mixed_qkv,
            "conv_state": conv_state,
            "conv_weight": conv_weight,
            "conv_bias": conv_bias,
            "A_log": A_log,
            "a": a,
            "dt_bias": dt_bias,
            "b": b,
            "ssm_state": ssm_state,
            "conv_state_indices": conv_state_indices,
            "ssm_state_indices": ssm_state_indices,
            "key_dim": key_dim,
            "value_dim": value_dim,
            "num_heads_qk": num_heads_qk,
            "num_heads_v": num_heads_v,
            "head_dim": head_dim,
        }
    
    @pytest.mark.parametrize("batch_size", [1])
    @pytest.mark.parametrize("num_heads_qk", [4])
    @pytest.mark.parametrize("num_heads_v", [8])
    @pytest.mark.parametrize("head_dim", [128])
    @pytest.mark.parametrize("seqlen", [1])
    @pytest.mark.parametrize("conv_width", [4])
    @pytest.mark.parametrize("has_bias", [True])
    def test_gluon_vs_reference(
        self,
        batch_size,
        num_heads_qk,
        num_heads_v,
        head_dim,
        seqlen,
        conv_width,
        has_bias,
        device,
        dtype,
    ):
        """Test that Gluon kernel produces the same results as reference."""
        key_dim = num_heads_qk * head_dim
        value_dim = num_heads_v * head_dim
        
        inputs = self.create_test_inputs(
            batch_size, key_dim, value_dim, num_heads_qk, num_heads_v, head_dim,
            seqlen, conv_width, device, dtype, has_bias
        )
        
        # Clone inputs for each run
        inputs_ref = {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
        inputs_gluon = {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
        
        # Run reference implementation
        output_ref = gdn_fwd_decode_ref(**inputs_ref)
        conv_state_ref = inputs_ref["conv_state"]
        
        # Run Gluon implementation
        output_gluon = fused_gdn_fwd_decode_gluon(**inputs_gluon)
        conv_state_gluon = inputs_gluon["conv_state"]
        
        # Compare outputs
        rtol, atol = 1e-2, 5e-2
        assert torch.allclose(output_gluon, output_ref, rtol=rtol, atol=atol), \
            f"Output mismatch: max diff = {(output_gluon - output_ref).abs().max()}"
        
        # Compare conv_states
        assert torch.allclose(conv_state_gluon, conv_state_ref, rtol=rtol, atol=atol), \
            f"Conv state mismatch: max diff = {(conv_state_gluon - conv_state_ref).abs().max()}"
        
        print(f"✓ Gluon kernel test passed")
    
    @pytest.mark.parametrize("batch_size", [1])
    @pytest.mark.parametrize("num_heads_qk", [4])
    @pytest.mark.parametrize("num_heads_v", [8])
    @pytest.mark.parametrize("head_dim", [128])
    @pytest.mark.parametrize("seqlen", [1])
    @pytest.mark.parametrize("conv_width", [4])
    @pytest.mark.parametrize("has_bias", [True])
    def test_gluon_vs_triton(
        self,
        batch_size,
        num_heads_qk,
        num_heads_v,
        head_dim,
        seqlen,
        conv_width,
        has_bias,
        device,
        dtype,
    ):
        """Test that Gluon kernel produces the same results as Triton kernel."""
        key_dim = num_heads_qk * head_dim
        value_dim = num_heads_v * head_dim
        
        inputs = self.create_test_inputs(
            batch_size, key_dim, value_dim, num_heads_qk, num_heads_v, head_dim,
            seqlen, conv_width, device, dtype, has_bias
        )
        
        # Clone inputs for each run
        inputs_triton = {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
        inputs_gluon = {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
        
        # Run Triton implementation
        output_triton = fused_gdn_fwd_decode(**inputs_triton)
        conv_state_triton = inputs_triton["conv_state"]
        
        # Run Gluon implementation
        output_gluon = fused_gdn_fwd_decode_gluon(**inputs_gluon)
        conv_state_gluon = inputs_gluon["conv_state"]
        
        # Compare outputs
        rtol, atol = 1e-3, 1e-3
        assert torch.allclose(output_gluon, output_triton, rtol=rtol, atol=atol), \
            f"Output mismatch: max diff = {(output_gluon - output_triton).abs().max()}"
        
        # Compare conv_states
        assert torch.allclose(conv_state_gluon, conv_state_triton, rtol=rtol, atol=atol), \
            f"Conv state mismatch: max diff = {(conv_state_gluon - conv_state_triton).abs().max()}"
        
        print(f"✓ Gluon vs Triton test passed")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])

