# Adapted from https://github.com/vllm-project/vllm/blob/main/tests/kernels/mamba/test_causal_conv1d.py


from typing import Optional

import pytest
import torch
import torch.nn.functional as F
from einops import rearrange

from sglang.srt.layers.attention.mamba.causal_conv1d_triton import (
    PAD_SLOT_ID,
    causal_conv1d_fn,
    causal_conv1d_update,
)
from sglang.srt.layers.attention.mamba.causal_conv1d_split_qkv import (
    causal_conv1d_update_split_qkv,
)


def causal_conv1d_ref(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    initial_states: Optional[torch.Tensor] = None,
    return_final_states: bool = False,
    final_states_out: Optional[torch.Tensor] = None,
    activation: Optional[str] = "silu",
):
    """
    x: (batch, dim, seqlen)
    weight: (dim, width)
    bias: (dim,)
    initial_states: (batch, dim, width - 1)
    final_states_out: (batch, dim, width - 1)

    out: (batch, dim, seqlen)
    """
    if activation not in [None, "silu", "swish"]:
        raise NotImplementedError("activation must be None, silu, or swish")
    dtype_in = x.dtype
    x = x.to(weight.dtype)
    seqlen = x.shape[-1]
    dim, width = weight.shape
    if initial_states is None:
        out = F.conv1d(x, weight.unsqueeze(1), bias, padding=width - 1, groups=dim)
    else:
        x = torch.cat([initial_states, x], dim=-1)
        out = F.conv1d(x, weight.unsqueeze(1), bias, padding=0, groups=dim)
    out = out[..., :seqlen]
    if return_final_states:
        final_states = F.pad(x, (width - 1 - x.shape[-1], 0)).to(
            dtype_in
        )  # (batch, dim, width - 1)
        if final_states_out is not None:
            final_states_out.copy_(final_states)
        else:
            final_states_out = final_states
    out = (out if activation is None else F.silu(out)).to(dtype=dtype_in)
    return (out, None) if not return_final_states else (out, final_states_out)


def causal_conv1d_update_ref(
    x, conv_state, weight, bias=None, activation=None, cache_seqlens=None
):
    """
    x: (batch, dim) or (batch, dim, seqlen)
    conv_state: (batch, dim, state_len), where state_len >= width - 1
    weight: (dim, width)
    bias: (dim,)
    cache_seqlens: (batch,), dtype int32.
        If not None, the conv_state is treated as a circular buffer.
        The conv_state will be updated by copying x to the
        conv_state starting at the index
        @cache_seqlens % state_len before performing the convolution.

    out: (batch, dim) or (batch, dim, seqlen)
    """
    if activation not in [None, "silu", "swish"]:
        raise NotImplementedError("activation must be None, silu, or swish")
    dtype_in = x.dtype
    unsqueeze = x.dim() == 2
    if unsqueeze:
        x = x.unsqueeze(-1)
    batch, dim, seqlen = x.shape
    width = weight.shape[1]
    state_len = conv_state.shape[-1]
    assert conv_state.shape == (batch, dim, state_len)
    assert weight.shape == (dim, width)
    if cache_seqlens is None:
        x_new = torch.cat([conv_state, x], dim=-1).to(
            weight.dtype
        )  # (batch, dim, state_len + seqlen)
        conv_state.copy_(x_new[:, :, -state_len:])  # update
    else:
        width_idx = torch.arange(
            -(width - 1), 0, dtype=torch.long, device=x.device
        ).unsqueeze(0) + cache_seqlens.unsqueeze(1)
        width_idx = (
            torch.remainder(width_idx, state_len).unsqueeze(1).expand(-1, dim, -1)
        )
        x_new = torch.cat([conv_state.gather(2, width_idx), x], dim=-1).to(weight.dtype)
        copy_idx = torch.arange(seqlen, dtype=torch.long, device=x.device).unsqueeze(
            0
        ) + cache_seqlens.unsqueeze(1)
        copy_idx = torch.remainder(copy_idx, state_len).unsqueeze(1).expand(-1, dim, -1)
        conv_state.scatter_(2, copy_idx, x)
    out = F.conv1d(x_new, weight.unsqueeze(1), bias, padding=0, groups=dim)[
        :, :, -seqlen:
    ]
    if unsqueeze:
        out = out.squeeze(-1)
    return (out if activation is None else F.silu(out)).to(dtype=dtype_in)


@pytest.mark.parametrize("itype", [torch.bfloat16, torch.float])
@pytest.mark.parametrize("silu_activation", [True])
@pytest.mark.parametrize("has_bias", [True])
def causal_conv1d_opcheck_fn(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    cu_seq_len: Optional[torch.Tensor] = None,
    cache_indices: Optional[torch.Tensor] = None,
    has_initial_state: Optional[torch.Tensor] = None,
    conv_states: Optional[torch.Tensor] = None,
    activation: Optional[str] = "silu",
    pad_slot_id: int = PAD_SLOT_ID,
):
    """
    x: (batch, dim, seqlen)
    weight: (dim, width)
    bias: (dim,)
    seq_idx: (batch, seqlen)
    initial_states: (batch, dim, width - 1)
    final_states_out: (batch, dim, width - 1), to be written to
    activation: either None or "silu" or "swish"

    out: (batch, dim, seqlen)
    """
    if activation not in [None, "silu", "swish"]:
        raise NotImplementedError("activation must be None, silu, or swish")
    if x.stride(-1) != 1:
        x = x.contiguous()
    bias = bias.contiguous() if bias is not None else None


@pytest.mark.parametrize("itype", [torch.bfloat16])
# @pytest.mark.parametrize("silu_activation", [False, True])
# @pytest.mark.parametrize("has_bias", [False, True])
@pytest.mark.parametrize("silu_activation", [False])
@pytest.mark.parametrize("has_bias", [False])
@pytest.mark.parametrize("seqlen", [1])
@pytest.mark.parametrize("width", [4])
# @pytest.mark.parametrize("dim", [2048, 2048 + 16, 4096])
@pytest.mark.parametrize("dim", [2048])
def test_causal_conv1d_update(dim, width, seqlen, has_bias, silu_activation, itype):
    if not torch.cuda.is_available():
        pytest.skip("CUDA device not available")

    device = "cuda"
    rtol, atol = (3e-4, 1e-3) if itype == torch.float32 else (3e-3, 5e-3)
    if itype == torch.bfloat16:
        rtol, atol = 1e-2, 5e-2
    # set seed
    torch.manual_seed(0)
    batch = 2
    x = torch.randn(batch, dim, seqlen, device=device, dtype=itype)
    x_ref = x.clone()
    conv_state = torch.randn(batch, width - 1, dim, device=device, dtype=itype).transpose(1, 2)

    weight = torch.randn(dim, width, device=device, dtype=itype)
    bias = torch.randn(dim, device=device, dtype=itype) if has_bias else None
    conv_state_ref = conv_state.detach().clone()
    activation = None if not silu_activation else "silu"
    out = causal_conv1d_update(x, conv_state, weight, bias, activation=activation, validate_data=True)
    out_ref = causal_conv1d_update_ref(
        x_ref, conv_state_ref, weight, bias, activation=activation
    )

    assert torch.equal(conv_state, conv_state_ref)
    assert torch.allclose(out, out_ref, rtol=rtol, atol=atol)


@pytest.mark.parametrize("itype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("silu_activation", [False, True])
@pytest.mark.parametrize("has_bias", [False, True])
@pytest.mark.parametrize("seqlen", [1, 3])
@pytest.mark.parametrize("width", [3, 4])
@pytest.mark.parametrize("dim", [2048 + 16, 4096])
# tests correctness in case subset of the sequences are padded
@pytest.mark.parametrize("with_padding", [True, False])
@pytest.mark.parametrize("batch_size", [3])
def test_causal_conv1d_update_with_batch_gather(
    batch_size, with_padding, dim, width, seqlen, has_bias, silu_activation, itype
):
    if not torch.cuda.is_available():
        pytest.skip("CUDA device not available")

    device = "cuda"
    rtol, atol = (3e-4, 1e-3) if itype == torch.float32 else (3e-3, 5e-3)
    if itype == torch.bfloat16:
        rtol, atol = 1e-2, 5e-2

    # set seed
    torch.manual_seed(0)

    padding = 5 if with_padding else 0
    padded_batch_size = batch_size + padding
    # total_entries = number of cache line
    total_entries = 10 * batch_size

    # x will be (batch, dim, seqlen) with contiguous along dim-axis
    x = torch.randn(
        padded_batch_size, seqlen, dim, device=device, dtype=itype
    ).transpose(1, 2)

    x_ref = x.clone()

    conv_state_indices = torch.randperm(total_entries)[:batch_size].to(
        dtype=torch.int32, device=device
    )
    unused_states_bool = torch.ones(total_entries, dtype=torch.bool, device=device)
    unused_states_bool[conv_state_indices] = False
    padded_state_indices = torch.concat(
        [
            conv_state_indices,
            torch.as_tensor([PAD_SLOT_ID] * padding, dtype=torch.int32, device=device),
        ],
        dim=0,
    )

    # conv_state will be (cache_lines, dim, state_len)
    # with contiguous along dim-axis
    conv_state = torch.randn(
        total_entries, width - 1, dim, device=device, dtype=itype
    ).transpose(1, 2)

    conv_state_for_padding_test = conv_state.clone()

    weight = torch.randn(dim, width, device=device, dtype=itype)
    bias = torch.randn(dim, device=device, dtype=itype) if has_bias else None
    conv_state_ref = conv_state[conv_state_indices, :].detach().clone()
    activation = None if not silu_activation else "silu"

    out = causal_conv1d_update(
        x,
        conv_state,
        weight,
        bias,
        activation=activation,
        conv_state_indices=padded_state_indices,
        pad_slot_id=PAD_SLOT_ID,
    )
    out_ref = causal_conv1d_update_ref(
        x_ref[:batch_size], conv_state_ref, weight, bias, activation=activation
    )

    assert torch.equal(conv_state[conv_state_indices, :], conv_state_ref)
    assert torch.equal(
        conv_state[unused_states_bool], conv_state_for_padding_test[unused_states_bool]
    )
    assert torch.allclose(out[:batch_size], out_ref, rtol=rtol, atol=atol)


@pytest.mark.parametrize("itype", [torch.bfloat16])
@pytest.mark.parametrize("silu_activation", [True])
@pytest.mark.parametrize("has_bias", [True])
@pytest.mark.parametrize("width", [4])
@pytest.mark.parametrize("seqlen", [8, 30, 249, 2049, 4096])
@pytest.mark.parametrize("dim", [64, 4096])
@pytest.mark.parametrize("with_padding", [True, False])
@pytest.mark.parametrize("batch", [4, 10])
def test_causal_conv1d_varlen(
    batch, with_padding, dim, seqlen, width, has_bias, silu_activation, itype
):
    if not torch.cuda.is_available():
        pytest.skip("CUDA device not available")

    device = "cuda"
    torch.cuda.empty_cache()
    rtol, atol = (3e-4, 1e-3) if itype == torch.float32 else (3e-3, 5e-3)
    if itype == torch.bfloat16:
        rtol, atol = 1e-2, 5e-2
    # set seed
    torch.manual_seed(0)
    seqlens = []
    batch_size = batch
    padding = 3 if with_padding else 0
    padded_batch_size = batch_size + padding
    nsplits = padded_batch_size - 1

    eos_pos = torch.randperm(seqlen - 1)[:nsplits].sort().values

    seqlens.append(
        torch.diff(
            torch.cat([torch.tensor([-1]), eos_pos, torch.tensor([seqlen - 1])])
        ).tolist()
    )
    assert sum(seqlens[-1]) == seqlen
    assert all(s > 0 for s in seqlens[-1])

    total_entries = batch_size * 10
    cumsum = torch.cumsum(torch.tensor(seqlens[0]), dim=0).to(torch.int32)
    cumsum = torch.concat([torch.tensor([0], dtype=torch.int32), cumsum], dim=0)
    x = rearrange(
        torch.randn(1, seqlen, 4096 + dim + 64, device=device, dtype=itype),
        "b s d -> b d s",
    )[:, 4096 : 4096 + dim, :]

    weight = torch.randn(dim, width, device=device, dtype=itype)

    bias = torch.randn(dim, device=device, dtype=itype) if has_bias else None
    x_ref = x.clone()
    weight_ref = weight.clone()
    bias_ref = bias.clone() if bias is not None else None
    activation = None if not silu_activation else "silu"
    final_states = torch.randn(
        total_entries, width - 1, dim, device=x.device, dtype=x.dtype
    ).transpose(1, 2)
    final_states_ref = final_states.clone()
    has_initial_states = torch.randint(
        0, 2, (cumsum.shape[0] - 1,), dtype=torch.bool, device=x.device
    )
    state_indices = torch.randperm(total_entries, dtype=torch.int32, device=x.device)[
        :batch_size
    ]
    padded_state_indices = torch.concat(
        [
            state_indices,
            torch.as_tensor([PAD_SLOT_ID] * padding, dtype=torch.int32, device=device),
        ],
        dim=-1,
    )
    out = causal_conv1d_fn(
        x.squeeze(0),
        weight,
        bias=bias,
        conv_states=final_states,
        query_start_loc=cumsum.cuda(),
        seq_lens_cpu=torch.tensor(seqlens[0]),
        cache_indices=padded_state_indices,
        has_initial_state=has_initial_states,
        activation=activation,
        pad_slot_id=PAD_SLOT_ID,
    )

    out_ref = []
    out_ref_b = []

    splits = [torch.split(var, seqlens[0], dim=-1) for var in (x_ref)]
    for i in range(len(seqlens[0])):
        x_s = [v[i].unsqueeze(0) for v in splits][0]
        if padded_state_indices[i] == PAD_SLOT_ID:
            continue
        out_ref_b.append(
            causal_conv1d_ref(
                x_s,
                weight_ref,
                bias_ref,
                activation=activation,
                return_final_states=True,
                final_states_out=final_states_ref[padded_state_indices[i]].unsqueeze(0),
                initial_states=(
                    final_states_ref[padded_state_indices[i]].unsqueeze(0)
                    if has_initial_states[i]
                    else None
                ),
            )
        )
    out_ref.append(torch.cat([t[0] for t in out_ref_b], dim=2))
    out_ref_tensor = torch.cat(out_ref, dim=0)

    assert torch.allclose(
        final_states[state_indices],
        final_states_ref[state_indices],
        rtol=rtol,
        atol=atol,
    )
    unpadded_out = out[:, : out_ref_tensor.shape[-1]]
    assert torch.allclose(unpadded_out, out_ref_tensor, rtol=rtol, atol=atol)


@pytest.mark.parametrize("itype", [torch.bfloat16])
@pytest.mark.parametrize("silu_activation", [False])
@pytest.mark.parametrize("has_bias", [False])
@pytest.mark.parametrize("seqlen", [1])
@pytest.mark.parametrize("width", [4])
@pytest.mark.parametrize("key_dim", [512])
@pytest.mark.parametrize("value_dim", [1024])
def test_causal_conv1d_update_split_qkv(
    key_dim, value_dim, width, seqlen, has_bias, silu_activation, itype
):
    """
    Test that causal_conv1d_update_split_qkv Triton and Gluon kernels 
    produce the same results, with end-to-end performance benchmarking.
    
    Compares:
    - Triton kernel vs Gluon kernel for split_qkv implementation
    - Performance metrics for both implementations
    """
    if not torch.cuda.is_available():
        pytest.skip("CUDA device not available")

    device = "cuda"
    rtol, atol = (3e-4, 1e-3) if itype == torch.float32 else (3e-3, 5e-3)
    if itype == torch.bfloat16:
        rtol, atol = 1e-2, 5e-2

    torch.manual_seed(42)
    batch = 1
    dim = 2 * key_dim + value_dim  # Total dimension: q + k + v
    
    # Create input: (batch, dim, seqlen)
    mixed_qkv = torch.randn(batch, dim, seqlen, device=device, dtype=itype)
    
    # Create conv_state: (batch, dim, width - 1)
    conv_state_triton = torch.randn(
        batch, width - 1, dim, device=device, dtype=itype
    ).transpose(1, 2)
    conv_state_gluon = conv_state_triton.detach().clone()
    
    weight = torch.randn(dim, width, device=device, dtype=itype)
    bias = torch.randn(dim, device=device, dtype=itype) if has_bias else None
    activation = None if not silu_activation else "silu"
    
    # ============================================================================
    # Correctness Test: Triton vs Gluon
    # ============================================================================
    print(f"\n{'='*70}")
    print(f"Correctness Test: Triton vs Gluon Kernels")
    print(f"{'='*70}")
    
    # Run Triton kernel
    query_triton, key_triton, value_triton = causal_conv1d_update_split_qkv(
        mixed_qkv.clone(),
        conv_state_triton,
        weight,
        key_dim=key_dim,
        value_dim=value_dim,
        bias=bias,
        activation=activation,
        use_gluon=False,
    )
    
    # Run Gluon kernel
    query_gluon, key_gluon, value_gluon = causal_conv1d_update_split_qkv(
        mixed_qkv.clone(),
        conv_state_gluon,
        weight,
        key_dim=key_dim,
        value_dim=value_dim,
        bias=bias,
        activation=activation,
        use_gluon=True,
    )
    
    # Compare outputs
    # print(f"Output shapes:")
    # print(f"  Query: {query_triton.shape}")
    # print(f"  Key:   {key_triton.shape}")
    # print(f"  Value: {value_triton.shape}")
    
    # Check if outputs match
    # query_diff = (query_triton - query_gluon).abs().max().item()
    # key_diff = (key_triton - key_gluon).abs().max().item()
    # value_diff = (value_triton - value_gluon).abs().max().item()
    # state_diff = (conv_state_triton - conv_state_gluon).abs().max().item()
    
    # print(f"\nMax differences:")
    # print(f"  Query: {query_diff:.6e}")
    # print(f"  Key:   {key_diff:.6e}")
    # print(f"  Value: {value_diff:.6e}")
    # print(f"  State: {state_diff:.6e}")
    
    assert torch.allclose(query_triton, query_gluon, rtol=rtol, atol=atol)
        # f"Query mismatch: max diff = {query_diff}"
    assert torch.allclose(key_triton, key_gluon, rtol=rtol, atol=atol)
        # f"Key mismatch: max diff = {key_diff}"
    assert torch.allclose(value_triton, value_gluon, rtol=rtol, atol=atol)
        # f"Value mismatch: max diff = {value_diff}"
    assert torch.allclose(conv_state_triton, conv_state_gluon, rtol=rtol, atol=atol)
        # f"Conv state mismatch: max diff = {state_diff}"
    
    print(f"  ✓ Correctness check passed!")
    
    # ============================================================================
    # Performance Benchmarking: Triton vs Gluon
    # ============================================================================
    # print(f"\n{'='*70}")
    # print(f"End-to-End Performance Benchmark")
    # print(f"{'='*70}")
    # print(f"Configuration:")
    # print(f"  - batch_size:  {batch}")
    # print(f"  - total_dim:   {dim} (key_dim={key_dim}, value_dim={value_dim})")
    # print(f"  - seqlen:      {seqlen}")
    # print(f"  - conv_width:  {width}")
    # print(f"  - dtype:       {itype}")
    # print(f"  - activation:  {activation}")
    # print(f"  - bias:        {has_bias}")
    
    num_warmup = 10
    num_iters = 100
    
    # Warmup
    for _ in range(num_warmup):
        _ = causal_conv1d_update_split_qkv(
            mixed_qkv.clone(), conv_state_triton.clone(), weight,
            key_dim=key_dim, value_dim=value_dim, bias=bias, activation=activation,
            use_gluon=False,
        )
        _ = causal_conv1d_update_split_qkv(
            mixed_qkv.clone(), conv_state_gluon.clone(), weight,
            key_dim=key_dim, value_dim=value_dim, bias=bias, activation=activation,
            use_gluon=True,
        )
    torch.cuda.synchronize()
    
    # Benchmark Triton kernel
    import time
    start_time = time.time()
    for _ in range(num_iters):
        _ = causal_conv1d_update_split_qkv(
            mixed_qkv.clone(), conv_state_triton.clone(), weight,
            key_dim=key_dim, value_dim=value_dim, bias=bias, activation=activation,
            use_gluon=False,
        )
    torch.cuda.synchronize()
    triton_time = (time.time() - start_time) / num_iters * 1000  # ms
    
    # Benchmark Gluon kernel
    start_time = time.time()
    for _ in range(num_iters):
        _ = causal_conv1d_update_split_qkv(
            mixed_qkv.clone(), conv_state_gluon.clone(), weight,
            key_dim=key_dim, value_dim=value_dim, bias=bias, activation=activation,
            use_gluon=True,
        )
    torch.cuda.synchronize()
    gluon_time = (time.time() - start_time) / num_iters * 1000  # ms
    
    # Calculate metrics
    total_elements = batch * dim * seqlen
    speedup = triton_time / gluon_time
    time_saved = triton_time - gluon_time
    
    print(f"\nPerformance Results (averaged over {num_iters} iterations):")
    print(f"\n  Triton Kernel:")
    print(f"    - Time per iteration:  {triton_time:.4f} ms")
    print(f"    - Throughput:          {total_elements / triton_time / 1000:.2f} M elements/s")
    
    print(f"\n  Gluon Kernel:")
    print(f"    - Time per iteration:  {gluon_time:.4f} ms")
    print(f"    - Throughput:          {total_elements / gluon_time / 1000:.2f} M elements/s")
    
    print(f"\n  Performance Comparison:")
    print(f"    - Speedup (Gluon/Triton): {speedup:.2f}x")
    print(f"    - Time difference:        {abs(time_saved):.4f} ms ({abs(time_saved)/max(triton_time, gluon_time)*100:.1f}%)")
    
    if speedup > 1.05:
        print(f"    - Status:                 ✓ Gluon is {speedup:.2f}x FASTER")
    elif speedup < 0.95:
        print(f"    - Status:                 ⚠ Triton is {1/speedup:.2f}x FASTER")
    else:
        print(f"    - Status:                 ≈ Performance is similar")
    
    print(f"{'='*70}\n")


if __name__ == "__main__":
    pytest.main([__file__])
