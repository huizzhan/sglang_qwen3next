#!/usr/bin/env python3
"""
独立的 causal_conv1d_update 性能测试脚本
不使用 pytest，可以直接运行
"""

import time
from typing import Optional
import torch
import torch.nn.functional as F
from sglang.srt.layers.attention.mamba.causal_conv1d_triton import (
    PAD_SLOT_ID,
    causal_conv1d_update,
    causal_conv1d_update_v2,
    causal_conv1d_update_persistent,
    causal_conv1d_update_persistent_v2,
)

# ============================================================================
# Kernel 配置：在此定义要测试的 kernel 列表
# ============================================================================
# 格式：(kernel_function, "显示名称")
# 
# 使用说明：
# 1. 第一个 kernel 将作为性能基准（baseline）
# 2. 可以添加任意数量的 kernel
# 3. 支持同一个 kernel 的不同配置版本
# 4. 如果某个 kernel 编译或运行失败，会自动跳过
#
# 示例配置：
# KERNELS_TO_TEST = [
#     (causal_conv1d_update, "Gluon Baseline"),           # 基准
#     (causal_conv1d_update_persistent, "Persistent v1"), # 比较对象 1
#     (causal_conv1d_update_v2, "Gluon V2"),              # 比较对象 2
#     (causal_conv1d_update_persistent, "Persistent v2"), # 可以重复测试
#     (your_new_kernel, "Your Custom Kernel"),            # 自定义 kernel
# ]
# ============================================================================

KERNELS_TO_TEST = [
    # (causal_conv1d_update, "Gluon Kernel"),
    (causal_conv1d_update_persistent, "Persistent Kernel"),
    (causal_conv1d_update_persistent_v2, "Persistent v2"),
    # (causal_conv1d_update_v2, "V2 Kernel"),
    # 添加更多 kernel（取消注释以启用）:
    # (causal_conv1d_update, "Gluon (Duplicate Test)"),  # 测试重复添加
    # (your_new_kernel_func, "Your Custom Kernel Name"),
]

# KERNELS_TO_TEST = [
#     (causal_conv1d_update, "Gluon Kernel"),
#     (causal_conv1d_update_persistent_v2, "Persistent v2"),
#     (causal_conv1d_update_persistent, "Persistent Kernel"),
#     (causal_conv1d_update_v2, "V2 Kernel"),
#     # 添加更多 kernel（取消注释以启用）:
#     # (causal_conv1d_update, "Gluon (Duplicate Test)"),  # 测试重复添加
#     # (your_new_kernel_func, "Your Custom Kernel Name"),
# ]

# 获取基准 kernel（第一个）
baseline_kernel_name = KERNELS_TO_TEST[0][1] if KERNELS_TO_TEST else None


def test_kernel_correctness(
    kernel_func,
    kernel_name,
    x,
    conv_state,
    weight,
    bias,
    activation,
    conv_state_indices,
    out_ref,
    conv_state_ref,
    rtol,
    atol,
):
    """
    测试单个 kernel 的正确性
    
    返回: (is_correct, error_message)
    """
    print(f"\n{'='*70}")
    print(f"正确性测试: {kernel_name} (连续批处理模式)")
    print(f"{'='*70}")
    
    try:
        conv_state_test = conv_state.detach().clone()
        
        out_test = kernel_func(
            x.clone(), conv_state_test, weight, bias,
            activation=activation, conv_state_indices=conv_state_indices
        )
        
        # 检查输出正确性 - 添加详细调试信息
        max_diff = (out_test - out_ref).abs().max().item()
        mean_diff = (out_test - out_ref).abs().mean().item()
        if not torch.allclose(out_test, out_ref, rtol=rtol, atol=atol):
            print(f"  ✗ 输出不匹配:")
            print(f"    - Max diff: {max_diff:.6e}")
            print(f"    - Mean diff: {mean_diff:.6e}")
            print(f"    - rtol: {rtol}, atol: {atol}")
            print(f"    - out_test shape: {out_test.shape}")
            print(f"    - out_ref shape: {out_ref.shape}")
            print(f"    - out_test sample: {out_test.flatten()[:5]}")
            print(f"    - out_ref sample: {out_ref.flatten()[:5]}")
        assert torch.allclose(out_test, out_ref, rtol=rtol, atol=atol), \
            f"{kernel_name} 输出与参考实现不匹配 (max_diff={max_diff:.6e}, mean_diff={mean_diff:.6e})"
        
        # 检查 conv_state 正确性
        assert torch.allclose(
            conv_state_test[conv_state_indices],
            conv_state_ref[conv_state_indices],
            rtol=rtol, atol=atol
        ), f"{kernel_name} conv_state 不匹配"
        
        print(f"  ✓ 使用 conv_state_indices 的正确性检查通过！")
        return True, None, conv_state_test
        
    except Exception as e:
        print(f"  ✗ 使用 conv_state_indices 的测试失败: {type(e).__name__}")
        print(f"    错误: {str(e)}")
        import traceback
        traceback.print_exc()
        print(f"    注意: 这可能是由于 Gluon 编译问题")
        return False, str(e), None


def test_kernel_performance(
    kernel_func,
    kernel_name,
    x,
    conv_state,
    weight,
    bias,
    activation,
    conv_state_indices,
    num_warmup,
    num_iters,
):
    """
    测试单个 kernel 的性能
    
    返回: kernel_time (ms) 或 None
    """
    try:
        print(f"\n正在预热 {kernel_name} ({num_warmup} 次迭代)...")
        for _ in range(num_warmup):
            _ = kernel_func(
                x.clone(), conv_state.clone(), weight, bias,
                activation=activation, conv_state_indices=conv_state_indices
            )
        torch.cuda.synchronize()
        
        print(f"正在测试 {kernel_name} ({num_iters} 次迭代)...")
        start_time = time.time()
        for _ in range(num_iters):
            _ = kernel_func(
                x.clone(), conv_state.clone(), weight, bias,
                activation=activation, conv_state_indices=conv_state_indices
            )
        torch.cuda.synchronize()
        kernel_time = (time.time() - start_time) / num_iters * 1000  # ms
        
        return kernel_time
        
    except Exception as e:
        print(f"  ⚠ {kernel_name} 基准测试失败: {type(e).__name__}")
        return None


def causal_conv1d_update_ref(
    x, conv_state, weight, bias=None, activation=None, cache_seqlens=None, conv_state_indices=None
):
    """
    参考实现的 causal_conv1d_update
    
    x: (batch, dim) or (batch, dim, seqlen)
    conv_state: (batch, dim, state_len) or (total_entries, dim, state_len), where state_len >= width - 1
    weight: (dim, width)
    bias: (dim,)
    cache_seqlens: (batch,), dtype int32.
        If not None, the conv_state is treated as a circular buffer.
        The conv_state will be updated by copying x to the
        conv_state starting at the index
        @cache_seqlens % state_len before performing the convolution.
    conv_state_indices: (batch,), dtype int32
        If not None, the conv_state is a larger tensor along the batch dim,
        and we are selecting the batch coords specified by conv_state_indices.
        Useful for a continuous batching scenario.

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
    
    # Handle conv_state_indices for continuous batching
    if conv_state_indices is not None:
        # Select the relevant states from the larger conv_state tensor
        selected_conv_state = conv_state[conv_state_indices]  # (batch, dim, state_len)
        assert selected_conv_state.shape == (batch, dim, state_len)
    else:
        selected_conv_state = conv_state
        assert conv_state.shape == (batch, dim, state_len)
    
    assert weight.shape == (dim, width)
    
    if cache_seqlens is None:
        x_new = torch.cat([selected_conv_state, x], dim=-1).to(
            weight.dtype
        )  # (batch, dim, state_len + seqlen)
        updated_state = x_new[:, :, -state_len:]  # update
        if conv_state_indices is not None:
            # Update the original conv_state tensor at the specified indices
            conv_state[conv_state_indices] = updated_state
        else:
            conv_state.copy_(updated_state)
    else:
        # Circular buffer mode
        width_idx = torch.arange(-(width - 1), 0, dtype=torch.long, device=x.device).unsqueeze(0)  + cache_seqlens.unsqueeze(1)
        width_idx = torch.remainder(width_idx, state_len).unsqueeze(1).expand(batch, dim, width - 1)
        
        if conv_state_indices is not None:
            batch_indices = conv_state_indices.unsqueeze(1).unsqueeze(2).expand(batch, dim, width - 1)
            dim_indices = torch.arange(dim, device=x.device).unsqueeze(0).unsqueeze(2).expand(batch, dim, width - 1)
            selected_conv_state = conv_state[batch_indices, dim_indices, width_idx]
        else:
            batch_indices = torch.arange(batch, device=x.device).unsqueeze(1).unsqueeze(2).expand(batch, dim, width - 1)
            dim_indices = torch.arange(dim, device=x.device).unsqueeze(0).unsqueeze(2).expand(batch, dim, width - 1)
            selected_conv_state = conv_state[batch_indices, dim_indices, width_idx]
        
        x_new = torch.cat([selected_conv_state, x], dim=-1).to(weight.dtype)
        
        # Update conv_state
        copy_idx = torch.remainder(cache_seqlens, state_len).unsqueeze(1).unsqueeze(2).expand(batch, dim, seqlen)
        
        if conv_state_indices is not None:
            batch_idx_for_copy = conv_state_indices.unsqueeze(1).unsqueeze(2).expand(batch, dim, seqlen)
            dim_idx_for_copy = torch.arange(dim, device=x.device).unsqueeze(0).unsqueeze(2).expand(batch, dim, seqlen)
            conv_state[batch_idx_for_copy, dim_idx_for_copy, copy_idx] = x.to(conv_state.dtype)
        else:
            batch_idx_for_copy = torch.arange(batch, device=x.device).unsqueeze(1).unsqueeze(2).expand(batch, dim, seqlen)
            dim_idx_for_copy = torch.arange(dim, device=x.device).unsqueeze(0).unsqueeze(2).expand(batch, dim, seqlen)
            conv_state[batch_idx_for_copy, dim_idx_for_copy, copy_idx] = x.to(conv_state.dtype)
    
    out = F.conv1d(x_new, weight.unsqueeze(1), bias, padding=0, groups=dim)[..., :seqlen]
    
    if activation in ["silu", "swish"]:
        out = F.silu(out)
    
    if unsqueeze:
        out = out.squeeze(-1)
    
    return out.to(dtype=dtype_in)


def benchmark_causal_conv1d_update(
    batch=128, 
    dim=2048, 
    width=4, 
    seqlen=1, 
    has_bias=False, 
    silu_activation=True, 
    itype=torch.bfloat16, 
    total_entries=128,
    num_warmup=10,
    num_iters=100
):
    """
    对 causal_conv1d_update 进行性能基准测试
    
    参数:
        batch: 批次大小
        dim: 维度大小
        width: 卷积核宽度
        seqlen: 序列长度
        has_bias: 是否使用偏置
        silu_activation: 是否使用 SiLU 激活函数
        itype: 数据类型
        total_entries: 状态缓存总大小（用于连续批处理场景）
        num_warmup: 预热迭代次数
        num_iters: 性能测试迭代次数
    """
    
    if not torch.cuda.is_available():
        print("错误: CUDA 设备不可用")
        return
    
    device = "cuda"
    rtol, atol = (3e-4, 1e-3) if itype == torch.float32 else (3e-3, 5e-3)
    if itype == torch.bfloat16:
        rtol, atol = 1e-2, 5e-2
    
    # 设置随机种子以保证可重复性
    torch.manual_seed(0)
    
    # 创建输入数据
    x = torch.randn(batch, dim, seqlen, device=device, dtype=itype)
    x_ref = x.clone()
    
    weight = torch.randn(dim, width, device=device, dtype=itype)
    bias = torch.randn(dim, device=device, dtype=itype) if has_bias else None
    activation = None if not silu_activation else "silu"
    
    # ============================================================================
    # 准备测试数据
    # ============================================================================
    # 设置连续批处理场景
    conv_state_indices = torch.randperm(total_entries)[:batch].to(
        dtype=torch.int32, device=device
    )
    print(f"\nconv_state_indices: {conv_state_indices}")
    print(f"conv_state_indices.shape: {conv_state_indices.shape}")
    print(f"conv_state_indices.dtype: {conv_state_indices.dtype}")
    
    # 创建更大的 conv_state 张量用于连续批处理
    conv_state_large = torch.randn(total_entries, width - 1, dim, device=device, dtype=itype).transpose(1, 2)
    conv_state_large_ref = conv_state_large.detach().clone()
    
    # 运行参考实现（使用 conv_state_indices）
    print(f"\n{'='*70}")
    print(f"计算参考实现结果...")
    print(f"{'='*70}")
    out_ref_indices = causal_conv1d_update_ref(
        x_ref.clone(), conv_state_large_ref, weight, bias, 
        activation=activation, conv_state_indices=conv_state_indices
    )
    print(f"  ✓ 参考实现计算完成")
    print(f"  - 状态缓存总条目数: {total_entries}")
    print(f"  - 活跃批次大小: {batch}")
    print(f"  - 缓存利用率: {batch}/{total_entries} ({100*batch/total_entries:.1f}%)")
    print(f"  - 测试的状态索引: {conv_state_indices[:5].tolist()}...")
    
    # ============================================================================
    # 测试所有 kernel 的正确性
    # ============================================================================
    if not KERNELS_TO_TEST:
        print("\n✗ 错误: 没有定义要测试的 kernel")
        return
    
    kernel_results = {}
    
    for kernel_func, kernel_name in KERNELS_TO_TEST:
        is_correct, error_msg, conv_state_test = test_kernel_correctness(
            kernel_func, kernel_name,
            x, conv_state_large, weight, bias, activation,
            conv_state_indices, out_ref_indices, conv_state_large_ref,
            rtol, atol
        )
        
        kernel_results[kernel_name] = {
            'is_correct': is_correct,
            'error_msg': error_msg,
            'conv_state': conv_state_test,
        }
    
    # 检查基准 kernel 是否通过
    if not kernel_results[baseline_kernel_name]['is_correct']:
        print(f"\n✗ 基准 kernel ({baseline_kernel_name}) 正确性检查失败，停止测试")
        return
    
    # ============================================================================
    # 性能基准测试
    # ============================================================================
    total_elements = batch * dim * seqlen
    
    print(f"\n{'='*70}")
    print(f"性能基准测试 (连续批处理模式)")
    print(f"{'='*70}")
    
    kernel_times = {}
    
    for kernel_func, kernel_name in KERNELS_TO_TEST:
        if not kernel_results[kernel_name]['is_correct']:
            print(f"\n跳过 {kernel_name} 性能测试（正确性检查未通过）")
            kernel_times[kernel_name] = None
            continue
        
        conv_state_test = kernel_results[kernel_name]['conv_state']
        kernel_time = test_kernel_performance(
            kernel_func, kernel_name,
            x, conv_state_test, weight, bias, activation,
            conv_state_indices, num_warmup, num_iters
        )
        kernel_times[kernel_name] = kernel_time
    
    # ============================================================================
    # 输出性能测试结果
    # ============================================================================
    print(f"\n{'='*70}")
    print(f"性能测试结果 (平均 {num_iters} 次迭代)")
    print(f"{'='*70}")
    
    # 计算吞吐量和加速比
    kernel_throughputs = {}
    kernel_speedups = {}
    
    baseline_time = kernel_times[baseline_kernel_name]
    
    # 遍历所有 kernel 并输出结果
    for kernel_func, kernel_name in KERNELS_TO_TEST:
        kernel_time = kernel_times[kernel_name]
        
        if kernel_time is not None:
            throughput = total_elements / kernel_time / 1000  # M elements/s
            speedup = baseline_time / kernel_time if baseline_time else 1.0
            kernel_throughputs[kernel_name] = throughput
            kernel_speedups[kernel_name] = speedup
            
            print(f"\n  {kernel_name}:")
            print(f"    - 每次迭代时间:    {kernel_time:.4f} ms")
            print(f"    - 吞吐量:          {throughput:.2f} M elements/s")
            if kernel_name != baseline_kernel_name:
                print(f"    - 相对 {baseline_kernel_name} 加速: {speedup:.3f}x")
        else:
            print(f"\n  {kernel_name}: 不可用 (编译问题或正确性检查失败)")
            kernel_throughputs[kernel_name] = None
            kernel_speedups[kernel_name] = None
    
    print(f"\n  配置:")
    print(f"    - 批次大小:        {batch}")
    print(f"    - 维度:            {dim}")
    print(f"    - 序列长度:        {seqlen}")
    print(f"    - 卷积核宽度:      {width}")
    print(f"    - 数据类型:        {itype}")
    print(f"    - 激活函数:        {activation}")
    print(f"    - 使用偏置:        {has_bias}")
    print(f"    - 总条目数:        {total_entries}")
    print(f"    - 缓存利用率:      {100*batch/total_entries:.1f}%")
    
    print(f"\n  总结:")
    print(f"    ○ 测试连续批处理模式")
    print(f"    ○ 基准 kernel: {baseline_kernel_name}")
    
    # 打印性能总结（跳过基准 kernel）
    for kernel_func, kernel_name in KERNELS_TO_TEST:
        if kernel_name == baseline_kernel_name:
            continue
        
        speedup = kernel_speedups.get(kernel_name)
        if speedup is not None:
            if speedup > 1.1:
                print(f"    ✓ {kernel_name} 比 {baseline_kernel_name} 快 {speedup:.2f}x")
            elif speedup < 0.9:
                print(f"    ⚠ {kernel_name} 比 {baseline_kernel_name} 慢 {1/speedup:.2f}x")
            else:
                print(f"    ≈ {kernel_name} 性能相似 ({speedup:.2f}x)")
    
    # 找出最快的 kernel
    valid_times = [(name, time) for name, time in kernel_times.items() if time is not None]
    if valid_times:
        fastest_kernel = min(valid_times, key=lambda x: x[1])
        print(f"\n    🏆 最快的 kernel: {fastest_kernel[0]} ({fastest_kernel[1]:.4f} ms)")
    
    print(f"{'='*70}\n")
    
    # 构建返回结果字典
    results = {}
    for kernel_func, kernel_name in KERNELS_TO_TEST:
        results[kernel_name] = {
            'time_ms': kernel_times.get(kernel_name),
            'throughput': kernel_throughputs.get(kernel_name),
            'speedup': kernel_speedups.get(kernel_name),
        }
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Causal Conv1d Update 性能基准测试')
    parser.add_argument('--batch', type=int, default=128, help='批次大小')
    parser.add_argument('--dim', type=int, default=2048, help='维度大小')
    parser.add_argument('--width', type=int, default=4, help='卷积核宽度')
    parser.add_argument('--seqlen', type=int, default=1, help='序列长度')
    parser.add_argument('--total-entries', type=int, default=128, help='状态缓存总大小')
    parser.add_argument('--has-bias', action='store_true', help='是否使用偏置')
    parser.add_argument('--no-silu', action='store_true', help='不使用 SiLU 激活函数')
    parser.add_argument('--dtype', type=str, default='bfloat16', 
                        choices=['float32', 'float16', 'bfloat16'], help='数据类型')
    parser.add_argument('--num-warmup', type=int, default=10, help='预热迭代次数')
    parser.add_argument('--num-iters', type=int, default=100, help='性能测试迭代次数')
    
    args = parser.parse_args()
    
    # 转换数据类型
    dtype_map = {
        'float32': torch.float32,
        'float16': torch.float16,
        'bfloat16': torch.bfloat16,
    }
    itype = dtype_map[args.dtype]
    
    print(f"\n{'='*70}")
    print(f"Causal Conv1d Update 性能基准测试")
    print(f"{'='*70}")
    print(f"\n参数配置:")
    print(f"  - batch: {args.batch}")
    print(f"  - dim: {args.dim}")
    print(f"  - width: {args.width}")
    print(f"  - seqlen: {args.seqlen}")
    print(f"  - total_entries: {args.total_entries}")
    print(f"  - has_bias: {args.has_bias}")
    print(f"  - silu_activation: {not args.no_silu}")
    print(f"  - dtype: {args.dtype}")
    print(f"  - num_warmup: {args.num_warmup}")
    print(f"  - num_iters: {args.num_iters}")
    
    results = benchmark_causal_conv1d_update(
        batch=args.batch,
        dim=args.dim,
        width=args.width,
        seqlen=args.seqlen,
        has_bias=args.has_bias,
        silu_activation=not args.no_silu,
        itype=itype,
        total_entries=args.total_entries,
        num_warmup=args.num_warmup,
        num_iters=args.num_iters
    )
    
    print("\n测试完成！")

