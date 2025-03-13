from typing import Tuple

import torch
import triton
import triton.language as tl
from triton import Config


@triton.jit
def act_quant_kernel(x_ptr, y_ptr, s_ptr, BLOCK_SIZE: tl.constexpr):
    """
    Quantizes the input tensor `x_ptr` and stores the result in `y_ptr` and the scaling factor in `s_ptr`.

    Args:
        x_ptr (triton.Pointer): Pointer to the input tensor.
        y_ptr (triton.Pointer): Pointer to the output tensor where quantized values will be stored.
        s_ptr (triton.Pointer): Pointer to the output tensor where scaling factors will be stored.
        BLOCK_SIZE (tl.constexpr): The size of the block to be processed by each program instance.

    Returns:
        None
    """
    pid = tl.program_id(axis=0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    x = tl.load(x_ptr + offs).to(tl.float32)
    s = tl.max(tl.abs(x)) / 448.
    y = x / s
    # y = y.to(y_ptr.dtype.element_ty)
    y = y.to(tl.int8)
    tl.store(y_ptr + offs, y)
    tl.store(s_ptr + pid, s)


def act_quant(x: torch.Tensor, block_size: int = 128) -> Tuple[torch.Tensor, torch.Tensor]:

    assert x.is_contiguous(), 'Input tensor must be contiguous'
    assert x.size   (-1) % block_size == 0, f'Last dimension size must be divisible by block_size (block_size={block_size})'
    # y = torch.empty_like(x, dtype=torch.float8_e4m3fn) 
    # 在3090卡上报错：AssertionError: fp8e4nv data type is not supported on CUDA arch < 89
    # 原因是：Triton 在编译时会检查 CUDA 架构，只有当 CUDA 架构达到 8.9 及以上时，才支持 fp8e4nv（对应 torch.float8_e4m3fn）数据类型
    # 解决办法是：将torch.float8_e4m3fn改成torch.int8
    y = torch.empty_like(x, dtype=torch.int8)
    s = x.new_empty(*x.size()[:-1], x.size(-1) // block_size, dtype=torch.float32)
    grid = lambda meta: (triton.cdiv(x.numel(), meta['BLOCK_SIZE']), )
    act_quant_kernel[grid](x, y, s, BLOCK_SIZE=block_size)
    return y, s

# 定义输入张量
input_tensor = torch.randn(1024, dtype=torch.float32, device='cuda')

# 调用量化函数
quantized_tensor, scaling_factors = act_quant(input_tensor, block_size=128)

# 打印结果
print("Input tensor:", input_tensor)
print("Quantized tensor:", quantized_tensor)
print("Scaling factors:", scaling_factors)