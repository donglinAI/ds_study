import torch
import triton
import triton.language as tl

@triton.jit
def act_quant_kernel(x_ptr, y_ptr, s_ptr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)  # 获取程序实例 ID
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)  # 计算数据偏移量
    x = tl.load(x_ptr + offs).to(tl.float32)  # 加载当前块数据
    s = tl.max(tl.abs(x)) / 448.  # 计算缩放因子
    y = x / s  # 量化数据
    y = y.to(tl.int8)  # 转换数据类型
    tl.store(y_ptr + offs, y)  # 存储量化结果
    tl.store(s_ptr + pid, s)  # 存储缩放因子

def act_quant(x: torch.Tensor, block_size: int = 128):
    assert x.is_contiguous()
    assert x.size(-1) % block_size == 0
    y = torch.empty_like(x, dtype=torch.int8)
    s = x.new_empty(*x.size()[:-1], x.size(-1) // block_size, dtype=torch.float32)
    grid = lambda meta: (triton.cdiv(x.numel(), meta['BLOCK_SIZE']), )  # 计算程序实例数量
    act_quant_kernel[grid](x, y, s, BLOCK_SIZE=block_size)  # 启动内核
    return y, s

# 示例输入：512 个元素的张量
input_tensor = torch.randn(512, dtype=torch.float32, device='cuda')
block_size = 128

# 分析 grid 的计算过程
total_elements = input_tensor.numel()  # 总元素数：512
program_instances = triton.cdiv(total_elements, block_size)  # 计算程序实例数：512 ÷ 128 = 4

print(f"总元素数: {total_elements}, 程序实例数: {program_instances}")

# 调用量化函数
quantized_y, scaling_s = act_quant(input_tensor, block_size)

# 模拟内核执行过程（手动复现逻辑）
for pid in range(program_instances):
    # 计算当前实例处理的偏移量
    offs = pid * block_size + torch.arange(0, block_size, device='cuda')
    # 模拟加载数据
    x_block = input_tensor[offs].to(torch.float32)
    # 计算缩放因子
    s = torch.max(torch.abs(x_block)) / 448.
    # 量化数据
    y_block = (x_block / s).to(torch.int8)
    # 模拟存储结果
    quantized_y[offs] = y_block
    scaling_s[pid] = s

print("量化结果 y:", quantized_y)
print("缩放因子 s:", scaling_s)