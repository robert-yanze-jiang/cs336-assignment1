import torch
from torch import Tensor
from jaxtyping import Float

def softmax(x: Float[Tensor, "..."], dim: int) -> Float[Tensor, "..."]:
    """
    应用softmax操作，使用数值稳定性技巧
    
    Args:
        x: 输入张量
        dim: 应用softmax的维度
        
    Returns:
        输出张量，形状与输入相同，指定维度上为概率分布
    """
    # 数值稳定性技巧：减去该维度的最大值
    # 1. 找到指定维度的最大值（保持维度）
    max_vals = torch.max(x, dim=dim, keepdim=True).values
    
    # 2. 减去最大值（数值稳定）
    x_stable = x - max_vals
    
    # 3. 计算指数
    exp_x = torch.exp(x_stable)
    
    # 4. 计算分母（该维度的指数和）
    sum_exp = torch.sum(exp_x, dim=dim, keepdim=True)
    
    # 5. 计算softmax
    softmax_result = exp_x / sum_exp
    
    return softmax_result