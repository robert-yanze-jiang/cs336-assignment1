import torch
import torch.nn as nn
import torch.nn.init as init
import math

class Linear(nn.Module):
    def __init__(self, in_features, out_features, device=None, dtype=None):
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        
        # 初始化权重矩阵：形状为 (out_features, in_features)
        weight = torch.empty(out_features, in_features, device=device, dtype=dtype)
        
        # 计算标准差：σ = sqrt(2 / (in_features + out_features))
        std = math.sqrt(2.0 / (in_features + out_features))
        
        # 使用截断正态分布初始化：N(0, σ²)，截断在[-3σ, 3σ]
        init.trunc_normal_(weight, mean=0.0, std=std, a=-3*std, b=3*std)
        
        self.W = nn.Parameter(weight)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        线性变换：y = Wx
        
        Args:
            x: 输入张量，形状为 (batch_size, ..., in_features)
            
        Returns:
            输出张量，形状为 (batch_size, ..., out_features)
        """
        # 修正einsum方程：使用单个字母而不是描述性名称
        # '...i,ji->...j' 其中：
        # i 表示输入特征维度
        # j 表示输出特征维度
        # ... 表示任意数量的批次维度
        return torch.einsum('...i,ji->...j', x, self.W)
        
        # 或者更简单的方法：直接使用矩阵乘法
        # return x @ self.W.T