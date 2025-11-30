import torch
import torch.nn as nn
import math
import torch.nn.init as init

class SwiGLUFeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int = None, device=None, dtype=None):
        """
        SwiGLU前馈网络
        
        Args:
            d_model: int, 模型的隐藏维度
            device: torch.device | None, 参数存储设备  
            dtype: torch.dtype | None, 参数数据类型
        """
        super().__init__()
        
        self.d_model = d_model
        
        if d_ff is None:
            d_ff_base = int(8/3 * d_model)
            self.d_ff = ((d_ff_base + 63) // 64) * 64
        else:
            self.d_ff = d_ff
        
        # 三个线性变换层（无偏置）
        self.W1 = nn.Linear(d_model, self.d_ff, bias=False, device=device, dtype=dtype)
        self.W2 = nn.Linear(self.d_ff, d_model, bias=False, device=device, dtype=dtype) 
        self.W3 = nn.Linear(d_model, self.d_ff, bias=False, device=device, dtype=dtype)
        
        # 初始化权重
        self._initialize_weights()
    
    def _initialize_weights(self):
        """初始化权重"""
        # W1和W3的初始化
        std_w1_w3 = math.sqrt(2.0 / (self.d_model + self.d_ff))
        init.trunc_normal_(self.W1.weight, mean=0.0, std=std_w1_w3, a=-3*std_w1_w3, b=3*std_w1_w3)
        init.trunc_normal_(self.W3.weight, mean=0.0, std=std_w1_w3, a=-3*std_w1_w3, b=3*std_w1_w3)
        
        # W2的初始化
        std_w2 = math.sqrt(2.0 / (self.d_ff + self.d_model))
        init.trunc_normal_(self.W2.weight, mean=0.0, std=std_w2, a=-3*std_w2, b=3*std_w2)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        SwiGLU前向传播: W2(SiLU(W1x) ⊙ W3x)
        
        Args:
            x: 输入张量，形状为 (batch_size, sequence_length, d_model)
            
        Returns:
            torch.Tensor: 输出张量，形状与输入相同
        """
        # SiLU(W1x) ⊙ W3x
        gate = torch.nn.functional.silu(self.W1(x)) * self.W3(x)
        
        # W2(gate)
        return self.W2(gate)


def swiglu_forward(
    x: torch.Tensor, 
    W1: torch.Tensor, 
    W2: torch.Tensor, 
    W3: torch.Tensor
) -> torch.Tensor:
    """
    SwiGLU前向传播的纯函数版本
    
    Args:
        x: 输入张量，形状为 (..., d_model)
        W1: 权重矩阵，形状为 (d_ff, d_model)
        W2: 权重矩阵，形状为 (d_model, d_ff)
        W3: 权重矩阵，形状为 (d_ff, d_model)
        
    Returns:
        torch.Tensor: 输出张量，形状与输入相同
    """
    # 计算d_ff
    d_ff = W1.shape[0]
    
    # SiLU(W1x) ⊙ W3x
    W1x = x @ W1.T  # (..., d_ff)
    W3x = x @ W3.T  # (..., d_ff)
    
    gate = torch.nn.functional.silu(W1x) * W3x  # (..., d_ff)
    
    # W2(gate)
    return gate @ W2.T  # (..., d_model)