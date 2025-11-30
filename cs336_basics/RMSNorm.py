import torch
import torch.nn as nn
import math

class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        """
        RMSNorm (Root Mean Square Layer Normalization)
        
        Args:
            d_model: int, 模型的隐藏维度
            eps: float, 数值稳定性的epsilon值，默认为1e-5
            device: torch.device | None, 参数存储设备
            dtype: torch.dtype | None, 参数数据类型
        """
        super().__init__()
        
        self.d_model = d_model
        self.eps = eps
        
        # 初始化gain参数为1（根据3.4.1节的初始化要求）
        gain = torch.ones(d_model, device=device, dtype=dtype)
        self.gain = nn.Parameter(gain)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        RMSNorm前向传播
        
        Args:
            x: 输入张量，形状为 (batch_size, sequence_length, d_model)
            
        Returns:
            归一化后的张量，形状与输入相同
        """
        # 保存原始数据类型
        original_dtype = x.dtype
        
        # 上转为float32防止溢出
        x = x.to(torch.float32)
        
        # 计算RMS：sqrt(mean(x^2) + eps)
        # x^2: 对最后一个维度（d_model）计算平方
        mean_squared = torch.mean(x**2, dim=-1, keepdim=True)
        rms = torch.sqrt(mean_squared + self.eps)
        
        # 归一化：x / RMS(a)
        normalized_x = x / rms
        
        # 应用gain参数进行缩放
        result = normalized_x * self.gain
        
        # 转回原始数据类型
        return result.to(original_dtype)