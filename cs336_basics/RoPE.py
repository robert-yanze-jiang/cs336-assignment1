import torch
import torch.nn as nn
import math

class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        """
        RoPE (Rotary Positional Embedding) 旋转位置编码
        
        Args:
            theta: float, RoPE的Θ参数
            d_k: int, 查询和键向量的维度
            max_seq_len: int, 最大序列长度
            device: torch.device, 设备
        """
        super().__init__()
        
        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len
        
        # 预计算sin和cos值（作为buffer，不参与训练）
        self._precompute_freqs(device)
    
    def _precompute_freqs(self, device):
        """预计算频率张量"""
        # 创建位置索引 (0 到 max_seq_len-1)
        positions = torch.arange(self.max_seq_len, device=device).float()
        
        # 创建维度索引 (0 到 d_k/2-1)
        dim_indices = torch.arange(0, self.d_k, 2, device=device).float()
        
        # 计算频率: theta^( -2i/d_k )，其中i是维度索引
        freqs = 1.0 / (self.theta ** (dim_indices / self.d_k))
        
        # 计算角度: position * freq
        # 形状: (max_seq_len, d_k/2)
        angles = positions.unsqueeze(1) * freqs.unsqueeze(0)
        
        # 计算cos和sin
        # 形状: (max_seq_len, d_k/2)
        cos = torch.cos(angles)
        sin = torch.sin(angles)
        
        # 注册为buffer（不参与训练）
        self.register_buffer('cos_cached', cos, persistent=False)
        self.register_buffer('sin_cached', sin, persistent=False)
    
    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        """
        应用RoPE旋转
        
        Args:
            x: 输入张量，形状 (..., seq_len, d_k)
            token_positions: 位置索引，形状 (..., seq_len)
            
        Returns:
            旋转后的张量，形状与输入相同
        """
        # 保存原始形状
        original_shape = x.shape
        seq_len = original_shape[-2]
        
        # 重塑x为(..., seq_len, d_k/2, 2)
        # 将最后维度分成d_k/2个二维向量
        x_reshaped = x.reshape(*original_shape[:-1], -1, 2)
        
        # 根据token_positions获取对应的cos和sin值
        # token_positions形状: (..., seq_len)
        cos = self.cos_cached[token_positions]  # (..., seq_len, d_k/2)
        sin = self.sin_cached[token_positions]  # (..., seq_len, d_k/2)
        
        # 扩展cos和sin的维度以匹配x_reshaped
        cos = cos.unsqueeze(-1)  # (..., seq_len, d_k/2, 1)
        sin = sin.unsqueeze(-1)  # (..., seq_len, d_k/2, 1)
        
        # 应用旋转: [x1, x2] -> [x1*cos - x2*sin, x1*sin + x2*cos]
        x1 = x_reshaped[..., 0]  # (..., seq_len, d_k/2)
        x2 = x_reshaped[..., 1]  # (..., seq_len, d_k/2)
        
        # 旋转后的结果
        rotated_x1 = x1 * cos.squeeze(-1) - x2 * sin.squeeze(-1)
        rotated_x2 = x1 * sin.squeeze(-1) + x2 * cos.squeeze(-1)
        
        # 合并结果
        rotated_reshaped = torch.stack([rotated_x1, rotated_x2], dim=-1)
        
        # 恢复原始形状
        return rotated_reshaped.reshape(original_shape)


# 更高效的实现版本（使用复数运算）
class RotaryPositionalEmbeddingEfficient(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        super().__init__()
        
        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len
        
        # 预计算复数形式的旋转因子
        self._precompute_complex_freqs(device)
    
    def _precompute_complex_freqs(self, device):
        """使用复数形式预计算"""
        positions = torch.arange(self.max_seq_len, device=device).float()
        dim_indices = torch.arange(0, self.d_k, 2, device=device).float()
        
        # 计算频率和角度
        freqs = 1.0 / (self.theta ** (dim_indices / self.d_k))
        angles = positions.unsqueeze(1) * freqs.unsqueeze(0)
        
        # 复数形式的旋转因子: e^(i*theta) = cos(theta) + i*sin(theta)
        freqs_cis = torch.polar(torch.ones_like(angles), angles)  # 幅度为1，角度为angles
        
        self.register_buffer('freqs_cis', freqs_cis, persistent=False)
    
    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        """复数版本的旋转实现"""
        # 将实数向量转为复数形式
        x_reshaped = x.reshape(*x.shape[:-1], -1, 2)
        x_complex = torch.view_as_complex(x_reshaped)
        
        # 获取旋转因子
        freqs_cis = self.freqs_cis[token_positions]  # (..., seq_len, d_k/2)
        
        # 复数乘法实现旋转
        rotated_complex = x_complex * freqs_cis
        
        # 转回实数形式
        rotated_real = torch.view_as_real(rotated_complex)
        return rotated_real.reshape(x.shape)