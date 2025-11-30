import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from cs336_basics.RoPE import RotaryPositionalEmbedding
from cs336_basics.scales_dot_product_attention import scaled_dot_product_attention


class MultiHeadAttention(nn.Module):
    """
    统一的多头自注意力模块
    根据是否提供token_positions自动决定是否使用RoPE
    """
    
    def __init__(self, d_model: int, num_heads: int, max_seq_len: int = 4096, theta: float = 10000.0):
        super().__init__()
        
        assert d_model % num_heads == 0, "d_model必须能被num_heads整除"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads  # d_k = d_v = d_model / h
        self.max_seq_len = max_seq_len
        self.theta = theta
        
        
        # 实际上图片中的 h*d_k × d_model 是转置后的视角
        
        # 线性投影层 - 保持 [d_model, d_model] 但是理解方式不同
        self.W_q = nn.Linear(d_model, d_model, bias=False)  # W_Q^T ∈ ℝ^{d_model × d_model}
        self.W_k = nn.Linear(d_model, d_model, bias=False)  # W_K^T ∈ ℝ^{d_model × d_model}
        self.W_v = nn.Linear(d_model, d_model, bias=False)  # W_V^T ∈ ℝ^{d_model × d_model}
        self.W_o = nn.Linear(d_model, d_model, bias=False)  # W_O ∈ ℝ^{d_model × d_model}
        
        # RoPE位置编码
        self.rope = RotaryPositionalEmbedding(
            theta=theta,
            d_k=self.head_dim,  # 每个头的维度 d_k
            max_seq_len=max_seq_len
        )
        
        # 预计算因果掩码
        self.register_buffer("causal_mask", None)
    
    def forward(self, x: torch.Tensor, token_positions: torch.Tensor | None = None) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入张量，形状 [batch_size, seq_len, d_model]
            token_positions: 位置索引，如果为None则不使用RoPE
            
        Returns:
            输出张量，形状 [batch_size, seq_len, d_model]
        """
        batch_size, seq_len, _ = x.shape
        
        # 1. 线性投影得到Q, K, V - 对应公式(14)中的 W_Qx, W_Kx, W_Vx
        # x: [batch_size, seq_len, d_model]
        # 经过线性层后: [batch_size, seq_len, d_model] = [batch_size, seq_len, h*d_k]
        Q = self.W_q(x)  # Q = x × W_Q^T
        K = self.W_k(x)  # K = x × W_K^T
        V = self.W_v(x)  # V = x × W_V^T
        
        # 2. 重塑为多头格式 - 对应分割操作
        # 从 [batch_size, seq_len, h*d_k] 重塑为 [batch_size, seq_len, h, d_k]
        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim)
        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim)
        V = V.view(batch_size, seq_len, self.num_heads, self.head_dim)
        
        # 3. 应用RoPE（如果提供了位置信息）
        if token_positions is not None:
            Q, K = self.apply_rope(Q, K, token_positions)
        
        # 4. 转置以匹配注意力计算格式
        # 从 [batch_size, seq_len, h, d_k] 转置为 [batch_size, h, seq_len, d_k]
        Q = Q.transpose(1, 2)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)
        
        # 5. 创建或获取因果掩码
        if self.causal_mask is None or self.causal_mask.size(-1) < seq_len:
            self._create_causal_mask(seq_len, x.device)
        
        causal_mask = self.causal_mask[:seq_len, :seq_len]
        
        # 6. 批量计算注意力 - 对应公式(12)-(13)
        attn_output = self._batched_attention(Q, K, V, causal_mask)
        
        # 7. 合并多头输出 - 对应公式(12)中的Concat操作
        # 从 [batch_size, h, seq_len, d_k] 转置回 [batch_size, seq_len, h, d_k]
        # 然后重塑为 [batch_size, seq_len, h*d_k] = [batch_size, seq_len, d_model]
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.d_model
        )
        
        # 8. 输出投影 - 对应公式(14)中的 W_O
        output = self.W_o(attn_output)
        
        return output
    
    def apply_rope(self, Q: torch.Tensor, K: torch.Tensor, token_positions: torch.Tensor) -> tuple:
        """应用RoPE旋转位置编码"""
        batch_size, seq_len, num_heads, head_dim = Q.shape
        
        # 处理位置信息格式
        if token_positions.dim() == 1:
            positions = token_positions.unsqueeze(0).expand(batch_size, -1)
        else:
            positions = token_positions
        
        # 逐个头应用RoPE
        Q_rotated = torch.zeros_like(Q)
        K_rotated = torch.zeros_like(K)
        
        for i in range(num_heads):
            Q_head = Q[:, :, i, :]  # [batch_size, seq_len, d_k]
            K_head = K[:, :, i, :]
            
            Q_rotated_head = self.rope(Q_head, positions)
            K_rotated_head = self.rope(K_head, positions)
            
            Q_rotated[:, :, i, :] = Q_rotated_head
            K_rotated[:, :, i, :] = K_rotated_head
        
        return Q_rotated, K_rotated
    
    def _create_causal_mask(self, seq_len: int, device: torch.device):
        """使用torch.triu创建因果掩码"""
        mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1)
        self.causal_mask = (mask == 0)  # True表示允许关注的位置
    
    def _batched_attention(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, mask: torch.Tensor):
        """批量处理所有头的注意力计算"""
        batch_size, num_heads, seq_len, head_dim = Q.shape
        
        # 重塑为 [batch_size * num_heads, seq_len, head_dim]
        Q_flat = Q.reshape(batch_size * num_heads, seq_len, head_dim)
        K_flat = K.reshape(batch_size * num_heads, seq_len, head_dim)
        V_flat = V.reshape(batch_size * num_heads, seq_len, head_dim)
        
        if mask is not None:
            mask_expanded = mask.unsqueeze(0).expand(batch_size * num_heads, -1, -1)
        else:
            mask_expanded = None
        
        attn_flat = scaled_dot_product_attention(Q_flat, K_flat, V_flat, mask_expanded)
        attn_output = attn_flat.reshape(batch_size, num_heads, seq_len, head_dim)
        
        return attn_output