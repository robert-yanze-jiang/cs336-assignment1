import torch
import torch.nn as nn
import torch.nn.functional as F
from cs336_basics.multihead_self_attention import MultiHeadAttention
from cs336_basics.RMSNorm import RMSNorm
from cs336_basics.SwiGLU import SwiGLUFeedForward


class TransformerBlock(nn.Module):
    """
    Transformer Block (Pre-norm架构)
    包含两个子层：多头自注意力和前馈网络
    遵循公式(15)的结构：y = x + SubLayer(RMSNorm(x))
    """
    
    def __init__(self, d_model: int, num_heads: int, d_ff: int, max_seq_len: int = 4096, theta: float = 10000.0):
        super().__init__()
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        
        # 第一个子层：多头自注意力（带Pre-norm）
        self.attention_norm = RMSNorm(d_model)
        self.multi_head_attention = MultiHeadAttention(
            d_model=d_model,
            num_heads=num_heads,
            max_seq_len=max_seq_len,
            theta=theta
        )
        
        # 第二个子层：前馈网络（带Pre-norm）
        self.ffn_norm = RMSNorm(d_model)
        self.feed_forward = SwiGLUFeedForward(
            d_model=d_model, d_ff=d_ff
        )
        
    
    def forward(self, x: torch.Tensor, token_positions: torch.Tensor | None = None) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入张量，形状 [batch_size, seq_len, d_model]
            token_positions: 位置索引，如果为None则不使用RoPE
            
        Returns:
            输出张量，形状 [batch_size, seq_len, d_model]
        """
        # 第一个子层：多头自注意力
        # 公式(15): y = x + MultiHeadSelfAttention(RMSNorm(x))
        x = self._attention_sublayer(x, token_positions)
        
        # 第二个子层：前馈网络
        # 类似公式: y = x + FeedForward(RMSNorm(x))
        x = self._feed_forward_sublayer(x)
        
        return x
    
    def _attention_sublayer(self, x: torch.Tensor, token_positions: torch.Tensor | None = None) -> torch.Tensor:
        """多头自注意力子层"""
        # Pre-norm: 先对输入进行归一化
        x_norm = self.attention_norm(x)
        
        # 多头自注意力
        attention_output = self.multi_head_attention(x_norm, token_positions)
        
        # 残差连接
        output = x + attention_output
        
        return output
    
    def _feed_forward_sublayer(self, x: torch.Tensor) -> torch.Tensor:
        """前馈网络子层"""
        # Pre-norm: 先对输入进行归一化
        x_norm = self.ffn_norm(x)
        
        # 前馈网络
        ffn_output = self.feed_forward(x_norm)
        
        # 残差连接
        output = x + ffn_output
        
        return output