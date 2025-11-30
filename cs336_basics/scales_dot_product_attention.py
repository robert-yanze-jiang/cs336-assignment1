import torch
from cs336_basics.softmax import softmax
from torch import Tensor
from jaxtyping import Float, Bool
import math

def scaled_dot_product_attention(
    queries: Float[Tensor, "... seq_len_q d_k"],
    keys: Float[Tensor, "... seq_len_k d_k"], 
    values: Float[Tensor, "... seq_len_v d_v"],
    mask: Bool[Tensor, "seq_len_q seq_len_k"] = None
) -> Float[Tensor, "... seq_len_q d_v"]:
    """
    实现带掩码的scaled dot-product attention
    
    Args:
        queries: 查询张量，形状 (..., seq_len_q, d_k)
        keys: 键张量，形状 (..., seq_len_k, d_k)  
        values: 值张量，形状 (..., seq_len_v, d_v)
        mask: 布尔掩码，形状 (seq_len_q, seq_len_k)，True表示允许关注
        
    Returns:
        注意力输出，形状 (..., seq_len_q, d_v)
    """
    # 1. 计算点积注意力分数
    # Q * K^T / sqrt(d_k)
    d_k = queries.size(-1)
    scores = torch.matmul(queries, keys.transpose(-2, -1)) / math.sqrt(d_k)
    
    # 2. 应用掩码（如果提供）
    if mask is not None:
        scores = scores.masked_fill(mask == False, -1e9)
        
    # 3. 计算注意力权重（使用数值稳定的softmax）
    attention_weights = softmax(scores, dim=-1)
    
    # 4. 应用注意力权重到values
    output = torch.matmul(attention_weights, values)
    
    return output