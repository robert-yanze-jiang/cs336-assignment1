import torch
import torch.nn as nn
import torch.nn.init as init

class Embedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
        """
        Embedding模块：将token ID映射为向量
        
        Args:
            num_embeddings: int, 词汇表大小
            embedding_dim: int, 嵌入向量的维度（d_model）
            device: torch.device | None, 参数存储设备
            dtype: torch.dtype | None, 参数数据类型
        """
        super().__init__()
        
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        
        # 初始化嵌入矩阵：形状为 (vocab_size, d_model)
        embedding_matrix = torch.empty(num_embeddings, embedding_dim, device=device, dtype=dtype)
        
        # 使用截断正态分布初始化：N(0, 1)，截断在[-3, 3]
        init.trunc_normal_(embedding_matrix, mean=0.0, std=1.0, a=-3.0, b=3.0)
        
        # 存储为nn.Parameter
        self.embedding_matrix = nn.Parameter(embedding_matrix)
    
    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        查找给定token IDs对应的嵌入向量
        
        Args:
            token_ids: torch.LongTensor, token ID张量，形状为 (batch_size, sequence_length)
            
        Returns:
            torch.Tensor: 嵌入向量，形状为 (batch_size, sequence_length, embedding_dim)
        """
        # 使用索引查找：token_ids中的每个ID对应embedding_matrix中的一行
        # token_ids: (batch_size, seq_len)
        # embedding_matrix: (vocab_size, embedding_dim)
        # 输出: (batch_size, seq_len, embedding_dim)
        return self.embedding_matrix[token_ids]