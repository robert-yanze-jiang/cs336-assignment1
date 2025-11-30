import torch
import torch.nn as nn
import torch.nn.functional as F
from cs336_basics.transformer_block import TransformerBlock
from cs336_basics.embedding import Embedding
from cs336_basics.RMSNorm import RMSNorm


class TransformerLM(nn.Module):
    """
    Transformer Language Model
    """

    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        theta: float = 10000.0
    ):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.d_model = d_model
        self.num_layers = num_layers
        
        # 1. Token Embedding 
        self.token_embedding = Embedding(vocab_size, d_model)
        
        # 2. num_layers个Transformer Blocks
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(
                d_model=d_model,
                num_heads=num_heads,
                d_ff=d_ff,
                max_seq_len=context_length,
                theta=theta
            )
            for _ in range(num_layers)
        ])
        
        # 3. Norm 使用RMSNorm
        self.norm = RMSNorm(d_model)
        
        # 4. Linear (Output Embedding) 
        self.output_projection = nn.Linear(d_model, vocab_size, bias=False)
        
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        前向传播，严格遵循Figure 1的数据流：
        Inputs → Token Embedding → Transformer Blocks → Norm → Linear → Softmax
        """
        batch_size, seq_len = input_ids.shape
        
        # 检查序列长度
        if seq_len > self.context_length:
            raise ValueError(f"序列长度{seq_len}超过了最大上下文长度{self.context_length}")
        
        # 1. Token Embedding 
        # Inputs → Token Embedding
        x = self.token_embedding(input_ids)  # [batch_size, seq_len, d_model]
        
        # RoPE 
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
        token_positions = positions.expand(batch_size, seq_len)
        
        # 2. 通过num_layers个Transformer Blocks
        # Token Embedding → Transformer Blocks
        for block in self.transformer_blocks:
            x = block(x, token_positions)
        
        # 3. Norm 
        # Transformer Blocks → Norm
        x = self.norm(x)  # [batch_size, seq_len, d_model]
        
        # 4. Linear (Output Embedding) (浅紫色模块)
        # Norm → Linear
        logits = self.output_projection(x)  # [batch_size, seq_len, vocab_size]
        
        return logits