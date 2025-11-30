import torch
import torch.nn as nn
import torch.nn.functional as F

def cross_entropy(logits, targets):
    """
    计算交叉熵损失函数
    
    参数:
        logits: 预测的对数几率，形状为 [batch_size, seq_len, vocab_size] 或 [batch_size, vocab_size]
        targets: 目标标签，形状为 [batch_size, seq_len] 或 [batch_size]
    
    返回:
        loss: 平均交叉熵损失（标量）
    """
    # 确保输入是浮点类型
    logits = logits.float()
    
    # 数值稳定性处理：减去每行的最大值
    max_logits, _ = torch.max(logits, dim=-1, keepdim=True)
    shifted_logits = logits - max_logits
    
    # 计算 log-sum-exp（数值稳定的方式）
    log_sum_exp = torch.log(torch.sum(torch.exp(shifted_logits), dim=-1, keepdim=True))
    
    # 利用数学恒等式：-log(softmax) = -logits + log(sum(exp(logits)))
    # 这里我们使用数值稳定的版本：- (shifted_logits - log_sum_exp)
    # 但更直接的方式是：log_sum_exp - shifted_logits[targets]
    
    # 获取目标位置的对数值
    
       
    target_logits = torch.gather(shifted_logits, dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)
        
    # 计算每个样本的损失
    per_position_loss = log_sum_exp.squeeze(-1) - target_logits
    
    # 计算平均损失
    loss = torch.mean(per_position_loss)
    
    return loss

