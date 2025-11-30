import numpy as np
import torch

def get_batch(x, batch_size, context_length, device='cpu'):
    """
    从标记序列中采样批次数据
    """
    
    max_start_idx = len(x) - context_length  
    
    # 随机选择起始位置
    start_indices = np.random.randint(0, max_start_idx, size=batch_size)
    
    # 创建索引矩阵
    indices = start_indices[:, None] + np.arange(context_length)
    
    # 提取输入序列
    inputs = x[indices]
    
    # 目标序列是输入序列向后移动一位
    target_indices = start_indices[:, None] + np.arange(1, context_length + 1)
    targets = x[target_indices]
    
    # 转换为PyTorch张量并移动到指定设备
    inputs_tensor = torch.from_numpy(inputs).to(device)
    targets_tensor = torch.from_numpy(targets).to(device)
    
    return inputs_tensor, targets_tensor