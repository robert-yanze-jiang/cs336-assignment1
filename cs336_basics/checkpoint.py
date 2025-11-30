import torch
import os
from typing import Union, BinaryIO, IO
from pathlib import Path

def save_checkpoint(model: torch.nn.Module, optimizer: torch.optim.Optimizer, 
                   iteration: int, out: Union[str, os.PathLike, BinaryIO, IO[bytes]]):
    """
    保存模型检查点
    
    参数:
        model: PyTorch模型
        optimizer: 优化器
        iteration: 当前迭代次数
        out: 输出路径或文件对象
    """
    # 收集所有需要保存的状态
    checkpoint = {
        'iteration': iteration,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }
    
    # 保存到文件
    torch.save(checkpoint, out)

def load_checkpoint(src: Union[str, os.PathLike, BinaryIO, IO[bytes]], 
                   model: torch.nn.Module, optimizer: torch.optim.Optimizer) -> int:
    """
    加载模型检查点
    
    参数:
        src: 检查点文件路径或文件对象
        model: 要恢复状态的模型
        optimizer: 要恢复状态的优化器
        
    返回:
        保存的迭代次数
    """
    # 加载检查点
    checkpoint = torch.load(src, map_location='cpu')
    
    # 恢复模型状态
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # 恢复优化器状态
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    # 返回迭代次数
    return checkpoint['iteration']

