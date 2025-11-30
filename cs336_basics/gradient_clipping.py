import torch

def gradient_clipping(parameters, max_norm):
    """
    Clips gradients of the given parameters to have a maximum norm of max_norm.

    Args:
        parameters (Iterable[torch.nn.Parameter]): Iterable of model parameters.
        max_norm (float): Maximum allowed norm for the gradients.
    """
    
    # 计算所有参数梯度的总L2范数
    total_norm = 0.0
    for param in parameters:
        if param.grad is not None:
            # 累加每个参数梯度的平方和
            param_norm = param.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    
    # 计算总范数
    total_norm = total_norm ** 0.5
    
    # 如果总范数超过最大范数，则进行裁剪
    clip_coef = max_norm / (total_norm + 1e-6)
    if total_norm >= max_norm:
        for param in parameters:
            if param.grad is not None:
                param.grad.data.mul_(clip_coef)