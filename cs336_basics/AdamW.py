import torch
from torch.optim import Optimizer

def get_adamw_cls():
    return AdamW    

class AdamW(Optimizer):
    """
    AdamW 优化器实现
    参考: Decoupled Weight Decay Regularization (ICLR 2019)
    """
    
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, 
                 weight_decay=0.01, correct_bias=True):
        """
        初始化 AdamW 优化器
        
        参数:
            params: 要优化的参数或参数组
            lr: 学习率 (α)
            betas: 动量系数 (β1, β2)
            eps: 数值稳定性的小常数
            weight_decay: 权重衰减系数 (λ)
            correct_bias: 是否进行偏差校正
        """
        if not 0.0 <= lr:
            raise ValueError(f"无效的学习率: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"无效的epsilon值: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"无效的beta参数: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"无效的beta参数: {betas[1]}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"无效的权重衰减值: {weight_decay}")
            
        defaults = dict(lr=lr, betas=betas, eps=eps, 
                       weight_decay=weight_decay, correct_bias=correct_bias)
        super(AdamW, self).__init__(params, defaults)
    
    def step(self, closure=None):
        """
        执行单次参数更新
        
        参数:
            closure: 重新计算损失的闭包（可选）
        """
        loss = None
        if closure is not None:
            loss = closure()
        
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('AdamW 不支持稀疏梯度')
                
                # 获取状态
                state = self.state[p]
                
                # 初始化状态
                if len(state) == 0:
                    state['step'] = 0
                    # 第一矩估计 (动量)
                    state['m'] = torch.zeros_like(p.data)
                    # 第二矩估计 (RMS)
                    state['v'] = torch.zeros_like(p.data)
                
                # 从状态中读取值
                m, v = state['m'], state['v']
                beta1, beta2 = group['betas']
                step = state['step'] + 1  # t 从 1 开始
                
                # 更新第一矩估计
                m.mul_(beta1).add_(grad, alpha=1.0 - beta1)
                # 更新第二矩估计  
                v.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)
                
                # 偏差校正
                if group['correct_bias']:
                    bias_correction1 = 1.0 - beta1 ** step
                    bias_correction2 = 1.0 - beta2 ** step
                    # 计算调整后的学习率
                    alpha_t = group['lr'] * (bias_correction2 ** 0.5) / bias_correction1
                else:
                    alpha_t = group['lr']
                
                # 更新参数 (Adam 部分)
                denom = v.sqrt().add_(group['eps'])
                p.data.addcdiv_(m, denom, value=-alpha_t)
                
                # 应用权重衰减 (AdamW 部分)
                if group['weight_decay'] > 0.0:
                    p.data.add_(p.data, alpha=-group['lr'] * group['weight_decay'])
                
                # 更新步数
                state['step'] = step
        
        return loss

