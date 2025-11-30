import math

def get_lr_cosine_schedule(max_learning_rate, min_learning_rate, warmup_iters, cosine_cycle_iters):
    """
    创建余弦学习率调度函数
    
    参数:
        max_learning_rate: 最大学习率 (alpha_max)
        min_learning_rate: 最小学习率 (alpha_min)  
        warmup_iters: 预热步数 (T_w)
        cosine_cycle_iters: 余弦退火总步数 (T_c)
    
    返回:
        一个函数，该函数接收步数it并返回对应的学习率
    """
    
    # 输入验证
    if warmup_iters < 0 or cosine_cycle_iters < 0:
        raise ValueError("warmup_iters和cosine_cycle_iters不能为负数")
    if warmup_iters > cosine_cycle_iters:
        raise ValueError("预热步数warmup_iters不能大于总步数cosine_cycle_iters")
    if min_learning_rate > max_learning_rate:
        raise ValueError("最小学习率不能大于最大学习率")
    
    def lr_schedule(it):
        """
        根据步数计算学习率
        
        参数:
            it: 当前训练步数
            
        返回:
            第it步的学习率
        """
        t = it
        
        if t < 0:
            raise ValueError("步数t不能为负数")
        
        # 1. 预热阶段 (Warm-up)
        if t < warmup_iters:
            # α_t = (t / warmup_iters) * max_learning_rate
            if warmup_iters == 0:
                return max_learning_rate
            return (t / warmup_iters) * max_learning_rate
        
        # 2. 余弦退火阶段 (Cosine annealing)
        elif t <= cosine_cycle_iters:
            # α_t = min_learning_rate + 0.5 * (1 + cos(π * (t - warmup_iters) / (cosine_cycle_iters - warmup_iters))) * (max_learning_rate - min_learning_rate)
            if cosine_cycle_iters == warmup_iters:
                return min_learning_rate
            
            progress = (t - warmup_iters) / (cosine_cycle_iters - warmup_iters)
            cosine_factor = 0.5 * (1 + math.cos(math.pi * progress))
            return min_learning_rate + cosine_factor * (max_learning_rate - min_learning_rate)
        
        # 3. 退火后阶段 (Post-annealing)
        else:
            return min_learning_rate
    
    return lr_schedule