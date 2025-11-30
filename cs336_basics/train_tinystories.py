# cs336_basics/train_tinystories_fixed.py
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import logging
from pathlib import Path
import sys
import os
import inspect

# 添加项目根目录到路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)

# 导入你已有的代码
try:
    from cs336_basics.transformer_lm import TransformerLM
    from cs336_basics.AdamW import get_adamw_cls
    from cs336_basics.learning_rate_schedule import get_lr_cosine_schedule
    from cs336_basics.get_batch import get_batch
    from cs336_basics.checkpoint import save_checkpoint, load_checkpoint
    from cs336_basics.cross_entropy import cross_entropy
    print("✅ 成功导入已有组件")
    
    # 检查TransformerLM的参数
    sig = inspect.signature(TransformerLM.__init__)
    print(f"TransformerLM参数: {list(sig.parameters.keys())}")
    
except ImportError as e:
    print(f"❌ 导入错误: {e}")
    sys.exit(1)

class CS336Trainer:
    def __init__(self, config):
        self.config = config
        self.device = config['device']
        
        # 检查TransformerLM的参数并正确初始化
        sig = inspect.signature(TransformerLM.__init__)
        init_params = {}
        
        # 只传递TransformerLM接受的参数
        for param_name in sig.parameters:
            if param_name == 'self':
                continue
            if param_name in config:
                init_params[param_name] = config[param_name]
        
        print(f"初始化TransformerLM参数: {init_params}")
        
        # 创建模型
        self.model = TransformerLM(**init_params).to(self.device)
        
        # 使用你已有的优化器
        AdamW = get_adamw_cls()
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=config['learning_rate'],
            betas=(config.get('beta1', 0.9), config.get('beta2', 0.999)),
            eps=config.get('epsilon', 1e-8),
            weight_decay=config.get('weight_decay', 0.01)
        )
        
        # 使用你已有的学习率调度器
        self.lr_schedule = get_lr_cosine_schedule(
            max_learning_rate=config['learning_rate'],
            min_learning_rate=config.get('min_learning_rate', config['learning_rate'] * 0.1),
            warmup_iters=config.get('warmup_iters', 1000),
            cosine_cycle_iters=config.get('total_iters', 10000)
        )
        
        self.iteration = 0
        self.train_data = self.load_data()
        
        print(f"✅ 模型参数: {sum(p.numel() for p in self.model.parameters()):,}")
    
    def load_data(self):
        """加载TinyStories数据"""
        data_path = self.config.get('train_data_path', 
            '/Users/jiangyanze/Desktop/CS336/Assignment1/data/TinyStoriesV2-GPT4-train.txt')
        
        print(f"📖 加载数据: {data_path}")
        
        try:
            # 读取文本文件
            with open(data_path, 'r', encoding='utf-8') as f:
                text = f.read()
            
            # 简单的字符级标记化
            vocab = {chr(i): i for i in range(256)}  # ASCII字符
            vocab_size = len(vocab)
            
            # 将文本转换为标记
            tokens = [vocab.get(c, vocab.get(' ', 32)) for c in text[:1000000]]  # 限制大小
            
            print(f"✅ 加载成功: {len(tokens):,} 个标记")
            # 确保使用int64类型
            return np.array(tokens, dtype=np.int64)
            
        except Exception as e:
            print(f"❌ 数据加载失败: {e}")
            print("使用模拟数据...")
            return self.create_simulated_data()
    
    def create_simulated_data(self):
        """创建模拟数据"""
        vocab_size = self.config['vocab_size']
        seq_length = 1000000
        # 确保使用int64类型
        return np.random.randint(0, vocab_size, size=seq_length, dtype=np.int64)
    
    def compute_loss(self, logits, targets):
        """使用你已有的交叉熵函数"""
        return cross_entropy(logits, targets)
    
    def get_batch_fixed(self, data, batch_size=None, context_length=None):
        """修复版本的get_batch，确保数据类型正确"""
        if batch_size is None:
            batch_size = self.config['batch_size']
        if context_length is None:
            context_length = self.config['context_length']
        
        # 计算可用的起始位置
        max_start_idx = len(data) - context_length - 1
        
        # 随机选择起始位置
        start_indices = np.random.randint(0, max_start_idx, size=batch_size)
        
        # 创建索引矩阵用于向量化提取
        indices = start_indices[:, None] + np.arange(context_length)
        
        # 提取输入序列
        inputs = data[indices]  # [batch_size, context_length]
        
        # 目标序列是输入序列向后移动一位
        target_indices = start_indices[:, None] + np.arange(1, context_length + 1)
        targets = data[target_indices]  # [batch_size, context_length]
        
        # 转换为PyTorch张量并移动到指定设备
        # 确保使用torch.int64类型
        inputs_tensor = torch.from_numpy(inputs).to(torch.int64).to(self.device)
        targets_tensor = torch.from_numpy(targets).to(torch.int64).to(self.device)
        
        return inputs_tensor, targets_tensor
    
    def train_step(self):
        """单个训练步骤"""
        self.model.train()
        
        # 使用修复的get_batch函数
        inputs, targets = self.get_batch_fixed(self.train_data)
        
        # 前向传播
        logits = self.model(inputs)
        loss = self.compute_loss(logits, targets)
        
        # 反向传播
        self.optimizer.zero_grad()
        loss.backward()
        
        # 梯度裁剪
        if self.config.get('max_grad_norm', 1.0) > 0:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), 
                self.config['max_grad_norm']
            )
        
        # 更新学习率
        current_lr = self.lr_schedule(self.iteration)
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = current_lr
        
        self.optimizer.step()
        self.iteration += 1
        
        return loss.item(), current_lr
    
    def evaluate(self):
        """评估模型"""
        self.model.eval()
        with torch.no_grad():
            inputs, targets = self.get_batch_fixed(self.train_data)
            logits = self.model(inputs)
            loss = self.compute_loss(logits, targets)
        return loss.item()
    
    def train(self, num_steps=None):
        """主训练循环"""
        total_iters = self.config.get('total_iters', 100)  # 先用100步测试
        train_losses = []
        val_losses = []
        
        print(f"🚀 开始训练 {total_iters} 步")
        print("=" * 60)
        
        for step in range(total_iters):
            train_loss, current_lr = self.train_step()
            train_losses.append(train_loss)
            
            # 记录日志
            if step % 10 == 0 or step == total_iters - 1:
                val_loss = self.evaluate()
                val_losses.append(val_loss)
                
                tokens_processed = (step + 1) * self.config['batch_size'] * self.config['context_length']
                print(f"Step {step:5d} | "
                      f"LR: {current_lr:.2e} | "
                      f"Train: {train_loss:.4f} | "
                      f"Val: {val_loss:.4f} | "
                      f"Tokens: {tokens_processed:,}")
            
            # 保存检查点
            if step % 100 == 0 and step > 0:
                checkpoint_path = f"checkpoint_step_{step}.pt"
                save_checkpoint(self.model, self.optimizer, step, checkpoint_path)
                print(f"💾 保存检查点: {checkpoint_path}")
        
        return train_losses, val_losses

def run_lr_sweep_experiment():
    """运行学习率扫描实验"""
    
    # 基础配置
    base_config = {
        # 数据配置
        'train_data_path': '/Users/jiangyanze/Desktop/CS336/Assignment1/data/TinyStoriesV2-GPT4-train.txt',
        'vocab_size': 10000,
        
        # 模型配置 - 只包含TransformerLM实际接受的参数
        'd_model': 512,
        'num_layers': 6,
        'num_heads': 8,
        'd_ff': 2048,
        'context_length': 256,  # 注意：这是TransformerLM需要的参数
        'theta': 10000,  # RoPE的theta参数
        
        # 训练配置
        'batch_size': 32,
        'total_iters': 100,  # 先用100步测试
        'max_grad_norm': 1.0,
        
        # 优化器配置
        'beta1': 0.9,
        'beta2': 0.999,
        'epsilon': 1e-8,
        'weight_decay': 0.01,
        
        # 系统配置
        'device': 'cpu',
        'min_learning_rate': 1e-5,
        'warmup_iters': 100
    }
    
    # 学习率扫描范围
    learning_rates = [1e-5, 1e-4, 1e-3]  # 先用少量学习率测试
    results = {}
    
    print("🎯 CS336 学习率扫描实验")
    print("=" * 60)
    
    for i, lr in enumerate(learning_rates):
        print(f"\n🔬 实验 {i+1}/{len(learning_rates)}: LR = {lr:.2e}")
        print("-" * 40)
        
        config = base_config.copy()
        config['learning_rate'] = lr
        
        try:
            trainer = CS336Trainer(config)
            train_losses, val_losses = trainer.train()
            
            final_val_loss = val_losses[-1] if val_losses else float('inf')
            results[lr] = {
                'train_losses': train_losses,
                'val_losses': val_losses,
                'final_val_loss': final_val_loss,
                'status': '正常'
            }
            
            print(f"✅ LR={lr:.2e} 完成 | 最终损失: {final_val_loss:.4f}")
            
        except Exception as e:
            print(f"❌ LR={lr:.2e} 失败: {e}")
            import traceback
            traceback.print_exc()  # 打印详细错误信息
            results[lr] = {
                'train_losses': [], 'val_losses': [], 
                'final_val_loss': float('inf'), 'status': '失败'
            }
    
    return results

if __name__ == "__main__":
    results = run_lr_sweep_experiment()
    
    # 分析结果
    if results:
        best_lr = None
        best_loss = float('inf')
        
        for lr, data in sorted(results.items()):
            status = "✅正常" if data['status'] == '正常' else "❌失败"
            loss_str = f"{data['final_val_loss']:.4f}" if data['final_val_loss'] < float('inf') else "失败"
            print(f"LR={lr:.2e}: {status}, 最终损失={loss_str}")
            
            if data['status'] == '正常' and data['final_val_loss'] < best_loss:
                best_lr = lr
                best_loss = data['final_val_loss']
        
        if best_lr:
            print(f"\n🏆 最佳结果:")
            print(f"   学习率: {best_lr:.2e}")
            print(f"   验证损失: {best_loss:.4f}")
            print(f"   是否达标: {'✅是' if best_loss <= 2.00 else '❌否'}")