#!/usr/bin/env python3
"""
CS336 训练脚本
将模型、优化器、数据加载和检查点功能整合在一起
"""

import argparse
import os
import time
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
import logging
from typing import Dict, Any

# 导入之前实现的组件
from cs336_basics.transformer_lm import TransformerLM
from cs336_basics.AdamW import AdamW
from cs336_basics.learning_rate_schedule import get_lr_cosine_schedule
from cs336_basics.get_batch import get_batch
from cs336_basics.checkpoint import save_checkpoint, load_checkpoint
from cs336_basics.cross_entropy import cross_entropy

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class Trainer:
    def __init__(self, config: Dict[str, Any]):
        """初始化训练器"""
        self.config = config
        self.device = config['device']
        self.setup_training()
        
    def setup_training(self):
        """设置训练环境"""
        # 设置随机种子
        torch.manual_seed(self.config['seed'])
        np.random.seed(self.config['seed'])
        
        # 创建模型
        self.model = TransformerLM(
            vocab_size=self.config['vocab_size'],
            d_model=self.config['d_model'],
            num_layers=self.config['num_layers'],
            num_heads=self.config['num_heads'],
            d_ff=self.config['d_ff'],
            dropout=self.config['dropout'],
            max_seq_len=self.config['max_seq_len']
        ).to(self.device)
        
        # 计算参数数量
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        logger.info(f"模型参数: 总计 {total_params:,}，可训练 {trainable_params:,}")
        
        # 创建优化器
        
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=self.config['learning_rate'],
            betas=(self.config['beta1'], self.config['beta2']),
            eps=self.config['epsilon'],
            weight_decay=self.config['weight_decay']
        )
        
        # 创建学习率调度器
        self.lr_schedule = get_lr_cosine_schedule(
            max_learning_rate=self.config['learning_rate'],
            min_learning_rate=self.config['min_learning_rate'],
            warmup_iters=self.config['warmup_iters'],
            cosine_cycle_iters=self.config['total_iters']
        )
        
        # 加载数据集
        self.load_datasets()
        
        # 恢复检查点（如果存在）
        self.iteration = 0
        if self.config['resume_from'] and os.path.exists(self.config['resume_from']):
            self.iteration = load_checkpoint(self.config['resume_from'], self.model, self.optimizer)
            logger.info(f"从检查点恢复训练，迭代次数: {self.iteration}")
        
    def load_datasets(self):
        """使用内存映射加载数据集"""
        logger.info("加载数据集...")
        
        # 训练数据
        if self.config['train_data'].endswith('.npy'):
            self.train_data = np.load(self.config['train_data'], mmap_mode='r')
        else:
            # 假设是文本文件，需要预处理
            self.train_data = self.load_text_data(self.config['train_data'])
        
        # 验证数据（如果有）
        if self.config.get('val_data'):
            if self.config['val_data'].endswith('.npy'):
                self.val_data = np.load(self.config['val_data'], mmap_mode='r')
            else:
                self.val_data = self.load_text_data(self.config['val_data'])
        else:
            self.val_data = None
            
        logger.info(f"训练数据大小: {len(self.train_data):,} tokens")
        if self.val_data is not None:
            logger.info(f"验证数据大小: {len(self.val_data):,} tokens")
    
    def load_text_data(self, filepath):
        """从文本文件加载数据（简化版本）"""
        # 这里需要根据实际数据格式实现
        # 假设每行是一个整数token
        with open(filepath, 'r') as f:
            tokens = [int(line.strip()) for line in f if line.strip()]
        return np.array(tokens, dtype=np.int64)
    
    def get_batch(self, data, batch_size=None, context_length=None):
        """获取批次数据"""
        if batch_size is None:
            batch_size = self.config['batch_size']
        if context_length is None:
            context_length = self.config['context_length']
            
        return get_batch(data, batch_size, context_length, self.device)
    
    def compute_loss(self, logits, targets):
        """计算交叉熵损失"""
        # 展平批次和序列维度
        logits_flat = logits.view(-1, logits.size(-1))
        targets_flat = targets.view(-1)
        
        # 计算交叉熵损失
        loss = cross_entropy(logits_flat, targets_flat)
        return loss
    
    def train_step(self):
        """执行一个训练步骤"""
        self.model.train()
        
        # 获取训练批次
        inputs, targets = self.get_batch(self.train_data)
        
        # 前向传播
        logits = self.model(inputs)
        loss = self.compute_loss(logits, targets)
        
        # 反向传播
        self.optimizer.zero_grad()
        loss.backward()
        
        # 梯度裁剪
        if self.config['max_grad_norm'] > 0:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), 
                self.config['max_grad_norm']
            )
        
        # 更新学习率
        current_lr = self.lr_schedule(self.iteration)
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = current_lr
        
        # 优化器步骤
        self.optimizer.step()
        
        return loss.item(), current_lr
    
    def evaluate(self):
        """在验证集上评估模型"""
        if self.val_data is None:
            return None
            
        self.model.eval()
        total_loss = 0
        num_batches = 0
        
        with torch.no_grad():
            for _ in range(self.config['eval_iters']):
                inputs, targets = self.get_batch(
                    self.val_data, 
                    self.config['eval_batch_size'],
                    self.config['context_length']
                )
                logits = self.model(inputs)
                loss = self.compute_loss(logits, targets)
                total_loss += loss.item()
                num_batches += 1
        
        return total_loss / num_batches if num_batches > 0 else 0
    
    def save_checkpoint(self, iteration=None):
        """保存检查点"""
        if iteration is None:
            iteration = self.iteration
            
        checkpoint_path = Path(self.config['checkpoint_dir']) / f"checkpoint_{iteration:08d}.pt"
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        
        save_checkpoint(self.model, self.optimizer, iteration, checkpoint_path)
        logger.info(f"保存检查点到: {checkpoint_path}")
        
        # 保留最新的几个检查点
        self.cleanup_old_checkpoints()
    
    def cleanup_old_checkpoints(self, keep_last=3):
        """清理旧的检查点，只保留最新的几个"""
        checkpoint_dir = Path(self.config['checkpoint_dir'])
        checkpoints = sorted(checkpoint_dir.glob("checkpoint_*.pt"))
        
        if len(checkpoints) > keep_last:
            for old_checkpoint in checkpoints[:-keep_last]:
                old_checkpoint.unlink()
                logger.debug(f"删除旧检查点: {old_checkpoint}")
    
    def train(self):
        """主训练循环"""
        logger.info("开始训练...")
        start_time = time.time()
        
        # 训练统计
        train_losses = []
        val_losses = []
        
        while self.iteration < self.config['total_iters']:
            # 训练步骤
            train_loss, current_lr = self.train_step()
            train_losses.append(train_loss)
            
            # 记录训练进度
            if self.iteration % self.config['log_interval'] == 0:
                avg_train_loss = np.mean(train_losses[-self.config['log_interval']:])
                elapsed_time = time.time() - start_time
                tokens_processed = (self.iteration + 1) * self.config['batch_size'] * self.config['context_length']
                
                logger.info(
                    f"迭代 {self.iteration:6d} | "
                    f"训练损失: {avg_train_loss:.4f} | "
                    f"学习率: {current_lr:.2e} | "
                    f"时间: {elapsed_time:.1f}s | "
                    f"tokens: {tokens_processed:,}"
                )
            
            # 验证
            if self.val_data is not None and self.iteration % self.config['eval_interval'] == 0:
                val_loss = self.evaluate()
                val_losses.append(val_loss)
                logger.info(f"验证损失: {val_loss:.4f}")
            
            # 保存检查点
            if self.iteration % self.config['checkpoint_interval'] == 0:
                self.save_checkpoint()
            
            self.iteration += 1
        
        # 训练完成，保存最终检查点
        self.save_checkpoint()
        total_time = time.time() - start_time
        logger.info(f"训练完成！总时间: {total_time:.1f}s")
        
        return train_losses, val_losses

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='CS336 语言模型训练脚本')
    
    # 数据参数
    parser.add_argument('--train-data', type=str, required=True, help='训练数据路径')
    parser.add_argument('--val-data', type=str, help='验证数据路径')
    parser.add_argument('--vocab-size', type=int, default=10000, help='词汇表大小')
    
    # 模型参数
    parser.add_argument('--d-model', type=int, default=512, help='模型维度')
    parser.add_argument('--num-layers', type=int, default=6, help='层数')
    parser.add_argument('--num-heads', type=int, default=8, help='注意力头数')
    parser.add_argument('--d-ff', type=int, default=2048, help='前馈网络维度')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout率')
    parser.add_argument('--max-seq-len', type=int, default=1024, help='最大序列长度')
    
    # 训练参数
    parser.add_argument('--batch-size', type=int, default=32, help='批次大小')
    parser.add_argument('--context-length', type=int, default=256, help='上下文长度')
    parser.add_argument('--total-iters', type=int, default=10000, help='总迭代次数')
    parser.add_argument('--learning-rate', type=float, default=1e-3, help='学习率')
    parser.add_argument('--min-learning-rate', type=float, default=1e-5, help='最小学习率')
    parser.add_argument('--warmup-iters', type=int, default=1000, help='预热迭代次数')
    parser.add_argument('--max-grad-norm', type=float, default=1.0, help='梯度裁剪范数')
    
    # 优化器参数
    parser.add_argument('--beta1', type=float, default=0.9, help='AdamW beta1')
    parser.add_argument('--beta2', type=float, default=0.999, help='AdamW beta2')
    parser.add_argument('--epsilon', type=float, default=1e-8, help='AdamW epsilon')
    parser.add_argument('--weight-decay', type=float, default=0.01, help='权重衰减')
    
    # 检查点和恢复
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints', help='检查点目录')
    parser.add_argument('--resume-from', type=str, help='从检查点恢复训练')
    
    # 日志和评估
    parser.add_argument('--log-interval', type=int, default=100, help='日志间隔')
    parser.add_argument('--eval-interval', type=int, default=1000, help='评估间隔')
    parser.add_argument('--checkpoint-interval', type=int, default=5000, help='检查点保存间隔')
    parser.add_argument('--eval-iters', type=int, default=10, help='评估迭代次数')
    parser.add_argument('--eval-batch-size', type=int, default=32, help='评估批次大小')
    
    # 系统参数
    parser.add_argument('--device', type=str, default='mps', help='设备 (cpu, cuda, mps)')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    
    args = parser.parse_args()
    
    # 创建训练器并开始训练
    trainer = Trainer(vars(args))
    trainer.train()

if __name__ == "__main__":
    main()