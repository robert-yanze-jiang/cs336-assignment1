import torch
import torch.nn.functional as F
from typing import List, Optional, Union
import numpy as np

class TextGenerator:
    def __init__(self, model, tokenizer, device='cuda' if torch.cuda.is_available() else 'cpu'):
        """
        初始化文本生成器
        
        Args:
            model: 训练好的语言模型
            tokenizer: 分词器
            device: 设备
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.model.to(device)
        self.model.eval()  # 设置为评估模式
        
        # 特殊标记
        self.eos_token_id = tokenizer.eos_token_id if tokenizer.eos_token_id else tokenizer.convert_tokens_to_ids('<|endoftext|>')
        
    def _apply_temperature(self, logits: torch.Tensor, temperature: float) -> torch.Tensor:
        """
        应用温度缩放
        
        Args:
            logits: 模型输出的logits
            temperature: 温度参数
            
        Returns:
            温度缩放后的logits
        """
        if temperature <= 0:
            raise ValueError("Temperature must be positive")
        
        return logits / temperature
    
    def _apply_top_p(self, probs: torch.Tensor, top_p: float) -> torch.Tensor:
        """
        应用top-p采样
        
        Args:
            probs: 概率分布
            top_p: top-p阈值
            
        Returns:
            经过top-p筛选后的概率分布
        """
        if top_p < 0 or top_p > 1.0:
            raise ValueError("top_p must be in [0, 1]")
        
        if top_p == 1.0:
            return probs
        
        # 按概率降序排序
        sorted_probs, sorted_indices = torch.sort(probs, descending=True)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
        
        # 移除累积概率超过top_p的部分
        sorted_indices_to_remove = cumulative_probs > top_p
        # 确保至少保留一个token
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        
        # 创建筛选后的概率分布
        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        filtered_probs = probs.clone()
        filtered_probs[indices_to_remove] = 0
        
        # 重新归一化
        if filtered_probs.sum() > 0:
            filtered_probs = filtered_probs / filtered_probs.sum()
        
        return filtered_probs
    
    def _sample_next_token(self, logits: torch.Tensor, temperature: float = 1.0, 
                          top_p: float = 1.0, top_k: int = 0) -> int:
        """
        从logits中采样下一个token
        
        Args:
            logits: 模型输出的logits (vocab_size,)
            temperature: 温度参数
            top_p: top-p采样参数
            top_k: top-k采样参数（可选）
            
        Returns:
            采样的token id
        """
        # 应用温度缩放
        if temperature != 1.0:
            logits = self._apply_temperature(logits, temperature)
        
        # 转换为概率
        probs = F.softmax(logits, dim=-1)
        
        # 应用top-p采样
        if top_p < 1.0:
            probs = self._apply_top_p(probs, top_p)
        
        # 可选：应用top-k采样
        if top_k > 0:
            top_k_probs, top_k_indices = torch.topk(probs, min(top_k, probs.size(-1)))
            probs = torch.zeros_like(probs)
            probs[top_k_indices] = top_k_probs
            probs = probs / probs.sum()
        
        # 从概率分布中采样
        next_token_id = torch.multinomial(probs, num_samples=1).item()
        
        return next_token_id
    
    def generate(self, 
                 prompt: str, 
                 max_length: int = 100,
                 temperature: float = 1.0,
                 top_p: float = 1.0,
                 top_k: int = 0,
                 num_return_sequences: int = 1,
                 do_sample: bool = True) -> Union[List[str], str]:
        """
        生成文本
        
        Args:
            prompt: 提示文本
            max_length: 最大生成长度
            temperature: 温度参数
            top_p: top-p采样参数
            top_k: top-k采样参数
            num_return_sequences: 返回的序列数量
            do_sample: 是否使用采样（False时使用贪婪解码）
            
        Returns:
            生成的文本列表或单个文本
        """
        all_sequences = []
        
        for _ in range(num_return_sequences):
            # 编码提示文本
            input_ids = self.tokenizer.encode(prompt, return_tensors='pt').to(self.device)
            generated_ids = input_ids.clone()
            
            with torch.no_grad():
                for _ in range(max_length):
                    # 获取模型输出
                    outputs = self.model(generated_ids)
                    logits = outputs.logits if hasattr(outputs, 'logits') else outputs
                    
                    # 取最后一个位置的logits
                    next_token_logits = logits[0, -1, :]
                    
                    # 采样下一个token
                    if do_sample:
                        next_token_id = self._sample_next_token(
                            next_token_logits, temperature, top_p, top_k
                        )
                    else:
                        # 贪婪解码：选择概率最大的token
                        next_token_id = torch.argmax(next_token_logits, dim=-1).item()
                    
                    # 将新token添加到序列中
                    generated_ids = torch.cat([
                        generated_ids, 
                        torch.tensor([[next_token_id]], device=self.device)
                    ], dim=1)
                    
                    # 如果生成了结束标记，停止生成
                    if next_token_id == self.eos_token_id:
                        break
            
            # 解码生成的文本
            generated_text = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
            all_sequences.append(generated_text)
        
        return all_sequences if num_return_sequences > 1 else all_sequences[0]
    
    def generate_with_details(self, 
                             prompt: str, 
                             max_length: int = 100,
                             temperature: float = 1.0,
                             top_p: float = 1.0) -> dict:
        """
        生成文本并返回详细信息
        
        Returns:
            包含生成文本和详细信息的字典
        """
        input_ids = self.tokenizer.encode(prompt, return_tensors='pt').to(self.device)
        generated_ids = input_ids.clone().tolist()[0]
        generation_details = []
        
        with torch.no_grad():
            for step in range(max_length):
                outputs = self.model(torch.tensor([generated_ids], device=self.device))
                logits = outputs.logits if hasattr(outputs, 'logits') else outputs
                
                next_token_logits = logits[0, -1, :]
                
                # 应用温度缩放
                if temperature != 1.0:
                    next_token_logits = self._apply_temperature(next_token_logits, temperature)
                
                # 获取概率
                probs = F.softmax(next_token_logits, dim=-1)
                
                # 应用top-p
                if top_p < 1.0:
                    probs = self._apply_top_p(probs, top_p)
                
                # 采样
                next_token_id = torch.multinomial(probs, num_samples=1).item()
                
                # 记录生成信息
                token_info = {
                    'step': step,
                    'token_id': next_token_id,
                    'token': self.tokenizer.decode([next_token_id]),
                    'top_probabilities': torch.topk(probs, 5).values.tolist(),
                    'top_tokens': [self.tokenizer.decode([idx]) for idx in torch.topk(probs, 5).indices.tolist()]
                }
                generation_details.append(token_info)
                
                generated_ids.append(next_token_id)
                
                if next_token_id == self.eos_token_id:
                    break
        
        return {
            'generated_text': self.tokenizer.decode(generated_ids, skip_special_tokens=True),
            'generation_details': generation_details,
            'total_tokens': len(generated_ids) - len(input_ids[0])
        }

# 使用示例
def example_usage():
    # 假设已经有训练好的model和tokenizer
    # model = YourTrainedModel()
    # tokenizer = YourTokenizer()
    
    # 创建生成器
    # generator = TextGenerator(model, tokenizer)
    
    # 基本生成
    # result = generator.generate("今天天气很好，", max_length=50)
    
    # 使用温度缩放
    # result = generator.generate("人工智能是", temperature=0.8, max_length=100)
    
    # 使用top-p采样
    # result = generator.generate("未来的技术发展", top_p=0.9, temperature=0.7, max_length=150)
    
    # 贪婪解码（不使用采样）
    # result = generator.generate("机器学习", do_sample=False, max_length=50)
    
    # 生成多个序列
    # results = generator.generate("你好", num_return_sequences=3, max_length=30)
    
    pass

# 测试函数
def test_generator():
    """测试生成器的基本功能"""
    # 这里需要实际的model和tokenizer来测试
    print("文本生成器实现完成")
    print("支持的功能：")
    print("- 温度缩放 (temperature scaling)")
    print("- Top-p采样 (nucleus sampling)") 
    print("- Top-k采样")
    print("- 贪婪解码")
    print("- 批量生成")
    print("- 详细生成信息")

if __name__ == "__main__":
    test_generator()