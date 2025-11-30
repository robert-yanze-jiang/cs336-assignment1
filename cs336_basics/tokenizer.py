import json
import regex as re  # 使用regex库处理Unicode
from typing import Dict, List, Tuple, Iterable, Iterator, Optional, Any

class Tokenizer:
    def __init__(self, vocab: Dict[int, bytes], merges: List[Tuple[bytes, bytes]], special_tokens: Optional[List[str]] = None):
        self.vocab = vocab.copy()
        self.merges = merges
        self.special_token_ids = {}
        
        # 处理特殊token - 使用GPT-2的ID分配策略
        if special_tokens:
            base_id = 50256
            current_id = base_id
            
            for token in special_tokens:
                token_bytes = token.encode('utf-8')
                
                # 检查是否已存在
                existing_id = None
                for vid, vbytes in self.vocab.items():
                    if vbytes == token_bytes:
                        existing_id = vid
                        break
                
                if existing_id is not None:
                    self.special_token_ids[token] = existing_id
                else:
                    # 分配新ID，确保不冲突
                    while current_id in self.vocab:
                        current_id += 1
                    self.vocab[current_id] = token_bytes
                    self.special_token_ids[token] = current_id
                    current_id += 1
        
        # 创建反向词汇表
        self.inverse_vocab = {v: k for k, v in self.vocab.items()}
        
        # 预处理合并规则为快速查找格式
        self.merge_rules = {}
        for a, b in merges:
            self.merge_rules[(a, b)] = a + b

    def encode(self, text: str) -> List[int]:
        """编码文本为token IDs"""
        # 如果有特殊token，先进行特殊token处理
        if self.special_token_ids:
            return self._encode_with_special_tokens(text)
        else:
            return self._encode_normal(text)

    def _encode_with_special_tokens(self, text: str) -> List[int]:
        """处理包含特殊token的文本编码"""
        # 按长度排序特殊token（长的优先）
        special_tokens_sorted = sorted(self.special_token_ids.keys(), key=len, reverse=True)
        
        # 分割文本：普通文本和特殊token交替
        segments = []
        i = 0
        n = len(text)
        
        while i < n:
            found_special = False
            for special_token in special_tokens_sorted:
                special_len = len(special_token)
                if i + special_len <= n and text[i:i+special_len] == special_token:
                    # 找到特殊token
                    segments.append(('special', special_token))
                    i += special_len
                    found_special = True
                    break # 结束for循环
            
            if not found_special:
                # 查找下一个特殊token的位置
                next_special_pos = n
                for special_token in special_tokens_sorted:
                    pos = text.find(special_token, i)
                    if pos != -1 and pos < next_special_pos:
                        next_special_pos = pos
                
                if next_special_pos > i:
                    # 添加普通文本段
                    segments.append(('text', text[i:next_special_pos]))
                    i = next_special_pos
                else:
                    # 添加剩余文本
                    segments.append(('text', text[i:]))
                    break
        
        # 编码每个段
        ids = []
        for seg_type, content in segments:
            if seg_type == 'text':
                ids.extend(self._encode_normal(content))
            else:
                ids.append(self.special_token_ids[content])
        
        return ids

    def _encode_normal(self, text: str) -> List[int]:
        """编码普通文本（不包含特殊token）"""
        if not text:
            return []
        
        # GPT-2风格的预分词
        pre_tokens = self._gpt2_pretokenize(text)
        
        # 对每个pre-token独立应用BPE
        all_ids = []
        for pre_token in pre_tokens:
            if pre_token:  # 跳过空字符串
                ids = self._bpe_encode_single_token(pre_token)
                all_ids.extend(ids)
        
        return all_ids

    def _gpt2_pretokenize(self, text: str) -> List[str]:
        """GPT-2风格的预分词"""
        # 使用GPT-2的预分词策略
        # 这个正则表达式复制了GPT-2的分词行为
        pattern = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        
        try:
            # 使用regex库处理Unicode属性
            tokens = re.findall(pattern, text, flags=re.UNICODE)
        except:
            # 回退到简单空格分割
            tokens = text.split()
        
        return tokens

    def _bpe_encode_single_token(self, token: str) -> List[int]:
        """对单个pre-token应用BPE编码"""
        if not token:
            return []
        
        # 转换为字节序列
        try:
            bytes_seq = token.encode('utf-8')
        except UnicodeEncodeError:
            bytes_seq = token.encode('utf-8', errors='replace')
        
        # 初始化为单个字节的token
        word = [bytes([b]) for b in bytes_seq]
        
        # 应用所有合并规则
        # 注意：这里需要按照merges列表的顺序应用
        merges = self._get_all_possible_merges(word)
        
        # 按merges列表的顺序应用合并
        for merge_pair in self.merges:
            if merge_pair in merges:
                # 应用这个合并
                new_word = []
                i = 0
                while i < len(word):
                    if i < len(word) - 1 and word[i] == merge_pair[0] and word[i+1] == merge_pair[1]:
                        new_word.append(merge_pair[0] + merge_pair[1])
                        i += 2
                    else:
                        new_word.append(word[i])
                        i += 1
                word = new_word
                
                # 重新计算可能的合并
                merges = self._get_all_possible_merges(word)
        
        # 将token映射为ID
        ids = []
        for token_bytes in word:
            if token_bytes in self.inverse_vocab:
                ids.append(self.inverse_vocab[token_bytes])
            else:
                # 处理未知token
                ids.extend(self._handle_unknown_token_bytes(token_bytes))
        
        return ids

    def _get_all_possible_merges(self, tokens: List[bytes]) -> set:
        """获取所有可能的合并对"""
        merges = set()
        for i in range(len(tokens) - 1):
            pair = (tokens[i], tokens[i+1])
            if pair in self.merge_rules:
                merges.add(pair)
        return merges

    def _handle_unknown_token_bytes(self, token_bytes: bytes) -> List[int]:
        """处理未知的字节token"""
        ids = []
        
        # 如果有<unk>特殊token，使用它
        if '<unk>' in self.special_token_ids:
            ids.append(self.special_token_ids['<unk>'])
        else:
            # 否则拆分为单个字节
            for b in token_bytes:
                byte_token = bytes([b])
                if byte_token in self.inverse_vocab:
                    ids.append(self.inverse_vocab[byte_token])
                else:
                    # 如果单字节也不在词汇表中，使用最小的可用ID
                    if self.vocab:
                        min_id = min(self.vocab.keys())
                        ids.append(min_id)
                    else:
                        ids.append(0)
        
        return ids

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        """流式编码"""
        for text in iterable:
            yield from self.encode(text)

    def decode(self, ids: List[int]) -> str:
        """解码token IDs为文本"""
        byte_sequences = []
        
        for token_id in ids:
            if token_id in self.vocab:
                byte_sequences.append(self.vocab[token_id])
            else:
                # 未知ID使用Unicode替换字符
                byte_sequences.append(b'\xef\xbf\xbd')  # U+FFFD
        
        # 合并所有字节
        combined_bytes = b''.join(byte_sequences)
        
        try:
            # 尝试UTF-8解码
            return combined_bytes.decode('utf-8')
        except UnicodeDecodeError:
            # 替换无效字节序列
            return combined_bytes.decode('utf-8', errors='replace')

    @classmethod
    def from_files(cls, vocab_filepath: str, merges_filepath: str, special_tokens: Optional[List[str]] = None):
        """从文件加载分词器"""
        # 读取词汇表
        with open(vocab_filepath, 'r', encoding='utf-8') as f:
            vocab_data = json.load(f)
            vocab = {}
            for k, v in vocab_data.items():
                if isinstance(v, str):
                    vocab[int(k)] = v.encode('utf-8')
                else:
                    vocab[int(k)] = bytes(v)
        
        # 读取合并规则
        merges = []
        with open(merges_filepath, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                parts = line.split()
                if len(parts) >= 2:
                    a, b = parts[0], parts[1]
                    merges.append((a.encode('utf-8'), b.encode('utf-8')))
        
        return cls(vocab, merges, special_tokens)