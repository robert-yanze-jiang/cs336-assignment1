import os
from typing import BinaryIO
from multiprocessing import Pool
import regex as re
from collections import defaultdict
import heapq
from collections import Counter
import time

def train_bpe(
    input_path: str,
    vocab_size: int,
    special_tokens: list[str],
    num_processes: int = 4,
)->tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """
    Train a BPE tokenizer on the given input file.
    Returns the vocabulary and the list of merges.
    """
    print("=== Starting BPE Training ===")
    start_time = time.time()
    
    # 1. Initialize vocabulary with special tokens
    print("Step 1: Initializing vocabulary...")
    vocab = initialize_vocab(special_tokens)
    print(f"✓ Vocabulary initialized with {len(vocab)} tokens")

    # 2. Pre tokenization
    print("Step 2: Pre-tokenization...")
    pre_tokenize_start = time.time()
    
    with open(input_path, "rb") as f:
        file_size = os.path.getsize(input_path)
        print(f"File size: {file_size/(1024 * 1024):.2f} MB")
        
        boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")
        print(f"Found {len(boundaries)} boundaries")
        
        chunk_args = [(input_path, start, end, special_tokens) for start, end in 
                     zip(boundaries[:-1], boundaries[1:])]
        
        print(f"Processing {len(chunk_args)} chunks with {num_processes} processes...")
        
        # 并行处理每个分块
        with Pool(processes=num_processes) as pool:
            # 使用imap_unordered来显示进度
            chunk_results = []
            for i, result in enumerate(pool.imap_unordered(process_chunk_with_special_tokens, chunk_args)):
                chunk_results.append(result)
                if (i + 1) % max(1, len(chunk_args) // 5) == 0:  # 每20%显示进度
                    print(f"  Processed {i+1}/{len(chunk_args)} chunks")
    
    pre_tokenize_time = time.time() - pre_tokenize_start
    print(f"✓ Pre-tokenization completed in {pre_tokenize_time:.2f}s")
    
    # 3. 合并所有分块的预分词结果
    print("Step 3: Merging chunk results...")
    all_pre_tokens = []
    for i, chunk in enumerate(chunk_results):
        all_pre_tokens.extend(chunk)
        if (i + 1) % max(1, len(chunk_results) // 5) == 0:
            print(f"  Merged {i+1}/{len(chunk_results)} chunks")
    
    print(f"Total pre-tokens: {len(all_pre_tokens)}")

    # 4. 转化为字节
    print("Step 4: Converting to bytes...")
    byte_sequences = convert_pre_tokens_to_bytes(all_pre_tokens)
    print(f"Created {len(byte_sequences)} byte sequences")

    # 5. Merge
    print("Step 5: BPE merging...")
    merge_start = time.time()
    merges = perform_bpe_merging(byte_sequences, vocab_size, vocab)
    merge_time = time.time() - merge_start
    
    total_time = time.time() - start_time
    print(f"✓ BPE merging completed in {merge_time:.2f}s")
    print(f"✓ Total training time: {total_time:.2f}s")
    print(f"Final vocabulary size: {len(vocab)}")
    print(f"Number of merges: {len(merges)}")

    return vocab, merges

def perform_bpe_merging(
        pre_tokens: list[list[bytes]], 
        vocab_size: int, 
        vocab: dict[int,bytes]
) -> list[tuple[bytes, bytes]]:
    print("  Starting BPE merging process...")
    merges = []
    current_vocab_size = len(vocab)
    
    if current_vocab_size >= vocab_size:
        print("  Vocabulary already reached target size")
        return merges
    
    # 1. 初始化频率统计
    print("  Counting pair frequencies...")
    pair_freq = Counter()
    for i, token_seq in enumerate(pre_tokens):
        for j in range(len(token_seq) - 1):
            pair = (token_seq[j], token_seq[j + 1])
            pair_freq[pair] += 1
        if (i + 1) % max(1, len(pre_tokens) // 10) == 0:
            print(f"    Counted pairs for {i+1}/{len(pre_tokens)} sequences")
    
    print(f"  Found {len(pair_freq)} unique pairs")
    
    # 2. 开始合并循环
    iteration = 0
    while current_vocab_size < vocab_size and pair_freq:
        iteration += 1
        if iteration % 100 == 0:
            print(f"  Iteration {iteration}, vocab size: {current_vocab_size}, pairs remaining: {len(pair_freq)}")
        
        # 找到最频繁的对
        best_pair = find_most_frequent_pair(pair_freq)
        
        if best_pair is None:
            print("  No more pairs to merge")
            break
            
        # 应用合并
        merged_count = 0
        for sequence in pre_tokens:
            i = 0
            while i < len(sequence) - 1:
                if sequence[i] == best_pair[0] and sequence[i + 1] == best_pair[1]:
                    # 合并
                    new_token = best_pair[0] + best_pair[1]
                    sequence[i] = new_token
                    del sequence[i + 1]
                    merged_count += 1
                    
                    # 更新受影响的频率
                    update_affected_pairs(sequence, i, best_pair[0], best_pair[1], new_token, pair_freq)
                else:
                    i += 1
        
        if merged_count > 0:
            merges.append(best_pair)
            vocab[current_vocab_size] = best_pair[0] + best_pair[1]
            current_vocab_size += 1
            del pair_freq[best_pair]
        else:
            del pair_freq[best_pair]
    
    print(f"  BPE completed after {iteration} iterations")
    return merges

def update_affected_pairs(sequence: list[bytes], index: int, a: bytes, b: bytes, new_token: bytes, pair_freq: dict[tuple[bytes, bytes], int]):
    if index > 0:
        left_token = sequence[index - 1]
        old_left_pair = (left_token, a)
        new_left_pair = (left_token, new_token)

        if old_left_pair in pair_freq:
            pair_freq[old_left_pair] -= 1
        if pair_freq[old_left_pair] <= 0:
            del pair_freq[old_left_pair]
        
        pair_freq[new_left_pair] = pair_freq.get(new_left_pair, 0) + 1

    if index < len(sequence) - 1:
        right_token = sequence[index + 1]
        old_right_pair = (b, right_token)
        new_right_pair = (new_token, right_token)

        if old_right_pair in pair_freq:
            pair_freq[old_right_pair] -= 1
        if pair_freq[old_right_pair] <= 0:
            del pair_freq[old_right_pair]
        
        pair_freq[new_right_pair] = pair_freq.get(new_right_pair, 0) + 1

def find_most_frequent_pair(pair_freq: dict[tuple[bytes, bytes], int]) -> tuple[bytes, bytes]:
    if not pair_freq:
        return None
    max_freq = max(pair_freq.values())

    candidates = [pair for pair, freq in pair_freq.items() if freq == max_freq]
    candidates.sort(reverse=True)
    return candidates[0]

def convert_pre_tokens_to_bytes(pre_tokens: list[str]) -> list[list[bytes]]:
    """将预分词转换为字节序列"""
    print("  Converting pre-tokens to bytes...")
    byte_sequences = []
    for i, token in enumerate(pre_tokens):
        token_bytes = token.encode('utf-8')
        byte_list = [bytes([b]) for b in token_bytes]
        byte_sequences.append(byte_list)
        if (i + 1) % max(1, len(pre_tokens) // 10) == 0:
            print(f"    Converted {i+1}/{len(pre_tokens)} tokens")
    return byte_sequences

def initialize_vocab(special_tokens: list[str]) -> dict[int, bytes]:
    """
    Initialize the vocabulary with special tokens.
    """
    vocab = {}

    for i in range(256):
        vocab[i] = bytes([i])
    next_id = 256

    for token in special_tokens:
        vocab[next_id] = token.encode('utf-8')
        next_id += 1
    return vocab

def process_chunk_with_special_tokens(args:tuple) -> list[str]:
    input_path, start, end, special_tokens = args
    
    try:
        with open(input_path, "rb") as f:
            f.seek(start)
            chunk_bytes = f.read(end - start)
            chunk_text = chunk_bytes.decode("utf-8", errors="ignore")
            
            tokens = pre_tokenize_with_special_tokens(chunk_text, special_tokens)
            return tokens
    except Exception as e:
        print(f"Error processing chunk {start}-{end}: {e}")
        return []

def pre_tokenize_with_special_tokens(text: str, special_tokens: list[str]) -> list[str]:
    segments = split_by_special_tokens(text, special_tokens)
    all_tokens = []

    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    for segment in segments:
        if segment.strip():
            matches = re.finditer(PAT, segment)
            tokens = [match.group() for match in matches]
            all_tokens.extend(tokens)
    return all_tokens

def split_by_special_tokens(text: str, special_tokens: list[str]) -> list[str]:
    if not special_tokens:
        return [text] if text.strip() else []
    
    escaped_tokens = [re.escape(token) for token in special_tokens]
    pattern = '|'.join(escaped_tokens) 

    segments = re.split(pattern, text)
    segments =  [segment for segment in segments if segment.strip()]

    return segments

def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    print(f"  Finding boundaries for {file_size} bytes, {desired_num_chunks} chunks")

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)
        while True:
            mini_chunk = file.read(mini_chunk_size)

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    unique_boundaries = sorted(set(chunk_boundaries))
    print(f"  Final boundaries: {len(unique_boundaries)} unique points")
    return unique_boundaries

# 测试函数
def test_with_small_file():
    """用小文件测试"""
    print("=== Testing with small file ===")
    
    # 创建测试文件
    test_content = "<|endoftext|>This is a test sentence. Hello world! <|endoftext|>Another test document."
    with open("test_small.txt", "w") as f:
        for _ in range(100):  # 创建100个文档
            f.write(test_content + "\n")
    
    vocab, merges = train_bpe("test_small.txt", 1000, ["<|endoftext|>"], num_processes=2)
    
    if vocab and merges:
        print("✓ Small file test passed!")
        # 分析结果
        longest_token = max(vocab.values(), key=len)
        print(f"Longest token: {longest_token.decode('utf-8', errors='replace')} (length: {len(longest_token)})")
    else:
        print("✗ Small file test failed!")

if __name__ == "__main__":
    # 先测试小文件
    test_with_small_file()
    
    print("\n" + "="*50 + "\n")
    
    # 然后训练TinyStories
    input_file = "TinyStories-train.txt"
    if os.path.exists(input_file):
        print("=== Training on TinyStories ===")
        vocab, merges = train_bpe(input_file, 10000, ["<|endoftext|>"], num_processes=4)
    else:
        print(f"Error: {input_file} not found")