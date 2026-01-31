import time
from reasoners.tokenizer import QwenTokenizer

def benchmark_cpp_tokenizer(tokenizer, text, num_runs=100):
    total_time = 0
    
    for _ in range(num_runs):
        start_time = time.perf_counter()
        tokens = tokenizer.encode(text)
        end_time = time.perf_counter()
        total_time += (end_time - start_time)
    
    avg_time = total_time / num_runs
    return len(tokens), avg_time, len(tokens) / avg_time

def benchmark_hf_tokenizer(model_name, text, num_runs=100):

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    total_time = 0
    token_count = 0
    
    for _ in range(num_runs):
        start_time = time.perf_counter()
        result = tokenizer.encode(text)
        end_time = time.perf_counter()
        total_time += (end_time - start_time)
        token_count = len(result)
    
    avg_time = total_time / num_runs
    return token_count, avg_time, token_count / avg_time

def main():
    long_text = "The quick brown fox jumps over the lazy dog. " * 50
    
    print("benchmarking reasoners vs qwen tokenizer")
    print(f"length: {len(long_text)} characters")
    print()
    
    print("loading reasoners tokenizer")
    cpp_tokenizer = QwenTokenizer()
    
    print("benchmarking reasoners tokenizer")
    cpp_tokens, cpp_time, cpp_tps = benchmark_cpp_tokenizer(cpp_tokenizer, long_text)
    
    print("benchmarking qwen tokenizer")
    hf_tokens, hf_time, hf_tps = benchmark_hf_tokenizer("Qwen/Qwen2.5-1.5B-Instruct", long_text)
    
    print()
    print("results")
    print(f"reasoners tokenizer:")
    print(f"  tokens:              {cpp_tokens}")
    print(f"  avg time:            {cpp_time*1000:.4f} ms")
    print(f"  tokens/second:       {cpp_tps:,.0f}")
    print()

    if hf_tokens is not None:
        print(f"qwen Tokenizer:")
        print(f"  tokens:              {hf_tokens}")
        print(f"  avg time:            {hf_time*1000:.4f} ms")
        print(f"  tokens/second:       {hf_tps:,.0f}")
        print()
        
        speedup = hf_time / cpp_time
        print(f"impl is {speedup:.2f}x faster")
    else:
        print(f"qwen tokenizer: Skipped (not available)")
    
    print()
    print("decoding reasoners tokenizer")
    decoded = cpp_tokenizer.decode(cpp_tokenizer.encode(long_text[:100]))
    print(f"original: {long_text[:100]}")
    print(f"encoded:  {cpp_tokenizer.encode(long_text[:100])}")
    print(f"decoded:  {decoded}")

if __name__ == "__main__":
    main()
