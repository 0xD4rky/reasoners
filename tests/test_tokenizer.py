import sys
import os
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from reasoners.tokenizer.qwen_tokenizer import QwenTokenizer


def test_basic_encode_decode():
    """Test basic encoding and decoding."""
    tokenizer = QwenTokenizer()
    
    test_text = "Hello, world! This is a test."
    print(f"Input: {test_text}")
    
    tokens = tokenizer.encode(test_text)
    print(f"Tokens ({len(tokens)}): {tokens}")
    
    decoded = tokenizer.decode(tokens)
    print(f"Decoded: {decoded}")
    
    assert decoded == test_text, f"Decode mismatch: {decoded} != {test_text}"
    print("Basic encode/decode: PASSED\n")


def test_special_tokens():
    """Test special token IDs are set correctly."""
    tokenizer = QwenTokenizer()
    
    assert tokenizer.eos_token_id == 151643
    assert tokenizer.pad_token_id == 151643
    assert tokenizer.im_start_token_id == 151644
    assert tokenizer.im_end_token_id == 151645
    assert tokenizer.bos_token_id is None  # Qwen doesn't use BOS
    
    print(f"eos_token_id: {tokenizer.eos_token_id}")
    print(f"pad_token_id: {tokenizer.pad_token_id}")
    print(f"im_start_token_id: {tokenizer.im_start_token_id}")
    print(f"im_end_token_id: {tokenizer.im_end_token_id}")
    print("Special tokens: PASSED\n")


def test_apply_chat_template():
    """Test chat template formatting."""
    tokenizer = QwenTokenizer()
    
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is 2+2?"},
        {"role": "assistant", "content": "2+2 equals 4."}
    ]
    
    # Test string output
    formatted = tokenizer.apply_chat_template(messages, tokenize=False)
    print(f"Formatted chat:\n{formatted}")
    
    expected = (
        "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
        "<|im_start|>user\nWhat is 2+2?<|im_end|>\n"
        "<|im_start|>assistant\n2+2 equals 4.<|im_end|>\n"
    )
    assert formatted == expected, f"Template mismatch:\nGot: {formatted}\nExpected: {expected}"
    
    # Test tokenized output
    tokens = tokenizer.apply_chat_template(messages, tokenize=True)
    assert isinstance(tokens, list)
    assert all(isinstance(t, int) for t in tokens)
    print(f"Tokenized length: {len(tokens)}")
    
    # Test generation prompt
    gen_prompt = tokenizer.apply_chat_template(
        messages[:2],  # Only system + user
        tokenize=False,
        add_generation_prompt=True
    )
    assert gen_prompt.endswith("<|im_start|>assistant\n")
    print(f"Generation prompt ends correctly: PASSED")
    
    print("Chat template: PASSED\n")


def test_hf_compatible_call():
    """Test HuggingFace-compatible __call__ interface."""
    tokenizer = QwenTokenizer()
    
    text = "Hello, this is a test message."
    
    # Basic call
    result = tokenizer(text)
    assert "input_ids" in result
    assert "attention_mask" in result
    assert len(result["input_ids"]) == len(result["attention_mask"])
    print(f"Basic call - tokens: {len(result['input_ids'])}")
    
    # With truncation
    result_truncated = tokenizer(text, truncation=True, max_length=5)
    assert len(result_truncated["input_ids"]) == 5
    print(f"Truncated to 5 tokens: {result_truncated['input_ids']}")
    
    # Batch call
    texts = ["Hello world", "Another test message here"]
    batch_result = tokenizer(texts)
    assert len(batch_result["input_ids"]) == 2
    print(f"Batch sizes: {[len(ids) for ids in batch_result['input_ids']]}")
    
    print("HF-compatible call: PASSED\n")


def test_dataset_compatibility():
    """Test that tokenizer works with ReasoningDataset expectations."""
    tokenizer = QwenTokenizer()
    
    # Simulate what ReasoningDataset does
    messages = [
        {"role": "user", "content": "What is the capital of France?"},
        {"role": "assistant", "content": "The capital of France is Paris."}
    ]
    
    # 1. apply_chat_template with tokenize=False
    formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    assert isinstance(formatted, str)
    
    # 2. Call tokenizer with HF-style args
    encodings = tokenizer(
        formatted,
        truncation=True,
        max_length=4096,
        padding=False,
        return_tensors=None
    )
    
    input_ids = encodings["input_ids"]
    assert isinstance(input_ids, list)
    assert all(isinstance(t, int) for t in input_ids)
    
    # 3. Access special token IDs
    pad_id = tokenizer.pad_token_id
    eos_id = tokenizer.eos_token_id
    assert pad_id == 151643
    assert eos_id == 151643
    
    print(f"Dataset compatibility - {len(input_ids)} tokens")
    print("Dataset compatibility: PASSED\n")


if __name__ == "__main__":
    print("=" * 50)
    print("QwenTokenizer Tests")
    print("=" * 50 + "\n")
    
    test_basic_encode_decode()
    test_special_tokens()
    test_apply_chat_template()
    test_hf_compatible_call()
    test_dataset_compatibility()
