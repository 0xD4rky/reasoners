import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'build'))
from reasoners.tokenizer import QwenTokenizer

def test_tokenizer():
    tokenizer_path = os.path.join(os.path.dirname(__file__), "..", "reasoners/models/Qwen/Qwen2.5-1.5B-Instruct/tokenizer.json")
    
    tokenizer = QwenTokenizer(tokenizer_path)
    
    test_text = "Hello, world! This is a test."
    print(f"Input: {test_text}")
    
    tokens = tokenizer.encode(test_text)
    print(f"Tokens ({len(tokens)}): {tokens}")
    
    decoded = tokenizer.decode(tokens)
    print(f"Decoded: {decoded}")
    
    assert decoded == test_text, f"Decode mismatch: {decoded} != {test_text}"
    print("\nTest passed!")

if __name__ == "__main__":
    test_tokenizer()

