import os
from typing import List, Set, Union, Optional

try:
    from qwen_tokenizer_cpp import QwenBPETokenizer as _QwenBPETokenizerCpp
    _CPP_AVAILABLE = True
except ImportError:
    _CPP_AVAILABLE = False


class QwenTokenizer:
    def __init__(self, tokenizer_json_path: str):
        if not _CPP_AVAILABLE:
            raise RuntimeError(
                "C++ tokenizer not available. Please build it with: "
                "mkdir build && cd build && cmake .. && make"
            )
        
        if not os.path.exists(tokenizer_json_path):
            raise FileNotFoundError(f"Tokenizer file not found: {tokenizer_json_path}")
        
        self._tokenizer = _QwenBPETokenizerCpp(tokenizer_json_path)
    
    def encode(self, text: str, allowed_special: Optional[Set[str]] = None) -> List[int]:
        if allowed_special is None:
            allowed_special = set()
        return self._tokenizer.encode(text, allowed_special)
    
    def decode(self, ids: List[int]) -> str:
        return self._tokenizer.decode(ids)
    
    def token_to_id(self, token: str) -> int:
        return self._tokenizer.token_to_id(token)
    
    def __call__(self, text: Union[str, List[str]], **kwargs) -> Union[List[int], List[List[int]]]:
        if isinstance(text, str):
            return self.encode(text, kwargs.get('allowed_special'))
        return [self.encode(t, kwargs.get('allowed_special')) for t in text]

