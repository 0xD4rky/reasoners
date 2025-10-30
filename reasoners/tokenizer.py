import os
import sys
from pathlib import Path
from typing import List, Set, Optional, Union

_CPP_AVAILABLE = False
_QwenBPETokenizerCpp = None

try:
    sys.path.insert(0, str(Path(__file__).parent))
    from qwen_tokenizer_cpp import QwenBPETokenizer as _QwenBPETokenizerCpp
    _CPP_AVAILABLE = True
except ImportError:
    pass


class QwenTokenizer:
    def __init__(self, tokenizer_path: Optional[str] = None):
        if not _CPP_AVAILABLE:
            raise RuntimeError(
                "C++ tokenizer module not found. "
                "Make sure qwen_tokenizer_cpp.so is in the reasoners/ directory."
            )
        
        if tokenizer_path is None:
            tokenizer_path = os.path.join(
                Path(__file__).parent,
                "models/Qwen/Qwen2.5-1.5B-Instruct/tokenizer.json"
            )
        
        if not os.path.exists(tokenizer_path):
            raise FileNotFoundError(f"Tokenizer file not found: {tokenizer_path}")
        
        self._tokenizer = _QwenBPETokenizerCpp(tokenizer_path)
        self.tokenizer_path = tokenizer_path
    
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
    
    def __repr__(self):
        return f"QwenTokenizer(tokenizer_path='{self.tokenizer_path}')"

