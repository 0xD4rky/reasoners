import os
import sys
import torch
from pathlib import Path
from typing import Any, Dict, List, Set, Optional, Union

_CPP_AVAILABLE = False
_QwenBPETokenizerCpp = None

try:
    sys.path.insert(0, str(Path(__file__).parent))
    from qwen_tokenizer_cpp import QwenBPETokenizer as _QwenBPETokenizerCpp
    _CPP_AVAILABLE = True
except ImportError:
    pass

QWEN_SPECIAL_TOKENS = {
    "eos_token": "<|endoftext|>",
    "pad_token": "<|endoftext|>",
    "im_start_token": "<|im_start|>",
    "im_end_token": "<|im_end|>",
}
QWEN_SPECIAL_TOKEN_IDS = {
    "eos_token_id": 151643,
    "pad_token_id": 151643,
    "bos_token_id": None,
    "im_start_token_id": 151644,
    "im_end_token_id": 151645,
}

class QwenTokenizer:
    """
    Fast tokenizer for Qwen 2.5 models.

    supports:
    - encode/decode
    - applying chat template (qwen chatml format)
    - hf style __call__ returning {'input_ids': [...]}
    - special token properties (eos_token_id, pad_token_id, etc)
    """
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

        # special tokens
        self.eos_token = QWEN_SPECIAL_TOKENS["eos_token"]
        self.pad_token = QWEN_SPECIAL_TOKENS["pad_token"]
        self.im_start_token = QWEN_SPECIAL_TOKENS["im_start_token"]
        self.im_end_token = QWEN_SPECIAL_TOKENS["im_end_token"]

        # Special token IDs
        self.eos_token_id = QWEN_SPECIAL_TOKEN_IDS["eos_token_id"]
        self.pad_token_id = QWEN_SPECIAL_TOKEN_IDS["pad_token_id"]
        self.bos_token_id = QWEN_SPECIAL_TOKEN_IDS["bos_token_id"]
        self.im_start_token_id = QWEN_SPECIAL_TOKEN_IDS["im_start_token_id"]
        self.im_end_token_id = QWEN_SPECIAL_TOKEN_IDS["im_end_token_id"]
        
        # default special tokens to allow during encoding
        self._default_allowed_special = {
            "<|endoftext|>", "<|im_start|>", "<|im_end|>"
        }
    
    def encode(
        self, 
        text: str, 
        allowed_special: Optional[Set[str]] = None,
        add_special_tokens: bool = True
    ) -> List[int]:

        if allowed_special is None:
            allowed_special = self._default_allowed_special if add_special_tokens else set()
        return self._tokenizer.encode(text, allowed_special)
    
    def decode(
        self, 
        ids: List[int],
        skip_special_tokens: bool = False
    ) -> str:
        if skip_special_tokens:
            ids = [i for i in ids if i not in {self.eos_token_id, self.im_start_token_id, self.im_end_token_id}]
        return self._tokenizer.decode(ids)
    
    def token_to_id(self, token: str) -> int:
        return self._tokenizer.token_to_id(token)
    
    def apply_chat_template(
        self,
        messages: List[Dict[str,str]],
        tokenize: bool = True,
        add_generation_prompt: bool = False,
        **kwargs
    ) -> Union[str, List[int]]:
        """
        Apply Qwen chat template to a list of messages

        Format:
            <|im_start|>system
            {system_message}<|im_end|>
            <|im_start|>user
            {user_message}<|im_end|>
            <|im_start|>assistant
            {assistant_message}<|im_end|>
        
        Args:
            messages: List of {"role": str, "content": str} dicts
            tokenize: If True, return token IDs; if False, return formatted string
            add_generation_prompt: If True, add assistant prompt for generation
        
        Returns:
            Formatted string or list of token IDs
        """

        formatted_text = "".join(map(
        lambda m: f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n", 
        messages
        ))

        if add_generation_prompt:
            formatted_text += "<|im_start|>assistant\n"

        if tokenize:
            return self.encode(formatted_text)

        return formatted_text
    
    def __call__(
        self, 
        text: Union[str, List[str]],
        truncation: bool = False,
        max_length: Optional[int] = None,
        padding: Union[bool, str] = False,
        return_tensors: Optional[str] = None,
        add_special_tokens: bool = True,
        **kwargs
    ) -> Dict[str, Any]:
        """
       HF compatible __call__ interface

        Args:
            text: String or list of strings to tokenize
            truncation: If True, truncate the text to max_length
            max_length: Maximum length of the text
            padding: If True, pad the text to max_length
            return_tensors: If not None, return tensors of the specified type
            **kwargs: Additional arguments passed to the tokenizer

        Returns:
            Dictionary with 'input_ids' and optionally 'attention_mask'
        """

        is_single_input = isinstance(text, str)
        input_texts = [text] if is_single_input else text # conv into a list

        all_ids = []
        all_attention_mask = []

        for t in input_texts:
            ids = self.encode(t, add_special_tokens=add_special_tokens)
            if truncation and max_length is not None:
                ids = ids[:max_length]
            attention_mask = [1] * len(ids)
            all_ids.append(ids)
            all_attention_mask.append(attention_mask)
        
        actual_max_len = max(len(ids) for ids in all_ids)
        if padding == "max_length" and max_length:
            target_len = max_length
        elif padding:
            target_len = actual_max_len
        else:
            target_len = actual_max_len 
        
        if return_tensors == "pt":
            batch_size = len(all_ids)
            input_ids_tensor = torch.full((batch_size, target_len), self.pad_token_id, dtype=torch.long)
            attention_mask_tensor = torch.zeros((batch_size, target_len), dtype=torch.long)

            for i, ids in enumerate(all_ids):
                curr_len = len(ids)
                input_ids_tensor[i, :curr_len] = torch.tensor(ids)
                attention_mask_tensor[i, :curr_len] = 1

            if is_single_input:
                input_ids_tensor = input_ids_tensor.squeeze(0)
                attention_mask_tensor = attention_mask_tensor.squeeze(0)

            return {"input_ids": input_ids_tensor, "attention_mask": attention_mask_tensor}

        # non-tensor return
        if padding:
            all_ids = [ids + [self.pad_token_id] * (target_len - len(ids)) for ids in all_ids]
            all_masks = [[1] * len(ids) + [0] * (target_len - len(ids)) for ids in all_ids] 
        else:
            all_masks = [[1] * len(ids) for ids in all_ids]

        return {
            "input_ids": all_ids[0] if is_single_input else all_ids,
            "attention_mask": all_masks[0] if is_single_input else all_masks
        }

    def __len__(self) -> int:
        return 151665

    def __repr__(self):
        return f"QwenTokenizer(tokenizer_path='{self.tokenizer_path}')"

