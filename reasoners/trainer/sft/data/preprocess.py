import re
import torch
import numpy as np
import yaml

from pathlib import Path
from dataclasses import dataclass
from torch.utils.data import Dataset
from typing import List, Dict, Optional, Tuple


@dataclass
class ReasoningExample:
  user_query: str
  reasoning_chain: str
  final_answer: str
  raw_messages: List[Dict[str, str]]

  def get_full_response(self) -> str:
    return f"{self.reasoning_chain}\n\n{self.final_answer}"


class ReasoningParser:

  """
  class to extract reasoning chains from assistant's responses, handles <think>,</think> tokens
  """

  def __init__(self):

    # extracting patterns from ans

    self.think_pattern = re.compile(r'<think>(.*?)</think>', re.DOTALL)
    self.solution_pattern = re.compile(r'<\|begin_of_solution\|>(.*?)<\|end_of_solution\|>', re.DOTALL)
    self.boxed_pattern = re.compile(r'\\boxed\{([^}]+)\}')

  def parse(self, assistant_message: str) -> Tuple[str,str]:
    """
    extracts reasoning_chain and final_answer from the assistant's response
    returns: (reasoning_chain,final_answer)
    """

    reasoning = ""
    answer = ""

    think_match = self.think_pattern.search(assistant_message)
    if think_match:
      reasoning = think_match.group(1)

    solution_match = self.solution_pattern.search(assistant_message)
    if solution_match:
      answer = solution_match.group(1)
    else:
      answer = assistant_message

    return reasoning, answer
    

class ReasoningDataset(Dataset):

  """
  dataset class for reasoning tasks, handles system prompt, reasoning chain extraction, and tokenization
  """

  def __init__(
      self,
      data: List[Dict],
      tokenizer,
      max_length: int = 4096,
      parse_reasoning: bool = True,
      include_reasoning_in_loss: bool = True,
      system_prompt_path: Optional[str] = None
  ):

    self.data = data
    self.tokenizer = tokenizer
    self.max_length = max_length
    self.include_reasoning_in_loss = include_reasoning_in_loss
    self.parser = ReasoningParser() if parse_reasoning else None

    self.bos_token_id = tokenizer.bos_token_id
    self.eos_token_id = tokenizer.eos_token_id
    self.pad_token_id = tokenizer.pad_token_id

    self.system_prompt = self._load_system_prompt(system_prompt_path)
    self.examples = self._preprocess_data(data)
  
  def _load_system_prompt(self, path: Optional[str]) -> Optional[str]:
    if not path:
      return None
    
    prompt_path = Path(path)
    if not prompt_path.exists():
      return None
    
    with open(prompt_path, 'r') as f:
      config = yaml.safe_load(f)
    
    return config.get('system_prompt', '').strip()
  
  def _preprocess_data(
      self,
      data: List[Dict]
  ) -> List[ReasoningExample]:

    examples = []

    for item in data:
      messages = item.get('messages', [])
      if len(messages) < 2:
        continue
      
      user_message = next(message["content"] for message in messages if message["role"] == "user")
      assistant_message = next(message["content"] for message in messages if message["role"] == "assistant")
      if not user_message or not assistant_message:
        continue

      if self.parser:
        reasoning, answer = self.parser.parse(assistant_message)
      else:
        reasoning, answer = "", assistant_message
      
      examples.append(ReasoningExample(
                user_query=user_message,
                reasoning_chain=reasoning,
                final_answer=answer,
                raw_messages=messages
            ))
      
    return examples
  
  def __len__(self) -> int:
        return len(self.examples)
  
  def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:

    example = self.examples[idx]
    messages = []
    
    if self.system_prompt:
      messages.append({"role": "system", "content": self.system_prompt})
    
    messages.extend([
      {"role": "user", "content": example.user_query},
      {"role": "assistant", "content": example.get_full_response()}
    ])
    
    formatted = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False
        )

    encodings = self.tokenizer(
        formatted,
        truncation = True,
        max_length = self.max_length,
        padding = False,
        return_tensors=None
    )

    input_ids = encodings['input_ids']

    # create labels acc to our case i.e. mask user tokens, keep assistant tokens
    labels = self._create_labels(input_ids, messages)

    return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'labels': torch.tensor(labels, dtype=torch.long),
            'attention_mask': torch.ones(len(input_ids), dtype=torch.long)
        }
  
  def _create_labels(
      self,
      input_ids: List[int],
      messages: List[Dict[str, str]]
  ) -> List[int]:
  
    assistant_start_idx = next((i for i, msg in enumerate(messages) if msg["role"] == "assistant"), -1)
    
    if assistant_start_idx == -1:
      return [-100] * len(input_ids)
    
    prompt_messages = messages[:assistant_start_idx]
    prompt_formatted = self.tokenizer.apply_chat_template(
        prompt_messages,
        tokenize=False,
        add_generation_prompt=True
    )
    prompt_tokens = self.tokenizer(
        prompt_formatted,
        add_special_tokens=False
    )['input_ids']

    labels = [-100] * len(input_ids)
    prompt_len = len(prompt_tokens)
    
    if prompt_len < len(input_ids):
      labels[prompt_len:] = input_ids[prompt_len:]
    
    return labels
