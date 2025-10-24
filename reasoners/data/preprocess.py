import re
import torch
import numpy as np

from dataclasses import dataclass
from collections import defaultdict
from torch.utils.data import Dataset, DataLoader, BatchSampler
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
    
