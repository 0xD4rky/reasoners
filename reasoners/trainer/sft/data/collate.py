"""
Data collation and batching utilities for SFT training.

This module provides:
- Collator: Basic dynamic padding collator
- LengthBatchSampler: Simple length-based batching (DEPRECATED, use bucketing module)
- create_dataloader: Factory function for creating data loaders

For advanced bucketing with token-level optimization, use the bucketing module:
    from reasoners.trainer.sft.data.bucketing import create_bucketed_dataloader
"""

import torch
import numpy as np

from torch.utils.data import DataLoader, BatchSampler
from typing import List, Dict, Optional, Tuple, Any

# Re-export bucketing components for convenience
from reasoners.trainer.sft.data.bucketing import (
    BucketConfig,
    TokenBudgetConfig,
    TokenLengthCache,
    BucketedBatchSampler,
    TokenBudgetBatchSampler,
    InstrumentedCollator,
    create_bucketed_dataloader,
    compute_lr_scale,
)

__all__ = [
    # Legacy exports
    'Collator',
    'LengthBatchSampler',
    'create_dataloader',
    # New bucketing exports
    'BucketConfig',
    'TokenBudgetConfig',
    'TokenLengthCache',
    'BucketedBatchSampler',
    'TokenBudgetBatchSampler',
    'InstrumentedCollator',
    'create_bucketed_dataloader',
    'compute_lr_scale',
]


class Collator:

  """
  efficient batching of reasoning tokens and dynamic padding implementation in this class
  """

  def __init__(
      self,
      pad_token_id: int,
      padding_side: str = "right"
  ):
    self.pad_token_id = pad_token_id
    self.padding_side = padding_side

  def __call__(
      self,
      batch: List[Dict[str, torch.Tensor]]
  ) -> Dict[str, torch.Tensor]:

    max_length = max(item["input_ids"].size(0) for item in batch)
    batch_size = len(batch)

    # pre-fill the tensors
    input_ids = torch.full(
      (batch_size, max_length),
      self.pad_token_id,
      dtype = torch.long
    )
    labels = torch.full(
        (batch_size, max_length),
        -100,
        dtype = torch.long
    )
    attention_mask = torch.zeros(
        (batch_size, max_length),
        dtype = torch.long
    )

    # fill the tensors
    for i, item in enumerate(batch):
      seq_length = item["input_ids"].size(0)

      if self.padding_side == "right":
        input_ids[i, :seq_length] = item["input_ids"]
        labels[i, :seq_length] = item["labels"]
        attention_mask[i, :seq_length] = item['attention_mask']
      else: # as in left padding
        input_ids[i, -seq_length:] = item["input_ids"]
        labels[i, -seq_length:] = item["labels"]
        attention_mask[i, -seq_length:] = item['attention_mask']

    return {
            'input_ids': input_ids,
            'labels': labels,
            'attention_mask': attention_mask
        }


                                            ######## Data Loader Utils ###########


def seed_worker(worker_id):

  torch.set_num_threads(1)
  worker_seed = torch.initial_seed() % 2**32 
  np.random.seed(worker_seed + worker_id)

def get_optimal_workers(): # func to fetch ideal no of workers
  import multiprocessing as mp
  cpu_count = mp.cpu_count()

  return 2 if cpu_count <= 4 else 4 if cpu_count <= 8 else 6

def create_dataloader(
    dataset,
    batch_size: int,
    shuffle: bool = True,
    num_workers: int = 4,
    group_by_length: bool = True,
    drop_last: bool = False,
    # New bucketing options
    use_token_bucketing: bool = False,
    tokenizer = None,
    use_token_budget: bool = False,
    max_tokens_per_batch: int = 16384,
    bucket_boundaries: Optional[List[int]] = None,
    cache_dir: Optional[str] = None,
    seed: int = 42,
) -> Tuple[DataLoader, Optional[Dict[str, Any]]]:
  """
  Create a DataLoader for training.
  
  Args:
      dataset: ReasoningDataset instance
      batch_size: Batch size (fixed unless use_token_budget=True)
      shuffle: Whether to shuffle data
      num_workers: Number of DataLoader workers
      group_by_length: Whether to group by length (deprecated, use use_token_bucketing)
      drop_last: Whether to drop incomplete batches
      
      # Token-level bucketing options (recommended)
      use_token_bucketing: Enable proper token-level bucketing (requires tokenizer)
      tokenizer: Tokenizer instance (required if use_token_bucketing=True)
      use_token_budget: Variable batch size with fixed token budget
      max_tokens_per_batch: Token budget per batch (if use_token_budget=True)
      bucket_boundaries: Custom bucket boundaries (default: [256, 512, 1024, 2048, 4096, 8192])
      cache_dir: Directory to cache token lengths
      seed: Random seed for reproducibility
  
  Returns:
      If use_token_bucketing=True: Tuple of (DataLoader, statistics_dict)
      Otherwise: DataLoader (for backward compatibility)
  
  Example:
      # Basic usage (legacy, uses character-level approximation)
      loader = create_dataloader(dataset, batch_size=8)
      
      # Recommended: Token-level bucketing
      loader, stats = create_dataloader(
          dataset, 
          batch_size=8,
          use_token_bucketing=True,
          tokenizer=tokenizer
      )
      
      # Token budget batching (variable batch size)
      loader, stats = create_dataloader(
          dataset,
          batch_size=8,
          use_token_bucketing=True,
          use_token_budget=True,
          max_tokens_per_batch=16384,
          tokenizer=tokenizer
      )
  """
  
  # Use new token-level bucketing if requested
  if use_token_bucketing:
    if tokenizer is None:
      raise ValueError("tokenizer is required when use_token_bucketing=True")
    
    return create_bucketed_dataloader(
        dataset=dataset,
        tokenizer=tokenizer,
        batch_size=batch_size,
        bucket_boundaries=bucket_boundaries,
        use_token_budget=use_token_budget,
        max_tokens_per_batch=max_tokens_per_batch,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
        seed=seed,
        cache_dir=cache_dir,
        show_progress=True
    )
