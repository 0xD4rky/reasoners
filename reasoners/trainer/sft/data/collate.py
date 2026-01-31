"""
Data collation and batching utilities for SFT training.

This module provides:
- Collator: Basic dynamic padding collator
- create_dataloader: Factory function for creating data loaders with token-level bucketing

For direct access to bucketing components:
    from reasoners.trainer.sft.data.bucketing import (
        BucketedBatchSampler, TokenBudgetBatchSampler, ...
    )
"""

import torch
import numpy as np

from torch.utils.data import DataLoader
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
    'Collator',
    'create_dataloader',
    # Bucketing exports
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
  Dynamic padding collator for batching variable-length sequences.
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
      else: # left padding
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


def get_optimal_workers() -> int:
  """Get optimal number of DataLoader workers based on CPU count."""
  import multiprocessing as mp
  cpu_count = mp.cpu_count()
  return 2 if cpu_count <= 4 else 4 if cpu_count <= 8 else 6


def create_dataloader(
    dataset,
    tokenizer,
    batch_size: int = 8,
    shuffle: bool = True,
    num_workers: int = 4,
    drop_last: bool = False,
    use_token_budget: bool = False,
    max_tokens_per_batch: int = 16384,
    bucket_boundaries: Optional[List[int]] = None,
    cache_dir: Optional[str] = None,
    seed: int = 42,
    show_progress: bool = True,
) -> Tuple[DataLoader, Dict[str, Any]]:
  """
  Create a DataLoader with token-level bucketing.
  
  Args:
      dataset: ReasoningDataset instance
      tokenizer: Tokenizer instance (required for token-level bucketing)
      batch_size: Batch size (fixed unless use_token_budget=True)
      shuffle: Whether to shuffle within buckets
      num_workers: Number of DataLoader workers
      drop_last: Whether to drop incomplete batches
      use_token_budget: Variable batch size with fixed token budget
      max_tokens_per_batch: Token budget per batch (if use_token_budget=True)
      bucket_boundaries: Custom bucket boundaries (default: [256, 512, 1024, 2048, 4096, 8192])
      cache_dir: Directory to cache token lengths
      seed: Random seed for reproducibility
      show_progress: Show progress bar during pre-tokenization
  
  Returns:
      Tuple of (DataLoader, statistics_dict)
  
  Example:
      # Fixed batch size with token-level bucketing
      loader, stats = create_dataloader(
          dataset, 
          tokenizer,
          batch_size=8
      )
      
      # Token budget batching (variable batch size, maximizes GPU utilization)
      loader, stats = create_dataloader(
          dataset,
          tokenizer,
          batch_size=8,
          use_token_budget=True,
          max_tokens_per_batch=16384
      )
  """
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
      show_progress=show_progress
  )
