import torch
import numpy as np

from dataclasses import dataclass
from collections import defaultdict
from torch.utils.data import DataLoader, BatchSampler
from typing import List, Dict, Optional, Tuple


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


class LengthBatchSampler(BatchSampler):
  """
  groups similar-length sequences to minimize padding and reduces wasted compute
  """

  def __init__(
      self,
      dataset,
      batch_size: int,
      shuffle=True,
      drop_last=False
  ):
    self.batch_size = batch_size
    self.shuffle = shuffle
    self.drop_last = drop_last

    lengths = torch.zeros(len(dataset.examples), dtype=torch.int32)
    for i, item in enumerate(dataset.examples):
      lengths[i] = len(item.user_query) + len(item.get_full_response())
    
    self.sorted_indices = np.argsort(lengths).tolist()
  
  def __iter__(self):
    indices = self.sorted_indices.copy()

    # creating batches from sorted indices
    batches = []
    for i in range(0, len(indices), self.batch_size):
      batch = indices[i:i + self.batch_size]
      if len(batch) == self.batch_size or not self.drop_last:
          batches.append(batch)
    
    if self.shuffle:
      np.random.shuffle(batches)
    
    yield from batches
  
  def __len__(self):
    n = len(self.sorted_indices)
    if self.drop_last:
        return n // self.batch_size
    return (n + self.batch_size - 1) // self.batch_size




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
    drop_last: bool = False
):

  collator = Collator(
        pad_token_id=dataset.pad_token_id,
        padding_side="right"
    )

  batch_sampler = LengthBatchSampler(
    dataset=dataset,
    batch_size=batch_size,
    shuffle=shuffle,
    drop_last=drop_last
  )

  loader = DataLoader(
    dataset,
    batch_sampler=batch_sampler,
    num_workers=num_workers,
    collate_fn=collator,
    pin_memory=torch.cuda.is_available(),
    persistent_workers=num_workers > 0,
    worker_init_fn=seed_worker,
    prefetch_factor=2 if num_workers > 0 else None
  )

  return loader
