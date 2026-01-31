"""
Token-Level Bucketing System for Efficient Training

This module implements proper token-level bucketing with:
- Pre-tokenization and length caching
- Discrete bucket boundaries for minimal padding
- Token-budget batching (variable batch size, fixed token budget)
- Proper handling of bucket exhaustion
- LR scaling support for variable batch sizes
"""

import os
import json
import hashlib
import numpy as np
import torch

from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Iterator, Any, Union
from collections import defaultdict
from tqdm import tqdm


@dataclass
class BucketConfig:
    # bucket boundaries (upper bounds, exclusive)
    # seq with len < boundaries[0] go to bucket 0
    # seq with boundaries[i-1] <= len < boundaries[i] go to bucket i
    boundaries: List[int] = field(default_factory=lambda: [256, 512, 1024, 2048, 4096, 8192])
    
    #sampling strategy across buckets:
    #proportional: sample from buckets proportional to their size
    #uniform: equal probability for each bucket 
    #sequential: exhaust each bucket before moving to next
    cross_bucket_strategy: str = 'proportional'
    
    #shuffling within buckets for noise
    shuffle_within_bucket: bool = True
    
    # min samples per bucket to include it (filters tiny buckets)
    min_bucket_size: int = 1
    
    def get_bucket_idx(self, length: int) -> int:
        """
        get bucket index for a given sequence length (for a single example)
        """
        for i, boundary in enumerate(self.boundaries):
            if length < boundary:
                return i
        return len(self.boundaries) # case where length is greater than all boundaries
    
    def get_bucket_name(self, bucket_idx: int) -> str:
        if bucket_idx == 0:
            return f"[0, {self.boundaries[0]})"
        elif bucket_idx < len(self.boundaries):
            return f"[{self.boundaries[bucket_idx-1]}, {self.boundaries[bucket_idx]})"
        else:
            return f"[{self.boundaries[-1]}, ∞)"


@dataclass
class TokenBudgetConfig:
    """
    configuration for token-budget batching
    """
    
    max_tokens_per_batch: int = 16384
    min_batch_size: int = 1 # even tho it exceeds budget slightly
    max_batch_size: int = 64
    
    # count padding token -> accurate gpu utilization, false -> more samples per batch
    count_padding: bool = True
    drop_oversized: bool = False

class TokenLengthCache:
    """
    pre-tokenizes dataset and caches token lengths
    
    features:
    - lazy or eager tokenization
    - persistent caching to disk
    - cache invalidation based on dataset hash
    """
    
    def __init__(
        self,
        dataset,
        cache_dir: Optional[str] = None,
        eager: bool = True,
        show_progress: bool = True
    ):
        self.dataset = dataset
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.show_progress = show_progress
        
        # length cache: {example_index, token_length}
        self._lengths: Dict[int, int] = {}
        self._dataset_hash: Optional[str] = None
        
        if eager:
            self._compute_all_lengths()
    
    def _compute_dataset_hash(self) -> str:
        """
        this func is computing a fingerprint (hash) of the dataset to detect if the dataset has changed, 
        which would invalidate any cached pre-tokenized data.
        """
        # Use first/last few examples + length as fingerprint
        n = len(self.dataset)
        sample_indices = [0, n//4, n//2, 3*n//4, n-1] if n > 5 else list(range(n))
        
        fingerprint_parts = [str(n)]
        for idx in sample_indices:
            if idx < n:
                example = self.dataset.examples[idx]
                fingerprint_parts.append(example.user_query[:100])
                fingerprint_parts.append(example.get_full_response()[:100])
        
        fingerprint = "|".join(fingerprint_parts)
        return hashlib.md5(fingerprint.encode()).hexdigest()[:16]
    
    def _get_cache_path(self) -> Optional[Path]:
        if self.cache_dir is None:
            return None
        
        if self._dataset_hash is None:
            self._dataset_hash = self._compute_dataset_hash()
        
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        return self.cache_dir / f"token_lengths_{self._dataset_hash}.json"
    
    def _load_cache(self) -> bool:
        cache_path = self._get_cache_path()
        if cache_path is None or not cache_path.exists():
            return False
        
        try:
            with open(cache_path, 'r') as f:
                data = json.load(f)
            
            if data.get('dataset_hash') != self._dataset_hash:
                return False
            if data.get('length') != len(self.dataset):
                return False
            
            self._lengths = {int(k): v for k, v in data['lengths'].items()}
            return True
        except Exception:
            return False
    
    def _save_cache(self):
        cache_path = self._get_cache_path()
        if cache_path is None:
            return
        
        data = {
            'dataset_hash': self._dataset_hash,
            'length': len(self.dataset),
            'lengths': self._lengths
        }
        
        with open(cache_path, 'w') as f:
            json.dump(data, f)
    
    def _compute_all_lengths(self):
        """
        func to compute token lengths for all examples in the dataset
        """
        self._dataset_hash = self._compute_dataset_hash()
        
        if self._load_cache():
            if self.show_progress:
                print(f"Loaded {len(self._lengths)} token lengths from cache")
            return
        
        # Compute lengths
        iterator = range(len(self.dataset))
        if self.show_progress:
            iterator = tqdm(iterator, desc="Pre-tokenizing for length cache")
        
        for idx in iterator:
            self._lengths[idx] = self._compute_length(idx)
        
        self._save_cache()
        
        if self.show_progress:
            lengths = list(self._lengths.values()) # this list stores the length of each example in the dataset in terms of tokens
            print(f"Token length stats: min={min(lengths)}, max={max(lengths)}")
    
    def _compute_length(self, idx: int) -> int:
        # func calculates the token length for a single example
        # get the tokenized item from dataset
        item = self.dataset[idx]
        return item['input_ids'].size(0)
    
    def __getitem__(self, idx: int) -> int:
        if idx not in self._lengths:
            self._lengths[idx] = self._compute_length(idx)
        return self._lengths[idx]
    
    def __len__(self) -> int:
        return len(self.dataset)
    
    def get_all_lengths(self) -> np.ndarray:
        return np.array([self._lengths[i] for i in range(len(self.dataset))])
    
    def get_statistics(self) -> Dict[str, float]:
        lengths = self.get_all_lengths()
        return {
            'min': int(np.min(lengths)),
            'max': int(np.max(lengths)),
        }

class BucketManager:
    """
    manages discrete buckets of sequences grouped by token length.
    
    does:
    - bucket assignment based on token lengths
    - within-bucket shuffling
    - cross-bucket sampling strategies
    - bucket exhaustion
    """
    
    def __init__(
        self,
        length_cache: TokenLengthCache,
        config: BucketConfig,
        seed: int = 42
    ):
        self.length_cache = length_cache
        self.config = config
        self.seed = seed
        self.rng = np.random.RandomState(seed)
        
        # build buckets: {bucket_idx, list of dataset indices}
        self.buckets: Dict[int, List[int]] = defaultdict(list)
        self._build_buckets()
        
        # track bucket state for iteration
        self._bucket_positions: Dict[int, int] = {}
        self._active_buckets: List[int] = []
        self._reset_iteration_state()
    
    def _build_buckets(self):
        """
        func to assign each example to its bucket based on token length
        """
        for idx in range(len(self.length_cache)):
            length = self.length_cache[idx]
            bucket_idx = self.config.get_bucket_idx(length)
            self.buckets[bucket_idx].append(idx)
        
        self.buckets = {
            k: v for k, v in self.buckets.items()
            if len(v) >= self.config.min_bucket_size
        }
        
        self.bucket_indices = sorted(self.buckets.keys())
    
    def _reset_iteration_state(self):
        """
        func to reset the state for a new epoch
        """
        self._bucket_positions = {k: 0 for k in self.bucket_indices}
        self._active_buckets = list(self.bucket_indices)
                
        # shuffle within buckets if configured
        if self.config.shuffle_within_bucket:
            for bucket_idx in self.bucket_indices:
                self.rng.shuffle(self.buckets[bucket_idx])
    
    def get_bucket_statistics(self) -> Dict[str, Any]:
        stats = {
            'num_buckets': len(self.bucket_indices),
            'total_samples': sum(len(b) for b in self.buckets.values()),
            'buckets': {}
        }
        
        for bucket_idx in self.bucket_indices:
            bucket_data = self.buckets[bucket_idx]
            lengths = [self.length_cache[i] for i in bucket_data]
            stats['buckets'][self.config.get_bucket_name(bucket_idx)] = {
                'count': len(bucket_data),
                'min_length': min(lengths) if lengths else 0,
                'max_length': max(lengths) if lengths else 0,
                'mean_length': np.mean(lengths) if lengths else 0,
            }
        
        return stats
    
    def sample_from_bucket(self, bucket_idx: int) -> Optional[int]:
        """
        func to sample next index from a specific bucket
        returns None if exhausted
        """
        if bucket_idx not in self._active_buckets:
            return None
        
        pos = self._bucket_positions[bucket_idx]
        bucket = self.buckets[bucket_idx]
        
        if pos >= len(bucket):
            self._active_buckets.remove(bucket_idx)
            return None
        
        idx = bucket[pos]
        self._bucket_positions[bucket_idx] = pos + 1
        return idx
    
    def select_bucket(self) -> Optional[int]:
        """
        func to select which bucket to sample from based on strategy
        """
        if not self._active_buckets:
            return None
        
        strategy = self.config.cross_bucket_strategy
        
        if strategy == 'uniform':
            return self.rng.choice(self._active_buckets)
        
        elif strategy == 'proportional':
            # Weight by remaining samples
            weights = []
            for b in self._active_buckets:
                remaining = len(self.buckets[b]) - self._bucket_positions[b]
                weights.append(max(0, remaining))
            
            weights = np.array(weights, dtype=np.float64)
            total = weights.sum()
            
            # Handle edge case where all weights are 0
            if total == 0:
                return None
            
            weights /= total
            return self.rng.choice(self._active_buckets, p=weights)
        
        elif strategy == 'sequential':
            return self._active_buckets[0]
        
        else:
            raise ValueError(f"Unknown cross_bucket_strategy: {strategy}")
    
    def sample_batch_indices(self, batch_size: int) -> List[int]:
        """
        func to sample a batch of indices, trying to keep them from same bucket
        """
        indices = []
        
        # try to fill batch from same bucket for minimal padding
        bucket_idx = self.select_bucket()
        if bucket_idx is None:
            return indices
        
        while len(indices) < batch_size:
            idx = self.sample_from_bucket(bucket_idx)
            if idx is not None:
                indices.append(idx)
            else:
                # Bucket exhausted, try another
                bucket_idx = self.select_bucket()
                if bucket_idx is None:
                    break
        
        return indices
    
    def is_exhausted(self) -> bool:
        """
        func to check if all buckets are exhausted
        """
        return len(self._active_buckets) == 0
    
    def reset(self, new_seed: Optional[int] = None):
        """
        func to reset for a new epoch with optional new seed
        """
        if new_seed is not None:
            self.seed = new_seed
            self.rng = np.random.RandomState(new_seed)
        self._reset_iteration_state()


class BucketedBatchSampler:
    """
    Batch sampler using discrete buckets for minimal padding.
    
    Fixed batch size, samples from same bucket when possible.
    """
    
    def __init__(
        self,
        length_cache: TokenLengthCache,
        batch_size: int,
        bucket_config: Optional[BucketConfig] = None,
        drop_last: bool = False,
        seed: int = 42
    ):
        self.length_cache = length_cache
        self.batch_size = batch_size
        self.bucket_config = bucket_config or BucketConfig()
        self.drop_last = drop_last
        self.seed = seed
        
        self.bucket_manager = BucketManager(
            length_cache, self.bucket_config, seed
        )
        
        self._epoch = 0
    
    def __iter__(self) -> Iterator[List[int]]:
        # Reset with epoch-based seed for reproducibility
        self.bucket_manager.reset(self.seed + self._epoch)
        self._epoch += 1
        
        while not self.bucket_manager.is_exhausted():
            batch = self.bucket_manager.sample_batch_indices(self.batch_size)
            
            if len(batch) == 0:
                break
            
            if len(batch) < self.batch_size and self.drop_last:
                continue
            
            yield batch
    
    def __len__(self) -> int:
        total = sum(len(b) for b in self.bucket_manager.buckets.values())
        if self.drop_last:
            return total // self.batch_size
        return (total + self.batch_size - 1) // self.batch_size
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get bucket statistics."""
        return self.bucket_manager.get_bucket_statistics()


class TokenBudgetBatchSampler:
    """
    Batch sampler with fixed token budget (variable batch size).
    
    Maximizes GPU utilization by fitting as many sequences as possible
    within a token budget, while keeping sequences from same bucket.
    
    IMPORTANT: Variable batch sizes require careful handling:
    - Learning rate scaling: LR should scale with actual batch size
    - Gradient accumulation: Accumulate based on token count, not steps
    - Loss normalization: Normalize by tokens, not samples
    """
    
    def __init__(
        self,
        length_cache: TokenLengthCache,
        token_budget_config: TokenBudgetConfig,
        bucket_config: Optional[BucketConfig] = None,
        drop_last: bool = False,
        seed: int = 42
    ):
        self.length_cache = length_cache
        self.budget_config = token_budget_config
        self.bucket_config = bucket_config or BucketConfig()
        self.drop_last = drop_last
        self.seed = seed
        
        self.bucket_manager = BucketManager(
            length_cache, self.bucket_config, seed
        )
        
        self._epoch = 0
        
        # Track batch statistics for LR scaling
        self._last_batch_tokens: int = 0
        self._last_batch_size: int = 0
    
    def _compute_batch_tokens(self, indices: List[int]) -> int:
        """Compute total tokens for a batch (including padding if configured)."""
        if not indices:
            return 0
        
        lengths = [self.length_cache[i] for i in indices]
        
        if self.budget_config.count_padding:
            # Padded batch: all sequences padded to max length
            max_len = max(lengths)
            return max_len * len(indices)
        else:
            # Just sum of actual lengths
            return sum(lengths)
    
    def __iter__(self) -> Iterator[List[int]]:
        self.bucket_manager.reset(self.seed + self._epoch)
        self._epoch += 1
        
        budget = self.budget_config.max_tokens_per_batch
        min_size = self.budget_config.min_batch_size
        max_size = self.budget_config.max_batch_size
        
        # Track samples that couldn't fit in current batch (carry over)
        pending_sample: Optional[int] = None
        
        while not self.bucket_manager.is_exhausted() or pending_sample is not None:
            batch_indices = []
            current_tokens = 0
            
            # First, add any pending sample from previous iteration
            if pending_sample is not None:
                batch_indices.append(pending_sample)
                current_tokens = self._compute_batch_tokens(batch_indices)
                pending_sample = None
            
            # Select starting bucket
            bucket_idx = self.bucket_manager.select_bucket()
            if bucket_idx is None and len(batch_indices) == 0:
                break
            
            while len(batch_indices) < max_size and bucket_idx is not None:
                # Try to add one more sample
                idx = self.bucket_manager.sample_from_bucket(bucket_idx)
                
                if idx is None:
                    # Bucket exhausted, try another
                    bucket_idx = self.bucket_manager.select_bucket()
                    continue
                
                # Check if adding this sample exceeds budget
                test_indices = batch_indices + [idx]
                test_tokens = self._compute_batch_tokens(test_indices)
                
                if test_tokens > budget and len(batch_indices) >= min_size:
                    # Would exceed budget and we have enough samples
                    # Save this sample for next batch
                    pending_sample = idx
                    break
                
                # Add sample
                batch_indices.append(idx)
                current_tokens = test_tokens
                
                # Check budget
                if current_tokens >= budget and len(batch_indices) >= min_size:
                    break
            
            if len(batch_indices) == 0:
                break
            
            if len(batch_indices) < min_size and self.drop_last:
                continue
            
            # Track for LR scaling
            self._last_batch_size = len(batch_indices)
            self._last_batch_tokens = current_tokens
            
            yield batch_indices
    
    def __len__(self) -> int:
        # Approximate: total tokens / budget
        total_tokens = sum(self.length_cache[i] for i in range(len(self.length_cache)))
        return max(1, total_tokens // self.budget_config.max_tokens_per_batch)
    
    @property
    def last_batch_size(self) -> int:
        """
        size of the last yielded batch (for LR scaling)
        """
        return self._last_batch_size
    
    @property
    def last_batch_tokens(self) -> int:
        """
        token count of the last yielded batch
        """
        return self._last_batch_tokens
    
    def get_statistics(self) -> Dict[str, Any]:
        stats = self.bucket_manager.get_bucket_statistics()
        stats['token_budget'] = self.budget_config.max_tokens_per_batch
        stats['estimated_batches'] = len(self)
        return stats


@dataclass
class BatchInfo:
    batch_size: int
    total_tokens: int
    padding_tokens: int
    padding_ratio: float
    max_length: int
    min_length: int
    
    @property
    def effective_tokens(self) -> int:
        """
        actual non-padding tokens
        """
        return self.total_tokens - self.padding_tokens


class InstrumentedCollator:
    """
    Collator that tracks batch statistics for monitoring.
    
    Provides padding ratio, token counts, and other metrics
    useful for debugging and optimization.
    """
    
    def __init__(
        self,
        pad_token_id: int,
        padding_side: str = "right",
        track_statistics: bool = True
    ):
        self.pad_token_id = pad_token_id
        self.padding_side = padding_side
        self.track_statistics = track_statistics
        
        # Running statistics
        self._total_batches = 0
        self._total_padding_ratio = 0.0
        self._last_batch_info: Optional[BatchInfo] = None
    
    def __call__(
        self,
        batch: List[Dict[str, torch.Tensor]]
    ) -> Dict[str, torch.Tensor]:
        
        lengths = [item["input_ids"].size(0) for item in batch]
        max_length = max(lengths)
        batch_size = len(batch)
        
        # Pre-allocate tensors
        input_ids = torch.full(
            (batch_size, max_length),
            self.pad_token_id,
            dtype=torch.long
        )
        labels = torch.full(
            (batch_size, max_length),
            -100,
            dtype=torch.long
        )
        attention_mask = torch.zeros(
            (batch_size, max_length),
            dtype=torch.long
        )
        
        # Fill tensors
        for i, item in enumerate(batch):
            seq_length = lengths[i]
            
            if self.padding_side == "right":
                input_ids[i, :seq_length] = item["input_ids"]
                labels[i, :seq_length] = item["labels"]
                attention_mask[i, :seq_length] = item["attention_mask"]
            else:
                input_ids[i, -seq_length:] = item["input_ids"]
                labels[i, -seq_length:] = item["labels"]
                attention_mask[i, -seq_length:] = item["attention_mask"]
        
        result = {
            'input_ids': input_ids,
            'labels': labels,
            'attention_mask': attention_mask,
        }
        
        # Track statistics
        if self.track_statistics:
            total_tokens = batch_size * max_length
            actual_tokens = sum(lengths)
            padding_tokens = total_tokens - actual_tokens
            padding_ratio = padding_tokens / total_tokens if total_tokens > 0 else 0
            
            self._last_batch_info = BatchInfo(
                batch_size=batch_size,
                total_tokens=total_tokens,
                padding_tokens=padding_tokens,
                padding_ratio=padding_ratio,
                max_length=max_length,
                min_length=min(lengths)
            )
            
            self._total_batches += 1
            self._total_padding_ratio += padding_ratio
            
            # Add batch info to result for training loop
            result['_batch_info'] = self._last_batch_info
        
        return result
    
    @property
    def last_batch_info(self) -> Optional[BatchInfo]:
        return self._last_batch_info
    
    @property
    def average_padding_ratio(self) -> float:
        if self._total_batches == 0:
            return 0.0
        return self._total_padding_ratio / self._total_batches
    
    def reset_statistics(self):
        self._total_batches = 0
        self._total_padding_ratio = 0.0
        self._last_batch_info = None


def create_bucketed_dataloader(
    dataset,
    batch_size: int = 8,
    bucket_boundaries: Optional[List[int]] = None,
    use_token_budget: bool = False,
    max_tokens_per_batch: int = 16384,
    shuffle: bool = True,
    drop_last: bool = False,
    num_workers: int = 4,
    seed: int = 42,
    cache_dir: Optional[str] = None,
    show_progress: bool = True
) -> Tuple[torch.utils.data.DataLoader, Dict[str, Any]]:
    """
    Create a DataLoader with proper token-level bucketing.
    
    Args:
        dataset: ReasoningDataset instance
        batch_size: Batch size (ignored if use_token_budget=True)
        bucket_boundaries: List of bucket boundaries (default: [256, 512, 1024, 2048, 4096, 8192])
        use_token_budget: If True, use variable batch size with fixed token budget
        max_tokens_per_batch: Token budget per batch (only if use_token_budget=True)
        shuffle: Whether to shuffle within buckets
        drop_last: Whether to drop incomplete batches
        num_workers: Number of DataLoader workers
        seed: Random seed for reproducibility
        cache_dir: Directory to cache token lengths (None = no caching)
        show_progress: Show progress bar during pre-tokenization
    
    Returns:
        Tuple of (DataLoader, statistics_dict)
    """
    from torch.utils.data import DataLoader
    
    length_cache = TokenLengthCache(
        dataset=dataset,
        cache_dir=cache_dir,
        eager=True,
        show_progress=show_progress
    )
    
    bucket_config = BucketConfig(
        boundaries=bucket_boundaries or [256, 512, 1024, 2048, 4096, 8192],
        cross_bucket_strategy='proportional',
        shuffle_within_bucket=shuffle
    )
    
    # Create appropriate sampler
    if use_token_budget:
        budget_config = TokenBudgetConfig(
            max_tokens_per_batch=max_tokens_per_batch,
            min_batch_size=1,
            max_batch_size=batch_size * 4,  # Allow larger batches for short sequences
            count_padding=True
        )
        batch_sampler = TokenBudgetBatchSampler(
            length_cache=length_cache,
            token_budget_config=budget_config,
            bucket_config=bucket_config,
            drop_last=drop_last,
            seed=seed
        )
    else:
        batch_sampler = BucketedBatchSampler(
            length_cache=length_cache,
            batch_size=batch_size,
            bucket_config=bucket_config,
            drop_last=drop_last,
            seed=seed
        )
    
    # Create collator
    collator = InstrumentedCollator(
        pad_token_id=dataset.pad_token_id,
        padding_side="right",
        track_statistics=True
    )
    
    # Worker init function for reproducibility
    def seed_worker(worker_id):
        torch.set_num_threads(1)
        worker_seed = seed + worker_id
        np.random.seed(worker_seed)
        torch.manual_seed(worker_seed)
    
    # Create DataLoader
    loader = DataLoader(
        dataset,
        batch_sampler=batch_sampler,
        num_workers=num_workers,
        collate_fn=collator,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
        worker_init_fn=seed_worker if num_workers > 0 else None,
        prefetch_factor=2 if num_workers > 0 else None
    )
    
    # Gather statistics
    stats = {
        'length_stats': length_cache.get_statistics(),
        'bucket_stats': batch_sampler.get_statistics(),
        'use_token_budget': use_token_budget,
        'batch_size': batch_size if not use_token_budget else 'variable',
        'num_batches': len(batch_sampler),
    }
    
    return loader, stats


def compute_lr_scale(
    current_batch_tokens: int,
    reference_batch_tokens: int,
    scaling_type: str = 'linear'
) -> float:
    """
    Compute learning rate scaling factor for variable batch sizes.
    
    Args:
        current_batch_tokens: Tokens in current batch
        reference_batch_tokens: Reference batch tokens (e.g., from config)
        scaling_type: 'linear', 'sqrt', or 'none'
    
    Returns:
        Scaling factor to multiply with base LR
    """
    if scaling_type == 'none':
        return 1.0
    
    ratio = current_batch_tokens / reference_batch_tokens
    
    if scaling_type == 'linear':
        return ratio
    elif scaling_type == 'sqrt':
        return np.sqrt(ratio)
    else:
        raise ValueError(f"Unknown scaling_type: {scaling_type}")
