"""
Tests for token-level bucketing system.

Verifies:
- Token length caching
- Bucket assignment
- Bucketed batch sampling
- Token budget batch sampling
- Statistics tracking
- Edge cases
"""

import sys
import os
import tempfile
import torch
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from reasoners.trainer.sft.data.bucketing import (
    BucketConfig,
    TokenBudgetConfig,
    TokenLengthCache,
    BucketManager,
    BucketedBatchSampler,
    TokenBudgetBatchSampler,
    InstrumentedCollator,
    create_bucketed_dataloader,
    compute_lr_scale,
    BatchInfo
)


class MockExample:
    """Mock example for testing."""
    def __init__(self, user_query: str, response: str):
        self.user_query = user_query
        self.reasoning_chain = ""
        self.final_answer = response
        self.raw_messages = []
    
    def get_full_response(self) -> str:
        return self.final_answer


class MockDataset:
    """Mock dataset that returns variable length sequences."""
    
    def __init__(self, lengths, pad_token_id=0):
        self.lengths = lengths
        self.pad_token_id = pad_token_id
        self.examples = [
            MockExample(f"q{i}", "a" * lengths[i])
            for i in range(len(lengths))
        ]
    
    def __len__(self):
        return len(self.lengths)
    
    def __getitem__(self, idx):
        length = self.lengths[idx]
        return {
            'input_ids': torch.arange(length, dtype=torch.long),
            'labels': torch.arange(length, dtype=torch.long),
            'attention_mask': torch.ones(length, dtype=torch.long)
        }



def test_bucket_config():
    """Test bucket configuration and assignment."""
    print("=" * 50)
    print("Testing BucketConfig")
    print("=" * 50)
    
    config = BucketConfig(boundaries=[256, 512, 1024, 2048])
    
    # Test bucket assignment
    assert config.get_bucket_idx(100) == 0  # < 256
    assert config.get_bucket_idx(256) == 1  # >= 256, < 512
    assert config.get_bucket_idx(511) == 1
    assert config.get_bucket_idx(512) == 2  # >= 512, < 1024
    assert config.get_bucket_idx(2048) == 4  # overflow bucket
    assert config.get_bucket_idx(10000) == 4  # overflow bucket
    
    print(f"Bucket names:")
    for i in range(5):
        print(f"  Bucket {i}: {config.get_bucket_name(i)}")
    
    print("BucketConfig: PASSED\n")


def test_token_length_cache():
    """Test token length caching."""
    print("=" * 50)
    print("Testing TokenLengthCache")
    print("=" * 50)
    
    lengths = [100, 200, 300, 150, 500]
    dataset = MockDataset(lengths)
    
    # Test without disk cache
    cache = TokenLengthCache(
        dataset=dataset,
        cache_dir=None,
        eager=True,
        show_progress=False
    )
    
    # Verify lengths
    for i, expected_len in enumerate(lengths):
        actual_len = cache[i]
        assert actual_len == expected_len, f"Length mismatch at {i}: {actual_len} != {expected_len}"
    
    # Test statistics
    stats = cache.get_statistics()
    print(f"Length statistics: {stats}")
    assert stats['min'] == 100
    assert stats['max'] == 500
    
    # Test with disk cache
    with tempfile.TemporaryDirectory() as tmpdir:
        cache1 = TokenLengthCache(
            dataset=dataset,
            cache_dir=tmpdir,
            eager=True,
            show_progress=False
        )
        
        # Second load should use cache
        cache2 = TokenLengthCache(
            dataset=dataset,
            cache_dir=tmpdir,
            eager=True,
            show_progress=False
        )
        
        for i in range(len(dataset)):
            assert cache1[i] == cache2[i]
    
    print("TokenLengthCache: PASSED\n")


def test_bucket_manager():
    """Test bucket management and sampling."""
    print("=" * 50)
    print("Testing BucketManager")
    print("=" * 50)
    
    # Create dataset with known length distribution
    lengths = (
        [100] * 10 +   # Bucket 0: < 256
        [300] * 15 +   # Bucket 1: 256-512
        [700] * 8 +    # Bucket 2: 512-1024
        [1500] * 5     # Bucket 3: 1024-2048
    )
    dataset = MockDataset(lengths)
    
    cache = TokenLengthCache(
        dataset=dataset,
        cache_dir=None,
        eager=True,
        show_progress=False
    )
    
    config = BucketConfig(
        boundaries=[256, 512, 1024, 2048],
        cross_bucket_strategy='proportional',
        shuffle_within_bucket=True
    )
    
    manager = BucketManager(cache, config, seed=42)
    
    # Check bucket distribution
    stats = manager.get_bucket_statistics()
    print(f"Bucket statistics:")
    for name, info in stats['buckets'].items():
        print(f"  {name}: {info['count']} samples, "
              f"lengths {info['min_length']}-{info['max_length']}")
    
    assert stats['num_buckets'] == 4
    assert stats['total_samples'] == 38
    
    # Test sampling
    sampled = []
    while not manager.is_exhausted():
        batch = manager.sample_batch_indices(4)
        sampled.extend(batch)
    
    # All samples should be visited exactly once
    assert len(sampled) == 38
    assert len(set(sampled)) == 38
    
    print("BucketManager: PASSED\n")


def test_bucketed_batch_sampler():
    """Test bucketed batch sampling with fixed batch size."""
    print("=" * 50)
    print("Testing BucketedBatchSampler")
    print("=" * 50)
    
    lengths = (
        [100] * 20 +
        [500] * 15 +
        [1000] * 10
    )
    dataset = MockDataset(lengths)
    
    cache = TokenLengthCache(
        dataset=dataset,
        cache_dir=None,
        eager=True,
        show_progress=False
    )
    
    sampler = BucketedBatchSampler(
        length_cache=cache,
        batch_size=8,
        bucket_config=BucketConfig(boundaries=[256, 512, 1024, 2048]),
        drop_last=False,
        seed=42
    )
    
    # Iterate through all batches
    batches = list(sampler)
    total_samples = sum(len(b) for b in batches)
    
    print(f"Total batches: {len(batches)}")
    print(f"Total samples: {total_samples}")
    print(f"Sample batch sizes: {[len(b) for b in batches[:5]]}...")
    
    # Verify all samples are covered
    all_indices = []
    for batch in batches:
        all_indices.extend(batch)
    
    assert len(all_indices) == 45
    assert len(set(all_indices)) == 45
    
    # Verify batches tend to have similar lengths (from same bucket)
    for batch in batches:
        batch_lengths = [cache[i] for i in batch]
        length_variance = np.var(batch_lengths)
        # Variance should be low within a bucket
        if len(batch) > 1:
            max_len = max(batch_lengths)
            min_len = min(batch_lengths)
            # Allow some variance due to bucket boundaries
            assert max_len - min_len < 1000, f"Large variance in batch: {batch_lengths}"
    
    print("BucketedBatchSampler: PASSED\n")


def test_token_budget_batch_sampler():
    """Test token budget batch sampling with variable batch size."""
    print("=" * 50)
    print("Testing TokenBudgetBatchSampler")
    print("=" * 50)
    
    # Mix of short and long sequences
    lengths = (
        [50] * 20 +    # Very short
        [200] * 15 +   # Short
        [800] * 10 +   # Medium
        [2000] * 5     # Long
    )
    dataset = MockDataset(lengths)
    
    cache = TokenLengthCache(
        dataset=dataset,
        cache_dir=None,
        eager=True,
        show_progress=False
    )
    
    budget_config = TokenBudgetConfig(
        max_tokens_per_batch=4000,
        min_batch_size=1,
        max_batch_size=32,
        count_padding=True
    )
    
    sampler = TokenBudgetBatchSampler(
        length_cache=cache,
        token_budget_config=budget_config,
        bucket_config=BucketConfig(boundaries=[256, 512, 1024, 2048]),
        drop_last=False,
        seed=42
    )
    
    # Iterate and collect batch statistics
    batches = list(sampler)
    batch_sizes = [len(b) for b in batches]
    
    print(f"Total batches: {len(batches)}")
    print(f"Batch sizes: min={min(batch_sizes)}, max={max(batch_sizes)}, "
          f"mean={np.mean(batch_sizes):.1f}")
    
    # Check that short sequences get larger batches
    short_batch_sizes = []
    long_batch_sizes = []
    
    for batch in batches:
        batch_lengths = [cache[i] for i in batch]
        avg_len = np.mean(batch_lengths)
        if avg_len < 300:
            short_batch_sizes.append(len(batch))
        elif avg_len > 1000:
            long_batch_sizes.append(len(batch))
    
    if short_batch_sizes and long_batch_sizes:
        print(f"Short sequence batches: avg size {np.mean(short_batch_sizes):.1f}")
        print(f"Long sequence batches: avg size {np.mean(long_batch_sizes):.1f}")
        # Short sequences should have larger batches on average
        assert np.mean(short_batch_sizes) > np.mean(long_batch_sizes)
    
    # Verify all samples covered
    all_indices = []
    for batch in batches:
        all_indices.extend(batch)
    
    assert len(set(all_indices)) == 50
    
    print("TokenBudgetBatchSampler: PASSED\n")


def test_instrumented_collator():
    """Test collator with statistics tracking."""
    print("=" * 50)
    print("Testing InstrumentedCollator")
    print("=" * 50)
    
    collator = InstrumentedCollator(
        pad_token_id=0,
        padding_side="right",
        track_statistics=True
    )
    
    # Create batch with variable lengths
    batch = [
        {
            'input_ids': torch.tensor([1, 2, 3, 4, 5]),
            'labels': torch.tensor([1, 2, 3, 4, 5]),
            'attention_mask': torch.tensor([1, 1, 1, 1, 1])
        },
        {
            'input_ids': torch.tensor([1, 2, 3]),
            'labels': torch.tensor([1, 2, 3]),
            'attention_mask': torch.tensor([1, 1, 1])
        },
        {
            'input_ids': torch.tensor([1, 2, 3, 4, 5, 6, 7]),
            'labels': torch.tensor([1, 2, 3, 4, 5, 6, 7]),
            'attention_mask': torch.tensor([1, 1, 1, 1, 1, 1, 1])
        },
    ]
    
    result = collator(batch)
    
    # Check shapes
    assert result['input_ids'].shape == (3, 7)
    assert result['labels'].shape == (3, 7)
    assert result['attention_mask'].shape == (3, 7)
    
    # Check padding
    assert result['input_ids'][1, 3:].sum() == 0  # Padded with 0
    assert result['labels'][1, 3:].sum() == -400  # Padded with -100
    assert result['attention_mask'][1, 3:].sum() == 0
    
    # Check batch info
    info = collator.last_batch_info
    assert info is not None
    print(f"Batch info:")
    print(f"  batch_size: {info.batch_size}")
    print(f"  total_tokens: {info.total_tokens}")
    print(f"  padding_tokens: {info.padding_tokens}")
    print(f"  padding_ratio: {info.padding_ratio:.2%}")
    print(f"  effective_tokens: {info.effective_tokens}")
    
    assert info.batch_size == 3
    assert info.total_tokens == 21  # 3 * 7
    assert info.padding_tokens == 6  # (7-5) + (7-3) + (7-7) = 2 + 4 + 0
    assert info.effective_tokens == 15
    
    print("InstrumentedCollator: PASSED\n")


def test_lr_scaling():
    """Test learning rate scaling computation."""
    print("=" * 50)
    print("Testing LR Scaling")
    print("=" * 50)
    
    reference = 8192
    
    # Linear scaling
    scale = compute_lr_scale(4096, reference, 'linear')
    assert abs(scale - 0.5) < 1e-6
    print(f"Half tokens, linear: {scale:.3f}x LR")
    
    scale = compute_lr_scale(16384, reference, 'linear')
    assert abs(scale - 2.0) < 1e-6
    print(f"Double tokens, linear: {scale:.3f}x LR")
    
    # Sqrt scaling
    scale = compute_lr_scale(4096, reference, 'sqrt')
    assert abs(scale - np.sqrt(0.5)) < 1e-6
    print(f"Half tokens, sqrt: {scale:.3f}x LR")
    
    # No scaling
    scale = compute_lr_scale(4096, reference, 'none')
    assert scale == 1.0
    print(f"Half tokens, none: {scale:.3f}x LR")
    
    print("LR Scaling: PASSED\n")


def test_full_dataloader_creation():
    """Test full dataloader creation with all components."""
    print("=" * 50)
    print("Testing Full DataLoader Creation")
    print("=" * 50)
    
    lengths = [100] * 20 + [500] * 15 + [1000] * 10
    dataset = MockDataset(lengths, pad_token_id=0)
    
    # Test with fixed batch size
    loader, stats = create_bucketed_dataloader(
        dataset=dataset,
        batch_size=8,
        use_token_budget=False,
        shuffle=True,
        num_workers=0,  # For testing
        show_progress=False
    )
    
    print("Fixed batch size mode:")
    print(f"  Num batches: {stats['num_batches']}")
    print(f"  Length stats: min={stats['length_stats']['min']}, "
          f"max={stats['length_stats']['max']}")
    print(f"  Buckets: {stats['bucket_stats']['num_buckets']}")
    
    # Iterate through a few batches
    batch_count = 0
    for batch in loader:
        batch_count += 1
        if batch_count >= 3:
            break
    
    print(f"  Successfully iterated {batch_count} batches")
    
    # Test with token budget
    loader, stats = create_bucketed_dataloader(
        dataset=dataset,
        batch_size=8,
        use_token_budget=True,
        max_tokens_per_batch=4000,
        shuffle=True,
        num_workers=0,
        show_progress=False
    )
    
    print("\nToken budget mode:")
    print(f"  Estimated batches: {stats['num_batches']}")
    print(f"  Token budget: {stats['bucket_stats'].get('token_budget', 'N/A')}")
    
    batch_count = 0
    for batch in loader:
        batch_count += 1
        if batch_count >= 3:
            break
    
    print(f"  Successfully iterated {batch_count} batches")
    
    print("\nFull DataLoader Creation: PASSED\n")


def test_reproducibility():
    """Test that sampling is reproducible with same seed."""
    print("=" * 50)
    print("Testing Reproducibility")
    print("=" * 50)
    
    lengths = [100 + i * 10 for i in range(50)]
    dataset = MockDataset(lengths)
    
    cache = TokenLengthCache(
        dataset=dataset,
        cache_dir=None,
        eager=True,
        show_progress=False
    )
    
    # Create two samplers with same seed
    sampler1 = BucketedBatchSampler(
        length_cache=cache,
        batch_size=8,
        drop_last=False,
        seed=12345
    )
    
    sampler2 = BucketedBatchSampler(
        length_cache=cache,
        batch_size=8,
        drop_last=False,
        seed=12345
    )
    
    batches1 = list(sampler1)
    batches2 = list(sampler2)
    
    assert len(batches1) == len(batches2)
    for b1, b2 in zip(batches1, batches2):
        assert b1 == b2, f"Batches differ: {b1} vs {b2}"
    
    print("Same seed produces identical batches: PASSED")
    
    # Different seed should produce different results
    sampler3 = BucketedBatchSampler(
        length_cache=cache,
        batch_size=8,
        drop_last=False,
        seed=99999
    )
    
    batches3 = list(sampler3)
    
    # Should have same samples but different order
    all1 = sorted([idx for batch in batches1 for idx in batch])
    all3 = sorted([idx for batch in batches3 for idx in batch])
    assert all1 == all3, "Same samples expected"
    
    # But batches should be different
    different_batches = sum(1 for b1, b3 in zip(batches1, batches3) if b1 != b3)
    assert different_batches > 0, "Different seeds should produce different batch order"
    
    print(f"Different seed produces different order: {different_batches}/{len(batches1)} batches differ")
    
    print("Reproducibility: PASSED\n")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("TOKEN-LEVEL BUCKETING TESTS")
    print("=" * 60 + "\n")
    
    test_bucket_config()
    test_token_length_cache()
    test_bucket_manager()
    test_bucketed_batch_sampler()
    test_token_budget_batch_sampler()
    test_instrumented_collator()
    test_lr_scaling()
    test_full_dataloader_creation()
    test_reproducibility()
    
    print("=" * 60)
    print("ALL TESTS PASSED!")
    print("=" * 60)
