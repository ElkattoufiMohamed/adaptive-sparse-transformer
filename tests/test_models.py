"""
Comprehensive tests for our transformer models.
"""

import pytest
import torch
import numpy as np
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from models.transformer import AdaptiveSparseTransformer
from models.baseline_transformer import BaselineTransformer
from models.adaptive_attention import AdaptiveSparseAttention

class TestAdaptiveSparseAttention:
    """Test the core adaptive attention mechanism."""
    
    def test_attention_initialization(self):
        """Test that attention module initializes correctly."""
        attention = AdaptiveSparseAttention(
            dim=768,
            num_heads=12,
            dropout=0.1,
            local_window_size=32
        )
        
        assert attention.dim == 768
        assert attention.num_heads == 12
        assert attention.head_dim == 64  # 768 / 12
        
    def test_attention_forward_pass(self):
        """Test forward pass with different input sizes."""
        attention = AdaptiveSparseAttention(dim=768, num_heads=12)
        
        # Test different batch sizes and sequence lengths
        test_cases = [
            (1, 64, 768),    # Small batch, short sequence
            (4, 128, 768),   # Medium batch, medium sequence
            (2, 256, 768),   # Small batch, long sequence
        ]
        
        for batch_size, seq_len, dim in test_cases:
            x = torch.randn(batch_size, seq_len, dim)
            
            output, attention_info = attention(x)
            
            # Check output shape
            assert output.shape == (batch_size, seq_len, dim)
            
            # Check attention info
            assert 'pattern_weights' in attention_info
            assert attention_info['pattern_weights'].shape == (batch_size, 3)
            
            # Check pattern weights sum to 1 (softmax output)
            pattern_sums = attention_info['pattern_weights'].sum(dim=-1)
            assert torch.allclose(pattern_sums, torch.ones_like(pattern_sums), atol=1e-6)
    
    def test_attention_masks(self):
        """Test that attention masks work correctly."""
        attention = AdaptiveSparseAttention(dim=64, num_heads=4, local_window_size=8)
        
        batch_size, seq_len = 2, 16
        x = torch.randn(batch_size, seq_len, 64)
        
        # Create attention mask (mask out second half of second sample)
        attention_mask = torch.ones(batch_size, seq_len)
        attention_mask[1, seq_len//2:] = 0
        
        output, attention_info = attention(x, attention_mask)
        
        assert output.shape == (batch_size, seq_len, 64)
        
        # Check that masked positions don't affect output significantly
        # (This is a basic sanity check)
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()

class TestAdaptiveSparseTransformer:
    """Test the complete adaptive transformer model."""
    
    def test_model_initialization(self):
        """Test model initializes with correct parameters."""
        model = AdaptiveSparseTransformer(
            vocab_size=1000,
            dim=256,
            depth=6,
            num_heads=8,
            num_classes=2
        )
        
        assert model.dim == 256
        assert model.depth == 6
        assert len(model.blocks) == 6
        
        # Test parameter count
        total_params = sum(p.numel() for p in model.parameters())
        assert total_params > 0
        
    def test_model_forward_pass(self):
        """Test complete forward pass."""
        model = AdaptiveSparseTransformer(
            vocab_size=1000,
            dim=128,  # Smaller for testing
            depth=2,
            num_heads=4,
            num_classes=3
        )
        
        batch_size, seq_len = 2, 32
        input_ids = torch.randint(0, 1000, (batch_size, seq_len))
        attention_mask = torch.ones(batch_size, seq_len)
        
        # Forward pass
        outputs = model(input_ids, attention_mask, return_attention_info=True)
        
        # Check outputs
        assert 'logits' in outputs
        assert outputs['logits'].shape == (batch_size, 3)  # num_classes
        assert 'attention_info' in outputs
        assert len(outputs['attention_info']) == 2  # depth
        
    def test_model_training_mode(self):
        """Test that model behaves correctly in train/eval modes."""
        model = AdaptiveSparseTransformer(vocab_size=100, dim=64, depth=2, num_heads=4)
        
        input_ids = torch.randint(0, 100, (1, 16))
        
        # Test training mode
        model.train()
        output1 = model(input_ids)
        output2 = model(input_ids)
        
        # Outputs should be different due to dropout
        assert not torch.equal(output1['logits'], output2['logits'])
        
        # Test evaluation mode
        model.eval()
        with torch.no_grad():
            output3 = model(input_ids)
            output4 = model(input_ids)
        
        # Outputs should be identical in eval mode
        assert torch.allclose(output3['logits'], output4['logits'], atol=1e-6)

class TestDataLoading:
    """Test data loading functionality."""
    
    def test_dataset_creation(self):
        """Test custom dataset creation."""
        from src.data.datasets import TextClassificationDataset
        
        texts = ["This is a test sentence.", "Another test sentence."]
        labels = [0, 1]
        
        dataset = TextClassificationDataset(
            texts=texts,
            labels=labels,
            tokenizer_name="bert-base-uncased",
            max_length=32,
            cache_tokenization=False  # Skip caching for testing
        )
        
        assert len(dataset) == 2
        
        sample = dataset[0]
        assert 'input_ids' in sample
        assert 'attention_mask' in sample
        assert 'labels' in sample
        assert sample['input_ids'].shape[0] == 32  # max_length
        assert sample['labels'].item() == 0
    
    def test_dataloader_creation(self):
        """Test DataLoader creation and batching."""
        from src.data.datasets import DatasetLoader
        
        # Create a small test dataset
        loader = DatasetLoader(
            dataset_name='imdb',
            max_length=64,
            cache_dir='./test_cache'
        )
        
        # Test with very small subset
        train_dataset = loader.load_dataset('train', subset_size=10)
        assert len(train_dataset) == 10
        
        # Test dataloader creation
        from torch.utils.data import DataLoader
        dataloader = DataLoader(train_dataset, batch_size=4, shuffle=False)
        
        batch = next(iter(dataloader))
        assert batch['input_ids'].shape == (4, 64)
        assert batch['attention_mask'].shape == (4, 64)
        assert batch['labels'].shape == (4,)

def test_reproducibility():
    """Test that results are reproducible with same seed."""
    from src.utils.helpers import set_seed
    
    # Test 1: Same seed should give same results
    set_seed(42)
    model1 = AdaptiveSparseTransformer(vocab_size=100, dim=64, depth=2, num_heads=4)
    input_ids = torch.randint(0, 100, (1, 16))
    
    model1.eval()
    with torch.no_grad():
        output1 = model1(input_ids)
    
    # Reset and create identical model
    set_seed(42)
    model2 = AdaptiveSparseTransformer(vocab_size=100, dim=64, depth=2, num_heads=4)
    
    model2.eval()
    with torch.no_grad():
        output2 = model2(input_ids)
    
    # Results should be identical
    assert torch.allclose(output1['logits'], output2['logits'], atol=1e-6)
    
    print("✅ Reproducibility test passed!")

if __name__ == "__main__":
    # Run basic tests
    test_reproducibility()
    print("✅ All basic tests passed!")