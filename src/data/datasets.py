"""
Data loading and preprocessing for multiple NLP datasets.
Designed for flexibility and reproducibility.
"""

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from datasets import load_dataset, Dataset as HFDataset
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
from pathlib import Path
import pickle
import logging

logger = logging.getLogger(__name__)

class TextClassificationDataset(Dataset):
    """
    Generic text classification dataset with efficient tokenization.
    
    Features:
    - Lazy tokenization for memory efficiency
    - Caching for speed
    - Consistent preprocessing across experiments
    """
    
    def __init__(
        self,
        texts: List[str],
        labels: List[int],
        tokenizer_name: str = "bert-base-uncased",
        max_length: int = 512,
        cache_dir: Optional[str] = None,
        cache_tokenization: bool = True
    ):
        self.texts = texts
        self.labels = labels
        self.max_length = max_length
        self.cache_tokenization = cache_tokenization
        
        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name,
            cache_dir=cache_dir
        )
        
        # Add padding token if not present
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Cache tokenized data for efficiency
        self.tokenized_cache = {}
        if cache_tokenization:
            self._precompute_tokenization()
    
    def _precompute_tokenization(self):
        """Pre-tokenize all texts for faster training."""
        logger.info(f"Pre-tokenizing {len(self.texts)} texts...")
        
        for idx in range(len(self.texts)):
            if idx % 1000 == 0:
                logger.info(f"Tokenized {idx}/{len(self.texts)} texts")
            
            self.tokenized_cache[idx] = self._tokenize_text(self.texts[idx])
        
        logger.info("Tokenization complete!")
    
    def _tokenize_text(self, text: str) -> Dict[str, torch.Tensor]:
        """Tokenize a single text."""
        encoded = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoded['input_ids'].squeeze(0),
            'attention_mask': encoded['attention_mask'].squeeze(0)
        }
    
    def __len__(self) -> int:
        return len(self.texts)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # Use cached tokenization if available
        if self.cache_tokenization and idx in self.tokenized_cache:
            tokenized = self.tokenized_cache[idx]
        else:
            tokenized = self._tokenize_text(self.texts[idx])
        
        return {
            'input_ids': tokenized['input_ids'],
            'attention_mask': tokenized['attention_mask'],
            'labels': torch.tensor(self.labels[idx], dtype=torch.long)
        }

class DatasetLoader:
    """
    Handles loading and preprocessing of various NLP datasets.
    Supports IMDB, AG News, SST-2, and custom datasets.
    """
    
    SUPPORTED_DATASETS = {
        'imdb': ('imdb', None),
        'ag_news': ('ag_news', None),
        'sst2': ('glue', 'sst2'),
        'cola': ('glue', 'cola'),
        'rte': ('glue', 'rte')
    }
    
    def __init__(
        self,
        dataset_name: str,
        tokenizer_name: str = "bert-base-uncased",
        max_length: int = 512,
        cache_dir: str = "./data/cache",
        seed: int = 42
    ):
        self.dataset_name = dataset_name
        self.tokenizer_name = tokenizer_name
        self.max_length = max_length
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.seed = seed
        
        # Set random seed for reproducibility
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        logger.info(f"Initializing dataset loader for {dataset_name}")
    
    def load_dataset(
        self, 
        split: str = "train",
        subset_size: Optional[int] = None
    ) -> TextClassificationDataset:
        """
        Load dataset and return our custom dataset class.
        
        Args:
            split: 'train', 'validation', or 'test'
            subset_size: If provided, only load this many samples (for debugging)
        """
        
        # Check if dataset is supported
        if self.dataset_name not in self.SUPPORTED_DATASETS:
            raise ValueError(f"Dataset {self.dataset_name} not supported. "
                           f"Supported: {list(self.SUPPORTED_DATASETS.keys())}")
        
        # Load from HuggingFace
        hf_dataset_name, hf_config = self.SUPPORTED_DATASETS[self.dataset_name]
        
        logger.info(f"Loading {self.dataset_name} dataset (split: {split})")
        
        if hf_config:
            dataset = load_dataset(hf_dataset_name, hf_config, cache_dir=str(self.cache_dir))
        else:
            dataset = load_dataset(hf_dataset_name, cache_dir=str(self.cache_dir))
        
        # Handle different split names
        split_mapping = {
            'train': 'train',
            'validation': 'validation' if 'validation' in dataset else 'test',
            'test': 'test'
        }
        
        actual_split = split_mapping.get(split, split)
        if actual_split not in dataset:
            raise ValueError(f"Split {actual_split} not found in dataset. "
                           f"Available: {list(dataset.keys())}")
        
        data_split = dataset[actual_split]
        
        # Extract texts and labels based on dataset format
        texts, labels = self._extract_texts_and_labels(data_split, self.dataset_name)
        
        # Create subset if requested (useful for debugging)
        if subset_size and subset_size < len(texts):
            indices = np.random.choice(len(texts), size=subset_size, replace=False).tolist()
            texts = [texts[i] for i in indices]
            labels = [labels[i] for i in indices]
            logger.info(f"Using subset of {subset_size} samples")
        
        logger.info(f"Loaded {len(texts)} samples from {self.dataset_name} ({split})")
        
        return TextClassificationDataset(
            texts=texts,
            labels=labels,
            tokenizer_name=self.tokenizer_name,
            max_length=self.max_length,
            cache_dir=str(self.cache_dir)
        )
    
    def _extract_texts_and_labels(
        self, 
        dataset: HFDataset, 
        dataset_name: str
    ) -> Tuple[List[str], List[int]]:
        """Extract texts and labels from HuggingFace dataset."""
        
        if dataset_name == 'imdb':
            texts = dataset['text']
            labels = dataset['label']  # 0: negative, 1: positive
            
        elif dataset_name == 'ag_news':
            texts = dataset['text']
            labels = dataset['label']  # 0-3: World, Sports, Business, Tech
            
        elif dataset_name == 'sst2':
            texts = dataset['sentence']
            labels = dataset['label']  # 0: negative, 1: positive
            
        elif dataset_name == 'cola':
            texts = dataset['sentence']
            labels = dataset['label']  # 0: unacceptable, 1: acceptable
            
        elif dataset_name == 'rte':
            # Combine premise and hypothesis for textual entailment
            texts = [f"{premise} [SEP] {hypothesis}" 
                    for premise, hypothesis in zip(dataset['premise'], dataset['hypothesis'])]
            labels = dataset['label']  # 0: entailment, 1: not_entailment
            
        else:
            raise ValueError(f"Unknown dataset format for {dataset_name}")
        
        return texts, labels
    
    def get_num_classes(self) -> int:
        """Get number of classes for the dataset."""
        class_counts = {
            'imdb': 2,
            'ag_news': 4,
            'sst2': 2,
            'cola': 2,
            'rte': 2,
            'debug': 2
        }
        return class_counts[self.dataset_name]
    
    def create_dataloaders(
        self,
        batch_size: int = 16,
        num_workers: int = 4,
        pin_memory: bool = True,
        train_subset_size: Optional[int] = None,
        eval_subset_size: Optional[int] = None
    ) -> Dict[str, DataLoader]:
        """Create train and evaluation dataloaders."""
        
        # Load datasets
        train_dataset = self.load_dataset('train', subset_size=train_subset_size)
        eval_dataset = self.load_dataset('test', subset_size=eval_subset_size)
        
        # Create dataloaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=True  # For consistent batch sizes
        )
        
        eval_loader = DataLoader(
            eval_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=False
        )
        
        logger.info(f"Created dataloaders - Train: {len(train_loader)} batches, "
                   f"Eval: {len(eval_loader)} batches")
        
        return {
            'train': train_loader,
            'eval': eval_loader
        }

def test_data_loading():
    """Test data loading functionality."""
    print("🧪 Testing data loading...")
    
    # Test IMDB dataset loading
    loader = DatasetLoader(
        dataset_name='imdb',
        max_length=128,  # Small for testing
        cache_dir='./test_cache'
    )
    
    # Load small subset
    train_dataset = loader.load_dataset('train', subset_size=100)
    
    # Test a few samples
    print(f"Dataset size: {len(train_dataset)}")
    
    sample = train_dataset[0]
    print(f"Sample keys: {sample.keys()}")
    print(f"Input IDs shape: {sample['input_ids'].shape}")
    print(f"Attention mask shape: {sample['attention_mask'].shape}")
    print(f"Label: {sample['labels']}")
    
    # Test dataloader
    dataloaders = loader.create_dataloaders(batch_size=4, num_workers=0)
    
    batch = next(iter(dataloaders['train']))
    print(f"Batch input_ids shape: {batch['input_ids'].shape}")
    print(f"Batch labels shape: {batch['labels'].shape}")
    
    print("✅ Data loading test passed!")

if __name__ == "__main__":
    test_data_loading()