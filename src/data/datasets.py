import logging
from typing import Dict, Any, Optional, Tuple
from pathlib import Path

import datasets as hfds
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

logger = logging.getLogger(__name__)

# Map dataset name -> how to load + which fields contain text/label
# (All are public on Hugging Face Hub; no uploads needed.)
DATASET_ADAPTERS: Dict[str, Dict[str, Any]] = {
    # Binary sentiment
    "imdb": {
        "loader": lambda cache: hfds.load_dataset("imdb", cache_dir=cache),
        "splits": {"train": "train", "test": "test"},
        "text": ("text",), "label": "label", "num_classes": 2,
    },
    "yelp_polarity": {
        "loader": lambda cache: hfds.load_dataset("yelp_polarity", cache_dir=cache),
        "splits": {"train": "train", "test": "test"},
        "text": ("text",), "label": "label", "num_classes": 2,
    },
    "amazon_polarity": {
        "loader": lambda cache: hfds.load_dataset("amazon_polarity", cache_dir=cache),
        "splits": {"train": "train", "test": "test"},
        "text": ("content",), "label": "label", "num_classes": 2,
    },
    # Topic classification
    "ag_news": {
        "loader": lambda cache: hfds.load_dataset("ag_news", cache_dir=cache),
        "splits": {"train": "train", "test": "test"},
        "text": ("text",), "label": "label", "num_classes": 4,
    },
    "dbpedia_14": {
        "loader": lambda cache: hfds.load_dataset("dbpedia_14", cache_dir=cache),
        "splits": {"train": "train", "test": "test"},
        "text": ("content",), "label": "label", "num_classes": 14,
    },
    "trec": {  # 6 coarse classes
        "loader": lambda cache: hfds.load_dataset("trec", cache_dir=cache),
        "splits": {"train": "train", "test": "test"},
        "text": ("text",), "label": "coarse_label", "num_classes": 6,
    },
    # Emotion / sentiment fine-grained
    "emotion": {
        "loader": lambda cache: hfds.load_dataset("dair-ai/emotion", cache_dir=cache),
        "splits": {"train": "train", "test": "test"},
        "text": ("text",), "label": "label", "num_classes": 6,
    },
    # GLUE SST-2 (binary sentiment)
    "sst2": {
        "loader": lambda cache: hfds.load_dataset("glue", "sst2", cache_dir=cache),
        "splits": {"train": "train", "test": "validation"},  # GLUE uses 'validation'
        "text": ("sentence",), "label": "label", "num_classes": 2,
    },
}

def _concat_text(example: Dict[str, Any], text_fields: Tuple[str, ...]) -> str:
    # Join multiple fields with [SEP] to support pair tasks.
    parts = [example[f] for f in text_fields if f in example and example[f] is not None]
    return " [SEP] ".join(parts)

class _TorchMapDataset(torch.utils.data.Dataset):
    def __init__(self, ds, tokenizer, text_fields, label_field, max_len: int):
        self.ds = ds
        self.tok = tokenizer
        self.text_fields = text_fields
        self.label_field = label_field
        self.max_len = max_len

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        ex = self.ds[idx]
        text = _concat_text(ex, self.text_fields)
        enc = self.tok(
            text,
            truncation=True,
            max_length=self.max_len,
            padding="max_length",
            return_tensors="pt",
        )
        item = {
            "input_ids": enc["input_ids"][0],
            "attention_mask": enc["attention_mask"][0],
            "labels": torch.tensor(int(ex[self.label_field]), dtype=torch.long),
        }
        return item

class DatasetLoader:
    def __init__(self, dataset_name: str, tokenizer_name: str, cache_dir: str):
        self.dataset_name = dataset_name.lower()
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, cache_dir=str(self.cache_dir))
        if self.dataset_name not in DATASET_ADAPTERS:
            raise ValueError(f"Unknown dataset '{dataset_name}'. Available: {list(DATASET_ADAPTERS.keys())}")
        logger.info(f"Initializing dataset loader for {self.dataset_name}")

    def num_classes(self) -> int:
        """Expose number of classes so the model head can be sized correctly."""
        return int(DATASET_ADAPTERS[self.dataset_name]["num_classes"])

    def create_dataloaders(
        self,
        batch_size: int,
        num_workers: int = 4,
        train_subset_size: Optional[int] = None,
        eval_subset_size: Optional[int] = None,
        pin_memory: bool = False,
        max_seq_len: int = 256,
    ) -> Dict[str, DataLoader]:
        cfg = DATASET_ADAPTERS[self.dataset_name]
        raw = cfg["loader"](str(self.cache_dir))
        train_split = cfg["splits"]["train"]
        test_split = cfg["splits"]["test"]
        text_fields = cfg["text"]
        label_field = cfg["label"]

        train_ds = raw[train_split]
        eval_ds = raw[test_split]

        if train_subset_size:
            train_ds = train_ds.select(range(min(train_subset_size, len(train_ds))))
        if eval_subset_size:
            eval_ds = eval_ds.select(range(min(eval_subset_size, len(eval_ds))))

        t_train = _TorchMapDataset(train_ds, self.tokenizer, text_fields, label_field, max_len=max_seq_len)
        t_eval  = _TorchMapDataset(eval_ds,  self.tokenizer, text_fields, label_field, max_len=max_seq_len)

        dl_train = DataLoader(t_train, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=pin_memory)
        dl_eval  = DataLoader(t_eval,  batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=pin_memory)
        return {"train": dl_train, "eval": dl_eval}
