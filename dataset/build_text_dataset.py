from typing import List, Optional, Tuple, Dict
from dataclasses import dataclass
from pathlib import Path
import os
import json
import hashlib
import numpy as np
from glob import glob
import re

from argdantic import ArgParser
from pydantic import BaseModel
from datasets import load_dataset

from dataset.common import PuzzleDatasetMetadata


cli = ArgParser()


class TextDataProcessConfig(BaseModel):
    # Dataset configuration
    dataset_name: str = "PrimeIntellect/c4-tiny"  # or "nampdn-ai/mini-en"
    dataset_split: str = "train"
    output_dir: str = "data/text-tiny"
    
    # Text processing
    max_seq_len: int = 512
    min_seq_len: int = 64
    vocab_size: int = 32000
    
    # Training split
    train_ratio: float = 0.9
    seed: int = 42
    
    # Text cleaning
    min_words: int = 10
    max_words: int = 1000
    
    
@dataclass
class TextExample:
    text: str
    length: int
    word_count: int


def clean_text(text: str) -> str:
    """Clean and normalize text"""
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text.strip())
    
    # Remove URLs
    text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
    
    # Remove email addresses
    text = re.sub(r'\S+@\S+', '', text)
    
    # Remove special characters but keep basic punctuation
    text = re.sub(r'[^\w\s\.\,\!\?\;\:\-\(\)\[\]\{\}]', '', text)
    
    return text.strip()


def count_words(text: str) -> int:
    """Count words in text"""
    return len(text.split())


def is_valid_text(text: str, config: TextDataProcessConfig) -> bool:
    """Check if text meets quality criteria"""
    if not text or len(text.strip()) == 0:
        return False
    
    word_count = count_words(text)
    if word_count < config.min_words or word_count > config.max_words:
        return False
    
    # Check for repetitive content
    words = text.split()
    if len(set(words)) < len(words) * 0.3:  # Less than 30% unique words
        return False
    
    return True


def tokenize_text(text: str, vocab_size: int) -> List[int]:
    """Simple character-level tokenization for demonstration"""
    # Create a simple vocabulary
    chars = list(set(text))
    char_to_id = {char: i + 2 for i, char in enumerate(chars[:vocab_size-2])}  # Reserve 0 for PAD, 1 for UNK
    char_to_id['<PAD>'] = 0
    char_to_id['<UNK>'] = 1
    
    # Tokenize
    tokens = []
    for char in text:
        if char in char_to_id:
            tokens.append(char_to_id[char])
        else:
            tokens.append(char_to_id['<UNK>'])
    
    return tokens


def pad_or_truncate(tokens: List[int], max_len: int, pad_id: int = 0) -> List[int]:
    """Pad or truncate token sequence to max_len"""
    if len(tokens) > max_len:
        return tokens[:max_len]
    else:
        return tokens + [pad_id] * (max_len - len(tokens))


def convert_text_dataset(config: TextDataProcessConfig):
    """Convert text dataset to the puzzle format"""
    
    print(f"Loading dataset: {config.dataset_name}")
    
    # Load dataset from Hugging Face
    try:
        dataset = load_dataset(config.dataset_name, split=config.dataset_split)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("Trying alternative dataset...")
        # Try alternative dataset if the first one fails
        try:
            dataset = load_dataset("nampdn-ai/mini-en", split="train")
            print("Using alternative dataset: nampdn-ai/mini-en")
        except Exception as e2:
            print(f"Error loading alternative dataset: {e2}")
            raise Exception("Could not load any text dataset")
    
    # Check if dataset has length (not all datasets support len())
    dataset_length = None
    try:
        dataset_length = len(dataset)  # type: ignore
        print(f"Dataset loaded with {dataset_length} examples")
        has_length = True
    except (TypeError, AttributeError):
        print("Dataset loaded (length unknown - streaming dataset)")
        has_length = False
    
    # Extract and clean text
    texts = []
    for i, example in enumerate(dataset):
        if has_length and dataset_length and i % 1000 == 0:
            print(f"Processing example {i}/{dataset_length}")
        elif not has_length and i % 1000 == 0:
            print(f"Processing example {i}")
        
        # Extract text field (try common field names)
        text = None
        for field in ['text', 'content', 'sentence', 'document']:
            if field in example:
                text = example[field]
                break
        
        if text is None:
            # If no text field found, try to use the first string field
            for key, value in example.items():
                if isinstance(value, str):
                    text = value
                    break
        
        if text is None:
            continue
        
        # Clean text
        text = clean_text(text)
        
        if is_valid_text(text, config):
            texts.append(TextExample(
                text=text,
                length=len(text),
                word_count=count_words(text)
            ))
    
    print(f"Found {len(texts)} valid texts")
    
    # Sort by length for better batching
    texts.sort(key=lambda x: x.length)
    
    # Split into train/test
    np.random.seed(config.seed)
    np.random.shuffle(texts)
    
    split_idx = int(len(texts) * config.train_ratio)
    train_texts = texts[:split_idx]
    test_texts = texts[split_idx:]
    
    print(f"Train: {len(train_texts)} examples, Test: {len(test_texts)} examples")
    
    # Create output directories
    train_dir = os.path.join(config.output_dir, "train")
    test_dir = os.path.join(config.output_dir, "test")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    
    # Process splits
    for split_name, texts_split, output_dir in [
        ("train", train_texts, train_dir),
        ("test", test_texts, test_dir)
    ]:
        print(f"Processing {split_name} split...")
        
        # Tokenize all texts
        all_tokens = []
        for text_example in texts_split:
            tokens = tokenize_text(text_example.text, config.vocab_size)
            tokens = pad_or_truncate(tokens, config.max_seq_len)
            all_tokens.append(tokens)
        
        # Convert to numpy arrays
        inputs = np.array(all_tokens, dtype=np.int32)
        labels = inputs.copy()  # For language modeling, labels are the same as inputs
        
        # Create puzzle identifiers (each text is its own "puzzle")
        puzzle_identifiers = np.arange(len(texts_split), dtype=np.int32)
        
        # Create puzzle indices (each puzzle has one example)
        puzzle_indices = np.arange(len(texts_split) + 1, dtype=np.int32)
        
        # Create group indices (all texts in one group for simplicity)
        group_indices = np.array([0, len(texts_split)], dtype=np.int32)
        
        # Save arrays
        np.save(os.path.join(output_dir, "text__inputs.npy"), inputs)  # type: ignore
        np.save(os.path.join(output_dir, "text__labels.npy"), labels)  # type: ignore
        np.save(os.path.join(output_dir, "text__puzzle_identifiers.npy"), puzzle_identifiers)  # type: ignore
        np.save(os.path.join(output_dir, "text__puzzle_indices.npy"), puzzle_indices)  # type: ignore
        np.save(os.path.join(output_dir, "text__group_indices.npy"), group_indices)  # type: ignore
        
        # Create metadata
        metadata = PuzzleDatasetMetadata(
            pad_id=0,
            ignore_label_id=None,
            blank_identifier_id=-1,
            vocab_size=config.vocab_size,
            seq_len=config.max_seq_len,
            num_puzzle_identifiers=len(texts_split),
            total_groups=1,
            mean_puzzle_examples=len(texts_split),
            sets=["text"]
        )
        
        # Save metadata
        with open(os.path.join(output_dir, "dataset.json"), "w") as f:
            json.dump(metadata.model_dump(), f, indent=2)
        
        print(f"Saved {split_name} split with {len(texts_split)} examples")
    
    print(f"Dataset conversion complete. Output directory: {config.output_dir}")


@cli.command(singleton=True)
def main(config: TextDataProcessConfig):
    """Build text dataset for LLM training"""
    convert_text_dataset(config)


if __name__ == "__main__":
    cli() 