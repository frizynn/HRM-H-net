#!/usr/bin/env python3
"""
Text Language Model Training Script

This script trains a text language model using the novel HRM architecture
on small English datasets from Hugging Face.
"""

import os
import sys
import argparse
import yaml
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dataset.build_text_dataset import TextDataProcessConfig, convert_text_dataset
from pretrain import launch
from hydra_zen import launch as hydra_zen_launch
from omegaconf import DictConfig
import torch
from torch.utils.data import DataLoader
import numpy as np
import json
from dataset.common import PuzzleDatasetMetadata


class TextDataset(torch.utils.data.Dataset):
    """Text dataset compatible with the HRM training system"""
    
    def __init__(self, data_path: str, split: str = "train"):
        self.data_path = data_path
        self.split = split
        
        # Load metadata
        with open(os.path.join(data_path, split, "dataset.json"), "r") as f:
            self.metadata = PuzzleDatasetMetadata(**json.load(f))
        
        # Load data
        self.inputs = np.load(os.path.join(data_path, split, "text__inputs.npy"))
        self.labels = np.load(os.path.join(data_path, split, "text__labels.npy"))
        self.puzzle_identifiers = np.load(os.path.join(data_path, split, "text__puzzle_identifiers.npy"))
        self.puzzle_indices = np.load(os.path.join(data_path, split, "text__puzzle_indices.npy"))
        self.group_indices = np.load(os.path.join(data_path, split, "text__group_indices.npy"))
    
    def __len__(self):
        return len(self.inputs)
    
    def __getitem__(self, idx):
        return {
            "inputs": torch.tensor(self.inputs[idx], dtype=torch.long),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long),
            "puzzle_identifiers": torch.tensor(self.puzzle_identifiers[idx], dtype=torch.long),
            "puzzle_indices": torch.tensor(self.puzzle_indices, dtype=torch.long),
            "group_indices": torch.tensor(self.group_indices, dtype=torch.long)
        }


def create_text_dataloader(config_dict: dict, split: str, rank: int, world_size: int, **kwargs):
    """Create dataloader for text dataset"""
    dataset = TextDataset(
        data_path=config_dict['data_path'],
        split=split
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=config_dict['global_batch_size'] // world_size,
        shuffle=(split == "train"),
        num_workers=1,
        pin_memory=True
    )
    
    return dataloader, dataset.metadata


def build_dataset(config_path: str = "config/cfg_text_pretrain.yaml"):
    """Build the text dataset"""
    print("Building text dataset...")
    
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Create dataset config
    dataset_config = TextDataProcessConfig(
        dataset_name="PrimeIntellect/c4-tiny",  # Small English dataset
        output_dir=config['data_path'],
        max_seq_len=512,
        vocab_size=config['arch']['loss']['vocab_size'],
        train_ratio=0.9,
        seed=config['seed']
    )
    
    # Build dataset
    convert_text_dataset(dataset_config)
    print(f"Dataset built successfully at {config['data_path']}")


def create_text_model(config_dict: dict, train_metadata, world_size: int):
    """Create model specifically for text training without puzzle embeddings"""
    from pretrain import load_model_class
    import torch.distributed as dist
    import torch.nn as nn
    
    model_cfg = dict(
        **config_dict['arch'],
        batch_size=config_dict['global_batch_size'] // world_size,
        vocab_size=train_metadata.vocab_size,
        seq_len=train_metadata.seq_len,
        num_puzzle_identifiers=1,  # Minimal value for text models
        puzzle_emb_ndim=0,  # No puzzle embeddings for text
        causal=False  # Non-autoregressive
    )

    # Instantiate model with loss head
    model_cls = load_model_class(config_dict['arch']['name'])
    loss_head_cls = load_model_class(config_dict['arch']['loss']['name'])

    with torch.device("cuda"):
        model: nn.Module = model_cls(model_cfg)
        model = loss_head_cls(model, **config_dict['arch']['loss'])
        if "DISABLE_COMPILE" not in os.environ:
            model = torch.compile(model, dynamic=False)  # type: ignore

        # Broadcast parameters from rank 0
        if world_size > 1:
            with torch.no_grad():
                for param in list(model.parameters()) + list(model.buffers()):
                    dist.broadcast(param, src=0)

    # Optimizers and lr - only use Adam for text models (no puzzle embeddings)
    optimizers = [
        torch.optim.Adam(
            model.parameters(),
            lr=0,  # Needs to be set by scheduler
            weight_decay=config_dict['weight_decay'],
            betas=(config_dict['beta1'], config_dict['beta2'])
        )
    ]
    optimizer_lrs = [
        config_dict['lr']
    ]

    return model, optimizers, optimizer_lrs


def text_launch(config_dict: dict):
    """Custom launch function for text training"""
    from pretrain import init_train_state, train_batch, evaluate, save_train_state, compute_lr, PretrainConfig
    import torch.distributed as dist
    import wandb
    import tqdm
    import math
    
    # Initialize distributed training
    RANK = 0
    WORLD_SIZE = 1
    
    if "LOCAL_RANK" in os.environ:
        dist.init_process_group(backend="nccl")
        RANK = dist.get_rank()
        WORLD_SIZE = dist.get_world_size()
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    
    # Convert dict to PretrainConfig
    config = PretrainConfig(**config_dict)
    
    # Create dataloaders using text dataset
    train_loader, train_metadata = create_text_dataloader(
        config_dict, "train", rank=RANK, world_size=WORLD_SIZE
    )
    eval_loader, eval_metadata = create_text_dataloader(
        config_dict, "test", rank=RANK, world_size=WORLD_SIZE
    )
    
    # Initialize model and training state using custom text model creation
    model, optimizers, optimizer_lrs = create_text_model(config_dict, train_metadata, world_size=WORLD_SIZE)
    
    # Check GPU usage
    device = next(model.parameters()).device
    print(f"Model is on device: {device}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Training loop with detailed logging
    model.train()
    
    # Initialize carry state
    carry = None
    
    # Training statistics
    train_losses = []
    eval_losses = []
    
    for epoch in range(config_dict['epochs']):
        print(f"\n{'='*50}")
        print(f"Epoch {epoch + 1}/{config_dict['epochs']}")
        print(f"{'='*50}")
        
        # Training phase
        model.train()
        epoch_train_losses = []
        
        for batch_idx, batch in enumerate(train_loader):
            try:
                # Move batch to GPU
                batch = {k: v.cuda() for k, v in batch.items()}
                
                # Initialize carry if it's None
                if carry is None:
                    carry = model.initial_carry(batch)
                
                # Forward pass
                carry, loss, metrics, predictions, all_finished = model(carry, batch, return_keys=[])
                
                # Backward pass
                loss.backward()
                
                # Update weights
                for optimizer in optimizers:
                    optimizer.step()
                    optimizer.zero_grad()
                
                # Store loss
                loss_value = float(loss.detach().cpu())
                epoch_train_losses.append(loss_value)
                
                # Print progress
                if batch_idx % 5 == 0:
                    avg_loss = sum(epoch_train_losses[-10:]) / min(len(epoch_train_losses), 10)
                    print(f"  Train Batch {batch_idx:3d}, Loss: {loss_value:.4f}, Avg Loss: {avg_loss:.4f}")
                    
            except Exception as e:
                print(f"Error in training batch {batch_idx}: {e}")
                continue
        
        # Calculate average training loss for this epoch
        avg_train_loss = sum(epoch_train_losses) / len(epoch_train_losses) if epoch_train_losses else 0
        train_losses.append(avg_train_loss)
        
        # Evaluation phase
        model.eval()
        epoch_eval_losses = []
        
        print(f"\nEvaluating...")
        with torch.no_grad():
            eval_carry = None
            for batch_idx, batch in enumerate(eval_loader):
                try:
                    # Move batch to GPU
                    batch = {k: v.cuda() for k, v in batch.items()}
                    
                    # Initialize carry if it's None
                    if eval_carry is None:
                        eval_carry = model.initial_carry(batch)
                    
                    # Forward pass
                    eval_carry, loss, metrics, predictions, all_finished = model(eval_carry, batch, return_keys=[])
                    
                    # Store loss
                    loss_value = float(loss.detach().cpu())
                    epoch_eval_losses.append(loss_value)
                    
                    # Print progress
                    if batch_idx % 5 == 0:
                        print(f"  Eval Batch {batch_idx:3d}, Loss: {loss_value:.4f}")
                        
                except Exception as e:
                    print(f"Error in evaluation batch {batch_idx}: {e}")
                    continue
        
        # Calculate average evaluation loss for this epoch
        avg_eval_loss = sum(epoch_eval_losses) / len(epoch_eval_losses) if epoch_eval_losses else 0
        eval_losses.append(avg_eval_loss)
        
        # Print epoch summary
        print(f"\nEpoch {epoch + 1} Summary:")
        print(f"  Train Loss: {avg_train_loss:.4f}")
        print(f"  Eval Loss:  {avg_eval_loss:.4f}")
        print(f"  Train/Eval Ratio: {avg_train_loss/avg_eval_loss:.2f}" if avg_eval_loss > 0 else "  Train/Eval Ratio: N/A")
        
        # Memory usage
        if torch.cuda.is_available():
            memory_allocated = torch.cuda.memory_allocated(0) / 1e9
            memory_reserved = torch.cuda.memory_reserved(0) / 1e9
            print(f"  GPU Memory: {memory_allocated:.2f}GB allocated, {memory_reserved:.2f}GB reserved")
    
    # Final summary
    print(f"\n{'='*50}")
    print("Training completed!")
    print(f"{'='*50}")
    print(f"Final Train Loss: {train_losses[-1]:.4f}")
    print(f"Final Eval Loss:  {eval_losses[-1]:.4f}")
    print(f"Best Train Loss:  {min(train_losses):.4f}")
    print(f"Best Eval Loss:   {min(eval_losses):.4f}")
    
    if torch.cuda.is_available():
        memory_allocated = torch.cuda.memory_allocated(0) / 1e9
        print(f"Final GPU Memory: {memory_allocated:.2f}GB")


def train_model(config_path: str = "config/cfg_text_pretrain.yaml"):
    """Train the text language model"""
    print("Starting text LLM training...")
    
    # Set environment variables for training
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Use first GPU
    
    # Load config and create a proper config object
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    # Use custom text launch function
    text_launch(config_dict)
    
    print("Training completed!")


def main():
    parser = argparse.ArgumentParser(description="Train Text Language Model")
    parser.add_argument("--config", default="config/cfg_text_pretrain.yaml", 
                       help="Path to configuration file")
    parser.add_argument("--build-dataset", action="store_true",
                       help="Build the text dataset before training")
    parser.add_argument("--train-only", action="store_true",
                       help="Only train the model (skip dataset building)")
    
    args = parser.parse_args()
    
    # Check if config file exists
    if not os.path.exists(args.config):
        print(f"Configuration file {args.config} not found!")
        print("Creating default configuration...")
        
        # Create default config
        default_config = {
            "arch": {
                "name": "models.text_language_model.TextHRM_v1",
                "loss": {
                    "name": "models.text_language_model.TextLanguageModelHead",
                    "vocab_size": 32000
                }
            },
            "data_path": "data/text-tiny",
            "global_batch_size": 32,
            "epochs": 10,
            "lr": 1e-4,
            "lr_min_ratio": 0.1,
            "lr_warmup_steps": 1000,
            "weight_decay": 0.01,
            "beta1": 0.9,
            "beta2": 0.999,
            "puzzle_emb_lr": 1e-4,
            "puzzle_emb_weight_decay": 0.01,
            "project_name": "Text-LLM-HRM",
            "run_name": None,
            "checkpoint_path": None,
            "seed": 42,
            "checkpoint_every_eval": True,
            "eval_interval": 2,
            "eval_save_outputs": ["logits", "predictions"]
        }
        
        os.makedirs(os.path.dirname(args.config), exist_ok=True)
        with open(args.config, 'w') as f:
            yaml.dump(default_config, f, default_flow_style=False)
        
        print(f"Created default configuration at {args.config}")
    
    # Build dataset if requested or if dataset doesn't exist
    data_path = None
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
        data_path = config.get('data_path', 'data/text-tiny')
    
    if args.build_dataset or (not args.train_only and not os.path.exists(data_path)):
        build_dataset(args.config)
    
    # Train model
    if not args.build_dataset or not args.train_only:
        train_model(args.config)


if __name__ == "__main__":
    main() 