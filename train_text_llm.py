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


def train_model(config_path: str = "config/cfg_text_pretrain.yaml"):
    """Train the text language model"""
    print("Starting text LLM training...")
    
    # Set environment variables for training
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Use first GPU
    
    # Load config and create a proper config object
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    # Use hydra_zen.launch with the config dictionary
    result = hydra_zen_launch(
        config_dict,
        launch,
        overrides=[]
    )
    
    print(f"Training completed. Results saved to: {result.working_dir}")
    return result


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