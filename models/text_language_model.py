from typing import Optional, Any, Dict, List, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from models.hrm.hrm_act_v1 import HierarchicalReasoningModel_ACTV1 as HRM_ACT_v1
from models.losses import IGNORE_LABEL_ID


class TextLanguageModelHead(nn.Module):
    """Language model head for text generation"""
    
    def __init__(self, model: nn.Module, vocab_size: int, hidden_size: int, **kwargs):
        super().__init__()
        self.model = model
        self.vocab_size = vocab_size
        
        # Language modeling head
        self.lm_head = nn.Linear(hidden_size, vocab_size)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
    
    def initial_carry(self, batch: Dict[str, Tensor]) -> Any:
        """Initialize carry state for the model"""
        return self.model.initial_carry(batch)  # type: ignore
    
    def forward(self, carry: Any, batch: Dict[str, Tensor], return_keys: Optional[List[str]] = None) -> tuple:
        """
        Forward pass for language modeling
        
        Args:
            carry: Model carry state
            batch: Input batch with keys:
                - inputs: Input token ids [batch_size, seq_len]
                - labels: Target token ids [batch_size, seq_len]
            return_keys: Keys to return in predictions
            
        Returns:
            tuple: (new_carry, loss, metrics, predictions, all_finished)
        """
        inputs = batch["inputs"]
        labels = batch["labels"]
        
        # Forward through base model
        carry, hidden_states, metrics, preds, all_finished = self.model(carry, batch, return_keys)  # type: ignore
        
        # Apply language modeling head
        logits = self.lm_head(hidden_states)  # [batch_size, seq_len, vocab_size]
        
        # Compute loss
        loss = self._compute_loss(logits, labels)
        
        # Update metrics
        if metrics is None:
            metrics = {}
        metrics["loss"] = loss
        metrics["count"] = torch.tensor(inputs.size(0), device=inputs.device)
        
        # Update predictions
        if preds is None:
            preds = {}
        preds["logits"] = logits
        preds["predictions"] = torch.argmax(logits, dim=-1)
        
        return carry, loss, metrics, preds, all_finished
    
    def _compute_loss(self, logits: Tensor, labels: Tensor) -> Tensor:
        """Compute language modeling loss"""
        # Reshape for cross entropy
        batch_size, seq_len, vocab_size = logits.shape
        logits_flat = logits.view(-1, vocab_size)
        labels_flat = labels.view(-1)
        
        # Create mask for padding tokens
        mask = (labels_flat != IGNORE_LABEL_ID) & (labels_flat != 0)  # Exclude PAD and IGNORE tokens
        
        # Compute loss only on valid tokens
        if mask.sum() > 0:
            loss = F.cross_entropy(
                logits_flat[mask], 
                labels_flat[mask], 
                reduction='mean'
            )
        else:
            loss = torch.tensor(0.0, device=logits.device, requires_grad=True)
        
        return loss


class TextHRM_v1(HRM_ACT_v1):
    """Text-specific HRM model for language modeling"""
    
    def __init__(self, config: Dict[str, Any]):
        # Adapt config for text
        text_config = config.copy()
        text_config.update({
            "input_dim": config.get("vocab_size", 32000),
            "output_dim": config.get("vocab_size", 32000),
            "causal": True,  # Enable causal attention for language modeling
        })
        
        super().__init__(text_config)
    
    def forward(self, carry: Any, batch: Dict[str, Tensor], return_keys: Optional[List[str]] = None) -> tuple:
        """
        Forward pass for text processing
        
        Args:
            carry: Model carry state
            batch: Input batch with keys:
                - inputs: Input token ids [batch_size, seq_len]
                - labels: Target token ids [batch_size, seq_len]
            return_keys: Keys to return in predictions
            
        Returns:
            tuple: (new_carry, hidden_states, metrics, predictions, all_finished)
        """
        inputs = batch["inputs"]
        
        # Convert token ids to embeddings
        embeddings = self.input_embedding(inputs)  # type: ignore # [batch_size, seq_len, hidden_size]
        
        # Process through HRM layers
        carry, outputs = super().forward(carry, {"inputs": embeddings})  # type: ignore
        
        # Extract hidden states and other outputs
        hidden_states = outputs.get("logits", embeddings)  # Use logits as hidden states
        metrics = {}
        preds = outputs
        all_finished = torch.tensor(True)  # Default to finished
        
        return carry, hidden_states, metrics, preds, all_finished 