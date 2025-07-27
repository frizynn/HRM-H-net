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
        
        # Language modeling head - use bfloat16 for efficiency
        self.lm_head = nn.Linear(hidden_size, vocab_size, dtype=torch.bfloat16)
        
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
        
        # Ensure hidden_states has the correct shape [batch_size, seq_len, hidden_size]
        if hidden_states.dim() == 2:
            # If hidden_states is [batch_size * seq_len, hidden_size], reshape it
            batch_size = inputs.size(0)
            seq_len = inputs.size(1)
            hidden_states = hidden_states.view(batch_size, seq_len, -1)
        elif hidden_states.dim() == 3:
            # Already in correct shape [batch_size, seq_len, hidden_size]
            pass
        else:
            raise ValueError(f"Unexpected hidden_states shape: {hidden_states.shape}")
        
        # Apply language modeling head - use bfloat16 for efficiency
        hidden_states_bfloat16 = hidden_states.to(torch.bfloat16)
        logits = self.lm_head(hidden_states_bfloat16)  # [batch_size, seq_len, vocab_size]
        
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
        # Adapt config for text - ensure puzzle_emb_ndim is 0 for text models
        text_config = config.copy()
        text_config.update({
            "puzzle_emb_ndim": 0,  # No puzzle embeddings for text
            "num_puzzle_identifiers": 1,  # Minimal value since we don't use puzzle embeddings
            "batch_size": config.get("global_batch_size", 32),
            "seq_len": config.get("max_seq_len", 512),
            "forward_dtype": "bfloat16",  # Use bfloat16 for efficiency
        })
        
        super().__init__(text_config)
    
    @property
    def puzzle_emb(self):
        """Override puzzle_emb to return a dummy sparse embedding for text models"""
        # Return a dummy sparse embedding that has buffers() method
        # This is needed for compatibility with the optimizer
        if not hasattr(self, '_dummy_puzzle_emb'):
            from models.sparse_embedding import CastedSparseEmbedding
            self._dummy_puzzle_emb = CastedSparseEmbedding(
                num_embeddings=1,
                embedding_dim=1,
                batch_size=1,
                init_std=0.0,
                cast_to=torch.float32
            )
        return self._dummy_puzzle_emb
    
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
        # Add dummy puzzle_identifiers if not present
        if "puzzle_identifiers" not in batch:
            batch["puzzle_identifiers"] = torch.zeros(batch["inputs"].shape[0], dtype=torch.long, device=batch["inputs"].device)
        
        # Process through HRM layers
        carry, outputs = super().forward(carry, batch)  # type: ignore
        
        # Extract hidden states from the carry state (z_H contains the final hidden states)
        hidden_states = carry.inner_carry.z_H  # [batch_size, seq_len, hidden_size]
        
        # Create metrics
        metrics = {}
        
        # Create predictions
        preds = outputs
        
        # All sequences are finished for text generation
        all_finished = torch.tensor(True, device=hidden_states.device)
        
        return carry, hidden_states, metrics, preds, all_finished 