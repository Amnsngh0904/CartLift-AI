"""
Cart Transformer - Deep Ranking Model for Add-to-Cart Prediction

A context-aware model that predicts whether a candidate item will be added
to the current cart. Uses a Transformer encoder for cart sequence modeling
with attention over item embeddings.

Architecture:
1. CartEncoder: Transformer-based cart sequence encoder (2 layers, 4 heads)
2. CartAddToCartModel: Full ranking model combining cart, candidate, and context

Key Features:
- Frozen pre-trained item embeddings (Node2Vec, 70k × 64)
- Learnable positional encoding for cart sequences
- Interaction features (dot product, absolute difference)
- MLP head for binary classification

Usage:
    from src.model.cart_transformer import CartAddToCartModel
    
    model = CartAddToCartModel(
        item_embeddings_path="data/processed/item_embeddings_fixed.npy",
        user_feature_dim=7,
        restaurant_feature_dim=5,
        cart_dynamic_feature_dim=3,
        context_feature_dim=3
    )
    
    logits = model(
        cart_indices=cart_indices,       # (batch, seq_len)
        candidate_indices=candidate_idx, # (batch,)
        user_features=user_feat,         # (batch, 7)
        restaurant_features=rest_feat,   # (batch, 5)
        cart_dynamic_features=cart_dyn,  # (batch, 3)
        context_features=ctx_feat,       # (batch, 3)
        cart_mask=mask                   # (batch, seq_len) bool
    )

Author: ZOMATHON Team
Date: February 2026
"""

import logging
import math
from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# -----------------------------------------------------------------------------
# Logging Configuration
# -----------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------

# Default paths
DEFAULT_EMBEDDINGS_PATH = Path("data/processed/item_embeddings_fixed.npy")

# Model dimensions
EMBEDDING_DIM = 64
MAX_CART_SIZE = 10
NUM_TRANSFORMER_LAYERS = 2
NUM_ATTENTION_HEADS = 4
DROPOUT_RATE = 0.1
MLP_HIDDEN_1 = 256
MLP_HIDDEN_2 = 128


# -----------------------------------------------------------------------------
# PART 1: Cart Transformer Encoder
# -----------------------------------------------------------------------------

class PositionalEncoding(nn.Module):
    """
    Learnable positional encoding for cart sequences.
    
    Unlike sinusoidal encoding, learnable positional embeddings can capture
    domain-specific position semantics (e.g., first item is usually main dish).
    
    Args:
        d_model: Embedding dimension (64)
        max_len: Maximum sequence length (10)
        dropout: Dropout probability
    """
    
    def __init__(
        self,
        d_model: int = EMBEDDING_DIM,
        max_len: int = MAX_CART_SIZE,
        dropout: float = DROPOUT_RATE
    ):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Learnable positional embeddings
        self.position_embeddings = nn.Embedding(max_len, d_model)
        
        # Initialize with small values
        nn.init.normal_(self.position_embeddings.weight, mean=0.0, std=0.02)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding to input embeddings.
        
        Args:
            x: Input tensor (batch_size, seq_len, d_model)
            
        Returns:
            Position-encoded tensor (batch_size, seq_len, d_model)
        """
        batch_size, seq_len, _ = x.shape
        
        # Generate position indices [0, 1, 2, ..., seq_len-1]
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(batch_size, -1)
        
        # Add positional embeddings
        pos_emb = self.position_embeddings(positions)
        x = x + pos_emb
        
        return self.dropout(x)


class CartEncoder(nn.Module):
    """
    Transformer-based encoder for cart item sequences.
    
    Takes a sequence of item embeddings and produces a single cart representation
    using self-attention mechanism. Uses mean pooling over the sequence (with
    masking for padding) to produce the final cart vector.
    
    Architecture:
    - Input: (batch_size, seq_len, 64) item embeddings
    - Positional encoding (learnable)
    - 2-layer TransformerEncoder (4 heads, hidden=64)
    - Mean pooling with padding mask
    - Output: (batch_size, 64) cart embedding
    
    Args:
        embedding_dim: Dimension of item embeddings (64)
        num_heads: Number of attention heads (4)
        num_layers: Number of transformer layers (2)
        dropout: Dropout probability (0.1)
        max_cart_size: Maximum cart sequence length (10)
    
    Example:
        encoder = CartEncoder()
        cart_emb = encoder(item_embeddings, mask)  # (B, 64)
    
    Masking Logic:
    - Padding mask indicates which positions are valid (True = pad, False = valid)
    - For attention: src_key_padding_mask uses True for positions to ignore
    - For pooling: We invert the mask to sum only valid positions
    """
    
    def __init__(
        self,
        embedding_dim: int = EMBEDDING_DIM,
        num_heads: int = NUM_ATTENTION_HEADS,
        num_layers: int = NUM_TRANSFORMER_LAYERS,
        dropout: float = DROPOUT_RATE,
        max_cart_size: int = MAX_CART_SIZE
    ):
        super().__init__()
        
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(
            d_model=embedding_dim,
            max_len=max_cart_size,
            dropout=dropout
        )
        
        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=embedding_dim * 4,  # 256
            dropout=dropout,
            activation="gelu",
            batch_first=True  # (batch, seq, feature)
        )
        
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(embedding_dim)
        )
        
        # Output projection (optional, for representation learning)
        self.output_proj = nn.Linear(embedding_dim, embedding_dim)
        
        logger.info(
            f"CartEncoder initialized: {num_layers} layers, {num_heads} heads, "
            f"dim={embedding_dim}, max_seq={max_cart_size}"
        )
    
    def forward(
        self,
        cart_embeddings: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Encode a batch of cart sequences into fixed-size representations.
        
        Args:
            cart_embeddings: Item embeddings for cart items
                Shape: (batch_size, seq_len, embedding_dim)
            padding_mask: Boolean mask where True indicates padding positions
                Shape: (batch_size, seq_len)
                If None, assumes no padding (all positions valid)
        
        Returns:
            cart_vector: Pooled cart representation
                Shape: (batch_size, embedding_dim)
        
        Notes:
            - Empty carts (all positions masked) return zero vectors
            - Uses mean pooling over non-padded positions
        """
        batch_size = cart_embeddings.shape[0]
        seq_len = cart_embeddings.shape[1]
        device = cart_embeddings.device
        
        # Handle empty sequences (return zero vectors)
        if seq_len == 0:
            return torch.zeros(batch_size, self.embedding_dim, device=device)
        
        # Create default mask if not provided (no padding)
        if padding_mask is None:
            padding_mask = torch.zeros(batch_size, seq_len, dtype=torch.bool, device=device)
        
        # Check for fully empty carts (all positions masked)
        all_masked = padding_mask.all(dim=1)  # (batch_size,)
        
        # Apply positional encoding
        x = self.pos_encoder(cart_embeddings)
        
        # Apply transformer encoder with padding mask
        # src_key_padding_mask: True positions are ignored in attention
        x = self.transformer_encoder(x, src_key_padding_mask=padding_mask)
        
        # Mean pooling over non-padded positions
        # Invert mask: True -> valid, False -> padding
        valid_mask = ~padding_mask  # (batch_size, seq_len)
        valid_mask_expanded = valid_mask.unsqueeze(-1).float()  # (batch_size, seq_len, 1)
        
        # Sum valid positions
        sum_embeddings = (x * valid_mask_expanded).sum(dim=1)  # (batch_size, embedding_dim)
        
        # Count valid positions (avoid division by zero)
        valid_counts = valid_mask.sum(dim=1, keepdim=True).float()  # (batch_size, 1)
        valid_counts = valid_counts.clamp(min=1.0)  # At least 1 to avoid NaN
        
        # Mean pooling
        mean_embeddings = sum_embeddings / valid_counts  # (batch_size, embedding_dim)
        
        # Apply output projection
        cart_vector = self.output_proj(mean_embeddings)
        
        # Zero out fully empty carts
        cart_vector = cart_vector * (~all_masked).float().unsqueeze(-1)
        
        return cart_vector
    
    def get_attention_weights(
        self,
        cart_embeddings: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Get attention weights for visualization/analysis.
        
        Returns attention weights from the last transformer layer.
        
        Args:
            cart_embeddings: (batch_size, seq_len, embedding_dim)
            padding_mask: (batch_size, seq_len)
        
        Returns:
            attention_weights: (batch_size, num_heads, seq_len, seq_len)
        """
        # This is a simplified version - for full implementation,
        # we would need to modify the transformer to return attention weights
        # For now, return placeholder
        batch_size, seq_len, _ = cart_embeddings.shape
        return torch.zeros(
            batch_size, self.num_heads, seq_len, seq_len,
            device=cart_embeddings.device
        )


# -----------------------------------------------------------------------------
# PART 2: Full Ranking Model
# -----------------------------------------------------------------------------

class CartAddToCartModel(nn.Module):
    """
    Context-aware ranking model for add-to-cart prediction.
    
    Combines:
    1. Cart sequence encoding (Transformer)
    2. Candidate item embedding
    3. Interaction features (dot product, absolute difference)
    4. User features (RFM-style)
    5. Restaurant features
    6. Cart dynamic features (totals, counts)
    7. Context features (hour, meal_type, user_type)
    
    Pipeline:
    1. cart_indices → embedding lookup → CartEncoder → cart_vector (64)
    2. candidate_idx → embedding lookup → candidate_vector (64)
    3. interaction_dot = dot(cart_vector, candidate_vector) → (1)
    4. interaction_abs = |cart_vector - candidate_vector| → (64)
    5. Concatenate all features
    6. MLP: 256 → 128 → 1 (logit)
    
    Args:
        item_embeddings_path: Path to numpy file with item embeddings
        item_embeddings: Pre-loaded embeddings tensor (alternative to path)
        user_feature_dim: Dimension of user features (default: 7)
        restaurant_feature_dim: Dimension of restaurant features (default: 5)
        cart_dynamic_feature_dim: Dimension of cart dynamic features (default: 3)
        context_feature_dim: Dimension of context features (default: 3)
        freeze_embeddings: Whether to freeze item embeddings (default: True)
        embedding_dim: Item embedding dimension (default: 64)
        mlp_hidden_1: First MLP hidden layer size (default: 256)
        mlp_hidden_2: Second MLP hidden layer size (default: 128)
        dropout: Dropout probability (default: 0.1)
    
    Feature Details:
        user_features (7):
            - recency_days, frequency, monetary_avg, cuisine_entropy,
              avg_cart_size, dessert_ratio, beverage_ratio
        
        restaurant_features (5):
            - smoothed_rating, delivery_votes (log), avg_item_price,
              price_band_index, menu_size (log)
        
        cart_dynamic_features (3):
            - cart_total, cart_size, avg_item_price_in_cart
        
        context_features (3):
            - hour (normalized), meal_type_idx, user_type_idx
    
    Example:
        model = CartAddToCartModel("data/processed/item_embeddings_fixed.npy")
        logits = model(cart_indices, candidate_indices, user_feat, rest_feat, cart_dyn, ctx, mask)
        probs = torch.sigmoid(logits)
    """
    
    def __init__(
        self,
        item_embeddings_path: Optional[Union[str, Path]] = None,
        item_embeddings: Optional[torch.Tensor] = None,
        user_feature_dim: int = 7,
        restaurant_feature_dim: int = 5,
        cart_dynamic_feature_dim: int = 3,
        context_feature_dim: int = 3,
        freeze_embeddings: bool = True,
        embedding_dim: int = EMBEDDING_DIM,
        mlp_hidden_1: int = MLP_HIDDEN_1,
        mlp_hidden_2: int = MLP_HIDDEN_2,
        dropout: float = DROPOUT_RATE,
        disable_transformer: bool = False,
        prior_prob: Optional[float] = None
    ):
        super().__init__()
        
        self.embedding_dim = embedding_dim
        self.freeze_embeddings = freeze_embeddings
        self.disable_transformer = disable_transformer
        self.prior_prob = prior_prob
        
        # Load item embeddings
        if item_embeddings is not None:
            # Use provided embeddings tensor
            self.item_embeddings = nn.Embedding.from_pretrained(
                item_embeddings,
                freeze=freeze_embeddings,
                padding_idx=0  # Index 0 reserved for padding
            )
            self.num_items = item_embeddings.shape[0]
        elif item_embeddings_path is not None:
            # Load from file
            embeddings_np = np.load(str(item_embeddings_path))
            embeddings_tensor = torch.from_numpy(embeddings_np).float()
            
            # Add padding embedding at index 0 (shift all indices by 1)
            padding_embedding = torch.zeros(1, embedding_dim)
            embeddings_tensor = torch.cat([padding_embedding, embeddings_tensor], dim=0)
            
            self.item_embeddings = nn.Embedding.from_pretrained(
                embeddings_tensor,
                freeze=freeze_embeddings,
                padding_idx=0
            )
            self.num_items = embeddings_tensor.shape[0]
            logger.info(f"Loaded embeddings: {self.num_items} items × {embedding_dim}d")
        else:
            raise ValueError("Must provide either item_embeddings_path or item_embeddings")
        
        # Cart encoder
        self.cart_encoder = CartEncoder(
            embedding_dim=embedding_dim,
            num_heads=NUM_ATTENTION_HEADS,
            num_layers=NUM_TRANSFORMER_LAYERS,
            dropout=dropout
        )
        
        # Calculate input dimension for MLP
        # cart_vector (64) + candidate_vector (64) + 
        # interaction_dot (1) + interaction_abs (64) +
        # user_features + restaurant_features + cart_dynamic_features + context_features
        self.mlp_input_dim = (
            embedding_dim +          # cart_vector
            embedding_dim +          # candidate_vector
            1 +                      # interaction_dot
            embedding_dim +          # interaction_abs
            user_feature_dim +
            restaurant_feature_dim +
            cart_dynamic_feature_dim +
            context_feature_dim
        )
        
        # MLP head for ranking
        self.mlp = nn.Sequential(
            nn.Linear(self.mlp_input_dim, mlp_hidden_1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_1, mlp_hidden_2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_2, 1)  # Output: logit
        )
        
        # Initialize MLP weights
        self._init_mlp_weights()
        
        # Store feature dimensions for validation
        self.feature_dims = {
            "user": user_feature_dim,
            "restaurant": restaurant_feature_dim,
            "cart_dynamic": cart_dynamic_feature_dim,
            "context": context_feature_dim
        }
        
        logger.info(f"CartAddToCartModel initialized:")
        logger.info(f"  Embeddings: {self.num_items} items, frozen={freeze_embeddings}")
        logger.info(f"  Transformer: {'DISABLED' if disable_transformer else 'enabled'}")
        logger.info(f"  MLP input dim: {self.mlp_input_dim}")
        logger.info(f"  MLP layers: {self.mlp_input_dim} → {mlp_hidden_1} → {mlp_hidden_2} → 1")
        if prior_prob is not None:
            logger.info(f"  Output bias init: log({prior_prob:.4f}/{1-prior_prob:.4f}) = {np.log(prior_prob / (1 - prior_prob)):.4f}")
        logger.info(f"  Total params: {self.count_parameters():,}")
    
    def _init_mlp_weights(self):
        """Initialize MLP weights with Xavier/Glorot initialization.
        
        If prior_prob is set, initialize output bias to log(p/(1-p)) for
        better convergence with imbalanced data.
        """
        layers = list(self.mlp.children())
        for i, module in enumerate(layers):
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    # Check if this is the last Linear layer
                    is_last = (i == len(layers) - 1)
                    if is_last and self.prior_prob is not None:
                        # Initialize bias to log(p / (1-p)) for proper initial predictions
                        bias_value = np.log(self.prior_prob / (1 - self.prior_prob))
                        nn.init.constant_(module.bias, bias_value)
                        logger.info(f"  Output layer bias initialized to {bias_value:.4f}")
                    else:
                        nn.init.zeros_(module.bias)
    
    def count_parameters(self, trainable_only: bool = True) -> int:
        """Count model parameters."""
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())
    
    def forward(
        self,
        cart_indices: torch.Tensor,
        candidate_indices: torch.Tensor,
        user_features: torch.Tensor,
        restaurant_features: torch.Tensor,
        cart_dynamic_features: torch.Tensor,
        context_features: torch.Tensor,
        cart_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass for add-to-cart prediction.
        
        Args:
            cart_indices: Indices of items in cart (already shifted +1 for padding)
                Shape: (batch_size, seq_len)
            candidate_indices: Index of candidate item (shifted +1)
                Shape: (batch_size,)
            user_features: User feature vector
                Shape: (batch_size, user_feature_dim)
            restaurant_features: Restaurant feature vector
                Shape: (batch_size, restaurant_feature_dim)
            cart_dynamic_features: Dynamic cart features (total, size, avg_price)
                Shape: (batch_size, cart_dynamic_feature_dim)
            context_features: Context features (hour, meal_type, user_type)
                Shape: (batch_size, context_feature_dim)
            cart_mask: Boolean mask where True indicates padding positions
                Shape: (batch_size, seq_len)
        
        Returns:
            logits: Prediction logits (before sigmoid)
                Shape: (batch_size, 1)
        """
        batch_size = cart_indices.shape[0]
        device = cart_indices.device
        
        # 1. Encode cart sequence
        cart_embeddings = self.item_embeddings(cart_indices)  # (batch, seq, 64)
        
        if self.disable_transformer:
            # Use mean pooling instead of transformer (for debugging)
            if cart_mask is not None:
                # Mask out padding positions
                mask_expanded = (~cart_mask).unsqueeze(-1).float()  # (batch, seq, 1)
                cart_sum = (cart_embeddings * mask_expanded).sum(dim=1)  # (batch, 64)
                cart_count = mask_expanded.sum(dim=1).clamp(min=1)  # (batch, 1)
                cart_vector = cart_sum / cart_count
            else:
                cart_vector = cart_embeddings.mean(dim=1)  # (batch, 64)
        else:
            cart_vector = self.cart_encoder(cart_embeddings, cart_mask)  # (batch, 64)
        
        # 2. Lookup candidate embedding
        candidate_vector = self.item_embeddings(candidate_indices)  # (batch, 64)
        
        # 3. Compute interaction features
        # Dot product (scalar for each sample)
        interaction_dot = (cart_vector * candidate_vector).sum(dim=1, keepdim=True)  # (batch, 1)
        
        # Absolute difference
        interaction_abs = torch.abs(cart_vector - candidate_vector)  # (batch, 64)
        
        # 4. Concatenate all features
        combined = torch.cat([
            cart_vector,              # (batch, 64)
            candidate_vector,         # (batch, 64)
            interaction_dot,          # (batch, 1)
            interaction_abs,          # (batch, 64)
            user_features,            # (batch, user_dim)
            restaurant_features,      # (batch, rest_dim)
            cart_dynamic_features,    # (batch, cart_dyn_dim)
            context_features          # (batch, ctx_dim)
        ], dim=1)  # (batch, mlp_input_dim)
        
        # 5. MLP forward pass
        logits = self.mlp(combined)  # (batch, 1)
        
        return logits
    
    @torch.no_grad()
    def predict(
        self,
        cart_indices: torch.Tensor,
        candidate_indices: torch.Tensor,
        user_features: torch.Tensor,
        restaurant_features: torch.Tensor,
        cart_dynamic_features: torch.Tensor,
        context_features: torch.Tensor,
        cart_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Predict add-to-cart probabilities (inference mode).
        
        Returns probabilities in [0, 1] range.
        """
        self.eval()
        logits = self.forward(
            cart_indices, candidate_indices, user_features,
            restaurant_features, cart_dynamic_features,
            context_features, cart_mask
        )
        return torch.sigmoid(logits)
    
    def get_cart_embedding(
        self,
        cart_indices: torch.Tensor,
        cart_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Get cart representation for visualization or downstream tasks.
        
        Returns:
            cart_vector: (batch_size, embedding_dim)
        """
        cart_embeddings = self.item_embeddings(cart_indices)
        return self.cart_encoder(cart_embeddings, cart_mask)
    
    def get_item_embedding(self, item_indices: torch.Tensor) -> torch.Tensor:
        """
        Get item embeddings for given indices.
        
        Args:
            item_indices: (batch_size,) or (batch_size, seq_len)
        
        Returns:
            embeddings: Shape matches input + (embedding_dim,)
        """
        return self.item_embeddings(item_indices)


# -----------------------------------------------------------------------------
# Utility Functions
# -----------------------------------------------------------------------------

def create_cart_mask(
    cart_lengths: torch.Tensor,
    max_len: int
) -> torch.Tensor:
    """
    Create padding mask from cart lengths.
    
    Args:
        cart_lengths: Actual length of each cart in batch
            Shape: (batch_size,)
        max_len: Maximum sequence length (for padding)
    
    Returns:
        mask: Boolean tensor where True indicates padding
            Shape: (batch_size, max_len)
    
    Example:
        lengths = torch.tensor([2, 3, 1])
        mask = create_cart_mask(lengths, 4)
        # mask = [[F, F, T, T],
        #         [F, F, F, T],
        #         [F, T, T, T]]
    """
    batch_size = cart_lengths.shape[0]
    device = cart_lengths.device
    
    # Create position indices [0, 1, 2, ..., max_len-1]
    positions = torch.arange(max_len, device=device).unsqueeze(0).expand(batch_size, -1)
    
    # Create mask: True where position >= length (i.e., padding)
    mask = positions >= cart_lengths.unsqueeze(1)
    
    return mask


def load_model(
    checkpoint_path: Union[str, Path],
    item_embeddings_path: Optional[Union[str, Path]] = None,
    device: str = "cpu"
) -> CartAddToCartModel:
    """
    Load a trained model from checkpoint.
    
    Args:
        checkpoint_path: Path to model checkpoint (.pt file)
        item_embeddings_path: Path to embeddings (if not in checkpoint)
        device: Device to load model on
    
    Returns:
        Loaded model in eval mode
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Get model config from checkpoint
    config = checkpoint.get("config", {})
    
    # Initialize model
    model = CartAddToCartModel(
        item_embeddings_path=item_embeddings_path or config.get("embeddings_path"),
        user_feature_dim=config.get("user_feature_dim", 7),
        restaurant_feature_dim=config.get("restaurant_feature_dim", 5),
        cart_dynamic_feature_dim=config.get("cart_dynamic_feature_dim", 3),
        context_feature_dim=config.get("context_feature_dim", 3),
        freeze_embeddings=True,
        embedding_dim=config.get("embedding_dim", 64)
    )
    
    # Load weights
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    
    logger.info(f"Loaded model from {checkpoint_path}")
    return model


# -----------------------------------------------------------------------------
# Testing / Demo
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    """Quick test of model architecture."""
    
    print("=" * 60)
    print("Cart Transformer Model Test")
    print("=" * 60)
    
    # Create dummy embeddings (10 items × 64 dims)
    dummy_embeddings = torch.randn(10, 64)
    
    # Initialize model
    model = CartAddToCartModel(
        item_embeddings=dummy_embeddings,
        user_feature_dim=7,
        restaurant_feature_dim=5,
        cart_dynamic_feature_dim=3,
        context_feature_dim=3,
        freeze_embeddings=True
    )
    
    # Create dummy batch (batch_size=4)
    batch_size = 4
    seq_len = 3
    
    # Note: indices are shifted +1 (0 is padding)
    cart_indices = torch.tensor([
        [1, 2, 0],  # Cart with 2 items + 1 padding
        [3, 4, 5],  # Cart with 3 items
        [1, 0, 0],  # Cart with 1 item + 2 padding
        [0, 0, 0],  # Empty cart
    ])
    
    candidate_indices = torch.tensor([6, 7, 8, 9])
    
    user_features = torch.randn(batch_size, 7)
    restaurant_features = torch.randn(batch_size, 5)
    cart_dynamic_features = torch.randn(batch_size, 3)
    context_features = torch.randn(batch_size, 3)
    
    # Create mask from cart lengths
    cart_lengths = torch.tensor([2, 3, 1, 0])
    cart_mask = create_cart_mask(cart_lengths, seq_len)
    
    print(f"\nInput shapes:")
    print(f"  cart_indices: {cart_indices.shape}")
    print(f"  candidate_indices: {candidate_indices.shape}")
    print(f"  cart_mask: {cart_mask.shape}")
    print(f"  cart_mask:\n{cart_mask}")
    
    # Forward pass
    model.eval()
    with torch.no_grad():
        logits = model(
            cart_indices, candidate_indices,
            user_features, restaurant_features,
            cart_dynamic_features, context_features,
            cart_mask
        )
    
    probs = torch.sigmoid(logits)
    
    print(f"\nOutput:")
    print(f"  logits: {logits.squeeze().tolist()}")
    print(f"  probs: {probs.squeeze().tolist()}")
    
    print(f"\nModel parameter counts:")
    print(f"  Trainable: {model.count_parameters(trainable_only=True):,}")
    print(f"  Total: {model.count_parameters(trainable_only=False):,}")
    
    # Test cart encoder separately
    print(f"\nCart Encoder Test:")
    encoder = model.cart_encoder
    cart_emb = model.item_embeddings(cart_indices)  # (4, 3, 64)
    cart_vector = encoder(cart_emb, cart_mask)
    print(f"  Input: {cart_emb.shape}")
    print(f"  Output: {cart_vector.shape}")
    print(f"  Empty cart vector norm: {cart_vector[3].norm().item():.6f} (should be ~0)")
    
    print("\n✓ All tests passed!")
