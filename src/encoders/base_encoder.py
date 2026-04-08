"""Base encoder class for antibody representation."""

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class BaseEncoder(nn.Module, ABC):
    """
    Abstract base class for antibody sequence encoders.

    All encoders must:
    1. Accept heavy and light chain sequences
    2. Return fixed-size embeddings
    3. Support freezing/unfreezing for transfer learning
    """

    def __init__(
        self,
        use_heavy: bool = True,
        use_light: bool = True,
        pooling: str = "mean",
        freeze_epochs: int = 0,
    ):
        """
        Args:
            use_heavy: Whether to use heavy chain
            use_light: Whether to use light chain
            pooling: Pooling strategy (mean, cls, max, sliced_wasserstein, none)
                    - mean/cls/max/sliced_wasserstein: Returns pooled vectors
                    - none: Returns unpooled sequences (for sequence-level attention)
                    - None (from YAML null) or "None" (from CLI args): Converted to "none" automatically
            freeze_epochs: Number of epochs to keep encoder frozen
        """
        super().__init__()
        self.use_heavy = use_heavy
        self.use_light = use_light
        # Convert None (from YAML null) or string "None" (from CLI args) to "none" string
        self.pooling = "none" if (pooling is None or pooling == "None") else pooling
        self.freeze_epochs = freeze_epochs
        self._frozen = False

        # Register a buffer to track device (critical for precomputed embeddings with no parameters)
        # This ensures the encoder knows what device it's on even without trainable parameters
        self.register_buffer('_device_tracker', torch.zeros(1))

        # Validation: at least one chain must be used
        if not use_heavy and not use_light:
            raise ValueError("At least one of use_heavy or use_light must be True!")

        # Warning: antibody properties typically require both VH and VL chains
        if not use_heavy or not use_light:
            logger.warning(
                f"⚠️  Using only {'heavy' if use_heavy else 'light'} chain! "
                "Antibody developability properties typically depend on VH-VL interactions. "
                "Performance may be degraded without both chains."
            )

    @abstractmethod
    def forward(
        self,
        heavy_sequences: Optional[list] = None,
        light_sequences: Optional[list] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Encode antibody sequences to fixed-size embeddings.

        Args:
            heavy_sequences: List of heavy chain sequences
            light_sequences: List of light chain sequences

        Returns:
            Tuple of (vh_embeddings, vl_embeddings):
            - vh_embeddings: Tensor of shape (batch_size, [seq_len,] hidden_dim) or None
            - vl_embeddings: Tensor of shape (batch_size, [seq_len,] hidden_dim) or None
            At least one must be non-None.
        """
        pass

    def freeze(self):
        """
        Freeze encoder parameters for transfer learning.

        Only freezes the pretrained model weights, keeping projection layers trainable.
        This allows adapting the embeddings to the task while preserving pretrained knowledge.
        """
        # Freeze pretrained model (different attribute names per encoder type)
        pretrained_modules = []

        # ESM-C uses self.client
        if hasattr(self, 'client'):
            pretrained_modules.append(self.client)

        # ESM-2 and ProtT5 use self.model
        if hasattr(self, 'model'):
            pretrained_modules.append(self.model)

        # AntiBERTy uses self.antiberty
        if hasattr(self, 'antiberty'):
            pretrained_modules.append(self.antiberty)

        # MultiEncoder has multiple encoders in self.encoders
        if hasattr(self, 'encoders'):
            for encoder in self.encoders.values():
                encoder.freeze()  # Recursively freeze each encoder

        # Freeze only the pretrained model parameters
        for module in pretrained_modules:
            if module is not None:  # Skip if module is None (e.g., when using precomputed embeddings)
                # Skip if module doesn't have parameters() (e.g., ESM3ForgeInferenceClient API)
                if hasattr(module, 'parameters') and callable(getattr(module, 'parameters')):
                    for param in module.parameters():
                        param.requires_grad = False

        # Projection layers (heavy_projection, light_projection) remain trainable
        # Fusion module in MultiEncoder remains trainable

        self._frozen = True

    def unfreeze(self):
        """Unfreeze encoder parameters."""
        for param in self.parameters():
            param.requires_grad = True
        self._frozen = False

    def is_frozen(self) -> bool:
        """Check if encoder is frozen."""
        return self._frozen

    def get_embedding_dim(self) -> int:
        """
        Get output embedding dimension.

        Returns the actual output dimension after chain concatenation.
        This must be implemented by each encoder subclass to return:
        - native_dim * 2 if using both heavy and light chains
        - native_dim if using only one chain

        Raises:
            NotImplementedError: Must be implemented by subclass
        """
        raise NotImplementedError("Subclass must implement get_embedding_dim() to return native dimensions")

    def _get_device(self) -> torch.device:
        """
        Get the current device of the model.

        Uses the device tracker buffer, which is moved with the module even
        when there are no trainable parameters (e.g., precomputed embeddings).

        Returns:
            torch.device: The device where the module is located
        """
        return self._device_tracker.device

    def _apply(self, fn):
        """
        Override _apply to move precomputed embeddings dict when module is moved.

        This is called internally by .to(), .cuda(), .cpu(), etc.
        Ensures precomputed embeddings follow the model device automatically.
        """
        super()._apply(fn)

        # Move precomputed embeddings dict if it exists and not keeping on CPU
        if hasattr(self, 'precomputed_embeddings') and self.precomputed_embeddings is not None:
            if not getattr(self, 'keep_embeddings_on_cpu', False):
                # Move all embeddings in the dict to the target device
                self.precomputed_embeddings = {
                    key: fn(emb) if isinstance(emb, torch.Tensor) else emb
                    for key, emb in self.precomputed_embeddings.items()
                }

        return self

    def _pool_embeddings(
        self,
        embeddings: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Pool sequence embeddings to fixed-size representation.

        Args:
            embeddings: Shape (batch_size, seq_len, hidden_dim)
            attention_mask: Shape (batch_size, seq_len)

        Returns:
            - If pooling != "none": Pooled embeddings of shape (batch_size, hidden_dim)
            - If pooling == "none": Unpooled sequences of shape (batch_size, seq_len, hidden_dim)
        """
        if self.pooling == "none":
            # No pooling - return sequences as-is for sequence-level attention
            return embeddings

        elif self.pooling == "cls":
            # Use [CLS] token embedding (first position)
            return embeddings[:, 0, :]

        elif self.pooling == "mean":
            # Mean pooling over sequence length
            if attention_mask is not None:
                # Mask padding tokens
                mask_expanded = attention_mask.unsqueeze(-1).expand(embeddings.size())
                sum_embeddings = torch.sum(embeddings * mask_expanded, dim=1)
                sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
                return sum_embeddings / sum_mask
            else:
                return torch.mean(embeddings, dim=1)

        elif self.pooling == "max":
            # Max pooling over sequence length
            if attention_mask is not None:
                embeddings = embeddings.clone()
                embeddings[attention_mask == 0] = -1e9
            return torch.max(embeddings, dim=1)[0]

        elif self.pooling == "sliced_wasserstein":
            # Sliced Wasserstein pooling (optimal transport-based)
            from ..utils.pooling import sliced_wasserstein_pool
            return sliced_wasserstein_pool(embeddings, mask=attention_mask)

        else:
            raise ValueError(f"Unknown pooling method: {self.pooling}")


def validate_and_load_embeddings(path: str, logger: Optional[logging.Logger] = None) -> Dict[str, torch.Tensor]:
    """
    Validate and load precomputed embeddings with corruption checking.

    Args:
        path: Path to the .pt file containing embeddings
        logger: Optional logger for status messages

    Returns:
        Dictionary of embeddings

    Raises:
        RuntimeError: If file is corrupted or invalid
        FileNotFoundError: If file doesn't exist
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    path_obj = Path(path)

    # Check file exists
    if not path_obj.exists():
        raise FileNotFoundError(f"Embeddings file not found: {path}")

    # Check file size
    file_size = path_obj.stat().st_size
    if file_size == 0:
        raise RuntimeError(f"Embeddings file is empty: {path}")

    logger.info(f"Loading precomputed embeddings from {path} ({file_size / 1e6:.1f} MB)")

    # Try to load the file
    try:
        embeddings = torch.load(path, weights_only=False)
    except RuntimeError as e:
        if "failed finding central directory" in str(e):
            raise RuntimeError(
                f"Embeddings file is CORRUPTED: {path}\n"
                f"Error: {str(e)}\n"
                f"This usually happens when the file wasn't fully written or got interrupted.\n"
                f"Please regenerate the embeddings file."
            )
        else:
            raise RuntimeError(f"Failed to load embeddings from {path}: {e}")
    except Exception as e:
        raise RuntimeError(f"Unexpected error loading embeddings from {path}: {e}")

    # Validate structure
    if not isinstance(embeddings, dict):
        raise RuntimeError(
            f"Invalid embeddings structure in {path}: expected dict, got {type(embeddings)}"
        )

    if len(embeddings) == 0:
        raise RuntimeError(f"Embeddings file is empty (0 entries): {path}")

    # Validate first embedding
    try:
        first_key = next(iter(embeddings.keys()))
        first_emb = embeddings[first_key]

        if not isinstance(first_emb, torch.Tensor):
            raise RuntimeError(
                f"Invalid embedding type in {path}: expected torch.Tensor, got {type(first_emb)}"
            )

        if first_emb.dim() < 2:
            raise RuntimeError(
                f"Invalid embedding shape in {path}: expected at least 2D, got {first_emb.shape}"
            )

        # Check for NaN/Inf
        if torch.isnan(first_emb).any():
            raise RuntimeError(f"Embeddings contain NaN values: {path}")
        if torch.isinf(first_emb).any():
            raise RuntimeError(f"Embeddings contain Inf values: {path}")

    except StopIteration:
        raise RuntimeError(f"Cannot validate embeddings: dictionary is empty in {path}")

    logger.info(
        f"✓ Loaded {len(embeddings)} valid embeddings, "
        f"shape={first_emb.shape}, dtype={first_emb.dtype}"
    )

    return embeddings
