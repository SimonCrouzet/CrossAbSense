"""AntiBERTy encoder for antibody sequences."""

from typing import Optional, Tuple

import torch

from .base_encoder import BaseEncoder


class AntiBERTyEncoder(BaseEncoder):
    """
    AntiBERTy-based encoder for antibody sequences.

    Uses the official AntiBERTy package: https://github.com/jeffreyruffolo/AntiBERTy
    Pre-trained on 558M natural antibody sequences from OAS.

    Reference:
    Ruffolo et al. (2021) "Deciphering antibody affinity maturation with language models"
    arXiv:2112.07782
    """

    def __init__(
        self,
        model_name: str = "alchemab/antiberty",  # Kept for API compatibility
        use_heavy: bool = True,
        use_light: bool = True,
        pooling: str = "mean",
        freeze_epochs: int = 0,
        precomputed_embeddings_path: Optional[str] = None,
        source_csv_path: Optional[str] = None,
        keep_embeddings_on_cpu: bool = False,
    ):
        """
        Args:
            model_name: Model identifier (kept for compatibility, always uses AntiBERTyRunner)
            use_heavy: Whether to encode heavy chain
            use_light: Whether to encode light chain
            pooling: Pooling strategy (mean, cls, max)
            freeze_epochs: Epochs to keep frozen during fine-tuning
            precomputed_embeddings_path: Path to precomputed embeddings (optional)
            source_csv_path: Path to source CSV for precomputed embeddings (optional)
            keep_embeddings_on_cpu: Keep precomputed embeddings on CPU to save GPU memory (default: False)
        """
        super().__init__(use_heavy, use_light, pooling, freeze_epochs)

        # Store precomputed embeddings settings
        self.precomputed_embeddings_path = precomputed_embeddings_path
        self.source_csv_path = source_csv_path
        self.precomputed_embeddings = None
        self.keep_embeddings_on_cpu = keep_embeddings_on_cpu

        # Load precomputed embeddings if provided
        if precomputed_embeddings_path:
            import logging
            from .base_encoder import validate_and_load_embeddings
            logger = logging.getLogger(__name__)

            self.precomputed_embeddings = validate_and_load_embeddings(
                precomputed_embeddings_path, logger=logger
            )
            # Determine hidden size from first embedding
            first_key = next(iter(self.precomputed_embeddings.keys()))
            first_emb = self.precomputed_embeddings[first_key]
            self.hidden_size = first_emb.shape[-1]
            self.antiberty = None  # No model needed

            # Move embeddings to model device if not keeping on CPU
            if not keep_embeddings_on_cpu:
                # Will be moved to GPU when model.cuda() is called (via _apply hook)
                pass

            logger.info(f"✓ Loaded {len(self.precomputed_embeddings)} precomputed embeddings, hidden_size={self.hidden_size}")
        else:
            # Load official AntiBERTy runner
            try:
                from antiberty import AntiBERTyRunner
                self.antiberty = AntiBERTyRunner()
            except ImportError:
                raise ImportError(
                    "AntiBERTy package not found. Install with: pip install antiberty"
                )

            # AntiBERTy hidden size is 512
            self.hidden_size = 512

        # No projection layers - output native dimensions directly

    def get_embedding_dim(self) -> int:
        """
        Get output embedding dimension per chain (native dimensions, no projection).

        Returns:
            hidden_size (dimension of each chain's embeddings)
        """
        return self.hidden_size

    def forward(
        self,
        heavy_sequences: Optional[list] = None,
        light_sequences: Optional[list] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Encode antibody sequences using AntiBERTy.

        Args:
            heavy_sequences: List of heavy chain sequences (batch_size,)
            light_sequences: List of light chain sequences (batch_size,)

        Returns:
            Tuple of (vh_embeddings, vl_embeddings):
            - If pooling != "none": Each is (batch_size, hidden_size) or None
            - If pooling == "none": Each is (batch_size, max_seq_len, hidden_size) or None
        """
        vh_emb = None
        vl_emb = None

        # Encode heavy chain (no projection, native dimensions)
        if self.use_heavy and heavy_sequences is not None:
            vh_emb = self._encode_sequences(heavy_sequences, chain_type="VH")

        # Encode light chain (no projection, native dimensions)
        if self.use_light and light_sequences is not None:
            vl_emb = self._encode_sequences(light_sequences, chain_type="VL")

        # Validate at least one chain present
        if vh_emb is None and vl_emb is None:
            raise ValueError("Must provide at least one chain (heavy or light)")

        return (vh_emb, vl_emb)

    def _encode_sequences(self, sequences: list, chain_type: str = None) -> torch.Tensor:
        """
        Encode a batch of sequences using official AntiBERTy runner.

        Args:
            sequences: List of protein sequences
            chain_type: Chain type prefix ("VH" or "VL") for precomputed embeddings lookup

        Returns:
            - If pooling != "none": Pooled embeddings of shape (batch_size, hidden_size)
            - If pooling == "none": Unpooled sequences of shape (batch_size, max_seq_len, hidden_size)
        """
        # Use precomputed embeddings if available
        if self.precomputed_embeddings is not None:
            sequence_embeddings = []  # List of (seq_len, hidden_size) tensors

            for seq in sequences:
                # Add chain type prefix if provided (for precomputed embeddings)
                lookup_key = f"{chain_type}:{seq}" if chain_type else seq

                if lookup_key not in self.precomputed_embeddings:
                    raise ValueError(
                        f"Sequence not found in precomputed embeddings: {lookup_key[:50]}...\n"
                        f"Available sequences: {len(self.precomputed_embeddings)}\n"
                        f"Make sure the precomputed embeddings match the input CSV."
                    )

                # Get precomputed embeddings: (seq_len, hidden_size)
                embeddings = self.precomputed_embeddings[lookup_key]

                # Ensure it's a tensor
                if not isinstance(embeddings, torch.Tensor):
                    embeddings = torch.tensor(embeddings, device=self._get_device())

                sequence_embeddings.append(embeddings)

            # Handle pooling
            if self.pooling == "none":
                # Pad sequences to same length for batching
                max_len = max(emb.shape[0] for emb in sequence_embeddings)
                padded_embeddings = []

                for emb in sequence_embeddings:
                    seq_len = emb.shape[0]
                    if seq_len < max_len:
                        # Pad with zeros: (max_len, hidden_size)
                        padding = torch.zeros(max_len - seq_len, emb.shape[1],
                                            dtype=emb.dtype, device=emb.device)
                        emb = torch.cat([emb, padding], dim=0)
                    padded_embeddings.append(emb)

                # Stack into batch: (batch_size, max_seq_len, hidden_size)
                batch_embeddings = torch.stack(padded_embeddings)
            else:
                # Pool each sequence individually, then stack
                pooled_embeddings = []
                for emb in sequence_embeddings:
                    # Add batch dimension: (1, seq_len, hidden_size)
                    emb_batch = emb.unsqueeze(0)
                    # Pool: (1, hidden_size)
                    pooled = self._pool_embeddings(emb_batch)
                    # Remove batch dimension: (hidden_size,)
                    pooled = pooled.squeeze(0)
                    pooled_embeddings.append(pooled)

                # Stack batch: (batch_size, hidden_size)
                batch_embeddings = torch.stack(pooled_embeddings)

            # If keeping embeddings on CPU, move batch to model device now
            if self.keep_embeddings_on_cpu:
                batch_embeddings = batch_embeddings.to(self._get_device())

            return batch_embeddings

        # Otherwise use AntiBERTy model
        # Use official AntiBERTy embed method
        # Returns list of tensors, each of shape [(seq_len + 2) x 512]
        with torch.set_grad_enabled(not self._frozen):
            embeddings_list = self.antiberty.embed(sequences)

        # Pool embeddings for each sequence
        pooled_embeddings = []
        for emb in embeddings_list:
            # emb shape: (seq_len + 2, 512)
            # Convert to tensor if not already
            if not isinstance(emb, torch.Tensor):
                emb = torch.tensor(emb)

            # Add batch dimension: (1, seq_len + 2, 512)
            emb = emb.unsqueeze(0)

            # Use base class pooling method (supports mean, cls, max, sliced_wasserstein)
            pooled = self._pool_embeddings(emb)  # Returns (1, 512)

            # Remove batch dimension: (512,)
            pooled = pooled.squeeze(0)

            pooled_embeddings.append(pooled)

        # Stack into batch: (batch_size, 512)
        batch_embeddings = torch.stack(pooled_embeddings)

        # Move to appropriate device
        if torch.cuda.is_available():
            batch_embeddings = batch_embeddings.to("cuda")

        return batch_embeddings

    def freeze(self):
        """Freeze AntiBERTy model parameters."""
        # Freeze the underlying BERT model in AntiBERTyRunner
        if self.antiberty is not None and hasattr(self.antiberty, 'model'):
            for param in self.antiberty.model.parameters():
                param.requires_grad = False
        self._frozen = True

    def unfreeze(self):
        """Unfreeze AntiBERTy model parameters."""
        # Unfreeze the underlying BERT model in AntiBERTyRunner
        if self.antiberty is not None and hasattr(self.antiberty, 'model'):
            for param in self.antiberty.model.parameters():
                param.requires_grad = True
        self._frozen = False
