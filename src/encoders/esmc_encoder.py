"""ESM-C encoder for antibody sequences."""

import os
from typing import Optional, Tuple

import torch

from .base_encoder import BaseEncoder


class ESMCEncoder(BaseEncoder):
    """
    ESM-C based encoder for antibody sequences.

    ESM-C is the latest protein language model from EvolutionaryScale.
    Available models: esmc_300m, esmc_600m (open), esmc_6b (via Forge API)
    GitHub: https://github.com/evolutionaryscale/esm
    """

    def __init__(
        self,
        model_name: str = "esmc_600m",
        use_heavy: bool = True,
        use_light: bool = True,
        pooling: str = "mean",
        freeze_epochs: int = 0,
        forge_token: Optional[str] = None,
        precomputed_embeddings_path: Optional[str] = None,
        source_csv_path: Optional[str] = None,
        keep_embeddings_on_cpu: bool = False,
    ):
        """
        Args:
            model_name: ESM-C model name (esmc_300m, esmc_600m, esmc_6b)
                       - esmc_300m: 300M params, 960 hidden dim (native)
                       - esmc_600m: 600M params, 1152 hidden dim (native, default)
                       - esmc_6b: 6B params, 2560 hidden dim (native, requires Forge API)
            use_heavy: Whether to encode heavy chain
            use_light: Whether to encode light chain
            pooling: Pooling strategy (mean, cls, max)
            freeze_epochs: Epochs to keep frozen during fine-tuning
            forge_token: Token for Forge API (required for esmc_6b)
            precomputed_embeddings_path: Path to precomputed embeddings (optional)
            source_csv_path: Path to source CSV for precomputed embeddings (optional)
            keep_embeddings_on_cpu: Keep precomputed embeddings on CPU to save GPU memory (default: False)
        """
        super().__init__(use_heavy, use_light, pooling, freeze_epochs)

        self.model_name = model_name

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
            self.hidden_size = first_emb.shape[-1]  # Last dimension is hidden_size
            self.client = None  # No client needed
            self.use_forge = False
            self.model_name = model_name
        else:
            # Map model names to match config files
            model_name_map = {
                "facebook/esmc_300m": "esmc_300m",
                "facebook/esmc_600m": "esmc_600m",
                "facebook/esmc_6b": "esmc-6b-2024-12",
                "esmc_300m": "esmc_300m",
                "esmc_600m": "esmc_600m",
                "esmc_6b": "esmc-6b-2024-12",
            }

            esm_model_name = model_name_map.get(model_name, "esmc_600m")
            self.model_name = esm_model_name

            # Determine if we need Forge API (for 6B model)
            self.use_forge = "6b" in esm_model_name

            if self.use_forge:
                # Use Forge API for 6B model
                from esm.sdk.forge import ESM3ForgeInferenceClient

                # Get token from parameter or environment
                token = forge_token or os.getenv("FORGE_TOKEN")
                if not token:
                    raise ValueError(
                        "Forge token required for esmc_6b. "
                        "Set FORGE_TOKEN environment variable or pass forge_token parameter."
                    )

                self.client = ESM3ForgeInferenceClient(
                    model=esm_model_name,
                    url="https://forge.evolutionaryscale.ai",
                    token=token,
                )
                # 6B model has 2560 hidden size
                self.hidden_size = 2560
            else:
                # Use local models for 300M and 600M
                from esm.models.esmc import ESMC

                self.client = ESMC.from_pretrained(esm_model_name)

                # Move to device if available
                if torch.cuda.is_available():
                    self.client = self.client.to("cuda")

                # Get hidden size based on model
                # Note: These are the native embedding dimensions from the models
                if "300m" in esm_model_name:
                    self.hidden_size = 960  # ESM-C 300M native dimension
                else:  # 600M
                    self.hidden_size = 1152  # ESM-C 600M native dimension

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
        Encode antibody sequences using ESM-C.

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
        Encode a batch of sequences using ESM-C.

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

        # Otherwise use API/model
        from esm.sdk.api import ESMProtein, LogitsConfig

        batch_embeddings = []

        # Process each sequence (ESM-C API processes one at a time)
        for seq in sequences:
            # Create ESMProtein object
            protein = ESMProtein(sequence=seq)

            # Encode protein
            with torch.set_grad_enabled(not self._frozen):
                protein_tensor = self.client.encode(protein)

                # Get embeddings using logits API
                logits_output = self.client.logits(
                    protein_tensor,
                    LogitsConfig(sequence=True, return_embeddings=True)
                )

                # Extract embeddings: (seq_len, hidden_size)
                embeddings = logits_output.embeddings

                # Ensure embeddings is a tensor
                if not isinstance(embeddings, torch.Tensor):
                    embeddings = torch.tensor(embeddings)

                # Handle different shapes: could be (seq_len, hidden_size) or (1, seq_len, hidden_size)
                if embeddings.dim() == 3:
                    embeddings = embeddings.squeeze(0)

                # Add batch dimension: (1, seq_len, hidden_size)
                embeddings = embeddings.unsqueeze(0)

                # Use base class pooling method (supports mean, cls, max, sliced_wasserstein)
                pooled = self._pool_embeddings(embeddings)  # Returns (1, hidden_size)

                # Remove batch dimension: (hidden_size,)
                pooled = pooled.squeeze(0)

                batch_embeddings.append(pooled)

        # Stack batch
        batch_embeddings = torch.stack(batch_embeddings)

        return batch_embeddings
