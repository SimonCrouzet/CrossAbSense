"""ProtT5 encoder for antibody sequences."""

from typing import Optional, Tuple

import torch
from transformers import AutoModel, AutoTokenizer

from .base_encoder import BaseEncoder


class ProtT5Encoder(BaseEncoder):
    """
    ProtT5 based encoder for antibody sequences.

    ProtT5 is a protein language model trained on UniRef50 using T5 architecture.
    Model: Rostlab/prot_t5_xl_uniref50
    """

    def __init__(
        self,
        model_name: str = "Rostlab/prot_t5_xl_uniref50",
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
            model_name: HuggingFace model identifier
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
            self.model = None  # No model needed
            self.tokenizer = None
        else:
            # Load pre-trained ProtT5 model and tokenizer
            self.model = AutoModel.from_pretrained(model_name)
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)

            # ProtT5-XL has a hidden size of 1024
            self.hidden_size = self.model.config.d_model

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
        Encode antibody sequences using ProtT5.

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
        Encode a batch of sequences using ProtT5.

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

        # Otherwise use ProtT5 model
        # ProtT5 requires sequences with spaces between amino acids
        spaced_sequences = [" ".join(list(seq)) for seq in sequences]

        # Tokenize sequences
        inputs = self.tokenizer(
            spaced_sequences,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=1024,
        )

        # Move to same device as model
        device = next(self.model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Get embeddings from ProtT5
        with torch.set_grad_enabled(not self._frozen):
            outputs = self.model(**inputs)
            # For T5 encoder-decoder models, we use the encoder's last hidden state
            sequence_embeddings = outputs.last_hidden_state

        # Pool embeddings
        pooled = self._pool_embeddings(
            sequence_embeddings,
            attention_mask=inputs.get("attention_mask"),
        )

        return pooled
