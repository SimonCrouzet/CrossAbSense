"""Gradient monitoring callback to detect vanishing/exploding gradients."""

import pytorch_lightning as pl
import torch
import logging

logger = logging.getLogger(__name__)


class GradientMonitor(pl.Callback):
    """Monitor gradient flow to detect vanishing/exploding gradients.

    This callback tracks gradient norms during training to help diagnose:
    - Vanishing gradients (learning stalled, dead neurons)
    - Exploding gradients (training instability)
    - Dead neurons (constant outputs)

    Logs to WandB:
    - grad_norm/global: L2 norm of all gradients
    - grad_norm/projection: Projection layer (encoder → decoder)
    - grad_norm/fusion: Fusion layer (MultiEncoder only)
    - grad_norm/attention/first: First attention layer
    - grad_norm/attention/last: Last attention layer
    - grad_norm/ffn/first: First FFN/MLP layer
    - grad_norm/ffn/last: Last FFN/MLP layer
    - grad_norm/output: Output head

    Warning thresholds:
    - global_norm < 1e-6: Vanishing gradients (likely dead neurons)
    """

    def __init__(self, log_frequency: int = 50):
        """
        Args:
            log_frequency: Log gradients every N training steps
        """
        super().__init__()
        self.log_frequency = log_frequency

    def on_after_backward(self, trainer, pl_module):
        """Log gradient statistics after backward pass."""
        if trainer.global_step % self.log_frequency != 0:
            return

        grad_norms = []
        component_grads = {
            'projection': [],
            'fusion': [],
            'attention': {},  # {layer_idx: [grads]}
            'ffn': {},        # {layer_idx: [grads]}
            'output': [],
        }

        for name, param in pl_module.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm(2).item()
                grad_norms.append(grad_norm)

                # Categorize by component
                if 'projection' in name:
                    component_grads['projection'].append(grad_norm)
                elif 'fusion' in name:
                    component_grads['fusion'].append(grad_norm)
                elif 'decoder' in name:
                    if 'output' in name:
                        component_grads['output'].append(grad_norm)
                    else:
                        # Parse attention vs FFN layers
                        layer_type, layer_idx = self._parse_decoder_component(name)
                        if layer_type == 'attention' and layer_idx is not None:
                            if layer_idx not in component_grads['attention']:
                                component_grads['attention'][layer_idx] = []
                            component_grads['attention'][layer_idx].append(grad_norm)
                        elif layer_type == 'ffn' and layer_idx is not None:
                            if layer_idx not in component_grads['ffn']:
                                component_grads['ffn'][layer_idx] = []
                            component_grads['ffn'][layer_idx].append(grad_norm)

        if grad_norms:
            # Compute global norm with numerical stability
            # Clip individual norms to prevent overflow in sum of squares
            grad_tensor = torch.tensor(grad_norms).clamp(max=1e6)
            global_norm = grad_tensor.norm(2).item()

            # Safety check for NaN/Inf
            if not (torch.isfinite(torch.tensor(global_norm))):
                logger.warning(f"⚠️  Non-finite global gradient norm detected: {global_norm}")
                global_norm = 0.0

            # Log global gradient norm
            pl_module.log("grad_norm/global", global_norm, on_step=True, on_epoch=False)

            # Log projection layer
            if component_grads['projection']:
                proj_avg = sum(component_grads['projection']) / len(component_grads['projection'])
                if torch.isfinite(torch.tensor(proj_avg)):
                    pl_module.log("grad_norm/projection", proj_avg, on_step=True, on_epoch=False)

            # Log fusion layer (only if MultiEncoder is used)
            if component_grads['fusion']:
                fusion_avg = sum(component_grads['fusion']) / len(component_grads['fusion'])
                if torch.isfinite(torch.tensor(fusion_avg)):
                    pl_module.log("grad_norm/fusion", fusion_avg, on_step=True, on_epoch=False)

            # Log attention layers (first and last)
            if component_grads['attention']:
                attn_indices = sorted(component_grads['attention'].keys())
                if attn_indices:
                    # First attention layer
                    first_attn = component_grads['attention'][attn_indices[0]]
                    first_attn_avg = sum(first_attn) / len(first_attn)
                    pl_module.log("grad_norm/attention/first", first_attn_avg, on_step=True, on_epoch=False)

                    # Last attention layer (if different from first)
                    if len(attn_indices) > 1:
                        last_attn = component_grads['attention'][attn_indices[-1]]
                        last_attn_avg = sum(last_attn) / len(last_attn)
                        pl_module.log("grad_norm/attention/last", last_attn_avg, on_step=True, on_epoch=False)

            # Log FFN layers (first and last)
            if component_grads['ffn']:
                ffn_indices = sorted(component_grads['ffn'].keys())
                if ffn_indices:
                    # First FFN layer
                    first_ffn = component_grads['ffn'][ffn_indices[0]]
                    first_ffn_avg = sum(first_ffn) / len(first_ffn)
                    pl_module.log("grad_norm/ffn/first", first_ffn_avg, on_step=True, on_epoch=False)

                    # Last FFN layer (if different from first)
                    if len(ffn_indices) > 1:
                        last_ffn = component_grads['ffn'][ffn_indices[-1]]
                        last_ffn_avg = sum(last_ffn) / len(last_ffn)
                        pl_module.log("grad_norm/ffn/last", last_ffn_avg, on_step=True, on_epoch=False)

            # Log output head
            if component_grads['output']:
                output_avg = sum(component_grads['output']) / len(component_grads['output'])
                pl_module.log("grad_norm/output", output_avg, on_step=True, on_epoch=False)

            # Warning for problematic gradients
            if global_norm < 1e-6:
                logger.warning(f"⚠️  Vanishing gradients detected! Global norm: {global_norm:.2e}")
            elif global_norm > 100:
                logger.debug(f"Large gradients detected: {global_norm:.2e} (clipped at {trainer.gradient_clip_val})")

    def _parse_decoder_component(self, param_name: str):
        """Parse decoder parameter name to extract component type and layer index.

        Examples:
            decoder.vh_cross_attn.0.in_proj_weight → ('attention', 0)
            decoder.self_attn.2.in_proj_weight → ('attention', 2)
            decoder.vh_ffn.0.0.weight → ('ffn', 0)
            decoder.ffn.1.weight → ('ffn', 1)
            decoder.output.0.weight → (None, None) - handled separately

        Returns:
            tuple: (component_type, layer_index)
                   component_type: 'attention' or 'ffn' or None
                   layer_index: int or None
        """
        parts = param_name.split(".")

        # Identify component type
        layer_type = None
        for part in parts:
            if 'attn' in part.lower():
                layer_type = 'attention'
                break
            elif 'ffn' in part.lower() or 'mlp' in part.lower():
                layer_type = 'ffn'
                break

        # Extract layer index (first numeric part after component type)
        layer_idx = None
        if layer_type:
            for part in parts:
                if part.isdigit():
                    layer_idx = int(part)
                    break

        return layer_type, layer_idx

