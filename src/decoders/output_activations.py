"""Output activation functions for decoders."""

import torch
import torch.nn as nn


class ScaledSigmoid(nn.Module):
    """
    Scaled sigmoid activation that maps model output to target range.
    
    Maps unbounded model output to [min_target, max_target]:
    - act(0) = mean_target (centered)
    - act(-inf) → min_target
    - act(+inf) → max_target
    
    Formula: output = min + (max - min) * sigmoid(input)
    
    This ensures predictions stay within observed data bounds while allowing
    the model to output unbounded values.
    """
    
    def __init__(self, min_val: float, max_val: float, mean_val: float = None):
        """
        Args:
            min_val: Minimum target value
            max_val: Maximum target value
            mean_val: Mean target value (optional, for centering)
        """
        super().__init__()
        self.min_val = min_val
        self.max_val = max_val
        self.range = max_val - min_val
        
        # Compute bias to center sigmoid at mean
        # sigmoid(0) = 0.5, so to get mean at input=0:
        # min + range * sigmoid(0 + bias) = mean
        # sigmoid(bias) = (mean - min) / range
        # bias = logit((mean - min) / range)
        if mean_val is not None:
            normalized_mean = (mean_val - min_val) / self.range
            # Clamp to avoid inf/nan
            normalized_mean = torch.clamp(torch.tensor(normalized_mean), 0.01, 0.99)
            self.bias = torch.logit(normalized_mean).item()
        else:
            self.bias = 0.0
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply scaled sigmoid activation.
        
        Args:
            x: Input tensor (unbounded)
            
        Returns:
            Output tensor bounded to [min_val, max_val]
        """
        return self.min_val + self.range * torch.sigmoid(x + self.bias)
    
    def __repr__(self):
        return (f"ScaledSigmoid(min={self.min_val:.3f}, max={self.max_val:.3f}, "
                f"bias={self.bias:.3f})")


def get_output_activation(activation_name: str, target_stats: dict = None):
    """
    Get output activation function.
    
    Args:
        activation_name: Name of activation (none, softplus, sigmoid, exp, scaled_sigmoid)
        target_stats: Dict with 'min', 'max', 'mean' for scaled_sigmoid
        
    Returns:
        Activation module or None
    """
    if activation_name == "softplus":
        return nn.Softplus()
    elif activation_name == "sigmoid":
        return nn.Sigmoid()
    elif activation_name == "exp":
        return lambda x: torch.exp(x)
    elif activation_name == "scaled_sigmoid":
        if target_stats is None:
            raise ValueError("scaled_sigmoid requires target_stats dict with min, max, mean")
        return ScaledSigmoid(
            min_val=target_stats['min'],
            max_val=target_stats['max'],
            mean_val=target_stats.get('mean')
        )
    elif activation_name == "none" or activation_name is None:
        return None
    else:
        raise ValueError(
            f"Unknown output activation: {activation_name}. "
            f"Choose from: none, softplus, sigmoid, exp, scaled_sigmoid"
        )
