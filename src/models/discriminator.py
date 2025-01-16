# models/discriminator.py
import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional, Tuple

class DiscriminatorN_IN(nn.Module):
    """Modern implementation of PatchGAN discriminator with instance normalization"""
    def __init__(
        self,
        input_channels: int = 3,
        num_filters: int = 64,
        n_layers: int = 3,
        use_noise: bool = False,
        noise_sigma: float = 0.2,
        norm_layer: str = 'instance_norm',
        use_bias: bool = True
    ):
        super().__init__()
        
        self.use_noise = use_noise
        self.noise_sigma = noise_sigma
        
        # Select normalization layer
        if norm_layer == 'batch_norm':
            norm = nn.BatchNorm2d
        elif norm_layer == 'instance_norm':
            norm = nn.InstanceNorm2d
        else:
            norm = None
            
        # Initial convolution block
        self.initial = self._make_block(
            input_channels,
            num_filters,
            4, 2, 1,
            use_bias,
            None,
            nn.LeakyReLU(0.2, True)
        )
        
        # Intermediate layers
        self.intermediate = nn.ModuleList()
        curr_filters = num_filters
        for i in range(1, n_layers):
            next_filters = min(curr_filters * 2, num_filters * 8)
            self.intermediate.append(
                self._make_block(
                    curr_filters,
                    next_filters,
                    4, 2, 1,
                    use_bias,
                    norm,
                    nn.LeakyReLU(0.2, True)
                )
            )
            curr_filters = next_filters
            
        # Pre-output layer
        next_filters = min(curr_filters * 2, num_filters * 8)
        self.pre_output = self._make_block(
            curr_filters,
            next_filters,
            4, 1, 1,
            use_bias,
            norm,
            nn.LeakyReLU(0.2, True)
        )
        
        # Output layer
        self.output = self._make_block(
            next_filters,
            1,
            4, 1, 1,
            use_bias,
            None,
            None
        )
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, m: nn.Module):
        """Initialize network weights"""
        if isinstance(m, nn.Conv2d):
            nn.init.normal_(m.weight.data, 0.0, 0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias.data, 0.0)
                
    def _make_block(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        padding: int,
        bias: bool,
        norm: Optional[nn.Module],
        activation: Optional[nn.Module]
    ) -> nn.Sequential:
        """Helper function to create convolution blocks"""
        layers = [
            nn.Conv2d(
                in_channels, out_channels,
                kernel_size, stride, padding, bias=bias
            )
        ]
        
        if norm:
            layers.append(norm(out_channels))
        if activation:
            layers.append(activation)
            
        return nn.Sequential(*layers)
        
    def forward(self, x: Tensor) -> Tuple[Tensor, None]:
        """Forward pass with optional noise"""
        if self.use_noise and self.training:
            noise = torch.randn_like(x) * self.noise_sigma
            x = x + noise
            
        # Process through layers
        out = self.initial(x)
        for layer in self.intermediate:
            out = layer(out)
        out = self.pre_output(out)
        out = self.output(out)
        
        return out, None  # Second return value is for compatibility with original code
