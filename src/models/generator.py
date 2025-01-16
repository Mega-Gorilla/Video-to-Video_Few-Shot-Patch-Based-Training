# models/generator.py
import torch
import torch.nn as nn
from typing import List, Optional, Union
from dataclasses import dataclass
from torch import Tensor

@dataclass
class UpsamplingLayer(nn.Module):
    """Upsampling layer with optional convolution"""
    def __init__(self, channels: int):
        super().__init__()
        self.layer = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

    def forward(self, x: Tensor) -> Tensor:
        return self.layer(x)

class ResNetBlock(nn.Module):
    """ResNet block with optional normalization"""
    def __init__(
        self,
        channels: int,
        norm_layer: Optional[str] = 'instance_norm',
        use_bias: bool = False
    ):
        super().__init__()
        
        # Normalization layer
        norm = None
        if norm_layer == 'batch_norm':
            norm = nn.BatchNorm2d
        elif norm_layer == 'instance_norm':
            norm = nn.InstanceNorm2d
            
        # First convolution block
        layers = [
            nn.ReLU(True),
            nn.Conv2d(channels, channels, 3, 1, 1, bias=use_bias),
        ]
        if norm:
            layers.append(norm(channels))
            
        # Second convolution block
        layers.extend([
            nn.ReLU(True),
            nn.Conv2d(channels, channels, 3, 1, 1, bias=use_bias)
        ])
        if norm:
            layers.append(norm(channels))
            
        self.block = nn.Sequential(*layers)
        
    def forward(self, x: Tensor) -> Tensor:
        return x + self.block(x)

class GeneratorJ(nn.Module):
    """Modern implementation of GeneratorJ"""
    def __init__(
        self,
        input_channels: int = 3,
        filters: List[int] = [32, 64, 128, 128, 128, 64],
        norm_layer: str = 'instance_norm',
        use_bias: bool = False,
        resnet_blocks: int = 7,
        tanh: bool = True,
        append_smoothers: bool = True,
        input_size: int = 256,
    ):
        super().__init__()
        
        self.input_size = input_size
        self.append_smoothers = append_smoothers
        
        # Normalization layer type
        norm = None
        if norm_layer == 'batch_norm':
            norm = nn.BatchNorm2d
        elif norm_layer == 'instance_norm':
            norm = nn.InstanceNorm2d
            
        # Initial convolution
        self.initial_conv = self._make_conv_block(
            input_channels, filters[0], 7, 1, 3,
            use_bias, norm, nn.LeakyReLU(0.2, True)
        )
        
        # Downsampling layers
        self.downsample1 = self._make_conv_block(
            filters[0], filters[1], 3, 2, 1,
            use_bias, norm, nn.LeakyReLU(0.2, True)
        )
        self.downsample2 = self._make_conv_block(
            filters[1], filters[2], 3, 2, 1,
            use_bias, norm, nn.LeakyReLU(0.2, True)
        )
        
        # ResNet blocks
        self.resnet_blocks = nn.ModuleList([
            ResNetBlock(filters[2], norm_layer, use_bias)
            for _ in range(resnet_blocks)
        ])
        
        # Upsampling layers
        self.upsample2 = self._make_upconv_block(
            filters[2] + filters[2], filters[4], 4, 2, 1,
            use_bias, norm, nn.ReLU(True)
        )
        self.upsample1 = self._make_upconv_block(
            filters[4] + filters[1], filters[4], 4, 2, 1,
            use_bias, norm, nn.ReLU(True)
        )
        
        # Final convolution layers
        conv11_channels = filters[0] + filters[4] + input_channels
        self.conv11 = nn.Sequential(
            nn.Conv2d(conv11_channels, filters[5], 7, 1, 3, bias=use_bias),
            nn.ReLU(True)
        )
        
        if self.append_smoothers:
            self.smoothers = nn.Sequential(
                nn.Conv2d(filters[5], filters[5], 3, padding=1, bias=use_bias),
                nn.ReLU(True),
                nn.BatchNorm2d(filters[5]),
                nn.Conv2d(filters[5], filters[5], 3, padding=1, bias=use_bias),
                nn.ReLU(True)
            )
            
        # Output layer
        output_layers = [nn.Conv2d(filters[5], 3, 1, bias=True)]
        if tanh:
            output_layers.append(nn.Tanh())
        self.output = nn.Sequential(*output_layers)
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, m: nn.Module):
        """Initialize network weights"""
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.normal_(m.weight.data, 0.0, 0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias.data, 0.0)
                
    def _make_conv_block(
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
        
    def _make_upconv_block(
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
        """Helper function to create upsampling blocks"""
        layers = [
            UpsamplingLayer(in_channels),
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=bias)
        ]
        
        if norm:
            layers.append(norm(out_channels))
        if activation:
            layers.append(activation)
            
        return nn.Sequential(*layers)
        
    def forward(self, x: Tensor) -> Tensor:
        # Initial convolution
        conv0 = self.initial_conv(x)
        
        # Downsampling
        conv1 = self.downsample1(conv0)
        conv2 = self.downsample2(conv1)
        
        # ResNet blocks with residual connection
        out = conv2
        for block in self.resnet_blocks:
            out = block(out)
            
        # Upsampling with skip connections
        out = self.upsample2(torch.cat([out, conv2], dim=1))
        out = self.upsample1(torch.cat([out, conv1], dim=1))
        out = self.conv11(torch.cat([out, conv0, x], dim=1))
        
        # Optional smoothing layers
        if self.append_smoothers:
            out = self.smoothers(out)
            
        # Output layer
        out = self.output(out)
        
        return out