# models/perception.py
import torch
import torch.nn as nn
from torchvision import models
from typing import List, Optional, Tuple
from torch import Tensor
import torch.nn.functional as F

class PerceptualVGG19(nn.Module):
    """
    VGG19-based perceptual loss network.
    Extracts features from specific layers for perceptual loss computation.
    """
    def __init__(
        self,
        feature_layers: List[int],
        use_normalization: bool = True,
        path: Optional[str] = None,
        num_classes: int = 40,
        requires_grad: bool = False
    ):
        """
        Initialize the VGG19 model for perceptual loss.

        Args:
            feature_layers: List of layer indices to extract features from
            use_normalization: Whether to normalize input images
            path: Optional path to pre-trained weights
            num_classes: Number of output classes if using custom weights
            requires_grad: Whether to compute gradients for VGG weights
        """
        super().__init__()
        
        # Create VGG19 model
        if path is not None:
            # Load custom pretrained model
            model = models.vgg19(pretrained=False)
            model.classifier = nn.Sequential(
                nn.Linear(512 * 8 * 8, 4096),
                nn.ReLU(True),
                nn.Dropout(),
                nn.Linear(4096, 4096),
                nn.ReLU(True),
                nn.Dropout(),
                nn.Linear(4096, num_classes),
            )
            model.load_state_dict(torch.load(path))
        else:
            # Load standard pretrained model
            model = models.vgg19(pretrained=True)
            
        # Convert to float and eval mode
        model = model.float().eval()
        
        # Store configuration
        self.model = model
        self.feature_layers = sorted(feature_layers)
        self.use_normalization = use_normalization
        
        # Register normalization parameters
        self.register_buffer(
            'mean',
            torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        )
        self.register_buffer(
            'std',
            torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        )
        
        # Freeze network if not computing gradients
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False
                
    def normalize(self, x: Tensor) -> Tensor:
        """
        Normalize input images to ImageNet range.
        
        Args:
            x: Input tensor in range [-1, 1]
            
        Returns:
            Normalized tensor
        """
        if not self.use_normalization:
            return x
            
        # Convert from [-1, 1] to [0, 1]
        x = (x + 1) / 2
        
        # Normalize using ImageNet statistics
        return (x - self.mean) / self.std
        
    def get_features(self, x: Tensor) -> Tensor:
        """
        Extract features from specified VGG layers.
        
        Args:
            x: Input tensor
            
        Returns:
            Concatenated features from specified layers
        """
        features = []
        current = x
        
        # Extract features from each specified layer
        for i in range(max(self.feature_layers) + 1):
            current = self.model.features[i](current)
            if i in self.feature_layers:
                features.append(current.view(current.size(0), -1))
                
        # Concatenate all features
        return torch.cat(features, dim=1)
        
    def forward(self, x: Tensor) -> Tuple[None, Tensor]:
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor in range [-1, 1]
            
        Returns:
            Tuple of (None, features) for compatibility with original interface
        """
        # Normalize input
        x = self.normalize(x)
        
        # Extract and return features
        features = self.get_features(x)
        return None, features
    
    def perceptual_loss(self, y_pred: Tensor, y_true: Tensor) -> Tensor:
        """
        Compute perceptual loss between predicted and target images.
        
        Args:
            y_pred: Predicted images
            y_true: Target images
            
        Returns:
            Perceptual loss value
        """
        # Get features for both inputs
        _, pred_features = self(y_pred)
        _, true_features = self(y_true)
        
        # Compute MSE loss between features
        return F.mse_loss(pred_features, true_features)