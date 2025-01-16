# data/dataset.py
import os
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
from typing import Dict, Optional, List, Tuple
import numpy as np
from .transforms import RGBConvert, GrayscaleConvert

class StyleTransferDataset(Dataset):
    def __init__(
        self,
        dir_pre: str,
        dir_post: str,
        dir_mask: str,
        patch_size: int,
        device: str,
        additional_channels: Optional[Dict[str, str]] = None
    ):
        super().__init__()
        self.dir_pre = dir_pre
        self.dir_post = dir_post
        self.dir_mask = dir_mask
        self.patch_size = patch_size
        self.device = device
        self.additional_channels = additional_channels or {}
        
        # Setup transforms
        self.transform = transforms.Compose([
            RGBConvert(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
            
        self.mask_transform = transforms.Compose([
            GrayscaleConvert(),
            transforms.ToTensor()
        ])
            
        # Load image paths
        self.image_paths = sorted([f for f in os.listdir(dir_pre) if f.endswith(('.png', '.jpg', '.jpeg'))])
        
        # Pre-load images and masks
        self.images_pre = []
        self.images_post = []
        self.valid_indices: List[torch.Tensor] = []
        self.valid_indices_left: List[List[int]] = []
        
        self._load_images()
        
    def _load_images(self):
        """Load all images and prepare valid patch indices"""
        for img_path in self.image_paths:
            # Load pre image
            pre_img = Image.open(os.path.join(self.dir_pre, img_path))
            pre_tensor = self.transform(pre_img)
            
            # Load additional channels if specified
            for channel_name, channel_dir in self.additional_channels.items():
                channel_img = Image.open(os.path.join(channel_dir, img_path))
                channel_tensor = self.transform(channel_img)
                pre_tensor = torch.cat([pre_tensor, channel_tensor], dim=0)
                
            self.images_pre.append(pre_tensor)
            
            # Load post image
            post_img = Image.open(os.path.join(self.dir_post, img_path))
            self.images_post.append(self.transform(post_img))
            
            # Load and process mask
            mask = Image.open(os.path.join(self.dir_mask, img_path))
            mask = mask.point(lambda p: p > 128 and 255)
            mask_tensor = self.mask_transform(mask).to(self.device)
            
            # Calculate valid indices for patches
            mask_tensor[mask_tensor < 0.4] = 0
            
            # Apply erosion
            erosion_weights = torch.ones((1, 1, 7, 7)).to(self.device)
            mask_conv = F.conv2d(
                mask_tensor.unsqueeze(0),
                erosion_weights,
                stride=1,
                padding=3
            )
            mask_conv[mask_conv < erosion_weights.numel()] = 0
            mask_conv /= erosion_weights.numel()
            
            # Get valid indices
            indices = mask_conv.squeeze().nonzero()
            self.valid_indices.append(indices)
            self.valid_indices_left.append(list(range(len(indices))))
            
    def _cut_patch(self, tensor: torch.Tensor, midpoint: torch.Tensor) -> torch.Tensor:
        """Extract a patch from tensor centered at given coordinates"""
        size = self.patch_size
        y, x = midpoint[0], midpoint[1]
        
        half_size = size // 2
        # Calculate boundaries
        hn = max(0, y - half_size)
        hx = min(y + half_size, tensor.size(1) - 1)
        xn = max(0, x - half_size)
        xx = min(x + half_size, tensor.size(2) - 1)
        
        # Extract patch
        patch = tensor[:, hn:hx, xn:xx]
        
        # If patch size is not correct, create zero tensor and copy patch
        if patch.size(1) != size or patch.size(2) != size:
            result = torch.zeros((tensor.size(0), size, size), device=patch.device)
            result[:, :patch.size(1), :patch.size(2)] = patch
            patch = result
            
        return patch

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        img_idx = idx % len(self.images_pre)
        
        # Get random valid patch center
        if not self.valid_indices_left[img_idx]:
            self.valid_indices_left[img_idx] = list(range(len(self.valid_indices[img_idx])))
        
        center_idx = np.random.randint(0, len(self.valid_indices_left[img_idx]))
        midpoint = self.valid_indices[img_idx][self.valid_indices_left[img_idx][center_idx]]
        del self.valid_indices_left[img_idx][center_idx]
        
        # Get random midpoint for adversarial training
        midpoint_r = self.valid_indices[img_idx][np.random.randint(0, len(self.valid_indices[img_idx]))]
        
        # Get patches
        pre_patch = self._cut_patch(self.images_pre[img_idx], midpoint)
        post_patch = self._cut_patch(self.images_post[img_idx], midpoint)
        random_patch = self._cut_patch(self.images_post[img_idx], midpoint_r)
        
        return {
            'pre': pre_patch,
            'post': post_patch,
            'already': random_patch
        }

    def __len__(self) -> int:
        return sum(len(indices) for indices in self.valid_indices) * 5  # Multiplier for more iterations