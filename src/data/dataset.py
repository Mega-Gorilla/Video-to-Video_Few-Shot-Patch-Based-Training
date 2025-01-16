# data/dataset.py
import os
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
from typing import Dict, Optional, List, Tuple
import numpy as np

class StyleTransferDataset(Dataset):
    def __init__(
        self,
        dir_pre: str,
        dir_post: str,
        dir_mask: str,
        patch_size: int,
        device: str,
        additional_channels: Optional[Dict[str, str]] = None,
        transform = None,
        mask_transform = None
    ):
        super().__init__()
        self.dir_pre = dir_pre
        self.dir_post = dir_post
        self.dir_mask = dir_mask
        self.patch_size = patch_size
        self.device = device
        self.additional_channels = additional_channels or {}
        
        # Setup transforms
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Lambda(lambda x: x if x.mode == 'RGB' else x.convert('RGB')),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])
        else:
            self.transform = transform
            
        if mask_transform is None:
            self.mask_transform = transforms.Compose([
                transforms.Lambda(lambda x: x if x.mode == 'L' else x.convert('L')),
                transforms.ToTensor()
            ])
        else:
            self.mask_transform = mask_transform
            
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
            mask_tensor = self.mask_transform(mask).to(self.device)
            
            # Calculate valid indices for patches
            mask_tensor[mask_tensor < 0.4] = 0
            valid_indices = self._get_valid_patch_indices(mask_tensor)
            self.valid_indices.append(valid_indices)
            self.valid_indices_left.append(list(range(len(valid_indices))))
    
    def _get_valid_patch_indices(self, mask: torch.Tensor) -> torch.Tensor:
        """Get indices of valid patches from mask"""
        # Use convolution to find valid patch centers
        kernel_size = 7
        padding = kernel_size // 2
        erosion_weights = torch.ones((1, 1, kernel_size, kernel_size)).to(self.device)
        
        mask = F.conv2d(
            mask.unsqueeze(0),
            erosion_weights,
            padding=padding
        )
        
        mask[mask < erosion_weights.numel()] = 0
        mask /= erosion_weights.numel()
        
        return mask.squeeze().nonzero()
    
    def _get_patch(self, tensor: torch.Tensor, center: torch.Tensor) -> torch.Tensor:
        """Extract a patch from tensor centered at given coordinates"""
        half_size = self.patch_size // 2
        y, x = center
        
        # Calculate patch boundaries
        top = max(0, y - half_size)
        bottom = min(tensor.shape[1], y + half_size)
        left = max(0, x - half_size)
        right = min(tensor.shape[2], x + half_size)
        
        patch = tensor[:, top:bottom, left:right]
        
        # Pad if necessary
        if patch.shape[1:] != (self.patch_size, self.patch_size):
            pad_top = max(0, half_size - y)
            pad_bottom = max(0, y + half_size - tensor.shape[1])
            pad_left = max(0, half_size - x)
            pad_right = max(0, x + half_size - tensor.shape[2])
            
            patch = F.pad(
                patch,
                (pad_left, pad_right, pad_top, pad_bottom),
                mode='reflect'
            )
            
        return patch

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        img_idx = idx % len(self.images_pre)
        
        # Get random valid patch center
        if not self.valid_indices_left[img_idx]:
            self.valid_indices_left[img_idx] = list(range(len(self.valid_indices[img_idx])))
        
        center_idx = np.random.choice(self.valid_indices_left[img_idx])
        self.valid_indices_left[img_idx].remove(center_idx)
        
        center = self.valid_indices[img_idx][center_idx]
        
        # Get patches
        pre_patch = self._get_patch(self.images_pre[img_idx], center)
        post_patch = self._get_patch(self.images_post[img_idx], center)
        
        # Get random patch for adversarial training
        random_center = self.valid_indices[img_idx][
            np.random.randint(len(self.valid_indices[img_idx]))
        ]
        random_patch = self._get_patch(self.images_post[img_idx], random_center)
        
        return {
            'pre': pre_patch,
            'post': post_patch,
            'already': random_patch
        }

    def __len__(self) -> int:
        return sum(len(indices) for indices in self.valid_indices) * 5  # Multiplier for more iterations
