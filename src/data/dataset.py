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
import platform

class StyleTransferDataset(Dataset):
    def __init__(
        self,
        dir_pre: str,          # 入力画像のディレクトリ
        dir_post: str,         # 変換後の目標画像のディレクトリ
        dir_mask: str,         # マスク画像のディレクトリ
        patch_size: int,       # 抽出するパッチのサイズ
        device: str,           # 使用するデバイス（CPU/GPU）
        additional_channels: Optional[Dict[str, str]] = None  # 追加の入力チャネル
    ):
        super().__init__()
        self.dir_pre = dir_pre
        self.dir_post = dir_post
        self.dir_mask = dir_mask
        self.patch_size = patch_size
        self.additional_channels = additional_channels or {}

        self.is_windows = platform.system() == 'Windows'
        self.device = device if self.is_windows else 'cpu'  # Windowsの場合は指定されたdevice、それ以外はCPU
        
        # 画像変換のパイプラインを設定
        self.transform = transforms.Compose([
            RGBConvert(),      # RGB形式に変換
            transforms.ToTensor(),  # テンソルに変換
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # [-1, 1]の範囲に正規化
        ])
            
        # マスク画像変換のパイプラインを設定
        self.mask_transform = transforms.Compose([
            GrayscaleConvert(),    # グレースケールに変換
            transforms.ToTensor()   # テンソルに変換
        ])
            
        # 画像ファイルのパスを取得
        self.image_paths = sorted([f for f in os.listdir(dir_pre) if f.endswith(('.png', '.jpg', '.jpeg'))])
        
        # データ保持用のリストを初期化
        self.images_pre = []       # 入力画像のリスト
        self.images_post = []      # 目標画像のリスト
        self.valid_indices = []    # 有効なパッチの中心座標
        self.valid_indices_left = []   # まだ使用していない有効インデックス
        
        self._load_images()        # 画像の読み込みを実行
        
    def _load_images(self):
        """全ての画像を読み込み、有効なパッチの位置を計算"""
        for img_path in self.image_paths:
            # 入力画像の読み込みと変換
            pre_img = Image.open(os.path.join(self.dir_pre, img_path))
            pre_tensor = self.transform(pre_img)
            
            # 追加チャネルがある場合は結合
            for channel_name, channel_dir in self.additional_channels.items():
                channel_img = Image.open(os.path.join(channel_dir, img_path))
                channel_tensor = self.transform(channel_img)
                pre_tensor = torch.cat([pre_tensor, channel_tensor], dim=0)
                
            self.images_pre.append(pre_tensor)
            
            # 目標画像の読み込みと変換
            post_img = Image.open(os.path.join(self.dir_post, img_path))
            self.images_post.append(self.transform(post_img))
            
            # マスク画像の読み込みと前処理
            mask = Image.open(os.path.join(self.dir_mask, img_path))
            mask = mask.point(lambda p: p > 128 and 255)  # 二値化処理
            mask_tensor = self.mask_transform(mask).to(self.device)
            
            # マスクの閾値処理
            mask_tensor[mask_tensor < 0.4] = 0
            
            # エロージョン処理でマスクの境界を縮小
            erosion_weights = torch.ones((1, 1, 7, 7)).to(self.device)
            mask_conv = F.conv2d(
                mask_tensor.unsqueeze(0),
                erosion_weights,
                stride=1,
                padding=3
            )
            
            # エロージョンの結果を正規化
            mask_conv[mask_conv < erosion_weights.numel()] = 0
            mask_conv /= erosion_weights.numel()
            
            # 有効なパッチの中心座標を取得
            indices = mask_conv.squeeze().nonzero()
            self.valid_indices.append(indices)
            self.valid_indices_left.append(list(range(len(indices))))
            
    def _cut_patch(self, tensor: torch.Tensor, midpoint: torch.Tensor) -> torch.Tensor:
        """指定された中心座標からパッチを切り出す"""
        size = self.patch_size
        y, x = midpoint[0], midpoint[1]
        
        half_size = size // 2
        # パッチの境界を計算（画像の端を考慮）
        hn = max(0, y - half_size)
        hx = min(y + half_size, tensor.size(1) - 1)
        xn = max(0, x - half_size)
        xx = min(x + half_size, tensor.size(2) - 1)
        
        # パッチを切り出し
        patch = tensor[:, hn:hx, xn:xx]
        
        # パッチサイズが正しくない場合、ゼロパディング
        if patch.size(1) != size or patch.size(2) != size:
            result = torch.zeros((tensor.size(0), size, size), device=patch.device)
            result[:, :patch.size(1), :patch.size(2)] = patch
            patch = result
            
        return patch

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """データセットからアイテムを取得"""
        # 画像のインデックスを計算
        img_idx = idx % len(self.images_pre)
        
        # パッチ位置の記録用リスト
        self.last_patch_positions = []

        # 有効なパッチの中心をランダムに選択
        if not self.valid_indices_left[img_idx]:
            # 使用可能なインデックスがなくなった場合、リセット
            self.valid_indices_left[img_idx] = list(range(len(self.valid_indices[img_idx])))
        
        # ランダムに中心点を選択し、使用済みリストから削除
        center_idx = np.random.randint(0, len(self.valid_indices_left[img_idx]))
        midpoint = self.valid_indices[img_idx][self.valid_indices_left[img_idx][center_idx]]

        # パッチ位置を記録
        self.last_patch_positions.append(midpoint.tolist())
        
        # ランダムなmidpointのパッチ位置も記録
        midpoint_r = self.valid_indices[img_idx][np.random.randint(0, len(self.valid_indices[img_idx]))]
        self.last_patch_positions.append(midpoint_r.tolist())
        
        # パッチの切り出し
        pre_patch = self._cut_patch(self.images_pre[img_idx], midpoint)
        post_patch = self._cut_patch(self.images_post[img_idx], midpoint)
        random_patch = self._cut_patch(self.images_post[img_idx], midpoint_r)
        
        return {
            'pre': pre_patch,
            'post': post_patch,
            'already': random_patch
        }

    def __len__(self) -> int:
        """データセットの長さを返す（有効なパッチ数 × 5）"""
        return sum(len(indices) for indices in self.valid_indices) * 5  # より多くの学習反復のため5倍に