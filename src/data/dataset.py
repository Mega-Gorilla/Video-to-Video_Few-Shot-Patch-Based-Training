# data/dataset.py
import os
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
from typing import Dict, Optional, List, Union
import numpy as np
from .transforms import RGBConvert, GrayscaleConvert
from omegaconf import DictConfig

class StyleTransferDataset(Dataset):
    def __init__(
        self,
        dir_pre: str,          # 入力画像のディレクトリ
        dir_post: str,         # 変換後の目標画像のディレクトリ
        dir_mask: str,         # マスク画像のディレクトリ
        patch_size: int,       # 抽出するパッチのサイズ
        augmentation_factor: int = 1,  # データ拡張倍数
        additional_channels: Optional[Dict[str, str]] = None  # 追加の入力チャネル
    ):
        super().__init__()
        self.dir_pre = dir_pre
        self.dir_post = dir_post
        self.dir_mask = dir_mask
        self.patch_size = patch_size
        self.additional_channels = additional_channels or {}
        self.augmentation_factor = max(1, augmentation_factor)  # 最小値を1に制限

        self.device =  'cpu'
        
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
        self.image_paths = sorted([
            f for f in os.listdir(dir_pre) 
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ])
        
        # データ保持用のリストを初期化
        self.images_pre = []       # 入力画像のリスト
        self.images_post = []      # 目標画像のリスト
        self.valid_indices = []    # 有効なパッチの中心座標
        self.valid_indices_left = []   # まだ使用していない有効インデックス
        
        self._load_images()        # 画像の読み込みを実行
    
    def _show_debug_image(self, img, title="Debug Image", is_tensor=False):
        import matplotlib.pyplot as plt
        """デバッグ用の画像表示関数"""
        plt.figure(figsize=(10, 10))
        plt.title(title)
        
        if is_tensor:
            # テンソルの場合、numpy配列に変換して正規化
            if isinstance(img, torch.Tensor):
                img = img.detach().cpu().numpy()
            if img.shape[0] in [1, 3, 4]:  # Channel first to channel last
                img = np.transpose(img, (1, 2, 0))
            # [-1, 1]の範囲から[0, 1]の範囲に変換
            img = (img - img.min()) / (img.max() - img.min())

        plt.imshow(img)
        plt.axis('off')
        plt.show(block=False)
        print(f"\nDisplaying {title}")
        print(f"Image shape: {img.shape}")
        if isinstance(img, np.ndarray):
            print(f"Value range: [{img.min():.3f}, {img.max():.3f}]")
        input("Press Enter to continue...")
        plt.close()

    def _find_corresponding_image(self, base_dir: Union[str, Dict, DictConfig], image_name: str) -> str:
        """
        指定されたベースディレクトリで、対応する画像ファイルを探す
        Args:
            base_dir: 文字列のパス、または設定辞書
            image_name: 画像ファイル名
        Returns:
            str: 完全なファイルパス
        """
        # DictConfigまたは辞書の場合、pathキーを使用
        if isinstance(base_dir, (dict, DictConfig)):
            base_dir = base_dir.get('path')
        
        # 元のファイル名から拡張子を除去
        base_name = os.path.splitext(image_name)[0]
        
        # サポートする拡張子のリスト
        extensions = ['.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG']
        
        # 各拡張子でファイルの存在をチェック
        for ext in extensions:
            file_path = os.path.join(base_dir, base_name + ext)
            if os.path.exists(file_path):
                return file_path
                
        # 対応するファイルが見つからない場合は元のパスを返す
        return os.path.join(base_dir, image_name)
    
    def _load_images(self):
        """画像読み込み処理"""
        print("\nStarting image loading process...")
        print(f"Number of image paths: {len(self.image_paths)}")
        
        # 追加チャネルのデータを保持するための辞書
        self.additional_channel_data = {
            channel_name: [] for channel_name in self.additional_channels.keys()
        }

        for img_path in self.image_paths:
            print(f"\nProcessing image: {img_path}")
            
            # 入力画像の読み込み
            pre_path = self._find_corresponding_image(self.dir_pre, img_path)
            print(f"Loading input image from: {pre_path}")
            try:
                pre_img = Image.open(pre_path)
                pre_tensor = self.transform(pre_img)
                self.images_pre.append(pre_tensor)
                print(f"Successfully loaded input image, shape: {pre_tensor.shape}")
            except Exception as e:
                print(f"Error loading input image: {e}")
                continue
                
            # 目標画像の読み込み
            post_path = self._find_corresponding_image(self.dir_post, img_path)
            print(f"Loading target image from: {post_path}")
            try:
                post_img = Image.open(post_path)
                post_tensor = self.transform(post_img)
                self.images_post.append(post_tensor)
                print(f"Successfully loaded target image, shape: {post_tensor.shape}")
            except Exception as e:
                print(f"Error loading target image: {e}")
                continue

            # マスク画像の読み込み
            mask_path = self._find_corresponding_image(self.dir_mask, img_path)
            print(f"Loading mask from: {mask_path}")
            try:
                mask = Image.open(mask_path)
                
                # マスク処理
                mask = mask.point(lambda p: p > 128 and 255)
                mask_tensor = self.mask_transform(mask).to(self.device)
                
                # エロージョン処理
                erosion_weights = torch.ones((1, 1, 7, 7)).to(self.device)
                mask_conv = F.conv2d(
                    mask_tensor.unsqueeze(0),
                    erosion_weights,
                    stride=1,
                    padding=3
                )
                
                # 有効なパッチの位置を取得
                indices = mask_conv.squeeze().nonzero(as_tuple=False)
                print(f"Found {len(indices)} valid patch positions")
                
                self.valid_indices.append(indices)
                self.valid_indices_left.append(list(range(len(indices))))
                
            except Exception as e:
                print(f"Error processing mask: {e}")
                # マスク処理に失敗した場合、既に追加した画像を削除
                if len(self.images_pre) > len(self.valid_indices):
                    self.images_pre.pop()
                    self.images_post.pop()
                continue

            # 追加チャネルの読み込み
            try:
                for channel_name, channel_config in self.additional_channels.items():
                    channel_path = self._find_corresponding_image(channel_config, img_path)
                    print(f"Loading {channel_name} channel from: {channel_path}")
                    try:
                        channel_img = Image.open(channel_path)
                        channel_tensor = self.transform(channel_img)
                        self.additional_channel_data[channel_name].append(channel_tensor)
                        print(f"Successfully loaded {channel_name} channel, shape: {channel_tensor.shape}")
                    except Exception as e:
                        print(f"Error loading {channel_name} channel: {e}")
                        raise  # エラーを再送出して処理を中断
            except Exception as e:
                print(f"Error loading additional image: {e}")
                # エラーが発生した場合、この画像セット全体をスキップ
                if len(self.images_pre) > len(self.valid_indices):
                    self.images_pre.pop()
                    self.images_post.pop()
                    # 既に追加された追加チャネルデータも削除
                    for ch_name in self.additional_channel_data:
                        if self.additional_channel_data[ch_name]:
                            self.additional_channel_data[ch_name].pop()
                continue
            
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
        
        # パッチサイズが正しくない場合、パディング
        if patch.size(1) != size or patch.size(2) != size:
            # チャネル数を保持したまま新しいテンソルを作成
            result = torch.zeros((tensor.size(0), size, size), device=tensor.device)
            # 既存のパッチをコピー
            result[:, :patch.size(1), :patch.size(2)] = patch
            patch = result
        
        return patch

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        データセットからアイテムを取得
        Args:
            idx (int): インデックス
        Returns:
            Dict[str, torch.Tensor]: 'pre'(入力), 'post'(出力), channel_{name}(追加チャネル)を含む辞書
        """
        # 画像のインデックスを計算
        img_idx = idx % len(self.images_pre)
        
        # パッチ位置の記録用リストを初期化
        self.last_patch_positions = []

        # 有効なパッチの中心をランダムに選択
        if not self.valid_indices_left[img_idx]:
            # 使用可能なインデックスがなくなった場合、リセット
            self.valid_indices_left[img_idx] = list(range(len(self.valid_indices[img_idx])))
        
        # ランダムに中心点を選択し、使用済みリストから削除
        center_idx = np.random.randint(0, len(self.valid_indices_left[img_idx]))
        midpoint = self.valid_indices[img_idx][self.valid_indices_left[img_idx][center_idx]]
        self.valid_indices_left[img_idx].pop(center_idx)
        
        # パッチ位置を記録
        self.last_patch_positions.append(midpoint.tolist())

        # 基本のパッチを切り出し
        result = {
            'pre': self._cut_patch(self.images_pre[img_idx], midpoint),
            'post': self._cut_patch(self.images_post[img_idx], midpoint),
        }
        
        # 追加チャネルのパッチを切り出し
        for channel_name in self.additional_channels:
            channel_patch = self._cut_patch(
                self.additional_channel_data[channel_name][img_idx], 
                midpoint
            )
            result[f'channel_{channel_name}'] = channel_patch
        
        # データ拡張処理（augmentation_factor > 1 の場合）
        if self.augmentation_factor > 1:
            # 別のランダムな位置からパッチを切り出し
            random_idx = np.random.randint(0, len(self.valid_indices[img_idx]))
            midpoint_r = self.valid_indices[img_idx][random_idx]
            
            # パッチ位置を記録
            self.last_patch_positions.append(midpoint_r.tolist())
            
            # 拡張データを追加
            result['already'] = self._cut_patch(self.images_post[img_idx], midpoint_r)
            
            # 追加チャネルの拡張データも追加（必要な場合）
            for channel_name in self.additional_channels:
                result[f'channel_{channel_name}_aug'] = self._cut_patch(
                    self.additional_channel_data[channel_name][img_idx], 
                    midpoint_r
                )
        
        return result

    def __len__(self) -> int:
        """データセットの長さを返す（有効なパッチ数 × 拡張倍数）"""
        return sum(len(indices) for indices in self.valid_indices) * self.augmentation_factor