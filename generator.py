import os
import torch
from PIL import Image, ImageDraw
import numpy as np
from pathlib import Path
import pytorch_lightning as pl
from omegaconf import DictConfig
import logging
from tqdm import tqdm
import torchvision.transforms as transforms
from src.data.transforms import RGBConvert, GrayscaleConvert
from lightning_model import StyleTransferModel
import hydra
import torch.nn.functional as F
from typing import List, Tuple, Optional

class StyleTransferInference:
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self._setup_logging()
        self._setup_transforms()
        self._setup_model()
        
        # パッチサイズを学習時の設定から取得
        self.patch_size = self.cfg.data.patch_size  # データ設定から取得
        self.debug_mode = cfg.inference.get('debug_mode', False)
        self.patch_positions = []

    def _setup_logging(self):
        """ロギングの設定"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)

    def _setup_transforms(self):
        """変換処理の設定"""
        self.transform = transforms.Compose([
            RGBConvert(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        
        self.mask_transform = transforms.Compose([
            GrayscaleConvert(),
            transforms.ToTensor()
        ])

    def _setup_model(self):
        """モデルの設定とロード"""
        # デバイスの設定
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() and self.cfg.inference.use_gpu else "cpu"
        )
        
        # モデルの初期化
        self.model = StyleTransferModel(
            generator_config=self.cfg.model.generator,
            discriminator_config=self.cfg.model.discriminator,
            training_config=self.cfg.training,
            optimizer_config=self.cfg.optimizer,
            data_config=self.cfg.data,
            perception_loss_config=self.cfg.model.perception_loss
        )
        
        # チェックポイントの読み込み
        checkpoint = torch.load(self.cfg.paths.checkpoint, map_location=self.device)
        self.model.load_state_dict(checkpoint['state_dict'], strict=True)
        self.model.to(self.device)
        self.model.eval()
        
        # FP16の使用（GPUの場合）
        if torch.cuda.is_available():
            self.model.generator = self.model.generator.half()

    def _draw_patches(
        self, 
        image: Image.Image, 
        positions: List[Tuple[int, int, int, int]], 
    ) -> Image.Image:
        """パッチ位置を画像上に描画（ランダムカラー）"""
        draw = ImageDraw.Draw(image)
        
        # 各パッチに対して異なるランダムカラーを生成
        for y_start, y_end, x_start, x_end in positions:
            # ランダムカラーの生成（明るめの色を生成）
            random_color = (
                np.random.randint(100, 256),  # R
                np.random.randint(100, 256),  # G
                np.random.randint(100, 256)   # B
            )
            
            # 矩形を描画
            draw.rectangle(
                [x_start, y_start, x_end, y_end],
                outline=random_color,
                width=2
            )
        
        return image

    def _process_mask(self, mask_tensor: torch.Tensor) -> torch.Tensor:
        """マスクの処理（dataset.pyと完全に同じ処理）"""
        device = mask_tensor.device
        
        # マスクの閾値処理
        mask_tensor[mask_tensor < 0.4] = 0
        
        # エロージョン処理
        erosion_weights = torch.ones((1, 1, 7, 7)).to(device)
        mask_conv = F.conv2d(
            mask_tensor.unsqueeze(0),  # (1, 1, H, W)の形状に
            erosion_weights,
            stride=1,
            padding=3
        )
        
        # エロージョンの結果を正規化
        mask_conv[mask_conv < erosion_weights.numel()] = 0
        mask_conv /= erosion_weights.numel()
        
        # デバッグ情報
        print(f"Mask tensor shape after erosion: {mask_conv.shape}")
        print(f"Unique values in processed mask: {torch.unique(mask_conv)}")
        
        return mask_conv.squeeze(0)

    def _get_valid_patch_positions(
        self, 
        mask_tensor: torch.Tensor,
        overlap_percent: float = 50.0
    ) -> List[Tuple[int, int, int, int]]:
        """マスクに基づいて有効なパッチ位置を取得
        Args:
            mask_tensor: マスクテンソル
            overlap_percent: パッチ間の重なりの割合（0-100%）
        Returns:
            List of tuples (y_start, y_end, x_start, x_end) representing patch boundaries
        """
        # オーバーラップからストライドを計算
        overlap = min(max(overlap_percent, 0.0), 100.0) / 100.0  # 0-1の範囲に正規化
        stride = int(self.patch_size * (1 - overlap))  # ストライドを計算
        stride = max(1, stride)  # 最小値は1に設定
        
        indices = mask_tensor.squeeze().nonzero()
        valid_positions = []
        used_positions = set()
        half_size = self.patch_size // 2
        
        h, w = mask_tensor.shape[-2:]
        
        for idx in range(0, len(indices), stride):
            y, x = indices[idx]
            pos_key = (y.item() // stride, x.item() // stride)
            
            if pos_key not in used_positions:
                y_start = max(0, y.item() - half_size)
                y_end = min(h, y.item() + half_size)
                x_start = max(0, x.item() - half_size)
                x_end = min(w, x.item() + half_size)
                
                valid_positions.append((y_start, y_end, x_start, x_end))
                used_positions.add(pos_key)
        
        # デバッグ情報
        print(f"Original number of nonzero positions: {len(indices)}")
        print(f"Number of selected patch positions: {len(valid_positions)}")
        print(f"Overlap: {overlap_percent}%")
        print(f"Stride: {stride}")
        print(f"Patch size: {self.patch_size}")
        print(f"Sample patch box: {valid_positions[0] if valid_positions else None}")
        
        return valid_positions

    def _cut_patch(self, tensor: torch.Tensor, midpoint: Tuple[int, int]) -> Tuple[torch.Tensor, Tuple[int, int, int, int]]:
        """指定された中心座標からパッチを切り出す（学習時と同じ処理）"""
        y, x = midpoint
        half_size = self.patch_size // 2
        
        h, w = tensor.shape[-2:]
        # パッチの境界を計算
        hn = max(0, y - half_size)
        hx = min(y + half_size, h - 1)
        xn = max(0, x - half_size)
        xx = min(x + half_size, w - 1)
        
        # パッチを切り出し
        patch = tensor[..., hn:hx, xn:xx]
        
        # パッチサイズが正しくない場合、ゼロパディング
        if patch.shape[-2:] != (self.patch_size, self.patch_size):
            result = torch.zeros(
                (*tensor.shape[:-2], self.patch_size, self.patch_size),
                device=patch.device,
                dtype=patch.dtype
            )
            result[..., :patch.shape[-2], :patch.shape[-1]] = patch
            patch = result
        
        return patch, (hn, hx, xn, xx)

    def process_large_image(
        self, 
        input_tensor: torch.Tensor, 
        mask_tensor: Optional[torch.Tensor] = None,
        overlap_percent: float = 30.0
    ) -> torch.Tensor:
        """画像を処理
        Args:
            input_tensor: 入力画像テンソル
            mask_tensor: マスクテンソル
            overlap_percent: パッチ間の重なりの割合（0-100%）
        """
        b, c, h, w = input_tensor.shape
        device = input_tensor.device
        dtype = input_tensor.dtype
        
        output = torch.zeros((b, c, h, w), dtype=dtype, device=device)
        weights = torch.zeros((b, 1, h, w), dtype=dtype, device=device)
        
        if mask_tensor is None:
            mask_tensor = torch.ones((b, 1, h, w), dtype=dtype, device=device)
        else:
            mask_tensor = mask_tensor.repeat(1, c, 1, 1)
        
        # パッチの境界ボックスを取得
        patch_boxes = self._get_valid_patch_positions(
            mask_tensor[:, 0:1, :, :],
            overlap_percent=overlap_percent
        )
        
        for y_start, y_end, x_start, x_end in tqdm(patch_boxes, desc="Processing patches"):
            # パッチを直接切り出し
            patch = input_tensor[..., y_start:y_end, x_start:x_end]
            
            with torch.no_grad():
                processed_patch = self.model.generator(patch)
                
                # ガウシアンウェイト
                patch_h, patch_w = y_end - y_start, x_end - x_start
                weight = torch.exp(-((torch.arange(patch_h, device=device) - patch_h/2)**2 / (patch_h/4)**2))[:, None] * \
                        torch.exp(-((torch.arange(patch_w, device=device) - patch_w/2)**2 / (patch_w/4)**2))[None, :]
                weight = weight.to(dtype)[None, None, :, :]
                
                # 重みを全チャネルに拡張
                weight = weight.repeat(1, c, 1, 1)
                
                # 出力テンソルに結果を追加
                output[..., y_start:y_end, x_start:x_end] += processed_patch * weight
                weights[..., y_start:y_end, x_start:x_end] += weight[:, 0:1]
                
                # パッチ位置を記録（デバッグ用）
                self.patch_positions.append((y_start, y_end, x_start, x_end))
        
        # 重みで正規化
        valid_mask = weights > 1e-8
        valid_mask = valid_mask.repeat(1, c, 1, 1)
        output[valid_mask] /= weights.repeat(1, c, 1, 1)[valid_mask]
        
        # マスクの適用
        output = input_tensor * (1 - mask_tensor) + output * mask_tensor
        
        return output

    def process_image(self, input_path: str, mask_path: str, save_path: str):
        """1枚の画像を処理"""
        try:
            # 入力画像の読み込み
            image = Image.open(input_path).convert('RGB')
            input_tensor = self.transform(image).unsqueeze(0)
            
            # マスクの読み込み
            mask_tensor = None
            if not self.cfg.paths.mask_dir.endswith("ignore"):
                # マスクの読み込み
                mask = Image.open(mask_path)
                # 二値化処理を学習時と同じに
                mask = mask.point(lambda p: p > 128 and 255)
                # グレースケール変換とテンソル化
                mask_tensor = self.mask_transform(mask)
                # マスクのエロージョン処理
                mask_tensor = self._process_mask(mask_tensor)
                
                # デバッグ用の情報出力
                print(f"Mask shape: {mask_tensor.shape}")
                print(f"Unique values in mask: {torch.unique(mask_tensor)}")
                
                # チャンネル次元の調整
                mask_tensor = mask_tensor.unsqueeze(0)  # バッチ次元追加
            
            # デバイスとデータ型の設定
            input_tensor = input_tensor.to(self.device)
            if mask_tensor is not None:
                mask_tensor = mask_tensor.to(self.device)
            
            if torch.cuda.is_available():
                input_tensor = input_tensor.half()
                if mask_tensor is not None:
                    mask_tensor = mask_tensor.half()
            
            # パッチベースで画像を処理
            output_tensor = self.process_large_image(input_tensor, mask_tensor)
            
            # 結果の保存
            output_tensor = output_tensor.float()
            
            # 値の範囲をチェックして修正
            print(f"Output tensor range: {output_tensor.min():.3f} to {output_tensor.max():.3f}")
            
            # 明示的にクリッピングを行う
            output_tensor = output_tensor.clamp(-1, 1)
            image_space = ((output_tensor + 1) * 127.5).clamp(0, 255)
            image_space = image_space.permute(0, 2, 3, 1)
            
            # float32からuint8への変換前にチェック
            print(f"Image space range: {image_space.min():.3f} to {image_space.max():.3f}")
            
            # 安全な型変換
            image_space = image_space.round().cpu().numpy()[0].astype(np.uint8)
            
            # 保存前の値の範囲を確認
            print(f"Final image range: {image_space.min()} to {image_space.max()}")
            
            # 生成画像の保存
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            output_image = Image.fromarray(image_space)
            output_image.save(save_path)
            
            # デバッグモードの場合、パッチ位置を描画した画像も保存
            if self.debug_mode:
                debug_path = str(Path(save_path).with_name(f"debug_{Path(save_path).name}"))
                debug_image = output_image.copy()
                debug_image = self._draw_patches(debug_image, self.patch_positions)
                debug_image.save(debug_path)
            
            # メモリの解放
            del input_tensor, output_tensor
            torch.cuda.empty_cache()
            
        except Exception as e:
            self.logger.error(f"Error processing {input_path}: {str(e)}")
            torch.cuda.empty_cache()
            raise

    def process_directory(self):
        """ディレクトリ内の全画像を処理"""
        input_dir = Path(self.cfg.paths.input_dir)
        mask_dir = Path(self.cfg.paths.mask_dir)
        output_dir = Path(self.cfg.paths.output_dir)
        
        if not mask_dir.exists() and not mask_dir.name.endswith("ignore"):
            raise FileNotFoundError(f"Mask directory not found: {mask_dir}")
        
        output_dir.mkdir(parents=True, exist_ok=True)
        self.logger.info(f"Processing images from {input_dir} to {output_dir}")
        
        image_files = list(input_dir.glob("*.[pj][np][g]"))
        self.logger.info(f"Found {len(image_files)} images to process")
        
        for input_path in tqdm(image_files, desc="Processing images"):
            mask_path = mask_dir / input_path.name
            output_path = output_dir / input_path.name
            
            try:
                self.process_image(str(input_path), str(mask_path), str(output_path))
            except Exception as e:
                self.logger.error(f"Failed to process {input_path.name}: {str(e)}")
                print(f"Error details: {e}")  # より詳細なエラー情報
                continue

@hydra.main(version_base=None, config_path="config", config_name="inference")
def main(cfg: DictConfig) -> None:
    """メイン関数"""
    try:
        inferencer = StyleTransferInference(cfg)
        inferencer.process_directory()
        print("Inference completed successfully!")
    except Exception as e:
        print(f"Error during inference: {str(e)}")
        raise

if __name__ == "__main__":
    main()