import os
import torch
from PIL import Image
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

class StyleTransferInference:
    """PyTorch Lightningを活用したスタイル変換の推論クラス"""
    
    def __init__(self, cfg: DictConfig):
        """初期化"""
        self.cfg = cfg
        self._setup_logging()
        self._setup_transforms()
        self._setup_model()

    def _setup_logging(self):
        """ロギングの設定"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)

    def _setup_transforms(self):
        """変換処理の設定"""
        # 入力画像の変換
        self.transform = transforms.Compose([
            RGBConvert(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        
        # マスク画像の変換
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

    def _process_mask(self, mask_tensor: torch.Tensor) -> torch.Tensor:
        """マスクの処理
        学習時と同様のエロージョン処理を適用
        """
        device = mask_tensor.device
        erosion_weights = torch.ones((1, 1, 7, 7)).to(device)
        
        mask_tensor = mask_tensor.unsqueeze(0)  # バッチ次元を追加
        mask_conv = torch.nn.functional.conv2d(
            mask_tensor,
            erosion_weights,
            padding=3
        )
        
        mask_conv[mask_conv < erosion_weights.numel()] = 0
        mask_conv = mask_conv / erosion_weights.numel()
        
        return mask_conv.squeeze(0)  # バッチ次元を削除

    def process_image(self, input_path: str, mask_path: str, save_path: str):
        """1枚の画像を処理"""
        try:
            # 入力画像の読み込み
            image = Image.open(input_path).convert('RGB')
            input_tensor = self.transform(image).unsqueeze(0)
            
            # マスクの読み込み（必要な場合）
            mask_tensor = None
            if not self.cfg.paths.mask_dir.endswith("ignore"):
                mask = Image.open(mask_path).convert('L')
                mask = mask.resize(image.size, Image.NEAREST)
                mask = mask.point(lambda p: p > 128 and 255)  # 二値化
                mask_tensor = self.mask_transform(mask)
                mask_tensor = self._process_mask(mask_tensor)
            
            # 推論の実行
            with torch.no_grad():
                input_tensor = input_tensor.to(self.device)
                # GPUの場合はFP16に変換
                if torch.cuda.is_available():
                    input_tensor = input_tensor.half()
                
                # 推論実行
                output_tensor = self.model.generator(input_tensor)
                
                # マスク適用（必要な場合）
                if mask_tensor is not None:
                    mask_tensor = mask_tensor.to(self.device)
                    if torch.cuda.is_available():
                        mask_tensor = mask_tensor.half()
                    mask_tensor = mask_tensor.repeat(1, 3, 1, 1)  # チャネル次元を複製
                    output_tensor = input_tensor * (1 - mask_tensor) + output_tensor * mask_tensor
            
            # 結果の保存
            output_tensor = output_tensor.float()  # 保存前にfloatに戻す
            image_space = ((output_tensor.clamp(-1, 1) + 1) * 127.5).permute(0, 2, 3, 1)
            image_space = image_space.cpu().numpy()[0].astype(np.uint8)
            
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            Image.fromarray(image_space).save(save_path)
            
        except Exception as e:
            self.logger.error(f"Error processing {input_path}: {str(e)}")
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
                torch.cuda.empty_cache()  # GPUメモリのクリーンアップ
            except Exception as e:
                self.logger.error(f"Failed to process {input_path.name}: {str(e)}")
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
    # 必要なモジュールのインポート
    import hydra
    from omegaconf import OmegaConf
    
    # Hydraを使用して実行
    main()