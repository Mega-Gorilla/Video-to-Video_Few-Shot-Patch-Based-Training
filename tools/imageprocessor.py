import os
import hydra
from hydra.core.config_store import ConfigStore
from dataclasses import dataclass, field
from typing import Optional
from omegaconf import MISSING
from PIL import Image
import numpy as np
from gaussian_images import GaussianMixtureGenerator, GaussianBlurConfig, OpacityConfig
from tqdm import tqdm

# プロジェクトルートの設定
os.environ["PROJECT_ROOT"] = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

@dataclass
class SizeConfig:
    mode: str = "width"  # "width" or "scale"
    value: float = 1024

@dataclass
class BackgroundColorConfig:
    r: int = 255
    g: int = 255
    b: int = 255

@dataclass
class GaussianBlurConfig:
    min_sigma: float = 0.3
    max_sigma: float = 1.0

@dataclass
class OpacityConfig:
    min: float = 0.2
    max: float = 0.5

@dataclass
class GaussianConfig:
    enabled: bool = False
    grid_size: int = 20
    blur: GaussianBlurConfig = field(default_factory=GaussianBlurConfig)
    opacity: OpacityConfig = field(default_factory=OpacityConfig)

@dataclass
class DatasetConfig:
    input_dir: str = MISSING
    output_dir: str = MISSING

@dataclass
class ProcessingConfig:
    size: SizeConfig = field(default_factory=SizeConfig)
    background_color: BackgroundColorConfig = field(default_factory=BackgroundColorConfig)

@dataclass
class ImageProcessorConfig:
    dataset: DatasetConfig
    processing: ProcessingConfig = field(default_factory=ProcessingConfig)
    gaussian: GaussianConfig = field(default_factory=GaussianConfig)

cs = ConfigStore.instance()
cs.store(name="image_processor_config", node=ImageProcessorConfig)

class ImageProcessor:
    def __init__(self, config: ImageProcessorConfig):
        self.config = config
        self.bg_color = (
            config.processing.background_color.r,
            config.processing.background_color.g,
            config.processing.background_color.b
        )
        
        # 入力ディレクトリのパス
        self.input_img_dir = os.path.join(config.dataset.input_dir, 'input')
        self.input_mask_dir = os.path.join(config.dataset.input_dir, 'mask')
        self.input_output_dir = os.path.join(config.dataset.input_dir, 'output')
        
        # 出力ディレクトリのパス
        self.output_img_dir = os.path.join(config.dataset.output_dir, 'input')
        self.output_mask_dir = os.path.join(config.dataset.output_dir, 'mask')
        self.output_output_dir = os.path.join(config.dataset.output_dir, 'output')
        self.output_gauss_dir = os.path.join(config.dataset.output_dir, 'gauss')
        
    def process_single_image(self, args):
        """1つの画像を処理するメソッド"""
        try:
            png_file, prev_points = args
            
            # PNG画像を読み込む
            png_path = os.path.join(self.input_img_dir, png_file)
            png_img = Image.open(png_path)
            
            # サイズモードに応じてリサイズパラメータを計算
            if self.config.processing.size.mode == "width":
                target_width = int(self.config.processing.size.value)
                target_height = int(target_width * png_img.height / png_img.width)
            else:  # scale
                target_width = int(png_img.width * self.config.processing.size.value)
                target_height = int(png_img.height * self.config.processing.size.value)

            # アルファチャンネルの処理
            if png_img.mode == 'RGBA':
                # マスク画像の作成
                alpha = png_img.split()[3]
                mask_resized = alpha.resize((target_width, target_height), Image.Resampling.LANCZOS)
                
                # マスク画像の保存
                mask_filename = os.path.splitext(png_file)[0] + '.jpg'
                mask_path = os.path.join(self.output_mask_dir, mask_filename)
                mask_resized.convert('RGB').save(mask_path, 'JPEG', quality=95)
                
                if self.config.gaussian.enabled:
                    # マスク配列の作成（ガウス混合用）
                    mask_array = np.array(mask_resized) > 128
                    
                    # ガウス混合の生成
                    gaussian_generator = GaussianMixtureGenerator(
                        grid_size=self.config.gaussian.grid_size,
                        blur_config=GaussianBlurConfig(
                            min_sigma=self.config.gaussian.blur.min_sigma,
                            max_sigma=self.config.gaussian.blur.max_sigma
                        ),
                        opacity_config=OpacityConfig(
                            min=self.config.gaussian.opacity.min,
                            max=self.config.gaussian.opacity.max
                        )
                    )
                    gaussian_image, current_points = gaussian_generator.generate_gaussian_mixture(
                        mask_array,
                        previous_points=prev_points
                    )
                    
                    # ガウス画像の保存
                    gauss_filename = os.path.splitext(png_file)[0] + '.jpg'
                    gauss_path = os.path.join(self.output_gauss_dir, gauss_filename)
                    Image.fromarray(gaussian_image).save(gauss_path, 'JPEG', quality=95)
                else:
                    current_points = None
                
                # 背景色の設定と合成
                bg = Image.new('RGB', png_img.size, self.bg_color)
                bg.paste(png_img, mask=alpha)
                png_img = bg
            else:
                current_points = None
            
            # 入力画像のリサイズと保存
            png_resized = png_img.resize((target_width, target_height), Image.Resampling.LANCZOS)
            png_output_path = os.path.join(self.output_img_dir, os.path.splitext(png_file)[0] + '.jpg')
            png_resized.convert('RGB').save(png_output_path, 'JPEG', quality=95)
            
            # 対応する出力画像の処理（存在する場合のみ）
            output_src_filename = os.path.splitext(png_file)[0] + '.png'
            output_src_path = os.path.join(self.input_output_dir, output_src_filename)
            
            if os.path.exists(output_src_path):
                output_img = Image.open(output_src_path)
                output_resized = output_img.resize((target_width, target_height), Image.Resampling.LANCZOS)
                
                output_dst_filename = os.path.splitext(png_file)[0] + '.jpg'
                output_dst_path = os.path.join(self.output_output_dir, output_dst_filename)
                output_resized.convert('RGB').save(output_dst_path, 'JPEG', quality=95)
            
            return True, current_points
        except Exception as e:
            print(f"Error processing {png_file}: {str(e)}")
            return False, None

    def process_images(self):
        """画像処理を実行"""
        try:
            # 入力チェック
            if not os.path.exists(self.input_img_dir):
                raise ValueError(f"Input directory does not exist: {self.input_img_dir}")
            
            # 出力ディレクトリの作成
            os.makedirs(self.output_img_dir, exist_ok=True)
            os.makedirs(self.output_mask_dir, exist_ok=True)
            os.makedirs(self.output_output_dir, exist_ok=True)
            if self.config.gaussian.enabled:
                os.makedirs(self.output_gauss_dir, exist_ok=True)

            # PNG画像のリストを取得
            png_files = [f for f in os.listdir(self.input_img_dir) if f.lower().endswith('.png')]
            if not png_files:
                print(f"No PNG files found in {self.input_img_dir}")
                return
            
            # 前フレームのポイントを保持する変数
            prev_points = None
            
            # 各画像を順番に処理（フレーム間の連続性を保つため）
            for png_file in tqdm(sorted(png_files), desc="Processing images"):
                success, current_points = self.process_single_image((png_file, prev_points))
                
                if self.config.gaussian.enabled:
                    prev_points = current_points

            print("Image processing completed successfully.")
            
        except Exception as e:
            print(f"Error during processing: {str(e)}")
            raise

@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: ImageProcessorConfig):
    # 設定値の表示
    print("\nCurrent configuration:")
    print(f"Input directory: {cfg.dataset.input_dir}")
    print(f"Output directory: {cfg.dataset.output_dir}")
    print(f"Size mode: {cfg.processing.size.mode}")
    print(f"Size value: {cfg.processing.size.value}")
    print(f"Gaussian enabled: {cfg.gaussian.enabled}")
    if cfg.gaussian.enabled:
        print(f"Gaussian blur range: {cfg.gaussian.blur.min_sigma} - {cfg.gaussian.blur.max_sigma}")
        print(f"Gaussian opacity range: {cfg.gaussian.opacity.min} - {cfg.gaussian.opacity.max}\n")
    
    processor = ImageProcessor(cfg)
    processor.process_images()

if __name__ == "__main__":
    main()