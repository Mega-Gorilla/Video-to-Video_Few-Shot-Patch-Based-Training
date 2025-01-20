import os
from pathlib import Path
from PIL import Image
import argparse
from typing import Union, Tuple, List
from tqdm import tqdm

class ImageProcessor:
    def __init__(self, 
                 input_dir: Union[str, Path],
                 output_dir: Union[str, Path],
                 size_mode: str = "width",
                 size_value: int = 512):
        """
        Parameters:
        -----------
        input_dir : str or Path
            入力画像のディレクトリパス
        output_dir : str or Path
            出力先のベースディレクトリパス
        size_mode : str
            "width" または "scale"
        size_value : int
            width modeの場合は目標幅、scale modeの場合は倍率
        """
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.size_mode = size_mode
        self.size_value = size_value

        # 入力サブディレクトリのパスを設定
        self.input_images_dir = self.input_dir / "input"
        self.input_output_dir = self.input_dir / "output"
        self.input_tracking_dir = self.input_dir / "tracking"

        # 出力ディレクトリの作成
        self.output_input_dir = self.output_dir / "input"
        self.output_mask_dir = self.output_dir / "mask"
        self.output_output_dir = self.output_dir / "output"
        self.output_tracking_dir = self.output_dir / "tracking"
        
        # ディレクトリが存在しない場合のみ作成
        for dir_path in [self.output_input_dir, self.output_mask_dir, 
                        self.output_output_dir, self.output_tracking_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)

    def calculate_target_size(self, img: Image.Image) -> Tuple[int, int]:
        """リサイズ後のサイズを計算"""
        if self.size_mode == "width":
            target_width = int(self.size_value)
            target_height = int(target_width * img.height / img.width)
        else:  # scale
            target_width = int(img.width * self.size_value)
            target_height = int(img.height * self.size_value)
        return target_width, target_height

    def process_input_image(self, img_path: Path) -> None:
        """入力画像の処理（リサイズ、マスク生成）"""
        try:
            # 画像を読み込む
            img = Image.open(img_path)
            target_width, target_height = self.calculate_target_size(img)

            # 入力画像のリサイズと保存
            img_resized = img.copy()
            if img.mode != 'RGB':
                img_resized = img_resized.convert('RGB')
            img_resized = img_resized.resize((target_width, target_height), 
                                           Image.Resampling.LANCZOS)
            
            output_img_path = self.output_input_dir / img_path.name
            img_resized.save(output_img_path, 'JPEG', quality=95)

            # マスク画像の作成と保存（入力画像がRGBAの場合のみ）
            if img.mode == 'RGBA':
                alpha = img.split()[3]
                mask_resized = alpha.resize((target_width, target_height), 
                                          Image.Resampling.LANCZOS)
                
                mask_filename = img_path.stem + '.jpg'
                mask_path = self.output_mask_dir / mask_filename
                mask_resized.convert('RGB').save(mask_path, 'JPEG', quality=95)

        except Exception as e:
            print(f"Error processing input image {img_path.name}: {e}")

    def process_other_image(self, img_path: Path, output_subdir: Path) -> None:
        """その他の画像の処理（リサイズのみ）"""
        try:
            # 画像を読み込む
            img = Image.open(img_path)
            target_width, target_height = self.calculate_target_size(img)

            # 画像のリサイズと保存
            img_resized = img.copy()
            if img.mode != 'RGB':
                img_resized = img_resized.convert('RGB')
            img_resized = img_resized.resize((target_width, target_height), 
                                           Image.Resampling.LANCZOS)
            
            output_img_path = output_subdir / img_path.name
            img_resized.save(output_img_path, 'JPEG', quality=95)

        except Exception as e:
            print(f"Error processing {img_path.name}: {e}")

    def get_image_files(self, directory: Path) -> List[Path]:
        """指定されたディレクトリから画像ファイルを取得"""
        if not directory.exists():
            return []
            
        extensions = {'.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG'}
        return [f for f in directory.iterdir() 
                if f.is_file() and f.suffix.lower() in {ext.lower() for ext in extensions}]

    def process_directory(self):
        """ディレクトリ内の全画像を処理"""
        print("Processing resize images...")
        
        # 入力ディレクトリとoutputディレクトリから画像ファイルを取得
        input_files = self.get_image_files(self.input_images_dir)
        output_files = self.get_image_files(self.input_output_dir)
        tracking_files = self.get_image_files(self.input_tracking_dir)

        total_files = len(input_files) + len(output_files) + len(tracking_files)
        if total_files == 0:
            print("No image files found in input directories")
            return

        # プログレスバーの設定
        pbar = tqdm(total=total_files, desc="Progress", unit="images")

        # input内の画像を処理（リサイズ、マスク生成）
        for img_path in input_files:
            self.process_input_image(img_path)
            pbar.update(1)
        
        # output内の画像を処理（リサイズのみ）
        for img_path in output_files:
            self.process_other_image(img_path, self.output_output_dir)
            pbar.update(1)

        # tracking内の画像を処理（リサイズのみ）
        for img_path in tracking_files:
            self.process_other_image(img_path, self.output_tracking_dir)
            pbar.update(1)

        pbar.close()
        print("\nProcessing complete!")

def main():
    # python tools\imageprocessor.py ./test_dataset/platinum_chan_gen_sorce/ ./test_dataset\PlatinumChan_x0.5_gen --size-mode scale --size-value 0.5
    parser = argparse.ArgumentParser(description="Image and Mask Generator")
    parser.add_argument("input_dir", help="Input directory containing images")
    parser.add_argument("output_dir", help="Output base directory")
    parser.add_argument("--size-mode", choices=["width", "scale"], default="width",
                      help="Resize mode: 'width' for target width or 'scale' for scaling factor")
    parser.add_argument("--size-value", type=float, default=512,
                      help="Target width or scale factor depending on size-mode")

    args = parser.parse_args()

    try:
        processor = ImageProcessor(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            size_mode=args.size_mode,
            size_value=args.size_value
        )
        processor.process_directory()
        return 0
        
    except Exception as e:
        print(f"Error: {e}")
        return 1

if __name__ == "__main__":
    exit(main())