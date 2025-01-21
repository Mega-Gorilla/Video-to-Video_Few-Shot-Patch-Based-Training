import argparse
from pathlib import Path
from typing import List,Union
import shutil

from imageprocessor import ImageProcessor
from optical_flow import OpticalFlowCalculator
from gaussian_filter import GaussianFilter

def process_pipeline(
    input_dir: str,
    output_base_dir: str,
    resize_mode: str = "width",
    resize_value: Union[int, float] = 512,
    use_gpu: bool = True,
    file_format: str = "%03d"
) -> None:
    """
    画像処理パイプライン実行関数
    
    Parameters:
    -----------
    input_dir : str
        入力画像のディレクトリパス
    output_base_dir : str
        出力ベースディレクトリパス
    resize_mode : str
        リサイズモード ("width" or "scale")
    resize_value : Union[int, float]
        リサイズ値（幅またはスケール）
    use_gpu : bool
        GPUを使用するかどうか
    file_format : str
        ファイル名フォーマット
    """
    # 出力ディレクトリの設定
    output_base_dir = Path(output_base_dir)
    input_dir_resized = output_base_dir / "input"
    mask_dir = output_base_dir / "mask"
    flow_fwd_dir = output_base_dir / "flow_fwd"
    flow_bwd_dir = output_base_dir / "flow_bwd"
    gauss_r10_s10_dir = output_base_dir / "gauss_r10_s10"
    gauss_r10_s15_dir = output_base_dir / "gauss_r10_s15"

    # Step 1: 画像のリサイズ処理
    print("\nStep 1: Resizing images...")
    image_processor = ImageProcessor(
        input_dir=input_dir,
        output_dir=str(output_base_dir),
        size_mode=resize_mode,
        size_value=resize_value
    )
    image_processor.process_directory()

    # Step 2: オプティカルフロー計算
    print("\nStep 2: Calculating optical flow...")
    flow_calculator = OpticalFlowCalculator(
        input_dir=str(input_dir_resized),
        flow_fwd_dir=str(flow_fwd_dir),
        flow_bwd_dir=str(flow_bwd_dir),
        file_format=file_format,
        use_gpu=use_gpu
    )
    flow_calculator.calculate_flows()

    # 入力画像の枚数を取得
    # 複数の画像フォーマットに対応
    input_images = []
    for ext in [".png", ".jpg", ".jpeg"]:
        input_images.extend(list(input_dir_resized.glob(f"*{ext}")))
    input_images = sorted(input_images)
    if not input_images:
        raise ValueError(f"No input images found in directory: {input_dir_resized}\n"
                        f"Please check if the images are present and have the correct file extension.")

    frame_first = 1
    frame_last = len(input_images)
    key_frames = list(range(frame_first, frame_last + 1, 10))  # 10フレームごとにキーフレーム
    if frame_last not in key_frames:
        key_frames.append(frame_last)

    # Step 3: ガウシアンフィルター処理 (r10_s10)
    print("\nStep 3: Applying Gaussian filter (r10_s10)...")
    gaussian_filter_s10 = GaussianFilter(
        mask_dir=str(mask_dir),
        flow_fwd_dir=str(flow_fwd_dir),
        flow_bwd_dir=str(flow_bwd_dir),
        output_dir=str(gauss_r10_s10_dir),
        frame_first=frame_first,
        frame_last=frame_last,
        key_frames=key_frames,
        radius=10.0,
        sigma=10.0,
        file_format=file_format
    )
    gaussian_filter_s10.process()

    # Step 4: ガウシアンフィルター処理 (r10_s15)
    print("\nStep 4: Applying Gaussian filter (r10_s15)...")
    gaussian_filter_s15 = GaussianFilter(
        mask_dir=str(mask_dir),
        flow_fwd_dir=str(flow_fwd_dir),
        flow_bwd_dir=str(flow_bwd_dir),
        output_dir=str(gauss_r10_s15_dir),
        frame_first=frame_first,
        frame_last=frame_last,
        key_frames=key_frames,
        radius=10.0,
        sigma=15.0,
        file_format=file_format
    )
    gaussian_filter_s15.process()

    print("\nProcessing complete!")
    print(f"Results saved in: {output_base_dir}")

def main():
    parser = argparse.ArgumentParser(description="Image Processing Pipeline")
    parser.add_argument("input_dir", help="Input directory containing images")
    parser.add_argument("output_dir", help="Output base directory")
    parser.add_argument("--resize-mode", choices=["width", "scale"], default="width",
                      help="Resize mode: 'width' for target width or 'scale' for scaling factor")
    parser.add_argument("--resize-value", type=float, default=512,
                      help="Target width (pixels) or scale factor (0.0-1.0) depending on resize-mode")
    parser.add_argument("--no-gpu", action="store_true",
                      help="Disable GPU usage")
    parser.add_argument("--file-format", default="%03d",
                      help="File name format (default: %03d)")

    args = parser.parse_args()

    try:
        process_pipeline(
            input_dir=args.input_dir,
            output_base_dir=args.output_dir,
            resize_mode=args.resize_mode,
            resize_value=args.resize_value,
            use_gpu=not args.no_gpu,
            file_format=args.file_format
        )
        return 0
    
    except Exception as e:
        print(f"Error: {e}")
        return 1

if __name__ == "__main__":
    exit(main())