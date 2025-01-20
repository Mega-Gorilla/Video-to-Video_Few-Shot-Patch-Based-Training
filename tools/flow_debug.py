import numpy as np
import cv2
from pathlib import Path
import argparse
import matplotlib.pyplot as plt
from typing import Tuple, Optional

class FlowDebugVisualizer:
    def __init__(self, 
                 input_dir: str,
                 flow_dir: str,
                 output_dir: str,
                 frame_start: int = 1,
                 frame_end: Optional[int] = None,
                 skip_frames: int = 1,
                 flow_scale: float = 1.0,
                 grid_size: int = 20):
        """
        Parameters
        ----------
        input_dir : str
            入力画像が格納されているディレクトリ
        flow_dir : str
            オプティカルフローデータ(.npy)が格納されているディレクトリ
        output_dir : str
            可視化結果の出力先ディレクトリ
        frame_start : int
            開始フレーム番号
        frame_end : int, optional
            終了フレーム番号（Noneの場合は全フレーム）
        skip_frames : int
            フレームのスキップ間隔
        flow_scale : float
            フローベクトルの表示スケール
        grid_size : int
            グリッドの間隔（ピクセル単位）
        """
        self.input_dir = Path(input_dir)
        self.flow_dir = Path(flow_dir)
        self.output_dir = Path(output_dir)
        self.frame_start = frame_start
        self.frame_end = frame_end
        self.skip_frames = skip_frames
        self.flow_scale = flow_scale
        self.grid_size = grid_size
        
        # 出力ディレクトリの作成
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _load_image(self, frame_num: int) -> Optional[np.ndarray]:
        """画像の読み込み"""
        extensions = ['.png', '.jpg', '.jpeg']
        for ext in extensions:
            path = self.input_dir / f"{frame_num:03d}{ext}"
            if path.exists():
                img = cv2.imread(str(path))
                if img is not None:
                    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return None

    def _load_flow(self, frame_num: int) -> Optional[np.ndarray]:
        """フローデータの読み込み"""
        flow_path = self.flow_dir / f"{frame_num:03d}.npy"
        if flow_path.exists():
            return np.load(str(flow_path))
        return None

    def _create_grid_points(self, shape: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray]:
        """グリッドポイントの生成"""
        h, w = shape[:2]
        y, x = np.mgrid[self.grid_size//2:h:self.grid_size, 
                       self.grid_size//2:w:self.grid_size]
        return x, y

    def visualize_frame(self, frame_num: int) -> bool:
        """1フレームの可視化（修正版: 画像ペアの重ね合わせ表示）"""
        # フローデータと対応する2枚の画像を読み込み
        flow = self._load_flow(frame_num)
        img1 = self._load_image(frame_num)      # 001.npy なら 001.png
        img2 = self._load_image(frame_num + 1)  # 001.npy なら 002.png
        
        if img1 is None or img2 is None or flow is None:
            print(f"Failed to load data for frame {frame_num}")
            return False

        # 画像の重ね合わせ
        # img1を50%透過させた画像を作成
        overlay = cv2.addWeighted(img1, 0.5, img2, 0.5, 0)

        # グリッドポイントの生成
        x, y = self._create_grid_points(img1.shape)
        
        # フローのサンプリング
        u = cv2.resize(flow[..., 0], (x.shape[1], x.shape[0]))
        v = cv2.resize(flow[..., 1], (x.shape[1], x.shape[0]))

        # 描画
        plt.figure(figsize=(12, 8))
        
        # 重ね合わせ画像の表示
        plt.imshow(overlay)
        
        # フローベクトルの描画
        plt.quiver(x, y, u * self.flow_scale, v * self.flow_scale, 
                color='r', scale_units='xy', scale=1, 
                angles='xy', width=0.003)
        
        # タイトルに使用した画像のペアを表示
        plt.title(f'Flow Visualization - Images {frame_num:03d}-{frame_num+1:03d}')
        plt.axis('off')
        
        # 保存
        output_path = self.output_dir / f"flow_{frame_num:03d}.png"
        plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
        plt.close()
        
        return True

    def visualize_all(self):
        """全フレームの可視化"""
        if self.frame_end is None:
            # フローファイルから最終フレームを決定
            flow_files = list(self.flow_dir.glob("*.npy"))
            if not flow_files:
                raise ValueError("No flow files found")
            self.frame_end = max(int(f.stem) for f in flow_files)

        print(f"Processing frames {self.frame_start} to {self.frame_end}")
        for frame in range(self.frame_start, self.frame_end + 1, self.skip_frames):
            print(f"Processing frame {frame}...", end='\r')
            success = self.visualize_frame(frame)
            if not success:
                print(f"\nSkipped frame {frame}")

        print("\nVisualization complete!")

def main():
    parser = argparse.ArgumentParser(description='Optical Flow Debug Visualizer')
    parser.add_argument('input_dir', help='Input image directory')
    parser.add_argument('flow_dir', help='Optical flow directory')
    parser.add_argument('output_dir', help='Output directory')
    parser.add_argument('--start', type=int, default=1, help='Start frame')
    parser.add_argument('--end', type=int, help='End frame')
    parser.add_argument('--skip', type=int, default=1, help='Frame skip interval')
    parser.add_argument('--scale', type=float, default=1.0, help='Flow vector scale')
    parser.add_argument('--grid', type=int, default=20, help='Grid size in pixels')

    args = parser.parse_args()

    visualizer = FlowDebugVisualizer(
        input_dir=args.input_dir,
        flow_dir=args.flow_dir,
        output_dir=args.output_dir,
        frame_start=args.start,
        frame_end=args.end,
        skip_frames=args.skip,
        flow_scale=args.scale,
        grid_size=args.grid
    )

    visualizer.visualize_all()

if __name__ == "__main__":
    main()