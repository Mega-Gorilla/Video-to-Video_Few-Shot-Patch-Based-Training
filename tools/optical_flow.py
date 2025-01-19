import cv2
import numpy as np
from pathlib import Path
from typing import Union, List, Tuple, Optional
from tqdm import tqdm

class OpticalFlowCalculator:
    def __init__(self,
                 input_dir: Union[str, Path],
                 flow_fwd_dir: Union[str, Path],
                 flow_bwd_dir: Union[str, Path],
                 file_format: str = '%03d',
                 use_gpu: bool = True):
        """
        Parameters:
        -----------
        input_dir : str or Path 
            入力画像が格納されているディレクトリ
        flow_fwd_dir : str or Path
            前方向フローの出力ディレクトリ
        flow_bwd_dir : str or Path
            後方向フローの出力ディレクトリ
        file_format : str
            ファイル名のフォーマット（デフォルト: '%03d'）
        use_gpu : bool
            GPUを使用するかどうか（デフォルト: True）
        """
        self.input_dir = Path(input_dir)
        self.flow_fwd_dir = Path(flow_fwd_dir)
        self.flow_bwd_dir = Path(flow_bwd_dir)
        self.file_format = file_format
        
        # ディレクトリの存在確認と作成
        self.flow_fwd_dir.mkdir(parents=True, exist_ok=True)
        self.flow_bwd_dir.mkdir(parents=True, exist_ok=True)
        
        # GPU利用可能性の確認と初期化
        self.use_gpu = use_gpu and cv2.cuda.getCudaEnabledDeviceCount() > 0
        if self.use_gpu:
            self.dis = cv2.cuda.createOptFlow_DIS()
            print("GPUモードで実行します")
        else:
            self.dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
            print("CPUモードで実行します")
        
        # フローバッファの初期化
        self.flow_buffer: Optional[np.ndarray] = None
        
        # 画像ファイルのリストを取得
        self.image_files = self._get_image_files()
        
        # 画像キャッシュの初期化
        self.image_cache = {}
        
    def _get_image_files(self) -> List[Path]:
        """入力ディレクトリから画像ファイルのリストを取得"""
        extensions = ('.png', '.jpg', '.jpeg')
        files = [f for f in self.input_dir.iterdir()
                if f.suffix.lower() in extensions]
        if not files:
            raise ValueError(f"画像ファイルが見つかりません: {self.input_dir}")
        return sorted(files)
    
    def _read_image(self, file_path: Path) -> np.ndarray:
        """画像をキャッシュから取得または読み込み"""
        if file_path in self.image_cache:
            return self.image_cache[file_path]
        
        img = cv2.imread(str(file_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"画像の読み込みに失敗: {file_path}")
            
        # キャッシュに追加
        self.image_cache[file_path] = img
        return img
    
    def _calculate_flow_gpu(self, img1: np.ndarray, img2: np.ndarray) -> np.ndarray:
        """GPU使用時のフロー計算"""
        img1_gpu = cv2.cuda_GpuMat(img1)
        img2_gpu = cv2.cuda_GpuMat(img2)
        
        if self.flow_buffer is not None:
            init_flow = cv2.cuda_GpuMat(self.flow_buffer)
            flow_gpu = self.dis.calc(img1_gpu, img2_gpu, init_flow)
        else:
            flow_gpu = self.dis.calc(img1_gpu, img2_gpu, None)
        
        return flow_gpu.download()
    
    def _calculate_flow_cpu(self, img1: np.ndarray, img2: np.ndarray) -> np.ndarray:
        """CPU使用時のフロー計算"""
        if self.flow_buffer is not None:
            return self.dis.calc(img1, img2, self.flow_buffer.copy())
        return self.dis.calc(img1, img2, None)
    
    def _save_flow(self, flow: np.ndarray, file_path: Path):
        """フローデータを保存"""
        np.save(str(file_path), flow)
    
    def calculate_direction(self, is_forward: bool = True):
        """指定した方向のフローを計算（修正版）"""
        if len(self.image_files) < 2:
            raise ValueError("画像が2枚以上必要です")
        
        direction = "前方向" if is_forward else "後方向"
        output_dir = self.flow_fwd_dir if is_forward else self.flow_bwd_dir
        
        # インデックスの設定を修正
        if is_forward:
            indices = range(len(self.image_files) - 1)
            get_output_name = lambda i: f"{self.file_format % (i+1)}.npy"  # 修正
        else:
            indices = range(len(self.image_files) - 1, 0, -1)
            get_output_name = lambda i: f"{self.file_format % (i)}.npy"    # 修正
        
        # フロー計算
        print(f"\n{direction}フローの計算中...")
        self.flow_buffer = None
        
        for i in tqdm(indices, desc=f"{direction}フロー"):
            if is_forward:
                img1 = self._read_image(self.image_files[i])
                img2 = self._read_image(self.image_files[i + 1])
            else:
                img1 = self._read_image(self.image_files[i])
                img2 = self._read_image(self.image_files[i - 1])
            
            # フロー計算
            flow = self._calculate_flow_gpu(img1, img2) if self.use_gpu \
                  else self._calculate_flow_cpu(img1, img2)
            
            # 保存
            output_path = output_dir / get_output_name(i)
            self._save_flow(flow, output_path)
            
            # バッファ更新
            self.flow_buffer = flow
    
    def calculate_flows(self):
        """両方向のフローを計算"""
        try:
            # 前方向フロー計算
            self.calculate_direction(is_forward=True)
            
            # 後方向フロー計算
            self.calculate_direction(is_forward=False)
            
            print("\n処理が完了しました")
            
        except Exception as e:
            print(f"エラーが発生しました: {e}")
            raise
        finally:
            # キャッシュのクリア
            self.image_cache.clear()

def main():
    """メイン関数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='オプティカルフロー計算')
    parser.add_argument('input_dir', help='入力画像ディレクトリ')
    parser.add_argument('flow_fwd_dir', help='前方向フロー出力ディレクトリ')
    parser.add_argument('flow_bwd_dir', help='後方向フロー出力ディレクトリ')
    parser.add_argument('--format', default='%03d', help='ファイル名フォーマット（デフォルト: %03d）')
    parser.add_argument('--no-gpu', action='store_true', help='GPUを使用しない')
    
    args = parser.parse_args()
    
    try:
        calculator = OpticalFlowCalculator(
            args.input_dir,
            args.flow_fwd_dir,
            args.flow_bwd_dir,
            args.format,
            not args.no_gpu
        )
        calculator.calculate_flows()
        return 0
        
    except Exception as e:
        print(f"エラー: {e}")
        return 1

if __name__ == "__main__":
    exit(main())