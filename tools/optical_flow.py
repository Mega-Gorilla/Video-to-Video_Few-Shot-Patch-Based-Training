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
                 mask_dir: Union[str, Path], 
                 file_format: str = '%03d'):
        """
        Parameters:
        -----------
        input_dir : str or Path 
            入力画像が格納されているディレクトリ
        flow_fwd_dir : str or Path
            前方向フローの出力ディレクトリ
        flow_bwd_dir : str or Path
            後方向フローの出力ディレクトリ
        mask_dir : str or Path
            マスク画像が格納されているディレクトリ
        file_format : str
            ファイル名のフォーマット（デフォルト: '%03d'）
        """
        self.input_dir = Path(input_dir)
        self.flow_fwd_dir = Path(flow_fwd_dir)
        self.flow_bwd_dir = Path(flow_bwd_dir)
        self.mask_dir = Path(mask_dir)
        self.file_format = file_format
        
        # ディレクトリの存在確認と作成
        self.flow_fwd_dir.mkdir(parents=True, exist_ok=True)
        self.flow_bwd_dir.mkdir(parents=True, exist_ok=True)
        
        # GPU利用可能性の確認と初期化
        self.dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
        
        # フローバッファの初期化
        self.flow_buffer: Optional[np.ndarray] = None
        
        # 画像ファイルのリストを取得
        self.image_files = self._get_image_files()
        
        # 画像キャッシュの初期化
        self.image_cache = {}

        # マスクのキャッシュ
        self.masks = {}  
        
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

    def _read_mask(self, index: int) -> np.ndarray:
        """マスク画像の読み込み"""
        if index in self.masks:
            return self.masks[index]
        
        mask_path = self.mask_dir / f"{self.file_format % index}.jpg"
        
        if not mask_path.exists():
            raise ValueError(f"マスクファイルが見つかりません: {mask_path}")
            
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        # 2値化（閾値は必要に応じて調整）
        _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
        # マスクを論理値に変換（Trueが追跡する領域）
        mask = mask == 255
        
        self.masks[index] = mask
        return mask

    def _calculate_flow_cpu(self, img1: np.ndarray, img2: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """マスク付きCPUフロー計算"""
        flow = self.dis.calc(img1, img2, None)
        # マスクされた領域のフローを0に設定
        flow[~mask] = 0
        return flow
    
    def _save_flow(self, flow: np.ndarray, file_path: Path):
        """フローデータを保存"""
        np.save(str(file_path), flow)
    
    def calculate_direction(self, is_forward: bool = True):
        """マスク対応版のフロー計算"""
        if len(self.image_files) < 2:
            raise ValueError("画像が2枚以上必要です")
        
        direction = "前方向" if is_forward else "後方向"
        output_dir = self.flow_fwd_dir if is_forward else self.flow_bwd_dir
        
        # インデックスの設定
        if is_forward:
            indices = range(len(self.image_files) - 1)
            get_output_name = lambda i: f"{self.file_format % (i+1)}.npy"
        else:
            indices = range(len(self.image_files) - 1, 0, -1)
            get_output_name = lambda i: f"{self.file_format % i}.npy"
        
        print(f"\n{direction}フローの計算中...")
        
        for i in tqdm(indices, desc=f"{direction}フロー"):
            # 現在のフレームのマスクを読み込み
            current_mask = self._read_mask(i + 1 if is_forward else i)
            
            if is_forward:
                img1 = self._read_image(self.image_files[i])
                img2 = self._read_image(self.image_files[i + 1])
            else:
                img1 = self._read_image(self.image_files[i])
                img2 = self._read_image(self.image_files[i - 1])
            
            # マスクを適用してフロー計算
            flow = self._calculate_flow_cpu(img1, img2, current_mask)
            
            # 保存
            output_path = output_dir / get_output_name(i)
            self._save_flow(flow, output_path)
    
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
    # python tools\optical_flow.py test_dataset\PlatinumChan_x0.5_gen\tracking test_dataset\PlatinumChan_x0.5_gen\mask  test_dataset\PlatinumChan_x0.5_gen\flow_fwd test_dataset\PlatinumChan_x0.5_gen\flow_bwd
    """メイン関数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='オプティカルフロー計算')
    parser.add_argument('input_dir', help='入力画像ディレクトリ')
    parser.add_argument('mask_dir', help='マスク画像ディレクトリ')
    parser.add_argument('flow_fwd_dir', help='前方向フロー出力ディレクトリ')
    parser.add_argument('flow_bwd_dir', help='後方向フロー出力ディレクトリ')
    parser.add_argument('--format', default='%03d', help='ファイル名フォーマット（デフォルト: %03d）')
    
    args = parser.parse_args()
    
    try:
        calculator = OpticalFlowCalculator(
            args.input_dir,
            args.flow_fwd_dir,
            args.flow_bwd_dir,
            args.mask_dir,
            args.format
        )
        calculator.calculate_flows()
        return 0
        
    except Exception as e:
        print(f"エラー: {e}")
        return 1

if __name__ == "__main__":
    exit(main())