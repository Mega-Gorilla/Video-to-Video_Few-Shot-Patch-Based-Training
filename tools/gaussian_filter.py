import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional, Union, Dict
import cv2
from numba import jit, prange
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from functools import partial
import warnings
warnings.filterwarnings('ignore')

@jit(nopython=True)
def _generate_points_numba(radius: float, x_min: np.ndarray, x_max: np.ndarray, 
                          mask: np.ndarray, max_attempts: int, seed: int) -> np.ndarray:
    """マスク領域を考慮したポイント生成"""
    np.random.seed(seed)
    cell_size = radius / np.sqrt(2)
    
    h, w = mask.shape
    grid_size = (
        int(np.ceil((x_max[0] - x_min[0]) / cell_size)),
        int(np.ceil((x_max[1] - x_min[1]) / cell_size))
    )
    
    grid = np.full(grid_size, -1, dtype=np.int32)
    
    # マスク領域内の有効なピクセルを探す
    valid_points = np.empty((h * w, 2), dtype=np.float64)
    num_valid = 0
    for y in range(h):
        for x in range(w):
            if mask[y, x] > 64:  # マスク閾値
                valid_points[num_valid] = np.array([x, y])
                num_valid += 1
    
    if num_valid == 0:
        return np.empty((0, 2), dtype=np.float64)
    
    valid_points = valid_points[:num_valid]
    
    # 最大点数の計算
    max_points = int((grid_size[0] * grid_size[1]) / (np.pi * (radius/cell_size)**2))
    samples = np.empty((max_points, 2), dtype=np.float64)
    active = np.empty(max_points, dtype=np.int32)
    num_samples = 0
    num_active = 0
    
    # 最初のサンプルをマスク領域内からランダムに選択
    first_idx = np.random.randint(0, num_valid)
    first_sample = valid_points[first_idx]
    cell = ((first_sample - x_min) / cell_size).astype(np.int32)
    grid[cell[0], cell[1]] = 0
    samples[0] = first_sample
    active[0] = 0
    num_samples = 1
    num_active = 1

    while num_active > 0:
        idx = np.random.randint(0, num_active)
        point = samples[active[idx]]
        
        found = False
        for _ in range(max_attempts):
            angle = np.random.random() * 2 * np.pi
            distance = np.random.uniform(radius, 2*radius)
            candidate = point + distance * np.array([np.cos(angle), np.sin(angle)])
            
            x, y = int(candidate[0]), int(candidate[1])
            if (0 <= x < w and 0 <= y < h and mask[y, x] > 64):  # マスク領域内のチェック
                cell = ((candidate - x_min) / cell_size).astype(np.int32)
                if 0 <= cell[0] < grid_size[0] and 0 <= cell[1] < grid_size[1]:
                    valid = True
                    
                    for i in range(max(0, cell[0]-2), min(grid_size[0], cell[0]+3)):
                        for j in range(max(0, cell[1]-2), min(grid_size[1], cell[1]+3)):
                            if grid[i,j] != -1:
                                other = samples[grid[i,j]]
                                if np.sum((candidate - other)**2) < radius**2:
                                    valid = False
                                    break
                        if not valid:
                            break
                    
                    if valid and num_samples < max_points:
                        grid[cell[0], cell[1]] = num_samples
                        samples[num_samples] = candidate
                        active[num_active] = num_samples
                        num_samples += 1
                        num_active += 1
                        found = True
                        break
        
        if not found:
            active[idx] = active[num_active - 1]
            num_active -= 1
    
    return samples[:num_samples]

@jit(nopython=True)
def _sample_bilinear_numba(flow: np.ndarray, point: np.ndarray) -> np.ndarray:
    """Numbaで最適化されたバイリニア補間"""
    h, w = flow.shape[:2]
    x, y = point
    
    ix = int(np.floor(x))
    iy = int(np.floor(y))
    
    s = x - ix
    t = y - iy
    
    ix0 = max(0, min(ix, w-1))
    ix1 = max(0, min(ix + 1, w-1))
    iy0 = max(0, min(iy, h-1))
    iy1 = max(0, min(iy + 1, h-1))
    
    return (1.0-s)*(1.0-t)*flow[iy0,ix0] + \
           (s)*(1.0-t)*flow[iy0,ix1] + \
           (1.0-s)*(t)*flow[iy1,ix0] + \
           (s)*(t)*flow[iy1,ix1]

@jit(nopython=True, parallel=True)
def _draw_points_numba(output: np.ndarray, size: Tuple[int, int], 
                      points: np.ndarray, sigma: float, colors: np.ndarray) -> None:
    """Numbaで最適化された点群描画"""
    for i in prange(len(points)):
        point = points[i]
        color = colors[i]
        x, y = point
        r = int(3 * sigma)
        
        y_min = max(0, int(y-r))
        y_max = min(size[0], int(y+r)+1)
        x_min = max(0, int(x-r))
        x_max = min(size[1], int(x+r)+1)
        
        for yi in range(y_min, y_max):
            for xi in range(x_min, x_max):
                weight = np.exp(-(((xi-x)**2 + (yi-y)**2))/(sigma*sigma))
                for c in range(3):
                    output[yi,xi,c] = output[yi,xi,c] * (1-weight) + color[c] * weight

class PoissonDiskSampling:
    def __init__(self, radius: float, x_min: np.ndarray, x_max: np.ndarray, 
                 max_attempts: int = 30, seed: int = 0):
        self.radius = radius
        self.x_min = x_min
        self.x_max = x_max
        self.max_attempts = max_attempts
        self.seed = seed

    def generate(self, mask: np.ndarray) -> np.ndarray:
        """マスクを考慮した点群生成"""
        return _generate_points_numba(
            self.radius, self.x_min, self.x_max, 
            mask, self.max_attempts, self.seed
        )

class GaussianFilter:
    """最適化されたガウシアンフィルタ実装"""
    def __init__(self, mask_dir, flow_fwd_dir, flow_bwd_dir, output_dir, 
                 frame_first, frame_last, key_frames, radius, sigma, 
                 file_format='%03d', num_workers=None, max_points=1000): 
        
        if not key_frames:
            raise ValueError("key_frames list is empty")
            
        self.max_points = max_points 
        self.mask_dir = Path(mask_dir)
        self.flow_fwd_dir = Path(flow_fwd_dir)
        self.flow_bwd_dir = Path(flow_bwd_dir)
        self.output_dir = Path(output_dir)
        self.frame_first = frame_first
        self.frame_last = frame_last
        self.key_frames = sorted(key_frames)
        self.radius = radius
        self.sigma = sigma
        self.file_format = file_format
        self.num_workers = num_workers or mp.cpu_count()
        
        # ディレクトリ確認
        for directory in [self.mask_dir, self.flow_fwd_dir, self.flow_bwd_dir]:
            if not directory.exists():
                raise ValueError(f"Directory does not exist: {directory}")
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 最初のマスク読み込み
        first_frame = self.key_frames[0]
        first_mask_path = self.mask_dir / f"{first_frame:03d}.jpg"
        
        first_mask = cv2.imread(str(first_mask_path), cv2.IMREAD_GRAYSCALE)
        if first_mask is None:
            raise ValueError(f"Failed to read initial mask: {first_mask_path}")
            
        self.size = first_mask.shape
        
        # 点群データ構造の初期化（辞書形式に変更）
        self.pts: Dict[int, Dict[int, np.ndarray]] = {
            k: {} for k in range(len(key_frames))
        }
        
        # フローデータキャッシュ
        self.flow_cache = {}
        self.flow_cache_lock = threading.Lock()

    def _get_flow(self, path: Path) -> Optional[np.ndarray]:
        """フローデータのスレッドセーフなキャッシング"""
        with self.flow_cache_lock:
            if path not in self.flow_cache:
                if not path.exists():
                    return None
                self.flow_cache[path] = np.load(str(path))
            return self.flow_cache[path]

    def generate_points(self, mask: np.ndarray) -> np.ndarray:
        """点数制限付きの点群生成"""
        h, w = mask.shape
        x_min = np.array([0, 0])
        x_max = np.array([w, h])
        
        sampler = PoissonDiskSampling(self.radius, x_min, x_max)
        points = sampler.generate(mask)
        
        # 点数が制限を超える場合はランダムに間引く
        if len(points) > self.max_points:
            indices = np.random.choice(len(points), self.max_points, replace=False)
            points = points[indices]
            
        return points

    def _get_last_valid_frame(self) -> int:
        """有効な最後のフレーム番号を取得"""
        last_forward_flow = max(
            (int(p.stem) for p in self.flow_bwd_dir.glob("*.npy")),
            default=self.frame_first
        )
        last_backward_flow = max(
            (int(p.stem) for p in self.flow_fwd_dir.glob("*.npy")),
            default=self.frame_first
        )
        return min(last_forward_flow, last_backward_flow, self.frame_last)

    def _process_output_frame(self, frame: int) -> str:
        """1フレームの出力処理"""
        try:
            output = np.zeros((*self.size, 3), dtype=np.float32)
            
            for k in range(len(self.key_frames)-1, -1, -1):
                if frame in self.pts[k]:
                    points = self.pts[k][frame]
                    if len(points) > 0:
                        np.random.seed(1337 + k)
                        colors = np.random.random((len(points), 3))
                        _draw_points_numba(output, self.size, points, self.sigma, colors)
            
            output_path = self.output_dir / f"{self.file_format % frame}.png"
            cv2.imwrite(str(output_path), (output * 255).astype(np.uint8))
            return f"Saved frame {frame}"
        
        except Exception as e:
            return f"Error processing frame {frame}: {e}"

    def process(self):
        """メイン処理（最適化版）"""
        try:
            print(f"\nStarting process with {len(self.key_frames)} key frames")
            
            # フレーム範囲の検証と調整
            self.frame_last = min(self.frame_last, self._get_last_valid_frame())
            print(f"Adjusted frame range: {self.frame_first} to {self.frame_last}")
            
            # キーフレームごとの処理
            for k, key_frame in enumerate(self.key_frames):
                print(f"\nProcessing key frame {key_frame} ({k+1}/{len(self.key_frames)})")
                
                # マスク読み込みと検証
                mask_path = self.mask_dir / f"{key_frame:03d}.jpg"
                if not mask_path.exists():
                    print(f"Warning: Skip key frame {key_frame} - Mask not found: {mask_path}")
                    continue
                    
                mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
                if mask is None:
                    print(f"Warning: Skip key frame {key_frame} - Failed to read mask: {mask_path}")
                    continue
                
                # 点群生成
                key_points = self.generate_points(mask)
                if len(key_points) == 0:
                    print(f"Warning: Skip key frame {key_frame} - No points generated")
                    continue
                    
                print(f"Generated {len(key_points)} points")
                self.pts[k][key_frame] = key_points

                # 前方伝播
                if self.frame_last > key_frame:
                    print(f"Forward propagation from frame {key_frame}")
                    points = key_points.copy()
                    
                    for frame in range(key_frame + 1, self.frame_last + 1):
                        try:
                            flow = self._get_flow(
                                self.flow_bwd_dir / f"{self.file_format % (frame-1)}.npy"
                            )
                            
                            if flow is None:
                                print(f"Warning: No flow data for frame {frame-1}")
                                break
                            
                            if len(points) > 0:
                                new_points = []
                                for p in points:
                                    if 0 <= p[0] < self.size[1] and 0 <= p[1] < self.size[0]:
                                        new_p = p + _sample_bilinear_numba(flow, p)
                                        if 0 <= new_p[0] < self.size[1] and 0 <= new_p[1] < self.size[0]:
                                            new_points.append(new_p)
                                            
                                points = np.array(new_points)
                                if len(points) > 0:
                                    self.pts[k][frame] = points.copy()
                                else:
                                    print(f"No valid points for frame {frame}")
                                    break
                            else:
                                print(f"No points to propagate at frame {frame}")
                                break
                                
                        except Exception as e:
                            print(f"Error in forward propagation at frame {frame}: {e}")
                            break

                # 後方伝播
                if self.frame_first < key_frame:
                    print(f"Backward propagation from frame {key_frame}")
                    points = key_points.copy()
                    
                    for frame in range(key_frame - 1, self.frame_first - 1, -1):
                        try:
                            flow = self._get_flow(
                                self.flow_fwd_dir / f"{self.file_format % frame}.npy"
                            )
                            
                            if flow is None:
                                print(f"Warning: No flow data for frame {frame}")
                                break
                            
                            if len(points) > 0:
                                new_points = []
                                for p in points:
                                    if 0 <= p[0] < self.size[1] and 0 <= p[1] < self.size[0]:
                                        new_p = p + _sample_bilinear_numba(flow, p)
                                        if 0 <= new_p[0] < self.size[1] and 0 <= new_p[1] < self.size[0]:
                                            new_points.append(new_p)
                                            
                                points = np.array(new_points)
                                if len(points) > 0:
                                    self.pts[k][frame] = points.copy()
                                else:
                                    print(f"No valid points for frame {frame}")
                                    break
                            else:
                                print(f"No points to propagate at frame {frame}")
                                break
                                
                        except Exception as e:
                            print(f"Error in backward propagation at frame {frame}: {e}")
                            break

            # 結果の描画と保存
            print("\nDrawing and saving results...")
            with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
                frame_range = range(self.frame_first, self.frame_last + 1)
                futures = []
                
                for frame in frame_range:
                    future = executor.submit(self._process_output_frame, frame)
                    futures.append(future)
                
                for future in as_completed(futures):
                    try:
                        result = future.result()
                        print(result)
                    except Exception as e:
                        print(f"Error in frame processing: {e}")

        except Exception as e:
            print(f"Error in process: {e}")
            raise

def main():
    """使用例"""
    import argparse
    parser = argparse.ArgumentParser(description="Optimized Gaussian Filter")
    parser.add_argument("--mask-dir", required=True, help="Mask directory")
    parser.add_argument("--flow-fwd-dir", required=True, help="Forward flow directory")
    parser.add_argument("--flow-bwd-dir", required=True, help="Backward flow directory")
    parser.add_argument("--output-dir", required=True, help="Output directory")
    parser.add_argument("--frame-first", type=int, required=True, help="First frame number")
    parser.add_argument("--frame-last", type=int, required=True, help="Last frame number")
    parser.add_argument("--key-frames", type=int, nargs="+", required=True, help="Key frame numbers")
    parser.add_argument("--radius", type=float, default=10.0, help="Sampling radius")
    parser.add_argument("--sigma", type=float, default=5.0, help="Gaussian sigma")
    parser.add_argument("--file-format", default="%03d", help="File format string")
    parser.add_argument("--num-workers", type=int, help="Number of worker processes")
    parser.add_argument("--max-points", type=int, default=1000,
                      help="Maximum number of points to generate per frame")
    
    args = parser.parse_args()
    
    filter = GaussianFilter(
        mask_dir=args.mask_dir,
        flow_fwd_dir=args.flow_fwd_dir,
        flow_bwd_dir=args.flow_bwd_dir,
        output_dir=args.output_dir,
        frame_first=args.frame_first,
        frame_last=args.frame_last,
        key_frames=args.key_frames,
        radius=args.radius,
        sigma=args.sigma,
        file_format=args.file_format,
        num_workers=args.num_workers,
        max_points=args.max_points
    )
    
    filter.process()

if __name__ == "__main__":
    main()