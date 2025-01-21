import cv2
import numpy as np
import os
from typing import Tuple, List
from itertools import product

class ColorGrid:
    def __init__(self, width: int, height: int, grid_size: int):
        self.width = width
        self.height = height
        self.grid_size = self._adjust_grid_size(width, height, grid_size)
        self.cols = max(1, width // self.grid_size)
        self.rows = max(1, height // self.grid_size)
        self.total_grids = self.rows * self.cols
        print(f"Grid size: {self.grid_size}, Rows: {self.rows}, Cols: {self.cols}, Total grids: {self.total_grids}")
    
    def _adjust_grid_size(self, width: int, height: int, requested_size: int) -> int:
        # 最小サイズの制限のみを設定
        min_grid_size = 4
        
        # 要求サイズを最小サイズ以上に調整
        size = max(min_grid_size, requested_size)
        
        return size
        
    def generate_unique_colors(self) -> List[np.ndarray]:
        if self.total_grids == 0:
            raise ValueError("Total number of grids cannot be zero")
        
        colors = []
        hue_step = max(1, 180 // self.total_grids)
        hue_values = np.linspace(0, 179, self.total_grids, dtype=np.uint8)
        np.random.shuffle(hue_values)
        
        saturations = np.random.randint(180, 256, self.total_grids)
        values = np.random.randint(180, 256, self.total_grids)
        
        for h, s, v in zip(hue_values, saturations, values):
            colors.append(np.array([h, s, v], dtype=np.uint8))
            
        return colors

def process_texture(input_path: str, output_path: str, grid_size: int = 32):
    img = cv2.imread(input_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        print(f"Error: Could not read image {input_path}")
        return
    
    if len(img.shape) < 3 or img.shape[2] != 4:
        print(f"Warning: Image {input_path} has no alpha channel")
        return
    
    print(f"\nProcessing {input_path}")
    print(f"Image size: {img.shape[1]}x{img.shape[0]}")
    
    try:
        # アルファチャンネルを取得
        alpha = img[:, :, 3]
        mask = alpha > 0
        
        # ColorGridインスタンスを作成
        color_grid = ColorGrid(img.shape[1], img.shape[0], grid_size)
        unique_colors = color_grid.generate_unique_colors()
        
        # 色マップを作成
        color_map = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
        
        # グリッドごとに色を割り当て
        for idx, color in enumerate(unique_colors):
            row = idx // color_grid.cols
            col = idx % color_grid.cols
            
            y_start = row * color_grid.grid_size
            y_end = min((row + 1) * color_grid.grid_size, img.shape[0])
            x_start = col * color_grid.grid_size
            x_end = min((col + 1) * color_grid.grid_size, img.shape[1])
            
            color_map[y_start:y_end, x_start:x_end] = color
        
        # HSVからBGRに変換
        color_map_bgr = cv2.cvtColor(color_map, cv2.COLOR_HSV2BGR)
        
        # 元の画像のRGBチャンネルをコピー
        result = img[:, :, :3].copy()
        
        # マスク領域に色マップを適用
        mask_3d = np.stack([mask] * 3, axis=-1)
        result = np.where(mask_3d, color_map_bgr, result)
        
        # アルファチャンネルを追加
        result = cv2.merge([result[:,:,0], result[:,:,1], result[:,:,2], alpha])
        
        # 結果を保存
        cv2.imwrite(output_path, result)
        print(f"Successfully saved texture map to {output_path}")
        
    except Exception as e:
        print(f"Error processing image {input_path}: {str(e)}")
        import traceback
        traceback.print_exc()
        return

def process_directory(input_dir: str, grid_size: int = 32):
    if not os.path.exists(input_dir):
        print(f"Error: Directory {input_dir} does not exist")
        return
    
    print(f"Processing directory: {input_dir}")
    print(f"Requested grid size: {grid_size}")
    
    for filename in os.listdir(input_dir):
        if filename.lower().endswith('.png'):
            input_path = os.path.join(input_dir, filename)
            name, ext = os.path.splitext(filename)
            output_filename = f"{name}_map{ext}"
            output_path = os.path.join(input_dir, output_filename)
            process_texture(input_path, output_path, grid_size)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate texture maps from PNG images with alpha channels")
    parser.add_argument("input_dir", help="Input directory containing PNG images")
    parser.add_argument("--grid-size", type=int, default=32, help="Size of the color grid (default: 32)")
    
    args = parser.parse_args()
    process_directory(args.input_dir, args.grid_size)