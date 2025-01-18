import numpy as np
from scipy.ndimage import gaussian_filter
from skimage.transform import resize
from sklearn.neighbors import NearestNeighbors
import cv2
from dataclasses import dataclass

@dataclass
class GaussianBlurConfig:
    min_sigma: float = 0.3
    max_sigma: float = 1.0

@dataclass
class OpacityConfig:
    min: float = 0.2
    max: float = 0.5

class GaussianMixtureGenerator:
    def __init__(self, grid_size=20, blur_config=None, opacity_config=None):
        """
        Args:
            grid_size (int): グリッドサイズ
            blur_config (GaussianBlurConfig): ぼかしの設定
            opacity_config (OpacityConfig): 不透明度の設定
        """
        self.grid_size = grid_size
        self.blur_config = blur_config or GaussianBlurConfig()
        self.opacity_config = opacity_config or OpacityConfig()
        
    def create_gaussian_parameters(self, point):
        """円形のガウス分布パラメータを生成"""
        # 回転は均一分布ではなく、より自然な分布に
        rotation = np.random.normal(0, np.pi/4)
        
        # ぼかしの強度をconfig範囲内でランダムに設定
        sigma = np.random.uniform(self.blur_config.min_sigma, self.blur_config.max_sigma)
        
        # 不透明度を設定範囲内でランダムに設定
        opacity = np.random.uniform(self.opacity_config.min, self.opacity_config.max)
        
        return rotation, sigma, opacity
        
    def create_gaussian_splat(self, size, params):
        """円形の2Dガウスカーネルを生成"""
        rotation, sigma, opacity = params
        x, y = np.meshgrid(np.linspace(-1, 1, size), np.linspace(-1, 1, size))
        
        # より円形に近いガウス分布を生成（sigmaでぼかしの強度を調整）
        d = np.sqrt(x**2 + y**2)
        gaussian = np.exp(-d**2 / (2.0 * sigma**2)) * opacity
        
        # ソフトな境界を持つ円形マスクを適用
        mask = np.exp(-d**2 * 2)
        gaussian = gaussian * mask
        return gaussian
        
    def generate_random_color(self):
        """色相に基づくランダムな色を生成"""
        hue = np.random.uniform(0, 1)
        # HSVからRGBに変換 (S=V=1で鮮やかな色)
        c = np.array(cv2.cvtColor(np.uint8([[[hue * 180, 255, 255]]]), 
                                cv2.COLOR_HSV2RGB)) / 255.0
        return c[0, 0]
        
    def create_grid_points(self, mask):
        """均一な密度で円形のガウス分布を生成するグリッドポイントを生成"""
        h, w = mask.shape
        
        # より細かいグリッドの生成
        grid_h = max(h // (self.grid_size * 2), 2)
        grid_w = max(w // (self.grid_size * 2), 2)
        
        points = []
        parameters = []
        colors = []
        
        # マスク領域の中心を計算
        y_indices, x_indices = np.nonzero(mask)
        if len(y_indices) > 0 and len(x_indices) > 0:
            center_y = (y_indices.min() + y_indices.max()) // 2
            center_x = (x_indices.min() + x_indices.max()) // 2
        
        for i in range(0, h, grid_h):
            for j in range(0, w, grid_w):
                region = mask[i:i+grid_h, j:j+grid_w]
                
                if region.any():
                    # 基本点の数を設定
                    num_points = 3  # 基本の点の数を増やす
                    
                    for _ in range(num_points):
                        # マスク内でランダムな位置を選択
                        valid_positions = np.argwhere(region)
                        if len(valid_positions) > 0:
                            offset = valid_positions[np.random.randint(len(valid_positions))]
                            point = [i + offset[0], j + offset[1]]
                            points.append(point)
                            parameters.append(self.create_gaussian_parameters(point))
                            colors.append(self.generate_random_color())
        
        return np.array(points), parameters, colors
        
    def track_gaussians(self, prev_points, prev_params, prev_colors, curr_points):
        """ガウスの追跡と補間"""
        if len(prev_points) == 0 or len(curr_points) == 0:
            return curr_points, [self.create_gaussian_parameters(p) for p in curr_points], \
                   [self.generate_random_color() for _ in curr_points]
        
        # 最近傍点の探索
        nbrs = NearestNeighbors(n_neighbors=1).fit(prev_points)
        distances, indices = nbrs.kneighbors(curr_points)
        
        # パラメータの補間
        new_params = []
        new_colors = []
        for i, (dist, idx) in enumerate(zip(distances, indices)):
            if dist < self.grid_size * 2:  # 閾値内なら補間
                # パラメータの線形補間
                prev_param = prev_params[idx[0]]
                curr_param = self.create_gaussian_parameters(curr_points[i])
                interp_param = tuple(0.7 * np.array(prev_param) + 0.3 * np.array(curr_param))
                # 色の補間
                prev_color = prev_colors[idx[0]]
                curr_color = self.generate_random_color()
                interp_color = 0.7 * prev_color + 0.3 * curr_color
                
                new_params.append(interp_param)
                new_colors.append(interp_color)
            else:  # 新規ガウス
                new_params.append(self.create_gaussian_parameters(curr_points[i]))
                new_colors.append(self.generate_random_color())
        
        return curr_points, new_params, new_colors
        
    def generate_gaussian_mixture(self, mask, previous_points=None, previous_params=None, 
                                previous_colors=None):
        """ガウス分布の混合を生成（Splatting方式）"""
        h, w = mask.shape
        
        # グリッドポイントとパラメータの生成
        grid_points, parameters, colors = self.create_grid_points(mask)
        
        # 前フレームの情報がある場合は追跡を行う
        if previous_points is not None and previous_params is not None and previous_colors is not None:
            grid_points, parameters, colors = self.track_gaussians(
                previous_points, previous_params, previous_colors, grid_points)
        
        # 出力画像の初期化
        output = np.zeros((h, w, 3), dtype=np.float32)
        
        # 各ガウスをSplatting
        gaussian_size = min(self.grid_size * 3, min(h, w) // 8)
        
        for point, params, color in zip(grid_points, parameters, colors):
            y, x = int(point[0]), int(point[1])
            
            # ガウスカーネルの生成と配置
            kernel = self.create_gaussian_splat(gaussian_size, params)
            
            # 境界チェックと配置
            y1, y2 = max(0, y-gaussian_size//2), min(h, y+gaussian_size//2)
            x1, x2 = max(0, x-gaussian_size//2), min(w, x+gaussian_size//2)
            
            ky1 = max(0, gaussian_size//2-y)
            ky2 = min(gaussian_size, gaussian_size//2+(h-y))
            kx1 = max(0, gaussian_size//2-x)
            kx2 = min(gaussian_size, gaussian_size//2+(w-x))
            
            # カラーチャンネルごとにSplatting
            kernel_region = kernel[ky1:ky2, kx1:kx2, np.newaxis]
            output[y1:y2, x1:x2] += kernel_region * color
        
        # 値の正規化と変換
        output = np.clip(output * 255, 0, 255).astype(np.uint8)
        
        # 追跡情報を返す
        return output, (grid_points, parameters, colors)