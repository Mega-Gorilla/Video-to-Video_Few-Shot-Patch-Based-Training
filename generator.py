# generator.py
import os
import torch
from PIL import Image, ImageDraw
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
import torch.nn.functional as F
from typing import List, Tuple, Optional

class StyleTransferInference:
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self._setup_logging()
        self._setup_transforms()
        self._load_data_config()
        self._setup_model()
        
        # パッチサイズを学習時の設定から取得
        self.patch_size = self.cfg.data.patch_size  # データ設定から取得
        self.debug_mode = cfg.inference.get('debug_mode', False)
        self.patch_positions = []

    def _setup_logging(self):
        """ロギングの設定"""
        if self.debug_mode:
            level = logging.DEBUG
        else:
            level = logging.INFO
        logging.basicConfig(
            level=level,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
    
    def _calculate_total_channels(self) -> int:
        """総入力チャネル数を計算"""
        base_channels = 3  # RGB基本チャネル
        additional_channel_depth = 0

        if hasattr(self.cfg.paths, 'additional_channels'):
            from omegaconf import OmegaConf
            additional_channels = OmegaConf.to_container(self.cfg.paths.additional_channels)
            
            for channel_name, channel_config in additional_channels.items():
                if isinstance(channel_config, dict):
                    depth = int(channel_config.get('depth', 1))
                else:
                    depth = 1
                    
                additional_channel_depth += depth
                self.logger.info(f"Channel {channel_name}: depth = {depth}")

        total_channels = base_channels + additional_channel_depth
        self.logger.info(f"Total channels: {total_channels} (RGB: 3 + Additional: {additional_channel_depth})")
        return total_channels

    def _validate_additional_channels(self):
        """追加チャネルの設定を検証"""
        if not hasattr(self.cfg.paths, 'additional_channels'):
            self.logger.info("No additional channels configured.")
            return

        from omegaconf import OmegaConf
        additional_channels = OmegaConf.to_container(self.cfg.paths.additional_channels)

        self.logger.info("\nValidating additional channels configuration:")
        for channel_name, channel_config in additional_channels.items():
            path = str(channel_config.get('path'))
            depth = int(channel_config.get('depth', 1))
            
            if not path:
                raise ValueError(f"Channel {channel_name}: 'path' is required")
            if not isinstance(depth, int) or depth < 1:
                raise ValueError(f"Channel {channel_name}: 'depth' must be a positive integer")
                
            self.logger.info(f"Channel '{channel_name}':")
            self.logger.info(f"  - Path: {path}")
            self.logger.info(f"  - Depth: {depth}")

    def _setup_transforms(self):
        """変換処理の設定"""
        # すべてのチャネル（RGB及び追加チャネル）に共通の変換処理
        self.transform = transforms.Compose([
            RGBConvert(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        
        # マスク用の変換処理
        self.mask_transform = transforms.Compose([
            GrayscaleConvert(),
            transforms.ToTensor()
        ])

    def _setup_model(self):
        """モデルの設定とロード"""
        try:
            self.logger.info("=== Setting up Model ===")
            
            # デバイスの設定
            self.device = torch.device(
                "cuda" if torch.cuda.is_available() and self.cfg.inference.use_gpu else "cpu"
            )
            
            # チェックポイントを先に読み込んでチャネル数を確認
            self.logger.info(f"Loading checkpoint from: {self.cfg.paths.checkpoint}")
            checkpoint = torch.load(self.cfg.paths.checkpoint, map_location=self.device)
            
            # 生成器の初期畳み込み層のチャネル数を確認
            initial_conv_weight = checkpoint['state_dict']['generator.initial_conv.0.weight']
            checkpoint_input_channels = initial_conv_weight.shape[1]
            
            # 現在の設定のチャネル数と比較
            if hasattr(self, 'total_channels') and self.total_channels != checkpoint_input_channels:
                raise ValueError(
                    f"Channel count mismatch! "
                    f"Checkpoint model expects {checkpoint_input_channels} channels, "
                    f"but current configuration has {self.total_channels} channels "
                    f"(RGB: {self.base_channels}, Additional: {sum(self.channel_info.values() if hasattr(self, 'channel_info') else [0])}). "
                    f"Please ensure the model was trained with the same channel configuration."
                )
            
            # OmegaConfの設定を辞書に変換してから再度OmegaConfオブジェクトに変換
            from omegaconf import OmegaConf
            
            # generator_configの処理
            generator_dict = OmegaConf.to_container(self.cfg.model.generator, resolve=True)
            if "args" not in generator_dict:
                generator_dict["args"] = {}
            
            generator_dict["args"].update({
                "input_channels": checkpoint_input_channels,
                "additional_channels": None  # チェックポイントのモデル構造に合わせる
            })
            generator_config = OmegaConf.create(generator_dict)
            
            # discriminator_configの処理
            discriminator_dict = OmegaConf.to_container(self.cfg.model.discriminator, resolve=True)
            discriminator_config = OmegaConf.create(discriminator_dict)
            
            # training_configの処理
            training_dict = OmegaConf.to_container(self.cfg.training, resolve=True)
            training_config = OmegaConf.create(training_dict)
            
            # optimizer_configの処理
            optimizer_dict = OmegaConf.to_container(self.cfg.optimizer, resolve=True)
            optimizer_config = OmegaConf.create(optimizer_dict)
            
            # data_configの処理
            data_dict = OmegaConf.to_container(self.cfg.data, resolve=True)
            data_config = OmegaConf.create(data_dict)
            
            # perception_loss_configの処理
            perception_loss_dict = OmegaConf.to_container(self.cfg.model.perception_loss, resolve=True)
            perception_loss_config = OmegaConf.create(perception_loss_dict)
            
            self.logger.info(f"Generator configuration:")
            self.logger.info(f"- Input channels from checkpoint: {checkpoint_input_channels}")
            self.logger.info(f"- Current total channels: {self.total_channels if hasattr(self, 'total_channels') else 'Not set'}")
            
            # モデルの初期化
            self.model = StyleTransferModel(
                generator_config=generator_config,
                discriminator_config=discriminator_config,
                training_config=training_config,
                optimizer_config=optimizer_config,
                data_config=data_config,
                perception_loss_config=perception_loss_config
            )
            
            # チェックポイントのロード
            self.model.load_state_dict(checkpoint['state_dict'], strict=True)
            self.model.to(self.device)
            self.model.eval()
            
            if torch.cuda.is_available():
                self.model.generator = self.model.generator.half()
                
            self.logger.info("Model setup completed successfully")
            
        except Exception as e:
            self.logger.error(f"Error during model setup: {str(e)}")
            raise RuntimeError(
                "Failed to initialize model. Please check the model configuration "
                "and ensure it matches the checkpoint architecture."
            ) from e
    
    def _load_data_config(self):
        """データ設定の読み込み - 追加チャネルを動的に処理"""
        self.logger.info("=== Loading Data Configuration ===")
        
        # RGB基本チャネル数
        self.base_channels = 3
        self.additional_channels = {}
        
        # OmegaConfをインポート
        from omegaconf import OmegaConf
        
        # 追加チャネルの処理
        if hasattr(self.cfg.paths, 'additional_channels'):
            # DictConfigを通常の辞書に変換
            additional_channels = OmegaConf.to_container(self.cfg.paths.additional_channels)
            
            # 追加チャネルの検証
            self._validate_additional_channels()
            
            # チャネル情報の初期化
            self.channel_info = {}
            
            for channel_name, channel_config in additional_channels.items():
                try:
                    # 設定がdict形式かどうかを確認
                    if isinstance(channel_config, dict):
                        channel_path = str(channel_config['path'])
                        channel_depth = int(channel_config.get('depth', 1))
                    else:
                        # 後方互換性のための処理
                        channel_path = str(channel_config)
                        channel_depth = 1

                    # チャネルパスの存在確認
                    channel_dir = Path(channel_path)
                    if not channel_dir.exists():
                        raise FileNotFoundError(f"Channel directory not found: {channel_dir}")

                    # サンプル画像を読み込んでチャネル数を確認
                    sample_files = list(channel_dir.glob("*.[pj][np][g]"))
                    if not sample_files:
                        raise FileNotFoundError(f"No images found in {channel_path}")
                        
                    sample_image = Image.open(sample_files[0])
                    actual_channels = len(sample_image.getbands())
                    
                    if actual_channels < channel_depth:
                        raise ValueError(
                            f"Channel {channel_name} has insufficient channels: "
                            f"expected {channel_depth}, but found {actual_channels}"
                        )
                    
                    # チャネル情報を保存
                    self.channel_info[channel_name] = {
                        'path': channel_path,
                        'depth': channel_depth,
                        'actual_channels': actual_channels
                    }
                    
                    # 追加チャネル情報を保存
                    self.additional_channels[channel_name] = channel_path
                    
                    self.logger.info(
                        f"Channel {channel_name}: path={channel_path}, "
                        f"depth={channel_depth}, actual_channels={actual_channels}"
                    )

                except Exception as e:
                    self.logger.error(f"Error processing channel {channel_name}: {str(e)}")
                    raise

            # 総チャネル数を計算
            self.total_channels = self._calculate_total_channels()
        else:
            self.total_channels = self.base_channels
            self.logger.info(f"No additional channels found. Using RGB ({self.base_channels} channels) only.")


    def _find_corresponding_image(self, base_dir: str, image_name: str) -> str:
        """
        指定されたベースディレクトリで、対応する画像ファイルを探す
        異なる拡張子（png, jpg, jpeg）でも対応する
        
        Args:
            base_dir: 検索するディレクトリのパス
            image_name: 元のファイル名
        Returns:
            str: 見つかった画像ファイルの完全なパス
        """
        # 元のファイル名から拡張子を除去
        base_name = os.path.splitext(os.path.basename(image_name))[0]
        
        # サポートする拡張子のリスト
        extensions = ['.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG']
        
        # 各拡張子でファイルの存在をチェック
        for ext in extensions:
            file_path = os.path.join(base_dir, base_name + ext)
            if os.path.exists(file_path):
                self.logger.info(f"Found corresponding image: {file_path}")
                return file_path
        
        # 対応するファイルが見つからない場合は元のパスを返す
        return os.path.join(base_dir, image_name)
    
    def _draw_patches(
        self, 
        image: Image.Image, 
        positions: List[Tuple[int, int, int, int]], 
    ) -> Image.Image:
        """パッチ位置を画像上に描画（ランダムカラー）"""
        draw = ImageDraw.Draw(image)
        
        # 各パッチに対して異なるランダムカラーを生成
        for y_start, y_end, x_start, x_end in positions:
            # ランダムカラーの生成（明るめの色を生成）
            random_color = (
                np.random.randint(100, 256),  # R
                np.random.randint(100, 256),  # G
                np.random.randint(100, 256)   # B
            )
            
            # 矩形を描画
            draw.rectangle(
                [x_start, y_start, x_end, y_end],
                outline=random_color,
                width=2
            )
        
        return image

    def _process_mask(self, mask_tensor: torch.Tensor) -> torch.Tensor:
        """マスクの処理（dataset.pyと完全に同じ処理）"""
        device = mask_tensor.device
        
        # マスクの閾値処理
        mask_tensor[mask_tensor < 0.4] = 0
        
        # エロージョン処理
        erosion_weights = torch.ones((1, 1, 7, 7)).to(device)
        mask_conv = F.conv2d(
            mask_tensor.unsqueeze(0),  # (1, 1, H, W)の形状に
            erosion_weights,
            stride=1,
            padding=3
        )
        
        # エロージョンの結果を正規化
        mask_conv[mask_conv < erosion_weights.numel()] = 0
        mask_conv /= erosion_weights.numel()
        
        # デバッグ情報
        self.logger.debug(f"Mask tensor shape after erosion: {mask_conv.shape}")
        self.logger.debug(f"Unique values in processed mask: {torch.unique(mask_conv)}")
        
        return mask_conv.squeeze(0)

    def _get_valid_patch_positions(
        self, 
        mask_tensor: torch.Tensor,
        overlap_percent: float = 50.0
    ) -> List[Tuple[int, int, int, int]]:
        """マスクに基づいて有効なパッチ位置を取得
        Args:
            mask_tensor: マスクテンソル
            overlap_percent: パッチ間の重なりの割合（0-100%）
        Returns:
            List of tuples (y_start, y_end, x_start, x_end) representing patch boundaries
        """
        # オーバーラップからストライドを計算
        overlap = min(max(overlap_percent, 0.0), 100.0) / 100.0  # 0-1の範囲に正規化
        stride = int(self.patch_size * (1 - overlap))  # ストライドを計算
        stride = max(1, stride)  # 最小値は1に設定
        
        indices = mask_tensor.squeeze().nonzero()
        valid_positions = []
        used_positions = set()
        half_size = self.patch_size // 2
        
        h, w = mask_tensor.shape[-2:]
        
        for idx in range(0, len(indices), stride):
            y, x = indices[idx]
            pos_key = (y.item() // stride, x.item() // stride)
            
            if pos_key not in used_positions:
                y_start = max(0, y.item() - half_size)
                y_end = min(h, y.item() + half_size)
                x_start = max(0, x.item() - half_size)
                x_end = min(w, x.item() + half_size)
                
                valid_positions.append((y_start, y_end, x_start, x_end))
                used_positions.add(pos_key)
        
        # デバッグ情報
        self.logger.debug(f"Original number of nonzero positions: {len(indices)}")
        self.logger.debug(f"Number of selected patch positions: {len(valid_positions)}")
        self.logger.debug(f"Overlap: {overlap_percent}%")
        self.logger.debug(f"Stride: {stride}")
        self.logger.debug(f"Patch size: {self.patch_size}")
        self.logger.debug(f"Sample patch box: {valid_positions[0] if valid_positions else None}")
        
        return valid_positions

    def _cut_patch(self, tensor: torch.Tensor, midpoint: Tuple[int, int]) -> Tuple[torch.Tensor, Tuple[int, int, int, int]]:
        """指定された中心座標からパッチを切り出す（学習時と同じ処理）"""
        y, x = midpoint
        half_size = self.patch_size // 2
        
        h, w = tensor.shape[-2:]
        # パッチの境界を計算
        hn = max(0, y - half_size)
        hx = min(y + half_size, h - 1)
        xn = max(0, x - half_size)
        xx = min(x + half_size, w - 1)
        
        # パッチを切り出し
        patch = tensor[..., hn:hx, xn:xx]
        
        # パッチサイズが正しくない場合、ゼロパディング
        if patch.shape[-2:] != (self.patch_size, self.patch_size):
            result = torch.zeros(
                (*tensor.shape[:-2], self.patch_size, self.patch_size),
                device=patch.device,
                dtype=patch.dtype
            )
            result[..., :patch.shape[-2], :patch.shape[-1]] = patch
            patch = result
        
        return patch, (hn, hx, xn, xx)

    def process_large_image(
        self, 
        input_tensor: torch.Tensor, 
        mask_tensor: Optional[torch.Tensor] = None,
        overlap_percent: float = 30.0
    ) -> torch.Tensor:
        """画像を処理"""
        # デバイスの確認と設定
        if torch.cuda.is_available() and self.cfg.inference.use_gpu:
            device = torch.device('cuda:0')
            if not input_tensor.is_cuda:
                input_tensor = input_tensor.cuda()
            if mask_tensor is not None and not mask_tensor.is_cuda:
                mask_tensor = mask_tensor.cuda()
        else:
            device = torch.device('cpu')
            if input_tensor.is_cuda:
                input_tensor = input_tensor.cpu()
            if mask_tensor is not None and mask_tensor.is_cuda:
                mask_tensor = mask_tensor.cpu()

        # モデルを正しいデバイスに移動
        self.model = self.model.to(device)
        
        b, c, h, w = input_tensor.shape
        dtype = input_tensor.dtype
        
        # 出力テンソルの初期化（RGBのみ）
        output = torch.zeros((b, 3, h, w), dtype=dtype, device=device)
        weights = torch.zeros((b, 1, h, w), dtype=dtype, device=device)
        
        if mask_tensor is None:
            mask_tensor = torch.ones((b, 1, h, w), dtype=dtype, device=device)
        
        # パッチの境界ボックスを取得
        patch_boxes = self._get_valid_patch_positions(
            mask_tensor,
            overlap_percent=overlap_percent
        )
        
        def ensure_valid_patch_size(patch: torch.Tensor) -> torch.Tensor:
            """パッチサイズが正しいことを確認し、必要に応じて調整"""
            _, _, h, w = patch.shape
            if h != self.patch_size or w != self.patch_size:
                # パディングまたはトリミングを行う
                new_patch = torch.zeros(
                    (patch.size(0), patch.size(1), self.patch_size, self.patch_size),
                    dtype=patch.dtype,
                    device=patch.device
                )
                
                # コピー可能な範囲を計算
                h_copy = min(h, self.patch_size)
                w_copy = min(w, self.patch_size)
                
                # 中央に配置するためのオフセットを計算
                h_offset = (self.patch_size - h_copy) // 2
                w_offset = (self.patch_size - w_copy) // 2
                
                # データをコピー
                new_patch[
                    :, :,
                    h_offset:h_offset + h_copy,
                    w_offset:w_offset + w_copy
                ] = patch[:, :, :h_copy, :w_copy]
                
                self.logger.warning(
                    f"Adjusted patch size from {h}x{w} to "
                    f"{self.patch_size}x{self.patch_size}"
                )
                return new_patch
            return patch

        for y_start, y_end, x_start, x_end in tqdm(patch_boxes, desc="Processing patches"):
            try:
                # パッチを直接切り出し
                patch = input_tensor[..., y_start:y_end, x_start:x_end]
                
                # パッチサイズの確認と調整
                patch = ensure_valid_patch_size(patch)
                
                # パッチが正しいデバイスにあることを確認
                if patch.device != device:
                    patch = patch.to(device)
                
                with torch.no_grad():
                    # パッチを生成器に渡す（全チャネルを使用）
                    try:
                        processed_patch = self.model.generator(patch)
                    except Exception as e:
                        self.logger.error(
                            f"Error in generator: input shape={patch.shape}, "
                            f"error={str(e)}"
                        )
                        raise
                    
                    # 出力パッチが正しいサイズであることを確認
                    processed_patch = ensure_valid_patch_size(processed_patch)
                    
                    # ガウシアンウェイト
                    patch_h, patch_w = y_end - y_start, x_end - x_start
                    weight = torch.exp(-((torch.arange(patch_h, device=device) - patch_h/2)**2 / (patch_h/4)**2))[:, None] * \
                            torch.exp(-((torch.arange(patch_w, device=device) - patch_w/2)**2 / (patch_w/4)**2))[None, :]
                    weight = weight.to(dtype)[None, None, :, :]
                    
                    # 出力パッチのサイズに合わせてウェイトを調整
                    if weight.shape[-2:] != processed_patch.shape[-2:]:
                        weight = torch.nn.functional.interpolate(
                            weight,
                            size=processed_patch.shape[-2:],
                            mode='bilinear',
                            align_corners=False
                        )
                    
                    # 出力テンソルに結果を追加
                    h_slice = slice(y_start, min(y_start + processed_patch.shape[2], output.shape[2]))
                    w_slice = slice(x_start, min(x_start + processed_patch.shape[3], output.shape[3]))
                    
                    output[..., h_slice, w_slice] += processed_patch[..., :h_slice.stop-h_slice.start, :w_slice.stop-w_slice.start] * \
                                                    weight[..., :h_slice.stop-h_slice.start, :w_slice.stop-w_slice.start]
                    weights[..., h_slice, w_slice] += weight[..., :h_slice.stop-h_slice.start, :w_slice.stop-w_slice.start]
                    
                    self.patch_positions.append((y_start, y_end, x_start, x_end))
                    
            except Exception as e:
                self.logger.error(
                    f"Error processing patch at position ({y_start}, {y_end}, {x_start}, {x_end}): {str(e)}"
                )
                raise
        
        # 重みで正規化
        valid_mask = weights > 1e-8
        output = output / weights.repeat(1, 3, 1, 1).where(valid_mask, torch.ones_like(weights))
        
        # マスクの適用
        rgb_input = input_tensor[:, :3]
        output = rgb_input * (1 - mask_tensor) + output * mask_tensor
        
        return output

    def process_image(self, input_path: str, mask_path: str, save_path: str):
        """1枚の画像を処理
        Args:
            input_path: 入力画像のパス
            mask_path: マスク画像のベースパス（拡張子は自動で検出）
            save_path: 出力画像の保存パス
        """
        try:
            self.logger.info("=== Processing Image ===")
            self.logger.info(f"Input: {input_path}")
            
            input_tensors = []
            # RGB画像の読み込み
            if not os.path.exists(input_path):
                raise FileNotFoundError(f"Input image not found: {input_path}")
            
            self.logger.info("Loading RGB image...")
            image = Image.open(input_path).convert('RGB')
            rgb_tensor = self.transform(image).unsqueeze(0)
            input_tensors.append(rgb_tensor)
            self.logger.info(f"RGB tensor shape: {rgb_tensor.shape}")
            
            # 追加チャネルの読み込み
            self.logger.info("Processing additional channels...")
            for channel_name, channel_dir in self.additional_channels.items():
                self.logger.info(f"Processing channel: {channel_name}")
                self.logger.info(f"Channel directory: {channel_dir}")
                
                # 対応する画像を探す
                channel_path = self._find_corresponding_image(channel_dir, input_path)
                self.logger.info(f"Full channel path: {channel_path}")
                
                if not os.path.exists(channel_path):
                    raise FileNotFoundError(
                        f"Required channel {channel_name} not found: {channel_path}"
                    )
                
                self.logger.info("Loading channel image...")
                channel_image = Image.open(channel_path)
                channel_tensor = self.transform(channel_image).unsqueeze(0)
                self.logger.info(f"Channel tensor shape: {channel_tensor.shape}")
                input_tensors.append(channel_tensor)

            # 入力テンソルの結合
            input_tensor = torch.cat(input_tensors, dim=1)
            self.logger.info(f"Combined input tensor shape: {input_tensor.shape}")
            
            if torch.cuda.is_available():
                input_tensor = input_tensor.half()
                self.logger.info("Converted input tensor to half precision")

            # マスクの読み込みと処理
            mask_dir = os.path.dirname(mask_path)
            mask_file = os.path.basename(mask_path)
            mask_path = self._find_corresponding_image(mask_dir, mask_file)
            
            if not os.path.exists(mask_path):
                raise FileNotFoundError(f"Mask file not found: {mask_path}")
            
            self.logger.info(f"Loading mask from: {mask_path}")
            mask = Image.open(mask_path)
            mask = mask.point(lambda p: p > 128 and 255)
            mask_tensor = self.mask_transform(mask)
            mask_tensor = self._process_mask(mask_tensor)
            mask_tensor = mask_tensor.unsqueeze(0)
            
            if torch.cuda.is_available():
                mask_tensor = mask_tensor.half()
                self.logger.info("Converted mask tensor to half precision")

            # パッチベースの処理
            self.logger.info("Starting patch-based processing...")
            output_tensor = self.process_large_image(input_tensor, mask_tensor)
            
            # 出力の保存（RGB部分のみ）
            self.logger.info("Saving output image...")
            output_rgb = output_tensor[:, :3]  # RGB channels only
            output_rgb = output_rgb.float().clamp(-1, 1)
            image_space = ((output_rgb + 1) * 127.5).clamp(0, 255)
            image_space = image_space.permute(0, 2, 3, 1)
            image_space = image_space.round().cpu().numpy()[0].astype(np.uint8)
            
            # 画像の保存
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            output_image = Image.fromarray(image_space)
            output_image.save(save_path)
            
            # デバッグモードの場合
            if self.debug_mode:
                debug_path = str(Path(save_path).with_name(f"debug_{Path(save_path).name}"))
                debug_image = output_image.copy()
                debug_image = self._draw_patches(debug_image, self.patch_positions)
                debug_image.save(debug_path)

            # メモリ解放
            del input_tensor, output_tensor
            torch.cuda.empty_cache()
            
            self.logger.info(f"Successfully processed and saved: {save_path}")

        except Exception as e:
            self.logger.error(f"Error processing {input_path}: {str(e)}")
            self.logger.error(f"Error type: {type(e)}")
            import traceback
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            raise

    def process_directory(self):
        """ディレクトリ内の全画像を処理"""
        input_dir = Path(self.cfg.paths.input_dir)
        mask_dir = Path(self.cfg.paths.mask_dir)
        output_dir = Path(self.cfg.paths.output_dir)
        
        # 入力ディレクトリのチェック
        if not input_dir.exists():
            raise FileNotFoundError(f"Input directory not found: {input_dir}")
        
        # マスクディレクトリのチェック（ignoreでない場合）
        if not mask_dir.name.endswith("ignore"):
            if not mask_dir.exists():
                raise FileNotFoundError(f"Mask directory not found: {mask_dir}. Creating full masks.")
        
        
        output_dir.mkdir(parents=True, exist_ok=True)
        self.logger.info(f"Processing images from {input_dir} to {output_dir}")
        
        image_files = list(input_dir.glob("*.[pj][np][g]"))
        self.logger.info(f"Found {len(image_files)} images to process")
        
        for input_path in tqdm(image_files, desc="Processing images"):
            mask_path = mask_dir / input_path.name
            output_path = output_dir / input_path.name
            
            try:
                self.process_image(str(input_path), str(mask_path), str(output_path))
            except Exception as e:
                self.logger.error(f"Failed to process {input_path.name}: {str(e)}")
                self.logger.error(f"Error details: {e}")  # より詳細なエラー情報
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
    main()