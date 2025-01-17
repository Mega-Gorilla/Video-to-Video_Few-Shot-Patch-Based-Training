# lightning_model.py
import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, Any, Optional, List
from src.data.dataset import StyleTransferDataset
import torchvision.utils as vutils
import numpy as np

class StyleTransferModel(pl.LightningModule):
    """スタイル変換を行うPyTorch Lightningモデル"""
    def __init__(
        self,
        generator_config: Dict[str, Any],          # 生成器の設定
        discriminator_config: Dict[str, Any],      # 識別器の設定
        training_config: Dict[str, Any],           # 学習の設定
        optimizer_config: Dict[str, Any],          # オプティマイザーの設定
        data_config: Dict[str, Any],              # データセットの設定
        perception_loss_config: Optional[Dict[str, Any]] = None  # 知覚損失の設定
    ):
        super().__init__()
        self.save_hyperparameters()
        self.automatic_optimization = False  # 手動で最適化を制御

        # ネットワークの初期化
        self.generator = self._build_generator(generator_config)
        self.discriminator = (
            self._build_discriminator(discriminator_config)
            if discriminator_config is not None else None
        )
        
        # 損失関数の設定
        # 再構成損失（L1Loss or MSELoss）の初期化
        self.reconstruction_criterion = getattr(
            nn, training_config["reconstruction_criterion"]
        )()
        # 敵対的損失の初期化
        self.adversarial_criterion = getattr(
            nn, training_config["adversarial_criterion"]
        )()
        
        # 知覚損失の設定（VGG19ベース）
        self.perception_loss_model = None
        if perception_loss_config:
            self.perception_loss_model = self._build_perception_model(
                perception_loss_config["perception_model"]
            )
            self.perception_loss_weight = perception_loss_config["weight"]
        
        # 設定の保存
        self.training_config = training_config
        self.optimizer_config = optimizer_config
        self.data_config = data_config

        # メトリクス計算用のバッファを追加
        self.validation_step_outputs = []

    def _build_generator(self, config: Dict[str, Any]) -> nn.Module:
        """生成器モデルの構築"""
        from src.models.generator import GeneratorJ
        return GeneratorJ(**config["args"])
        
    def _build_discriminator(self, config: Dict[str, Any]) -> nn.Module:
        """識別器モデルの構築"""
        from src.models.discriminator import DiscriminatorN_IN
        return DiscriminatorN_IN(**config["args"])
        
    def _build_perception_model(self, config: Dict[str, Any]) -> nn.Module:
        """知覚損失モデル（VGG19）の構築"""
        from src.models.perception import PerceptualVGG19
        return PerceptualVGG19(**config["args"])

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int):
        """1バッチの学習ステップ"""
        # オプティマイザーの取得
        opt_g, opt_d = self.optimizers()

        # パッチ位置の記録（データセットから提供される場合）
        if hasattr(self.train_dataset, 'last_patch_positions'):
            batch['patch_positions'] = self.train_dataset.last_patch_positions

        # 識別器の学習
        if self.discriminator is not None:
            opt_d.zero_grad()  # 勾配の初期化
            d_loss = self._discriminator_step(batch)
            self.manual_backward(d_loss['loss'])
            
            # 識別器の勾配クリッピング
            if self.training_config.get("use_gradient_clipping", False):
                torch.nn.utils.clip_grad_norm_(
                    self.discriminator.parameters(), 
                    self.training_config["gradient_clip_val"]
                )
                
            opt_d.step()  # パラメータの更新

        # 生成器の学習
        opt_g.zero_grad()  # 勾配の初期化
        g_loss = self._generator_step(batch)
        self.manual_backward(g_loss['loss'])
        
        # 生成器の勾配クリッピング
        if self.training_config.get("use_gradient_clipping", False):
            torch.nn.utils.clip_grad_norm_(
                self.generator.parameters(), 
                self.training_config["gradient_clip_val"]
            )
            
        opt_g.step()

        # batch_idxを使用してログのタイミングを制御
        if batch_idx % self.training_config.get("image_log_freq", 100) == 0:
            # 生成画像を取得
            with torch.no_grad():
                generated = self.generator(batch['pre'])
            self._log_images(batch, generated, batch_idx)

        return g_loss

    def _generator_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """生成器の学習ステップ"""
        generated = self.generator(batch['pre'])
        
        # 損失の計算
        losses = {}
        
        # 画像再構成損失
        if self.training_config["use_image_loss"]:
            image_loss = self.reconstruction_criterion(generated, batch['post'])
            losses['margin_loss'] = image_loss * self.training_config["reconstruction_weight"]
                
        # 知覚損失
        if self.perception_loss_model is not None:
            _, fake_features = self.perception_loss_model(generated)
            _, target_features = self.perception_loss_model(batch['post'])
            perception_loss = ((fake_features - target_features) ** 2).mean()
            losses['g_perception_loss'] = perception_loss * self.perception_loss_weight
                
        # 敵対的損失
        if self.discriminator is not None:
            fake_labels, _ = self.discriminator(generated)
            adversarial_loss = self.adversarial_criterion(
                fake_labels, torch.ones_like(fake_labels)
            )
            losses['g_adversarial_loss'] = adversarial_loss * self.training_config["adversarial_weight"]
                
        # 総損失
        total_g_loss = sum(losses.values())
        losses['g_total_loss'] = total_g_loss
            
        # メトリクスのロギング
        self._log_metrics(losses)
                
        return {'loss': total_g_loss, **losses}

    def _discriminator_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """識別器の学習ステップ"""
        with torch.no_grad():
            generated = self.generator(batch['pre'])
        
        # 真画像の損失
        real_labels, _ = self.discriminator(batch['post'])
        real_loss = self.adversarial_criterion(
            real_labels, torch.ones_like(real_labels)
        )
        
        # 生成画像の損失
        fake_labels, _ = self.discriminator(generated)
        fake_loss = self.adversarial_criterion(
            fake_labels, torch.zeros_like(fake_labels)
        )
        
        # 総損失
        d_loss = (real_loss + fake_loss) * 0.5
        
        # 損失のロギング
        self.log_dict({
            'd_real_loss': real_loss,
            'd_fake_loss': fake_loss,
            'd_total_loss': d_loss
        }, prog_bar=True)
        
        return {'loss': d_loss}

    def configure_optimizers(self):
        """オプティマイザーの設定"""
        # 生成器のオプティマイザー
        opt_g = torch.optim.Adam(
            self.generator.parameters(),
            **self.optimizer_config["generator"]
        )
        
        optimizers = [opt_g]
        
        # 識別器のオプティマイザー（存在する場合）
        if self.discriminator is not None:
            opt_d = torch.optim.Adam(
                self.discriminator.parameters(),
                **self.optimizer_config["discriminator"]
            )
            optimizers.append(opt_d)
            
        return optimizers

    def setup(self, stage: Optional[str] = None):
        """データセットのセットアップ"""
        if stage == "fit" or stage is None:
            self.train_dataset = StyleTransferDataset(
                **self.data_config,
                device=self.device
            )

    def train_dataloader(self) -> DataLoader:
        """訓練用データローダーの設定"""
        return DataLoader(
            self.train_dataset,
            batch_size=self.training_config["batch_size"],
            num_workers=self.training_config["num_workers"],
            shuffle=True,
            pin_memory=True
        )
    
    def _log_metrics(self, losses: Dict[str, torch.Tensor]):
        """メトリクスのロギング"""
        # 論文と同じメトリクス名を使用
        metric_mapping = {
            'margin_loss': 'g_image_loss',
            'g_perception_loss': 'g_perception_loss',
            'g_adversarial_loss': 'g_adversarial_loss',
            'g_total_loss': 'g_total_loss'
        }
        
        for name, value in losses.items():
            mapped_name = metric_mapping.get(name, name)
            self.log(mapped_name, value, prog_bar=True)
            
    def _log_images(self, batch: Dict[str, torch.Tensor], generated: torch.Tensor, batch_idx: int):
        """画像のロギング - 入力、生成、目標画像を並べて表示し、パッチ位置も可視化"""
        log_freq = self.training_config.get("image_log_freq", 100)
        if batch_idx % log_freq == 0:
            # 画像を[-1, 1]から[0, 1]の範囲に正規化
            def normalize_tensor(x):
                # バッチの最初の画像だけを使用
                if len(x.shape) == 4:  # (batch, channel, height, width)
                    x = x[0:1]  # 最初の1枚を保持 (1, channel, height, width)
                return (x.clamp(-1, 1) + 1) / 2

            def draw_patches_on_image(image_tensor, patch_positions, patch_size, color=(1, 0, 0)):
                """画像にパッチ位置を描画"""
                # バッチの最初の画像だけを使用
                if len(image_tensor.shape) == 4:
                    image_tensor = image_tensor[0:1]
                image = normalize_tensor(image_tensor.clone())
                
                if patch_positions:
                    pos = patch_positions[0]  # 最初のパッチ位置のみ使用
                    y, x = pos
                    half_size = patch_size // 2
                    # パッチの境界を描画（赤いボックス）
                    y_min = max(0, y - half_size)
                    y_max = min(image.shape[2] - 1, y + half_size)
                    x_min = max(0, x - half_size)
                    x_max = min(image.shape[3] - 1, x + half_size)
                    
                    # 横線を描画
                    if y_min >= 0 and y_min < image.shape[2]:
                        image[:, :, y_min, x_min:x_max+1] = torch.tensor(color).view(1, 3, 1)
                    if y_max >= 0 and y_max < image.shape[2]:
                        image[:, :, y_max, x_min:x_max+1] = torch.tensor(color).view(1, 3, 1)
                    
                    # 縦線を描画
                    if x_min >= 0 and x_min < image.shape[3]:
                        image[:, :, y_min:y_max+1, x_min] = torch.tensor(color).view(1, 3, 1)
                    if x_max >= 0 and x_max < image.shape[3]:
                        image[:, :, y_min:y_max+1, x_max] = torch.tensor(color).view(1, 3, 1)
                        
                return image

            # パッチ位置情報の取得
            patch_positions = batch.get('patch_positions', [])
            patch_size = getattr(self, 'patch_size', 64)  # デフォルトのパッチサイズ

            # 各画像の準備
            input_image = normalize_tensor(batch['pre'])
            input_with_patches = draw_patches_on_image(batch['pre'], patch_positions, patch_size)
            generated_image = normalize_tensor(generated)
            target_image = normalize_tensor(batch['post'])

            # グリッド表示用に画像を結合
            combined_images = torch.cat([
                input_with_patches,  # パッチ位置表示付きの入力
                generated_image,     # 生成画像
                target_image        # 目標画像
            ], dim=3)

            # グリッド作成
            grid = vutils.make_grid(
                combined_images,
                nrow=1,            # 1行に全ての画像を表示
                padding=2,
                normalize=False    # 既に正規化済み
            )

            # TensorBoardへのログ記録
            step = self.current_epoch * len(self.trainer.train_dataloader) + batch_idx

            # 個別の画像をログ
            self.logger.experiment.add_image('training/input', input_image[0], step)
            self.logger.experiment.add_image('training/input_with_patches', input_with_patches[0], step)
            self.logger.experiment.add_image('training/generated', generated_image[0], step)
            self.logger.experiment.add_image('training/target', target_image[0], step)

            # 比較グリッドをログ
            self.logger.experiment.add_image('training/comparison', grid, step)