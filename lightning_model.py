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
            
        opt_g.step()  # パラメータの更新

        return g_loss

    def _generator_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """生成器の学習ステップ"""
        # 画像の生成
        generated = self.generator(batch['pre'])
        
        # 損失の計算
        losses = {}
        
        # 画像再構成損失の計算
        if self.training_config["use_image_loss"]:
            image_loss = self.reconstruction_criterion(generated, batch['post'])
            losses['g_image_loss'] = image_loss * self.training_config["reconstruction_weight"]
            
            # PSNR計算
            with torch.no_grad():
                mse = torch.mean((generated - batch['post']) ** 2)
                psnr = 20 * torch.log10(2.0 / torch.sqrt(mse))  # [-1,1]範囲なので最大値は2
                self.log('psnr', psnr, on_step=True, on_epoch=True, prog_bar=True)
            
        # 知覚損失（VGG特徴量の差）の計算
        if self.perception_loss_model is not None:
            _, fake_features = self.perception_loss_model(generated)
            _, target_features = self.perception_loss_model(batch['post'])
            perception_loss = ((fake_features - target_features) ** 2).mean()
            losses['g_perception_loss'] = perception_loss * self.perception_loss_weight
            
        # 敵対的損失の計算
        if self.discriminator is not None:
            fake_labels, fake_features = self.discriminator(generated)
            
            # 生成画像を本物と判定させるように学習
            adversarial_loss = self.adversarial_criterion(
                fake_labels, torch.ones_like(fake_labels)
            )
            losses['g_adversarial_loss'] = adversarial_loss * self.training_config["adversarial_weight"]
            
            # 特徴マッチング損失（オプション）
            if self.training_config.get("use_feature_matching", False):
                _, real_features = self.discriminator(batch['post'])
                feature_matching_loss = sum(
                    torch.mean((f_r - f_f) ** 2)
                    for f_r, f_f in zip(real_features, fake_features)
                )
                losses['g_feature_matching_loss'] = (
                    feature_matching_loss * self.training_config.get("feature_matching_weight", 0.1)
                )
        
        # 総損失の計算
        total_g_loss = sum(losses.values())
        losses['g_total_loss'] = total_g_loss
        
        # メトリクスのロギング
        self._log_metrics(losses, prefix='generator/')
        
        # 画像のロギング
        self._log_images(batch, generated)
            
        return {'loss': total_g_loss, **losses}

    def _discriminator_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """識別器の学習ステップ"""
        # 生成画像の作成（勾配計算なし）
        with torch.no_grad():
            generated = self.generator(batch['pre'])
        
        # 本物画像の判定
        real_labels, _ = self.discriminator(batch['post'])
        real_loss = self.adversarial_criterion(
            real_labels, torch.ones_like(real_labels)
        )
        
        # 生成画像の判定
        fake_labels, _ = self.discriminator(generated)
        fake_loss = self.adversarial_criterion(
            fake_labels, torch.zeros_like(fake_labels)
        )
        
        # 識別器の総損失
        d_loss = (real_loss + fake_loss) * 0.5
        
        # 識別器の損失をログ記録
        losses = {
            'd_real_loss': real_loss,
            'd_fake_loss': fake_loss,
            'd_total_loss': d_loss
        }
        
        # 識別器の精度計算
        with torch.no_grad():
            real_accuracy = (real_labels > 0.5).float().mean()
            fake_accuracy = (fake_labels < 0.5).float().mean()
            accuracy = (real_accuracy + fake_accuracy) * 0.5
            
            self.log('d_real_accuracy', real_accuracy, on_step=True, on_epoch=True, prog_bar=True)
            self.log('d_fake_accuracy', fake_accuracy, on_step=True, on_epoch=True, prog_bar=True)
            self.log('d_accuracy', accuracy, on_step=True, on_epoch=True, prog_bar=True)
        
        # メトリクスのロギング
        self._log_metrics(losses, prefix='discriminator/')
        
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
    
    def _log_metrics(self, losses: Dict[str, torch.Tensor], prefix: str = ""):
        """メトリクスのロギング"""
        # 各損失値を個別にログ
        for name, value in losses.items():
            self.log(
                f"{prefix}{name}",
                value,
                on_step=True,
                on_epoch=True,
                prog_bar=True,
                logger=True,
                sync_dist=True
            )

        # 移動平均の計算とログ
        if not hasattr(self, "loss_history"):
            self.loss_history = {}
        
        window_size = 100
        for name, value in losses.items():
            if name not in self.loss_history:
                self.loss_history[name] = []
            
            self.loss_history[name].append(value.item())
            if len(self.loss_history[name]) > window_size:
                self.loss_history[name].pop(0)
            
            avg_value = np.mean(self.loss_history[name])
            self.log(
                f"{prefix}{name}_ma{window_size}",
                avg_value,
                on_step=True,
                on_epoch=True,
                prog_bar=True,
                logger=True,
                sync_dist=True
            )

    def _log_images(self, batch: Dict[str, torch.Tensor], generated: torch.Tensor):
        """画像のロギング"""
        if self.global_step % self.training_config.get("image_log_freq", 100) == 0:
            # 入力画像、生成画像、目標画像を並べて表示
            vis_images = []
            
            # 入力画像
            input_images = batch['pre']
            vis_images.append(input_images)
            
            # 生成画像
            vis_images.append(generated)
            
            # 目標画像
            target_images = batch['post']
            vis_images.append(target_images)
            
            # 画像を結合
            vis_images = torch.cat(vis_images, dim=3)  # 横に並べる
            
            # [-1,1]から[0,1]に変換
            vis_images = (vis_images + 1) / 2
            
            # グリッド形式で画像を保存
            grid = vutils.make_grid(
                vis_images,
                nrow=1,
                padding=2,
                normalize=False
            )
            
            # TensorBoardにログ
            self.logger.experiment.add_image(
                'comparison',
                grid,
                self.global_step
            )

            # 差分マップの生成と保存
            with torch.no_grad():
                diff_map = torch.abs(generated - target_images)
                diff_map = diff_map.mean(dim=1, keepdim=True)  # チャンネル方向の平均
                diff_map = diff_map / diff_map.max()  # [0,1]に正規化
                
                # カラーマップの適用（任意）
                diff_map = torch.cat([diff_map, torch.zeros_like(diff_map), torch.zeros_like(diff_map)], dim=1)
                
                self.logger.experiment.add_image(
                    'diff_map',
                    vutils.make_grid(diff_map, nrow=1, padding=2),
                    self.global_step
                )
