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
        self.automatic_optimization = False

        # データ設定の保存
        self.data_config = data_config
        
        # 追加チャネルの設定を保存
        self.additional_channels = data_config.get("additional_channels", {})
        
        # 設定の更新と検証
        self._validate_and_update_configs(generator_config, discriminator_config)
        
        # ネットワークの初期化
        self.generator = self._build_generator(generator_config)
        self.discriminator = (
            self._build_discriminator(discriminator_config)
            if discriminator_config is not None else None
        )
        
        # 設定の保存
        self.training_config = training_config
        self.optimizer_config = optimizer_config
        
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

        # メトリクス計算用のバッファを追加
        self.validation_step_outputs = []

    def _calculate_total_channels(self) -> int:
        """
        総入力チャネル数を計算
        Returns:
            int: 基本チャネル(RGB=3) + 追加チャネルの深さの合計
        """
        base_channels = 3  # RGB基本チャネル

        # 追加チャネルの深さを合計
        additional_channel_depth = 0
        for channel_name, channel_config in self.additional_channels.items():
            depth = channel_config.get('depth', 1)  # デフォルトは1
            additional_channel_depth += depth
            print(f"Channel {channel_name}: depth = {depth}")

        total_channels = base_channels + additional_channel_depth
        print(f"Total channels: {total_channels} (RGB: 3 + Additional: {additional_channel_depth})")
        return total_channels
    
    def _validate_additional_channels(self):
        """
        追加チャネルの設定を検証
        """
        if not self.additional_channels:
            print("No additional channels configured.")
            return

        print("\nValidating additional channels configuration:")
        for channel_name, channel_config in self.additional_channels.items():
            if isinstance(channel_config, dict):
                # 新しい形式での検証
                path = channel_config.get('path')
                depth = channel_config.get('depth', 1)
                
                if not path:
                    raise ValueError(f"Channel {channel_name}: 'path' is required")
                if not isinstance(depth, int) or depth < 1:
                    raise ValueError(f"Channel {channel_name}: 'depth' must be a positive integer")
                    
                print(f"Channel '{channel_name}':")
                print(f"  - Path: {path}")
                print(f"  - Depth: {depth}")
            else:
                # 後方互換性のための処理
                print(f"Channel '{channel_name}': {channel_config} (using default depth=1)")

    def _validate_and_update_configs(
        self,
        generator_config: Dict[str, Any],
        discriminator_config: Optional[Dict[str, Any]]
    ) -> None:
        """設定の検証と更新を行う"""
        from omegaconf import OmegaConf

        # まず追加チャネルの設定を検証
        self._validate_additional_channels()

        def process_model_config(config: Dict[str, Any], model_name: str) -> Dict[str, Any]:
            """モデル設定を処理"""
            if "args" not in config:
                config["args"] = {}
            
            args = OmegaConf.to_container(config["args"], resolve=True)
            current_channels = args.get("input_channels")
            
            # input_channelsの処理
            if current_channels == "auto":
                total_channels = self._calculate_total_channels()
                args["input_channels"] = total_channels
                # 追加チャネルの情報を正しい構文で設定
                args["additional_channels"] = {
                    channel_name: {
                        'depth': channel_config.get('depth', 1) if isinstance(channel_config, dict) else 1,
                        'path': channel_config.get('path', channel_config) if isinstance(channel_config, dict) else channel_config
                    }
                    for channel_name, channel_config in self.additional_channels.items()
                }
                print(f"\n{model_name} using auto input channels: {total_channels}")
            else:
                if current_channels is None:
                    args["input_channels"] = 3
                    print(f"\n{model_name} using default input channels: 3")
                else:
                    print(f"\n{model_name} using specified input channels: {current_channels}")
            
            config["args"] = OmegaConf.create(args)
            return config

        # Generator設定の処理
        generator_config = process_model_config(generator_config, "Generator")
        
        # Discriminator設定の処理（存在する場合）
        if discriminator_config is not None:
            discriminator_config = process_model_config(discriminator_config, "Discriminator")
        
        # 設定の整合性チェック
        self._validate_channel_configuration(generator_config, discriminator_config)

    def _validate_channel_configuration(
        self,
        generator_config: Dict[str, Any],
        discriminator_config: Optional[Dict[str, Any]]
    ) -> None:
        """チャネル設定の検証"""
        print("\nChannel Configuration Validation:")
        print(f"Base channels (RGB): 3")
        print(f"Additional channels config: {self.additional_channels}")
        
        g_channels = generator_config["args"]["input_channels"]
        print(f"Generator configured input channels: {g_channels}")
        
        if discriminator_config is not None:
            d_channels = discriminator_config["args"]["input_channels"]
            print(f"Discriminator configured input channels: {d_channels}")

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

        # 入力テンソルの準備
        input_tensors = [batch['pre']]
        
        # 追加チャネルの連結
        for channel_name in self.additional_channels:
            channel_key = f'channel_{channel_name}'
            if channel_key not in batch:
                raise ValueError(f"Channel {channel_name} not found in batch")
            input_tensors.append(batch[channel_key])
        
        # すべての入力を結合
        combined_input = torch.cat(input_tensors, dim=1)

        # 識別器の学習
        if self.discriminator is not None:
            opt_d.zero_grad()  # 勾配の初期化
            d_loss = self._discriminator_step(combined_input, batch)
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
        g_loss = self._generator_step(combined_input, batch)
        self.manual_backward(g_loss['loss'])
        
        # 生成器の勾配クリッピング
        if self.training_config.get("use_gradient_clipping", False):
            torch.nn.utils.clip_grad_norm_(
                self.generator.parameters(), 
                self.training_config["gradient_clip_val"]
            )
            
        opt_g.step()

        # ロギング用の画像生成（指定された間隔で）
        if batch_idx % self.training_config.get("image_log_freq", 100) == 0:
            with torch.no_grad():
                generated = self.generator(combined_input)
                self._log_images(batch, generated, batch_idx, combined_input)

        return g_loss

    def _generator_step(self, combined_input: torch.Tensor, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """生成器の学習ステップ"""
        generated = self.generator(combined_input)
        losses = {}
        
        # 画像再構成損失
        if self.training_config["use_image_loss"]:
            image_loss = self.reconstruction_criterion(generated, batch['post'])
            losses['margin_loss'] = image_loss * self.training_config["reconstruction_weight"]
                
        # 知覚損失
        if self.perception_loss_model is not None:
            _, fake_features = self.perception_loss_model(generated)
            _, target_features = self.perception_loss_model(batch['post'].detach())
            perception_loss = ((fake_features - target_features) ** 2).mean()
            losses['g_perception_loss'] = perception_loss * self.perception_loss_weight
                
        # 敵対的損失
        if self.discriminator is not None:
            fake_labels, _ = self.discriminator(generated)
            adversarial_loss = self.adversarial_criterion(
                fake_labels, torch.ones_like(fake_labels)
            )
            losses['g_adversarial_loss'] = adversarial_loss * self.training_config["adversarial_weight"]
                
        # 総損失の計算
        total_g_loss = sum(losses.values())
        losses['g_total_loss'] = total_g_loss
            
        # メトリクスのロギング
        self._log_metrics(losses)
                
        return {'loss': total_g_loss, **losses}

    def _discriminator_step(self, combined_input: torch.Tensor, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """識別器の学習ステップ"""
        with torch.no_grad():
            generated = self.generator(combined_input)
        
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
            self.train_dataset = StyleTransferDataset(**self.data_config)

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
            
    def _log_images(
        self,
        batch: Dict[str, torch.Tensor],
        generated: torch.Tensor,
        batch_idx: int,
        combined_input: torch.Tensor = None
    ):
        """画像のロギング - 複数の入力、生成、目標画像を並べて表示
        Args:
            batch: バッチデータ
            generated: 生成された画像
            batch_idx: バッチインデックス
            combined_input: 結合された入力テンソル
        """
        log_freq = self.training_config.get("image_log_freq", 100)
        if batch_idx % log_freq == 0:
            num_images = min(8, batch['pre'].size(0))
            
            def normalize_tensor(x: torch.Tensor, start_idx: int = 0, num_images: int = num_images) -> torch.Tensor:
                """テンソルを[0, 1]の範囲に正規化し、指定した数の画像を抽出"""
                x = x[start_idx:start_idx + num_images]
                return (x.clamp(-1, 1) + 1) / 2

            # 基本画像の準備
            input_images = normalize_tensor(batch['pre'])
            generated_images = normalize_tensor(generated)
            target_images = normalize_tensor(batch['post'])

            # 追加チャネルの準備
            additional_images = {}
            if combined_input is not None:
                channel_start_idx = 3  # RGB後の開始インデックス
                for channel_name, channel_config in self.additional_channels.items():
                    # チャネルの深さを取得
                    depth = channel_config.get('depth', 1) if isinstance(channel_config, dict) else 1
                    
                    # チャネルデータの抽出
                    channel_data = combined_input[:, channel_start_idx:channel_start_idx + depth]

                    # 複数チャネルの場合は平均を取る、または適切な可視化方法を選択
                    if depth == 1:
                        channel_data = channel_data.repeat(1, 3, 1, 1)
                    elif depth == 3:
                        pass
                    else:
                        channel_data = channel_data.mean(dim=1, keepdim=True).repeat(1, 3, 1, 1)
                    additional_images[channel_name] = normalize_tensor(channel_data)
                    
                    channel_start_idx += depth

            # グリッド作成のための画像配列を準備
            for img_idx in range(num_images):
                # 基本画像の行を作成
                row_images = [
                    input_images[img_idx:img_idx+1],
                    generated_images[img_idx:img_idx+1],
                    target_images[img_idx:img_idx+1]
                ]
                
                # 追加チャネルを行に追加
                for channel_name, channel_images in additional_images.items():
                    row_images.append(channel_images[img_idx:img_idx+1])

                # 1行に全ての画像を水平方向に結合
                combined_row = torch.cat(row_images, dim=3)

                # 最初の行の場合は新しいテンソルを作成、それ以外は既存のテンソルに追加
                if img_idx == 0:
                    combined_grid = combined_row
                else:
                    combined_grid = torch.cat([combined_grid, combined_row], dim=2)

            # グリッド作成
            grid = vutils.make_grid(
                combined_grid,
                nrow=1,  # 1行に1セットの画像を表示
                padding=2,
                normalize=False
            )

            # ステップ数の計算
            step = self.current_epoch * len(self.trainer.train_dataloader) + batch_idx

            # 画像の説明をログに追加
            header_text = "Input | Generated | Target"
            for channel_name in additional_images.keys():
                header_text += f" | {channel_name}"
            
            # まず基本的な画像をログ
            self.logger.experiment.add_image('training/comparison_grid', grid, step)
            self.logger.experiment.add_image('training/input', input_images[0], step)
            self.logger.experiment.add_image('training/generated', generated_images[0], step)
            self.logger.experiment.add_image('training/target', target_images[0], step)

            self.logger.experiment.add_text(
                'training/grid_description',
                header_text,
                step
            )
            
            # 最後に追加チャネルをログ
            for channel_name, channel_images in additional_images.items():
                self.logger.experiment.add_image(
                    f'training/channel_{channel_name}',
                    channel_images[0],
                    step
                )