import os
from pathlib import Path
import hydra
from omegaconf import DictConfig, OmegaConf
import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    LearningRateMonitor,
    EarlyStopping
)
from pytorch_lightning.loggers import TensorBoardLogger
from lightning_model import StyleTransferModel

def create_callbacks(cfg: DictConfig):
    """学習時のコールバック関数を作成する"""
    callbacks = []
    
    # モデルのチェックポイントを保存するコールバック
    # - g_total_lossが最小のモデルを最大3つ保存
    # - 最新のモデルも保存（last.ckpt）
    callbacks.append(
        ModelCheckpoint(
            dirpath=os.path.join(cfg.training.output_dir, "checkpoints"),
            filename="style_transfer-{epoch:02d}-{g_total_loss:.4f}",
            monitor="g_total_loss",  # 監視する指標
            mode="min",              # 最小値を記録
            save_top_k=3,           # 上位3つを保存
            save_last=True,         # 最新モデルも保存
        )
    )
    
    # 学習率の変化を記録するコールバック
    callbacks.append(
        LearningRateMonitor(logging_interval="step")
    )
    
    # Early Stoppingのコールバック（設定で有効な場合のみ）
    if cfg.training.early_stopping:
        callbacks.append(
            EarlyStopping(
                monitor="g_total_loss",    # 監視する指標
                patience=cfg.training.early_stopping_patience,  # 何エポック我慢するか
                mode="min",                # 最小値を監視
                verbose=True               # 詳細なログを出力
            )
        )
    
    return callbacks

@hydra.main(version_base=None, config_path="config", config_name="config")
def train(cfg: DictConfig) -> None:
    """メインの学習関数"""
    
    # 設定内容を表示
    print(OmegaConf.to_yaml(cfg))
    
    # 出力ディレクトリを作成
    os.makedirs(cfg.training.output_dir, exist_ok=True)
    
    # 設定をYAMLファイルとして保存
    config_path = os.path.join(cfg.training.output_dir, "config.yaml")
    with open(config_path, "w") as f:
        OmegaConf.save(cfg, f)
    
    # モデルの初期化
    # - Generator、Discriminator、学習設定、最適化設定、データ設定を渡す
    model = StyleTransferModel(
        generator_config=cfg.model.generator,
        discriminator_config=cfg.model.discriminator,
        training_config=cfg.training,
        optimizer_config=cfg.optimizer,
        data_config=cfg.data,
        perception_loss_config=cfg.model.perception_loss
    )
    
    # TensorBoardロガーの設定
    # - 学習の進捗をTensorBoardで可視化
    logger = TensorBoardLogger(
        save_dir=os.path.join(cfg.training.output_dir, "logs"),
        name="style_transfer",
        default_hp_metric=False
    )
    
    # コールバックの作成（チェックポイント、Early Stopping等）
    callbacks = create_callbacks(cfg)
    
    # PyTorch Lightningのトレーナーを初期化
    trainer = pl.Trainer(
        logger=logger,                    # ロガーの設定
        callbacks=callbacks,              # コールバックの設定
        max_epochs=cfg.training.max_epochs,  # 最大エポック数
        accelerator=cfg.training.accelerator,  # 使用するデバイス（CPU/GPU）
        devices=cfg.training.devices,     # デバイス数
        precision=cfg.training.precision,  # 計算精度（16/32/bf16）
        accumulate_grad_batches=cfg.training.accumulate_grad_batches,  # 勾配蓄積回数
        deterministic=cfg.training.deterministic,  # 決定論的な動作の有効化
        log_every_n_steps=cfg.training.log_every_n_steps,  # ログ出力の頻度
        enable_progress_bar=True,         # プログレスバーの表示
        enable_model_summary=True,        # モデルサマリーの表示
        enable_checkpointing=True,        # チェックポイントの有効化
    )
    
    # モデルの学習開始
    trainer.fit(model)
    
    print("Training completed!")

if __name__ == "__main__":
    train()