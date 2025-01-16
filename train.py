# train.py

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
    """Create training callbacks"""
    callbacks = []
    
    # Checkpoint callback
    callbacks.append(
        ModelCheckpoint(
            dirpath=os.path.join(cfg.training.output_dir, "checkpoints"),
            filename="style_transfer-{epoch:02d}-{g_total_loss:.4f}",
            monitor="g_total_loss",
            mode="min",
            save_top_k=3,
            save_last=True,
        )
    )
    
    # Learning rate monitoring
    callbacks.append(
        LearningRateMonitor(logging_interval="step")
    )
    
    # Early stopping
    if cfg.training.early_stopping:
        callbacks.append(
            EarlyStopping(
                monitor="g_total_loss",
                patience=cfg.training.early_stopping_patience,
                mode="min",
                verbose=True
            )
        )
        
    return callbacks

@hydra.main(version_base=None, config_path="config", config_name="config")
def train(cfg: DictConfig) -> None:
    """Main training function"""
    
    # Print config
    print(OmegaConf.to_yaml(cfg))
    
    # Create output directory
    os.makedirs(cfg.training.output_dir, exist_ok=True)
    
    # Save config
    config_path = os.path.join(cfg.training.output_dir, "config.yaml")
    with open(config_path, "w") as f:
        OmegaConf.save(cfg, f)
    
    # Initialize model
    model = StyleTransferModel(
        generator_config=cfg.model.generator,
        discriminator_config=cfg.model.discriminator,
        training_config=cfg.training,
        optimizer_config=cfg.optimizer,
        data_config=cfg.data,
        perception_loss_config=cfg.model.perception_loss
    )
    
    # Setup logger
    logger = TensorBoardLogger(
        save_dir=os.path.join(cfg.training.output_dir, "logs"),
        name="style_transfer",
        default_hp_metric=False
    )
    
    # Create callbacks
    callbacks = create_callbacks(cfg)
    
    # Initialize trainer (without gradient_clip_val)
    trainer = pl.Trainer(
        logger=logger,
        callbacks=callbacks,
        max_epochs=cfg.training.max_epochs,
        accelerator=cfg.training.accelerator,
        devices=cfg.training.devices,
        precision=cfg.training.precision,
        accumulate_grad_batches=cfg.training.accumulate_grad_batches,
        deterministic=cfg.training.deterministic,
        log_every_n_steps=cfg.training.log_every_n_steps,
        enable_progress_bar=True,
        enable_model_summary=True,
        enable_checkpointing=True,
    )
    
    # Train model
    trainer.fit(model)
    
    print("Training completed!")

if __name__ == "__main__":
    train()