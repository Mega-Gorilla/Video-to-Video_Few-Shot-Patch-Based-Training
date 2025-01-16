# lightning_model.py
import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, Any, Optional, List
from src.data.dataset import StyleTransferDataset

class StyleTransferModel(pl.LightningModule):
    def __init__(
        self,
        generator_config: Dict[str, Any],
        discriminator_config: Dict[str, Any],
        training_config: Dict[str, Any],
        optimizer_config: Dict[str, Any],
        data_config: Dict[str, Any],
        perception_loss_config: Optional[Dict[str, Any]] = None
    ):
        super().__init__()
        self.save_hyperparameters()
        self.automatic_optimization = False  # 手動最適化モードに設定
        
        # Initialize networks
        self.generator = self._build_generator(generator_config)
        self.discriminator = (
            self._build_discriminator(discriminator_config)
            if discriminator_config is not None else None
        )
        
        # Setup loss functions
        self.reconstruction_criterion = getattr(
            nn, training_config["reconstruction_criterion"]
        )()
        self.adversarial_criterion = getattr(
            nn, training_config["adversarial_criterion"]
        )()
        
        # Setup perception loss
        self.perception_loss_model = None
        if perception_loss_config:
            self.perception_loss_model = self._build_perception_model(
                perception_loss_config["perception_model"]
            )
            self.perception_loss_weight = perception_loss_config["weight"]
        
        # Training config
        self.training_config = training_config
        self.optimizer_config = optimizer_config
        self.data_config = data_config

    def _build_generator(self, config: Dict[str, Any]) -> nn.Module:
        # Import your generator model here
        from src.models.generator import GeneratorJ
        return GeneratorJ(**config["args"])
        
    def _build_discriminator(self, config: Dict[str, Any]) -> nn.Module:
        from src.models.discriminator import DiscriminatorN_IN
        return DiscriminatorN_IN(**config["args"])
        
    def _build_perception_model(self, config: Dict[str, Any]) -> nn.Module:
        from src.models.perception import PerceptualVGG19
        return PerceptualVGG19(**config["args"])

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int):
        # Get optimizers
        opt_g, opt_d = self.optimizers()

        # Train Discriminator
        if self.discriminator is not None:
            opt_d.zero_grad()
            d_loss = self._discriminator_step(batch)
            self.manual_backward(d_loss['loss'])
            
            # Manual gradient clipping for discriminator
            if self.training_config.get("use_gradient_clipping", False):
                torch.nn.utils.clip_grad_norm_(
                    self.discriminator.parameters(), 
                    self.training_config["gradient_clip_val"]
                )
                
            opt_d.step()

        # Train Generator
        opt_g.zero_grad()
        g_loss = self._generator_step(batch)
        self.manual_backward(g_loss['loss'])
        
        # Manual gradient clipping for generator
        if self.training_config.get("use_gradient_clipping", False):
            torch.nn.utils.clip_grad_norm_(
                self.generator.parameters(), 
                self.training_config["gradient_clip_val"]
            )
            
        opt_g.step()

        return g_loss

    def _generator_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        generated = self.generator(batch['pre'])
        
        # Calculate losses
        losses = {}
        
        # Image reconstruction loss
        if self.training_config["use_image_loss"]:
            image_loss = self.reconstruction_criterion(generated, batch['post'])
            losses['g_image_loss'] = image_loss * self.training_config["reconstruction_weight"]
            
        # Perceptual loss
        if self.perception_loss_model is not None:
            _, fake_features = self.perception_loss_model(generated)
            _, target_features = self.perception_loss_model(batch['post'])
            perception_loss = ((fake_features - target_features) ** 2).mean()
            losses['g_perception_loss'] = perception_loss * self.perception_loss_weight
            
        # Adversarial loss
        if self.discriminator is not None:
            fake_labels, _ = self.discriminator(generated)
            adversarial_loss = self.adversarial_criterion(
                fake_labels, torch.ones_like(fake_labels)
            )
            losses['g_adversarial_loss'] = adversarial_loss * self.training_config["adversarial_weight"]
            
        # Total generator loss
        total_g_loss = sum(losses.values())
        losses['g_total_loss'] = total_g_loss
        
        # Log metrics
        self.log_dict(losses, prog_bar=True)
        
        if self.global_step % 100 == 0:
            # Log images
            self.logger.experiment.add_images(
                'generated_images',
                (generated + 1) / 2,
                self.global_step
            )
            
        return {'loss': total_g_loss, **losses}

    def _discriminator_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        # Generate fake images
        with torch.no_grad():
            generated = self.generator(batch['pre'])
        
        # Real images
        real_labels, _ = self.discriminator(batch['post'])
        real_loss = self.adversarial_criterion(
            real_labels, torch.ones_like(real_labels)
        )
        
        # Fake images
        fake_labels, _ = self.discriminator(generated)
        fake_loss = self.adversarial_criterion(
            fake_labels, torch.zeros_like(fake_labels)
        )
        
        # Total discriminator loss
        d_loss = (real_loss + fake_loss) * 0.5
        
        # Log discriminator losses
        self.log_dict({
            'd_real_loss': real_loss,
            'd_fake_loss': fake_loss,
            'd_total_loss': d_loss
        }, prog_bar=True)
        
        return {'loss': d_loss}

    def configure_optimizers(self):
        opt_g = torch.optim.Adam(
            self.generator.parameters(),
            **self.optimizer_config["generator"]
        )
        
        optimizers = [opt_g]
        
        if self.discriminator is not None:
            opt_d = torch.optim.Adam(
                self.discriminator.parameters(),
                **self.optimizer_config["discriminator"]
            )
            optimizers.append(opt_d)
            
        return optimizers

    def setup(self, stage: Optional[str] = None):
        """Data setup"""
        if stage == "fit" or stage is None:
            self.train_dataset = StyleTransferDataset(
                **self.data_config,
                device=self.device
            )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.training_config["batch_size"],
            num_workers=self.training_config["num_workers"],
            shuffle=True,
            pin_memory=True
        )