import json
import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.utils.data import DataLoader
import numpy as np

import sys
import os
from pathlib import Path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# from ..data.dataset import LightCurveDataset
from models.encoder import EncoderCNN
from models.decoder import DecoderCNN
from models.losses import kl_divergence, linear_beta_scheduler, masked_mse
from data.dataset import LightCurveDataset

BASE = Path(__file__).resolve().parents[2]
DATA = BASE / "data" / "processed"

GAMMA = 0.7

class LitVAE(pl.LightningModule):
    def __init__(
        self,
        latent_dim: int = 32,
        lr: float = 1e-3,
        warmup_epochs: int = 10,
        beta_start: float = 0.1,
        beta_end: float = 0.25,
        kl_free_bits: float = 0.0,
        batch_size: int = 128,
        c_max: float = 6.0,
        stats_path = DATA / "stats.json",
        splits_path = DATA
    ):
        
        super().__init__()
        self.save_hyperparameters()
        
        torch.autograd.set_detect_anomaly(True)

        # Cargar stats
        with open(stats_path) as f:
            stats = json.load(f)
        
        # Preparar stats como antes
        self.meta_keys = list(stats["meta"].keys())
        mean_meta = torch.tensor([stats["meta"][k]["mean"] for k in self.meta_keys])
        std_meta = torch.tensor([stats["meta"][k]["std"] for k in self.meta_keys])
        
        # Registrar como buffers
        self.register_buffer("flux_mean", torch.tensor(stats["flux"]["mean"]))
        self.register_buffer("flux_std", torch.tensor(stats["flux"]["std"]))
        self.register_buffer("meta_mean", mean_meta)
        self.register_buffer("meta_std", std_meta)

        # Modelos
        # --- Encoder: (batch, T,6) -> mu, logvar
        self.encoder = EncoderCNN(latent_dim=latent_dim)

        # --- Decoder: (batch, T, latent_and_meta) -> (batch, T, 6)
        T_max = LightCurveDataset.T_max
        T0 = T_max // 4     # 4 = (2*2) -> encoder usa stride=2 2 veces
        self.decoder = DecoderCNN(latent_and_meta=latent_dim + len(self.meta_keys), T0=T0)

        # Crgar ids de los splits
        self.train_ids = np.load(splits_path / "train.npy")
        self.val_ids = np.load(splits_path / "val.npy") 

    def setup(self, stage=None):
        stats_tup = (
            self.flux_mean,   
            self.flux_std,    
            self.meta_mean,   
            self.meta_std,    
            self.meta_keys
        )
        self.train_ds = LightCurveDataset(self.train_ids, stats=stats_tup)
        self.val_ds = LightCurveDataset(self.val_ids, stats=stats_tup)

    def train_dataloader(self):
        train_loader = DataLoader(
            self.train_ds,
            batch_size=self.hparams.batch_size,
            shuffle=True,
            num_workers=14,  
            pin_memory=True,
            persistent_workers=True 
        )
        return train_loader
    
    def val_dataloader(self):
        val_loader = DataLoader(
            self.val_ds,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            num_workers=14,  
            pin_memory=True,
            persistent_workers=True 
        )
        return val_loader
    
    def forward(self, flux, mask, meta):

        # Encoder
        mu, logvar = self.encoder(flux)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)     # Generar ruido gaussiano
        z = mu + eps * std              # Reparametrization trick (clave para el VAE)

        # Decoder
        recon = self.decoder(torch.cat([z, meta], dim=-1))
        return recon, mu, logvar
    
    def training_step(self, batch, batch_idx):
        flux, mask, meta = batch
        recon, mu, logvar = self(flux, mask, meta)

        # Calcular pérdidas
        recon_loss = masked_mse(recon, flux, mask)
        kl_loss = kl_divergence(mu, logvar)
        
        # Capacity Annealing (Burgess et al., 2018):
        steps_per_epoch = self.trainer.num_training_batches
        warmup_epochs = self.hparams.warmup_epochs
        warmup_steps = steps_per_epoch * warmup_epochs

        c_max = self.hparams.c_max
        step = float(self.global_step)
        C_t = min(c_max, c_max * step / warmup_steps)

        # loss final
        loss = recon_loss + GAMMA * torch.abs(kl_loss - C_t)

        # loss = recon_loss + beta * kl_loss
        self.log("train/recon_loss", recon_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/kl_raw", kl_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/C_target", C_t, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=True)

        current_lr = self.optimizers().param_groups[0]['lr']
        self.log('train/lr', current_lr, on_step=False, on_epoch=True)
    
        sigma = torch.exp(0.5 * logvar)
        self.log('train/mu_abs_mean', mu.abs().mean(), on_step=True, on_epoch=True)
        self.log('train/sigma_mean', sigma.mean(), on_step=True, on_epoch=True)
    

        return loss
    
    def validation_step(self, batch, batch_idx):
        flux, mask, meta = batch
        recon, mu, logvar = self(flux, mask, meta)

        # 1) Reconstrucción y KL cruda
        recon_loss = masked_mse(recon, flux, mask)
        kl_raw = kl_divergence(mu, logvar)

        # 2) Recalcular C_t 
        steps_per_epoch = self.trainer.num_training_batches
        warmup_epochs = self.hparams.warmup_epochs
        warmup_steps = steps_per_epoch * warmup_epochs
        step = float(self.global_step)

        c_max = self.hparams.c_max
        C_t = min(c_max, c_max * step / float(warmup_steps))

        # 3) Val loss “canónica” con annealing
        val_loss = recon_loss + GAMMA * torch.abs(kl_raw - C_t)

        # 4) Logging
        self.log("val/recon_loss", recon_loss, on_epoch=True, prog_bar=True)
        self.log("val/kl_raw", kl_raw, on_epoch=True, prog_bar=True)
        self.log("val/C_target", C_t, on_epoch=True, prog_bar=True)
        self.log("val/loss", val_loss, on_epoch=True)

        sigma = torch.exp(0.5 * logvar)
    
        # Loggear estadísticas de validación
        self.log('val/mu_abs_mean', mu.abs().mean(), on_step=False, on_epoch=True)
        self.log('val/sigma_mean', sigma.mean(), on_step=False, on_epoch=True)
        

        return val_loss
        
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        scheduler1 = {
            'scheduler': torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100, eta_min=1e-6),
            'interval': 'epoch',
            'frequency': 1
        }
        scheduler2 = {
            'scheduler': torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=6, verbose=True),
            'monitor': 'val/recon_loss',
            'interval': 'epoch',
            'frequency': 1
        }
        steps_per_epoch = len(self.train_ds) // self.hparams.batch_size
        scheduler3 = {
            'scheduler': torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer,
                T_0=10 * steps_per_epoch,   # primer ciclo: 10 epochs
                T_mult=2,                   # dobla la longitud de cada ciclo
                eta_min=3e-5
            ),
            'interval': 'step',            # actualiza en cada batch
            'frequency': 1
        }
        return [optimizer], [scheduler3]