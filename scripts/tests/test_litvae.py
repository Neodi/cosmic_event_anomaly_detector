# tests/test_litvae_pipeline.py

import os
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

# Ajusta el path si hace falta para importar tu módulo
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT / "project" / "scripts"))

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from train.model_vae import LitVAE
from data.dataset   import LightCurveDataset

def test_litvae_end_to_end():
    # 1) Instanciar el modelo y preparar datasets
    print("1.🔍 Iniciando test LitVAE end-to-end pipeline...")
    model = LitVAE()
    model.setup(stage="fit")  # crea train_ds y val_ds

    # 2) Comprobar que los DataLoaders funcionan
    print("2.🔄 Comprobando DataLoaders...")
    train_dl = model.train_dataloader()
    batch = next(iter(train_dl))
    flux, mask, meta = batch

    assert isinstance(flux, torch.Tensor)
    assert isinstance(mask, torch.Tensor)
    assert isinstance(meta, torch.Tensor)

    # Shapes
    B, T, C = flux.shape
    assert C == 6, f"Esperaba 6 canales, vinieron {C}"
    assert mask.shape == (B, T, C)
    assert meta.shape[0] == B

    # 3) Hacer un forward manual
    print("3.🔄 Probando forward manual...")
    recon, mu, logvar = model(flux, mask, meta)
    # recon debe coincidir en shape con flux
    assert recon.shape == flux.shape
    # mu y logvar deben tener shape (B, latent_dim)
    D = model.hparams.latent_dim
    assert mu.shape == (B, D)
    assert logvar.shape == (B, D)

    # 4) Probar training_step
    print("4.🔄 Probando training_step...")
    loss = model.training_step(batch, batch_idx=0)
    assert torch.isfinite(loss), "La pérdida no debe ser NaN o Inf"

    # 5) Probar validation_step
    print("5.🔄 Probando validation_step...")
    val_loss = model.validation_step(batch, batch_idx=0)
    # validation_step no devuelve explícitamente, así que mejor comprobamos que no lanza error
    print("✅ LitVAE end-to-end pipeline OK")
    print(f"Loss train: {loss.item():.4f}")

if __name__ == "__main__":
    test_litvae_end_to_end()
