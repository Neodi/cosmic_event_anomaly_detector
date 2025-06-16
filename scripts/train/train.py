import argparse
import sys
from pathlib import Path

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

from pytorch_lightning.loggers import WandbLogger

ROOT = Path(__file__).resolve().parents[2] / "scripts"

from model_vae import LitVAE


def main():
    parser = argparse.ArgumentParser(description="Entrenar el VAE con el dataset PLAsTiCC")
    parser.add_argument("--latent-dim", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--warmup-epochs", type=int, default=10*3)
    parser.add_argument("--beta-start", type=float, default=0.1)
    parser.add_argument("--beta-end", type=float, default=0.25)  
    parser.add_argument("--kl-free-bits", type=float, default=0)  # 0.5 * latent_dim
    parser.add_argument("--C_max", type=float, default=12)  
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--max-epochs", type=int, default=200)
    parser.add_argument("--gpus", type=int, default=1)

    args = parser.parse_args()
    print("-------------------------------------")
    print(f"Parámetros de entrenamiento: {args}")
    print("-------------------------------------")


    model = LitVAE(
        batch_size=args.batch_size,
        latent_dim=args.latent_dim,
        warmup_epochs=args.warmup_epochs,
        beta_start=args.beta_start,
        beta_end=args.beta_end,
        lr=args.learning_rate,
        kl_free_bits=args.kl_free_bits,
        c_max=args.C_max,
    )

    # Callbacks
    checkpoint_cb = ModelCheckpoint(
        monitor="val/loss",
        # dirpath=ROOT / "checkpoints",
        filename="vae-{epoch}",
        save_top_k=3,
        mode="min",
    )

    early_stopping_cb = EarlyStopping(
        monitor="val/recon_loss",
        patience=13,
        mode="min",
        verbose=True,        
        check_finite=False   
    )

    # tensorboard --logdir=logs             desde la carpeta donde se ejecuta el script
    tensorboard_logger = pl.loggers.TensorBoardLogger( 
            save_dir="logs",
            name="vae_experiment"
    )
    
    wandb_logger = WandbLogger(
        project="vae_plasticc",
        name=f"vae_latent_{args.latent_dim}_batch_{args.batch_size}_lr_{args.learning_rate}",
        log_model=False, 
    )

    # Trainer
    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        devices=args.gpus if args.gpus > 0 else "auto", 
        accelerator="gpu" if args.gpus > 0 else "cpu",   
        callbacks=[checkpoint_cb, early_stopping_cb],  
        log_every_n_steps= 10,
        gradient_clip_val=1.0,
        precision="16-mixed",   # Limitar la precisión a 16 bits
        logger= [tensorboard_logger, wandb_logger], # 
        # accumulate_grad_batches=4, 
    )

    # Entrenar
    trainer.fit(model)

if __name__ == "__main__":
    main()