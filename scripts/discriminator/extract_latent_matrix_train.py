from pathlib import Path
import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse

BASE = Path(__file__).resolve().parents[2]
DATA_PROCESSED = BASE / "data" / "processed"
NPZ_ROOT = DATA_PROCESSED / "npz"
STATS_PATH = DATA_PROCESSED / "stats.json"
CHECKPOINT = BASE / "scripts" / "train" / "logs" / "vae_experiment" / "version_1" / "checkpoints" / "vae-epoch=82-val_loss=0.0000.ckpt"
BATCH_SIZE = 256

SPLIT_FILES = {
    "train": DATA_PROCESSED / "train.npy",
    "test": DATA_PROCESSED / "test.npy",
    "val": DATA_PROCESSED / "val.npy"
}

from data.dataset    import LightCurveDataset
from train.model_vae import LitVAE

def load_model_and_stats(checkpoint=CHECKPOINT, stats_path=STATS_PATH):
    """Load VAE model and statistics."""
    model = LitVAE.load_from_checkpoint(str(checkpoint))
    model.eval().cuda()
    latent_dim = model.hparams.latent_dim
    print(f"Modelo cargado. Latent dim = {latent_dim}")

    stats = json.load(open(stats_path))
    flux_mean = torch.tensor(stats["flux"]["mean"]).cuda()
    flux_std  = torch.tensor(stats["flux"]["std"]).cuda()
    meta_keys = list(stats["meta"].keys())
    meta_mean = torch.tensor([stats["meta"][k]["mean"] for k in meta_keys]).cuda()
    meta_std  = torch.tensor([stats["meta"][k]["std" ] for k in meta_keys]).cuda()
    
    stats_tup = (flux_mean, flux_std, meta_mean, meta_std, meta_keys)
    return model, latent_dim, stats_tup

def create_dataloader(split_ids, stats_tup):
    """Create dataset and dataloader."""
    dataset = LightCurveDataset(split_ids, stats=stats_tup, root=NPZ_ROOT)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE,
                       shuffle=False, 
                       num_workers=0,        
                       pin_memory=False)     
    return dataset, loader

def extract_latent_features(model, loader, latent_dim, dataset_size):
    """Extract latent features using the VAE encoder."""
    latent_mat = np.zeros((dataset_size, latent_dim), dtype=np.float32)
    cursor = 0
    
    with torch.no_grad():
        for flux, mask, meta in tqdm(loader, desc="Extrayendo µ"):
            flux = flux.cuda(non_blocking=True)
            mask = mask.cuda(non_blocking=True)
            meta = meta.cuda(non_blocking=True)

            mu, _ = model.encoder(flux)
            latent_mat[cursor:cursor+mu.size(0)] = mu.cpu().numpy()
            cursor += mu.size(0)
    
    return latent_mat

def save_results(latent_mat, split_ids, split):
    """Save latent matrix and object IDs."""
    latent_out_np = DATA_PROCESSED / f"latent_{split}.npy"
    latent_out_csv = DATA_PROCESSED / f"latent_{split}_idx.csv"
    
    np.save(latent_out_np, latent_mat)
    pd.DataFrame({"object_id": split_ids}).to_csv(latent_out_csv, index=False)
    print(f"Guardado {latent_out_np}  shape={latent_mat.shape}")
    print(f"Guardado {latent_out_csv}")

def main():
    """Main function to extract latent features from data."""
    parser = argparse.ArgumentParser(description="Extract latent features")
    parser.add_argument("split", choices=["train", "test", "val"], 
                       help="Data split to process")
    
    parser.add_argument("--checkpoint", type=str, default=str(CHECKPOINT),
                       help="Path to model checkpoint")
    args = parser.parse_args()

    checkpoint_path = args.checkpoint
    # Load model and statistics
    model, latent_dim, stats_tup = load_model_and_stats( checkpoint=CHECKPOINT)
    
    # Load split IDs and create dataloader
    split_ids = np.load(SPLIT_FILES[args.split])
    dataset, loader = create_dataloader(split_ids, stats_tup)
    
    # Extract latent features
    latent_mat = extract_latent_features(model, loader, latent_dim, len(dataset))
    
    # Save results
    save_results(latent_mat, split_ids, args.split)

if __name__ == "__main__":
    main()
