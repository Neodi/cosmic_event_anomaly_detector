import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import multiprocessing as mp

from data.dataset import LightCurveDataset
from train.model_vae import LitVAE

def load_model(checkpoint_path: str) -> LitVAE:
    """Carga el modelo VAE entrenado y lo deja en modo eval en GPU."""
    model = LitVAE.load_from_checkpoint(checkpoint_path)
    model.eval().cuda()
    return model

def load_statistics(stats_path: str):
    """Carga estadísticas de normalización (mean/std) de flux y metadatos."""
    stats = json.load(open(stats_path, 'r'))
    flux_mean = torch.tensor(stats['flux']['mean'], dtype=torch.float32).cuda()
    flux_std = torch.tensor(stats['flux']['std'], dtype=torch.float32).cuda()

    meta_keys = list(stats['meta'].keys())
    mean_meta = torch.tensor([stats['meta'][k]['mean'] for k in meta_keys], dtype=torch.float32).cuda()
    std_meta = torch.tensor([stats['meta'][k]['std'] for k in meta_keys], dtype=torch.float32).cuda()
    return flux_mean, flux_std, mean_meta, std_meta, meta_keys

def create_dataset(split_ids_file: str, stats_tuple: tuple, npz_root: str) -> LightCurveDataset:
    """Crea LightCurveDataset para los IDs de la partición dada."""
    ids = np.load(split_ids_file)
    return LightCurveDataset(object_ids=ids, stats=stats_tuple, root=Path(npz_root))

def load_target_info(meta_csv: str):
    """Carga mapas de target y true_target (opcionales)."""
    target_map, true_map = {}, {}
    p = Path(meta_csv)
    if p.exists():
        dfm = pd.read_csv(p, usecols=['object_id', 'target', 'true_target'])
        target_map = dict(zip(dfm['object_id'].values, dfm['target'].values))
        true_map = dict(zip(dfm['object_id'].values, dfm['true_target'].values))
    return target_map, true_map

def preload_flux_errors(dataset: LightCurveDataset, T_max: int) -> dict:
    """Pre-carga todos los flux_err en memoria para evitar I/O repetitivo."""
    print("Pre-cargando flux_err en memoria...")
    flux_errors = {}
    
    def load_single_flux_err(obj_id):
        npz_path = dataset._id_to_path(obj_id)
        data_npz = np.load(npz_path)
        flux_err_raw = data_npz['flux_err'].astype(np.float32)
        
        # Pad/truncate a T_max
        pad = np.zeros((T_max, 6), dtype=np.float32)
        Ti = flux_err_raw.shape[0]
        if Ti > T_max:
            pad = flux_err_raw[:T_max]
        else:
            pad[:Ti] = flux_err_raw[:min(Ti, T_max)]
        return obj_id, pad
    
    # Cargar en paralelo
    with ThreadPoolExecutor(max_workers=mp.cpu_count()) as executor:
        results = list(tqdm(
            executor.map(load_single_flux_err, dataset.object_ids),
            total=len(dataset.object_ids),
            desc="Pre-cargando flux_err"
        ))
    
    flux_errors = dict(results)
    return flux_errors

def calculate_chi2_batch(
    model: LitVAE,
    flux_batch: torch.Tensor,
    mask_batch: torch.Tensor,
    meta_batch: torch.Tensor,
    obj_ids: list,
    flux_errors_dict: dict,
    flux_std: torch.Tensor,
) -> list:
    """Calcula χ² para un batch completo de objetos."""
    batch_size = flux_batch.shape[0]
    
    with torch.no_grad():
        flux_rec_batch, _, _ = model(flux_batch, mask_batch, meta_batch)
    
    results = []
    for i in range(batch_size):
        obj_id = obj_ids[i]
        flux_norm = flux_batch[i]
        mask = mask_batch[i]
        flux_rec = flux_rec_batch[i]
        
        # Usar flux_err pre-cargado
        flux_err_pad = torch.from_numpy(flux_errors_dict[obj_id]).cuda()
        
        sigma = flux_err_pad / flux_std.unsqueeze(0)
        diff2 = (flux_rec - flux_norm).pow(2)
        denom = sigma.pow(2).clamp(min=1e-6)
        chi2_map = diff2 / denom
        chi2_masked = chi2_map * mask
        
        chi2_val = chi2_masked.sum().item()
        n_valid = int(mask.sum().item())
        chi2_pp = chi2_val / n_valid if n_valid > 0 else float('nan')
        
        results.append((obj_id, chi2_val, chi2_pp, n_valid))
    
    return results

def main():
    parser = argparse.ArgumentParser(description='Calcula χ² y χ² por punto para un split y extrae percentiles.')
    parser.add_argument('split', choices=['train', 'val', 'test'], help='Partición a procesar')
    parser.add_argument('--checkpoint', default='../../scripts/train/logs/vae_experiment/version_6/checkpoints/vae-epoch=21-val_loss=0.0000.ckpt')
    parser.add_argument('--stats', default='../../data/processed/stats.json')
    parser.add_argument('--splits_dir', default='../../data/processed')
    parser.add_argument('--meta', default='../../data/raw/plasticc_test_metadata.csv')
    parser.add_argument('--percentiles', default='90,95,97.5,99,99.5', help='Lista de percentiles CSV')
    parser.add_argument('--batch_size', type=int, default=4096*2, help='Batch size para procesamiento')
    parser.add_argument('--out_csv', default=None, help='Ruta de salida CSV')
    parser.add_argument('--out_json', default=None, help='Ruta de salida JSON')
    args = parser.parse_args()

    # Derivar rutas si no se dan
    splits_dir = Path(args.splits_dir)
    split_ids = splits_dir / f"{args.split}.npy"
    npz_root = splits_dir / 'npz'
    out_csv = Path(args.out_csv) if args.out_csv else splits_dir / f"{args.split}_chi2.csv"
    out_json = Path(args.out_json) if args.out_json else splits_dir / f"{args.split}_percentiles.json"


    model = load_model(args.checkpoint)
    
    flux_mean, flux_std, mean_meta, std_meta, meta_keys = load_statistics(args.stats)
    stats_tuple = (flux_mean, flux_std, mean_meta, std_meta, meta_keys)
    
    dataset = create_dataset(str(split_ids), stats_tuple, str(npz_root))
    
    target_map, true_map = load_target_info(args.meta)

    # Pre-cargar flux_errors
    T_max = model.hparams.get('T_max', 256)  # Ajustar según tu modelo
    flux_errors_dict = preload_flux_errors(dataset, T_max)

    # Crear DataLoader para procesamiento por batches
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,  
        pin_memory=False  
    )

    percentiles = [float(p) for p in args.percentiles.split(',')]
    all_results = []

    print("Calculando χ² por batches...")
    for batch_idx, (flux_batch, mask_batch, meta_batch) in enumerate(tqdm(dataloader, desc=f"Procesando batches ({args.split})")):
        # Mover a GPU
        flux_batch = flux_batch.cuda()
        mask_batch = mask_batch.cuda()
        meta_batch = meta_batch.cuda()
        
        # Obtener object_ids para este batch
        start_idx = batch_idx * args.batch_size
        end_idx = min(start_idx + args.batch_size, len(dataset))
        obj_ids = [int(dataset.object_ids[i]) for i in range(start_idx, end_idx)]
        
        # Calcular χ² para el batch
        batch_results = calculate_chi2_batch(
            model, flux_batch, mask_batch, meta_batch,
            obj_ids, flux_errors_dict, flux_std
        )
        
        # Convertir a formato final
        for obj_id, chi2_val, chi2_pp, n_valid in batch_results:
            all_results.append({
                'object_id': obj_id,
                'chi2': chi2_val,
                'chi2_pp': chi2_pp,
                'n_valid': n_valid,
                'target': target_map.get(obj_id, -1),
                'true_target': true_map.get(obj_id, -1),
            })

    # Guardar CSV
    df = pd.DataFrame(all_results)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"Guardado CSV: {out_csv} con {len(df)} filas.")

    # Calcular percentiles
    thresholds = {}
    chi2_values = df['chi2'].values
    chi2_pp_values = df['chi2_pp'].values
    
    for p in percentiles:
        key_suffix = int(p) if p.is_integer() else p
        thresholds[f'chi2_threshold_{key_suffix}'] = float(np.percentile(chi2_values, p))
        thresholds[f'chi2_pp_threshold_{key_suffix}'] = float(np.percentile(chi2_pp_values, p))

    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, 'w') as f:
        json.dump(thresholds, f, indent=4)
    print(f"Guardado JSON percentiles: {out_json}")

if __name__ == '__main__':
    main()