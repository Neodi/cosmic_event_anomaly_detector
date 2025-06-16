import warnings
import numpy as np
import json
from tqdm import tqdm
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

BASE = Path(__file__).resolve().parents[2]
NPZ = BASE / "data" / "processed" / "npz"
JSON = BASE / "data" / "processed" / "stats.json"

N_BANDS = 6
META = ['photo_z', 'photo_z_err', 'distmod', 'mwebv']

def process_file_stats(npz_file):
    """Procesa un archivo npz y retorna sus estadísticas"""
    with np.load(npz_file) as data:
        flux = data["flux"]
        mask = data["mask"]
        
        # Estadísticas de flujo
        flux_masked = flux * mask
        sum_flux = flux_masked.sum(axis=0)
        count_flux = mask.sum(axis=0)
        
        # Metadatos
        meta_values = np.array([float(data[key]) for key in META])
        
        return sum_flux, count_flux, meta_values

def process_file_variance(args):
    """Procesa un archivo para calcular varianza"""
    npz_file, mean_flux, mean_meta = args
    
    with np.load(npz_file) as data:
        flux = data["flux"]
        mask = data["mask"]
        
        # Varianza de flujo
        sum_sq_flux = np.zeros(N_BANDS, dtype=np.float64)
        for b in range(N_BANDS):
            if not np.isnan(mean_flux[b]):
                diff = (flux[:, b] - mean_flux[b]) * mask[:, b]
                sum_sq_flux[b] = (diff ** 2).sum()
        
        # Varianza de metadatos
        meta_values = np.array([float(data[key]) for key in META])
        sum_sq_meta = (meta_values - mean_meta) ** 2
        
        return sum_sq_flux, sum_sq_meta

def main():
    """Función principal que ejecuta el cálculo de estadísticas"""
    # Recogemos todos los archivos npz
    npz_files = list(NPZ.glob("**/*.npz"))
    print(f"Encontrados {len(npz_files)} archivos .npz")

    # Procesamiento paralelo para medias
    n_cores = min(mp.cpu_count(), len(npz_files))
    sum_flux = np.zeros(N_BANDS, dtype=np.float64)
    count_flux = np.zeros(N_BANDS, dtype=np.int64)
    sum_meta = np.zeros(len(META), dtype=np.float64)
    count_meta = 0

    print("Calculando medias en paralelo...")
    with ProcessPoolExecutor(max_workers=n_cores) as executor:
        futures = [executor.submit(process_file_stats, f) for f in npz_files]
        
        for future in tqdm(as_completed(futures), total=len(futures)):
            file_sum_flux, file_count_flux, file_meta = future.result()
            sum_flux += file_sum_flux
            count_flux += file_count_flux
            sum_meta += file_meta
            count_meta += 1

    # Cálculo de medias
    mean_flux = np.where(count_flux > 0, sum_flux / count_flux, np.nan)
    mean_meta = sum_meta / count_meta if count_meta > 0 else np.full(len(META), np.nan)

    # Procesamiento paralelo para varianzas
    sum_sq_flux = np.zeros(N_BANDS, dtype=np.float64)
    sum_sq_meta = np.zeros(len(META), dtype=np.float64)

    print("Calculando varianzas en paralelo...")
    variance_args = [(f, mean_flux, mean_meta) for f in npz_files]

    with ProcessPoolExecutor(max_workers=n_cores) as executor:
        futures = [executor.submit(process_file_variance, args) for args in variance_args]
        
        for future in tqdm(as_completed(futures), total=len(futures)):
            file_sum_sq_flux, file_sum_sq_meta = future.result()
            sum_sq_flux += file_sum_sq_flux
            sum_sq_meta += file_sum_sq_meta

    # Cálculo de desviaciones
    std_flux = np.where(count_flux > 0, np.sqrt(np.maximum(0, sum_sq_flux / count_flux)), np.nan)
    std_meta = np.sqrt(np.maximum(0, sum_sq_meta / count_meta)) if count_meta > 0 else np.full(len(META), np.nan)

    # Creación del diccionario de estadísticas
    flux_stats = {
        "mean": [m if not np.isnan(m) else None for m in mean_flux],
        "std":  [s if not np.isnan(s) else None for s in std_flux]
    }

    meta_stats = {}
    for i, key in enumerate(META):
        meta_stats[key] = {
            "mean": mean_meta[i].item() if not np.isnan(mean_meta[i]) else None,
            "std":  std_meta[i].item() if not np.isnan(std_meta[i]) else None
        }

    stats = {"flux": flux_stats, "meta": meta_stats}

    JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(JSON, "w") as f:
        json.dump(stats, f, indent=4)

    print(f"Estadísticas guardadas en {JSON}")

if __name__ == '__main__':
    mp.freeze_support()  
    main()