import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path

from torch.nn import functional as F

BASE = Path(__file__).resolve().parents[2]
ROOT = BASE / "data" / "processed" / "npz"
class LightCurveDataset(Dataset):
    """
    Dataset que carga archivos .npz de curvas de luz,
    normaliza flux y metadatos con stats.json
    y aplica padding/trim a longitud fija T_max.

    Devuelve (flux_norm, mask, meta_norm) como tensores.

    Parámetros
    ----------
    object_ids : np.ndarray (N,)
        Lista de IDs que componen el split (train / val / test).
    stats      : tuple(torch.Tensor) ó dict
        (mean_flux, std_flux, mean_meta, std_meta, meta_keys)
        pasados desde LitVAE.setup().
    root       : str | Path
        Carpeta raíz donde están los archivos .npz (default: data/processed/npz).
    """

    T_max = 256

    def __init__(self, object_ids, stats, root: str | Path = ROOT):
        
        self.object_ids = object_ids.astype(int)
        self.root = Path(root)

        self.mean_flux, self.std_flux, self.mean_meta, self.std_meta, self.meta_keys = stats
        # Convertir stats a tensores
        self.mean_flux = self.mean_flux.float()
        self.std_flux  = self.std_flux.float()
        self.mean_meta = self.mean_meta.float()
        self.std_meta  = self.std_meta.float()

        if isinstance(self.mean_flux, torch.Tensor):
            self.device = self.mean_flux.device
        else:
            print("Warning: stats are not tensors, using CPU.")
            self.device = torch.device("cpu")

    def __len__(self):
        return len(self.object_ids)
    
    def _id_to_path(self, obj_id: int) -> Path:
        """
        Convierte 123456 → data/processed/npz/01234/123456.npz
        """
        subdir = f"{obj_id//10000:05d}"
        return self.root / subdir / f"{obj_id}.npz"
    
    def __getitem__(self, idx):
        object_id = self.object_ids[idx]
        npz_path = self._id_to_path(object_id)

        
        with np.load(npz_path) as data:
            flux = torch.from_numpy(data["flux"].astype(np.float32)).to(self.device)
            mask = torch.from_numpy(data["mask"].astype(np.float32)).to(self.device)
            meta_vals = [float(data[k]) for k in self.meta_keys]
            meta = torch.tensor(meta_vals, dtype=torch.float32).to(self.device)

        # Normalizar flux
        flux = (flux - self.mean_flux) / self.std_flux
        flux = flux * mask  # Aplicar máscara para que no afecte el padding
        
        # Padding/Trimming a T_max
        T_i = flux.size(0)
        if T_i > self.T_max:
            # Trimming
            flux = flux[:self.T_max]
            mask = mask[:self.T_max]
        elif T_i < self.T_max:
            # Padding
            pad_size = self.T_max - T_i
            pad_flux = torch.zeros(pad_size, 6).to(self.device)
            pad_mask = torch.zeros(pad_size, 6).to(self.device)

            flux = torch.cat((flux, pad_flux), dim=0)
            mask = torch.cat((mask, pad_mask), dim=0)

        # Normalizar metadatos
        meta = (meta - self.mean_meta) / self.std_meta   # z-score
        
        return flux, mask, meta 