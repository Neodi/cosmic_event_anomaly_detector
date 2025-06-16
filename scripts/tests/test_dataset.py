import numpy as np
import torch
from torch.utils.data import DataLoader
from pathlib import Path
import sys
import os
# importa tu Dataset
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.dataset import LightCurveDataset

# ---------- parámetros que ya tienes ----------
BASE = Path(__file__).resolve().parents[2]
STATS_JSON = BASE / "data/processed/stats.json"
NPZ_ROOT   = BASE / "data/processed/npz"
TRAIN_IDS  = np.load(BASE / "data/processed/train.npy")

# ---------- preparar stats ----------
import json
stats = json.load(open(STATS_JSON))
mean_flux = torch.tensor(stats["flux"]["mean"])
std_flux  = torch.tensor(stats["flux"]["std"])
meta_keys = list(stats["meta"].keys())
mean_meta = torch.tensor([stats["meta"][k]["mean"] for k in meta_keys])
std_meta  = torch.tensor([stats["meta"][k]["std"]  for k in meta_keys])
stats_tup = (mean_flux, std_flux, mean_meta, std_meta, meta_keys)

# ---------- Dataset y DataLoader ----------
ds  = LightCurveDataset(TRAIN_IDS[:200], stats_tup, root=NPZ_ROOT)  # usa 200 ids de prueba
dl  = DataLoader(ds, batch_size=8, shuffle=True, num_workers=0)

# ---------- obtener un batch ----------
flux, mask, meta = next(iter(dl))

print("Shapes  -> flux:", flux.shape, "mask:", mask.shape, "meta:", meta.shape)
assert flux.shape == (8, ds.T_max, 6)
assert mask.shape == (8, ds.T_max, 6)
assert meta.shape == (8, len(meta_keys))

# ---------- checks de normalización ----------
# Valores donde mask==1 deberían estar aproximadamente en N(0,1)
valid = mask.bool()
sample_vals = flux[valid][:1000]      # 1000 valores aleatorios
print("Media ~", sample_vals.mean().item(), "  Desv ~", sample_vals.std().item())

# ---------- check de padding ----------
# Todas las posiciones donde mask==0 deberían ser *exactamente* 0 en flux
assert torch.all(flux[mask == 0] == 0), "Padding en flux no es 0"

print("✅ DataLoader test passed")