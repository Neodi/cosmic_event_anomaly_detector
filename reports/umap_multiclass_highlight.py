#!/usr/bin/env python
"""
umap_multiclass_highlight.py

UMAP 2D de los espacios latentes para val/test, y subplots que
resaltan cada clase individualmente contra el fondo gris.

Salida:
  reports/figures/umap_highlight/umap_{split}_highlight.png
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import umap
import matplotlib.pyplot as plt
from matplotlib import cm

BASE = Path(__file__).resolve().parents[1]
DATA_PROC = BASE / "data" / "processed"
FIG_DIR = BASE / "reports" / "figures" / "umap_highlight"

# Mapa de clase → nombre (igual que antes)
CLASS_NAME = {
    90:  "SNIa",     67:  "SNIa-91bg", 52:  "SNIax",    42:  "SNII",
    62:  "SNIbc",    95:  "SLSN-I",    15:  "TDE",      64:  "KN",
    88:  "AGN",      92:  "RRL",       65:  "M-dwarf",  16:  "EB",
    53:  "Mira",     6:   "μLens-Single", 
    991: "μLens-Binary", 992: "ILOT",   993: "CaRT",
    994: "PISN",     995: "μLens-String"
}

def load_data(split: str, data_dir: Path):
    X = np.load(data_dir / f"latent_{split}.npy")
    idx = pd.read_csv(data_dir / f"latent_{split}_idx.csv")["object_id"].values
    df_inf = pd.read_csv(data_dir / f"{split}_table_final.csv").set_index("object_id")
    y = df_inf.loc[idx, "true_target"].values
    return X, y

def plot_highlight(X2d, y, split, out_dir: Path, n_cols=4, marker_small=2, marker_big=10):
    # prepara la figura
    classes = np.unique(y)
    n = len(classes)
    n_rows = (n + n_cols - 1)//n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 4*n_rows), squeeze=False)
    cmap = cm.get_cmap("tab20", n)

    for i, cls in enumerate(classes):
        r = i // n_cols
        c = i % n_cols
        ax = axes[r][c]
        # fondo gris
        mask_bg = y != cls
        ax.scatter(X2d[mask_bg,0], X2d[mask_bg,1],
                   c="lightgray", s=marker_small, alpha=0.3, linewidths=0)
        # puntos de la clase
        mask_fg = y == cls
        ax.scatter(X2d[mask_fg,0], X2d[mask_fg,1],
                   c=[cmap(i)], s=marker_big, alpha=0.8, linewidths=0,
                   label=CLASS_NAME.get(cls, str(cls)))
        ax.set_title(CLASS_NAME.get(cls,str(cls)))
        ax.set_xticks([]); ax.set_yticks([])
    # elimina ejes vacíos
    for j in range(i+1, n_rows*n_cols):
        fig.delaxes(axes[j//n_cols][j%n_cols])

    plt.suptitle(f"UMAP highlight classes ({split})", fontsize=16)
    plt.tight_layout(rect=[0,0,1,0.96])
    out = out_dir / f"umap_{split}_highlight.png"
    out_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Guardado {out}")

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", type=Path, default=DATA_PROC)
    p.add_argument("--out_dir",  type=Path, default=FIG_DIR)
    p.add_argument("--split",    choices=["val","test","both"], default="both")
    p.add_argument("--n_cols",   type=int, default=4,
                   help="Número de columnas en la cuadrícula de subplots")
    args = p.parse_args()

    splits = ["val", "test"] if args.split == "both" else [args.split]
    
    for split in splits:

        X, y = load_data(split, args.data_dir)
        reducer = umap.UMAP(n_components=2, 
                            random_state=42, 
                            n_neighbors=40,
                            min_dist=0.01,
                            target_metric="categorical",
                            )
        X2d = reducer.fit_transform(X,  y=y)

        # 2) dibuja
        plot_highlight(X2d, y, split, args.out_dir, n_cols=args.n_cols)

if __name__=="__main__":
    main()
