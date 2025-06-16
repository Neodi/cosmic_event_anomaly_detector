import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import umap
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.patches import Patch


BASE = Path(__file__).resolve().parents[1]
DATA_PROC = BASE / "data" / "processed"
FIG_DIR = BASE / "reports" / "figures" / "umap_multiclass"

CLASS_NAME = {
    90:  "SNIa",
    67:  "SNIa-91bg",
    52:  "SNIax",
    42:  "SNII",
    62:  "SNIbc",
    95:  "SLSN-I",
    15:  "TDE",
    64:  "KN",
    88:  "AGN",
    92:  "RRL",
    65:  "M-dwarf",
    16:  "EB",
    53:  "Mira",
    6:   "μLens-Single",
    991: "μLens-Binary",
    992: "ILOT",
    993: "CaRT",
    994: "PISN",
    995: "μLens-String"
}

def load_data(split: str, data_dir: Path):

    latent_np = data_dir / f"latent_{split}.npy"
    idx_csv = data_dir / f"latent_{split}_idx.csv"
    inf_csv = data_dir / f"{split}_table_final.csv"

    X = np.load(latent_np)
    ids = pd.read_csv(idx_csv)["object_id"].values
    df_inf = pd.read_csv(inf_csv).set_index("object_id")
    y = df_inf.loc[ids, "true_target"].values
    return X, y



def plot_umap_classes(X, y, split: str, out_dir: Path):
    # UMAP supervisado: pasar 'y' y 'target_metric'
    reducer = umap.UMAP(n_components=2, 
                        random_state=42, 
                        n_neighbors=40,
                        min_dist=0.01,
                        target_metric="categorical",
                        )
    X_umap = reducer.fit_transform(X, y=y)

    classes = np.unique(y)
    n_classes = len(classes)
    cmap = plt.colormaps.get_cmap('tab20').resampled(n_classes)

    colors = {}
    labels = {}
    for i, cls in enumerate(classes):
        colors[cls] = cmap(i)
        labels[cls] = CLASS_NAME.get(cls, str(cls))  

    fig, ax = plt.subplots(figsize=(8, 6))
    for cls in classes:
        mask = (y == cls)
        ax.scatter(
            X_umap[mask, 0], X_umap[mask, 1],
            c=[colors[cls]], label=labels[cls], s=10, alpha=0.7
        )

    ax.set_title(f"UMAP projection by class ({split})")
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")

    # Construir leyenda
    handles = [Patch(color=colors[cls], label=labels[cls]) for cls in classes]
    ax.legend(handles=handles, title='True class', bbox_to_anchor=(1.05,1), loc='upper left')

    plt.tight_layout()
    out_path = out_dir / f"umap_{split}_classes.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"UMAP multiclass (con nombres) {split} guardado en {out_path}")


def main():
    p = argparse.ArgumentParser(
        description="UMAP multiclass de espacios latentes"
    )
    p.add_argument(
        "--data_dir", type=Path, default=DATA_PROC,
    )
    p.add_argument(
        "--out_dir", type=Path, default=FIG_DIR,
    )
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for split in ("val", "test"):
        X, y = load_data(split, args.data_dir)
        plot_umap_classes(X, y, split, args.out_dir)

if __name__ == '__main__':
    main()
