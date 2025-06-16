import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import umap
import matplotlib.pyplot as plt

BASE = Path(__file__).resolve().parents[1]
DATA_PROC = BASE / "data" / "processed"
FIG_DIR = BASE / "reports" / "figures" / "umap"


def load_data(split: str, data_dir: Path):

    latent_np = data_dir / f"latent_{split}.npy"
    idx_csv = data_dir / f"latent_{split}_idx.csv"
    inf_csv = data_dir / f"{split}_table_final.csv"

    # Carga latentes
    X = np.load(latent_np)

    # Carga índice de object_id para alinear filas
    ids = pd.read_csv(idx_csv)["object_id"].values

    # Carga tabla de inferencia y extrae la etiqueta
    df_inf = pd.read_csv(inf_csv).set_index("object_id")
    y = df_inf.loc[ids, "is_event"].values
    z = df_inf.loc[ids, "true_target"].values

    return X, y, z

def plot_umap(X, y, z, split, out_dir: Path):

    reducer = umap.UMAP(n_components=2, 
                        random_state=42, 
                        n_neighbors=40,
                        min_dist=0.01,
                        target_metric="categorical",
                        )
    X_umap = reducer.fit_transform(X,  y=z)

    _, ax = plt.subplots(figsize=(6, 6))
    scatter = ax.scatter(
        X_umap[:, 0], X_umap[:, 1],
        c=y, cmap="coolwarm", s=5, alpha=0.3
    )
    ax.set_title(f"UMAP projection ({split})")
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    # Leyenda manual
    handles, _ = scatter.legend_elements()
    labels = ["known", "unknown"]
    ax.legend(handles, labels, title="Event type", loc="best")
    plt.tight_layout()

    out_path = out_dir / f"umap_{split}.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"UMAP {split} saved to {out_path}")


def main():
    p = argparse.ArgumentParser(
        description="Visualizar UMAP de espacios latentes (val y test)"
    )
    p.add_argument(
        "--data_dir", type=Path, default=DATA_PROC,
    )
    p.add_argument(
        "--out_dir", type=Path, default=FIG_DIR
    )
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    for split in ("val", "test"):
        X, y, z = load_data(split, args.data_dir)
        plot_umap(X, y, z, split, args.out_dir)


if __name__ == "__main__":
    main()