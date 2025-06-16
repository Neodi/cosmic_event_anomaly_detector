import numpy as np
import pandas as pd
import joblib
import json
from pathlib import Path
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler


BASE = Path(__file__).resolve().parents[2]
DATA_PROC = BASE / "data" / "processed"
LATENT_NP = DATA_PROC / "latent_train.npy"
MODEL_OUT = BASE  / "models" / "iso_model.pkl"
THRESH_JSON = BASE  / "models" / "threshold_iso.json"

def load_latent(split: str, data_dir: Path):
    X = np.load(data_dir / f"latent_{split}.npy")
    ids = pd.read_csv(data_dir / f"latent_{split}_idx.csv")["object_id"].values
    return X, ids

def train_if(X_train, n_estimators=200, contamination=0.05):
    scaler = StandardScaler().fit(X_train)
    X_std = scaler.transform(X_train)
    iso = IsolationForest(
        n_estimators=n_estimators,
        contamination=contamination,
        random_state=42,
        n_jobs=-1,
    ).fit(X_std)
    scores = iso.score_samples(X_std)
    return iso, scaler, scores

def percentiles(scores, ps=(1, 3, 5, 10)):
    return {f"score_threshold_p{p}": float(np.percentile(scores, p)) for p in ps}

def score_split(split, iso, scaler, data_dir: Path, out_dir: Path):
    X, ids = load_latent(split, data_dir)
    scores = iso.score_samples(scaler.transform(X))
    out = pd.DataFrame({"object_id": ids, "score_if": scores})
    out_path = out_dir / f"{split}_if_scores.csv"
    out.to_csv(out_path, index=False)
    print(f"Scores para {split} guardados en {out_path}")
    return scores

def main():

    X, ids = load_latent("train", DATA_PROC)
    print(f"Latent matrix {X.shape}")

    iso, scaler, scores = train_if(X)
    print(f"Modelo IsolationForest entrenado con {len(scores)} muestras.")
    print(f"Umbral de contaminación: {iso.contamination:.4f}")

    # Calcular varios umbrales
    thresholds = percentiles(scores)


    joblib.dump({
        "iso": iso,
        "scaler": scaler,
    }, MODEL_OUT)

    json.dump(thresholds, open(THRESH_JSON, 'w'), indent=4)
    print(f"Umbrales guardados en {THRESH_JSON}")
    print(f"Modelo guardado en {MODEL_OUT}")

    # Evaluar en splits
    for split in ["val", "test"]:
        scores = score_split(split, iso, scaler, DATA_PROC, DATA_PROC)
        print(f"Umbral p5 para {split}: {thresholds['score_threshold_p5']:.4f}")

if __name__ == "__main__":
    main()