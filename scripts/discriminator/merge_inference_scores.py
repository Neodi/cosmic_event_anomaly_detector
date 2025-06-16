import argparse
import json
from pathlib import Path

import pandas as pd

# {
#     "score_threshold_p1": -0.7306759199282322,
#     "score_threshold_p3": -0.6100732574275483,
#     "score_threshold_p5": -0.5320583139424371,
#     "score_threshold_p10": -0.4269160492575467
# }

# {
#     "chi2_threshold_90": 964077.1625,
#     "chi2_pp_threshold_90": 3944.4415283203134,
#     "chi2_threshold_95": 1393725.799999999,
#     "chi2_pp_threshold_95": 5636.728640950521,
#     "chi2_threshold_97.5": 1941402.2812499995,
#     "chi2_pp_threshold_97.5": 8005.237792490037,
#     "chi2_threshold_99": 3125152.144999985,
#     "chi2_pp_threshold_99": 12402.524911764689,
#     "chi2_threshold_99.5": 4349586.249999996,
#     "chi2_pp_threshold_99.5": 17558.118404641486
# }

def get_threshold(thr_json: Path, key: str | None):
    thr_data = json.load(open(thr_json, 'r'))
    for key_ex, value in thr_data.items():
        print(f"  {key_ex}: {value}")
    if key is None:
        # Último key ordenado ⇒ percentil más alto
        key = sorted(thr_data.keys())[-1]
    return thr_data[key], key


BASE = Path(__file__).resolve().parents[2]
DATA_PROC = BASE / "data" / "processed"
MODELS_DIR = BASE / "models"

CHI2_THR_JSON = DATA_PROC / "val_percentiles.json"
IF_THR_JSON = MODELS_DIR / "threshold_iso.json"

CHI2_KEY = "chi2_pp_threshold_97.5"  # Por defecto, percentil 97.5
IF_KEY = "score_threshold_p3"  # Por defecto, percentil 3

def main():
    p = argparse.ArgumentParser(description='Fusiona χ² + IF y etiqueta anomalías')
    p.add_argument('--split', choices=['val', 'test'], required=True)

    chi2_csv = DATA_PROC / f"{p.parse_args().split}_chi2.csv"
    if_csv = DATA_PROC / f"{p.parse_args().split}_if_scores.csv"
    out_csv = DATA_PROC / f"{p.parse_args().split}_table_final.csv"

    # Cargar CSVs
    chi2_df = pd.read_csv(chi2_csv)
    if_df = pd.read_csv(if_csv)

    # Cargar umbrales
    chi2_thr, chi2_key = get_threshold(CHI2_THR_JSON, CHI2_KEY)
    
    # Cargar y mostrar valores del JSON de Isolation Forest
    if_data = json.load(open(IF_THR_JSON, 'r'))    
    if_thr, if_key = get_threshold(IF_THR_JSON, IF_KEY)

    # Fusionar DataFrames
    df = chi2_df.merge(if_df, on="object_id")
    df["anom_chi2"] = (df["chi2_pp"] > chi2_thr).astype(int)
    df["anom_if"] = (df["score_if"] < if_thr).astype(int)
    df["is_event"] = (df["anom_chi2"] | df["anom_if"]).astype(int)

    # Guardar CSV final
    df.to_csv(out_csv, index=False)

    print("\nMergeado:")
    print(f"  χ² csv:  {chi2_csv}")
    print(f"  IF csv:  {if_csv}")
    print(f"  χ² thr:  {chi2_thr:.4f}  ({chi2_key})")
    print(f"  IF  thr: {if_thr:.4f}  ({if_key})")
    print(f"\n\nTabla final: {out_csv}\n")

if __name__ == "__main__":
    main()