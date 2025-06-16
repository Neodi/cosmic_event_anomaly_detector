"""
Une, limpia y guarda training_set(+metadata) ⇒ clean.pkl
"""

from pathlib import Path
import pandas as pd
import argparse

pd.options.mode.chained_assignment = None       


BASE = Path(__file__).resolve().parents[2]
RAW =  BASE / "data" / "raw"
OUT = RAW / "challenge_train_df.pkl"

# Cargar datos con dtypes explícitos 
dtype_obs = {
    "object_id": "int32",
    "mjd":        "float32",
    "passband":   "int8",
    "flux":       "float32",
    "flux_err":   "float32",
    "detected":   "int8",
}
dtype_meta = {
    "object_id":          "int32",
    "ra":                 "float32",
    "decl":               "float32",
    "gal_l":              "float32",
    "gal_b":              "float32",
    "ddf_bool":           "int8",
    "hostgal_specz":      "float32",
    "hostgal_photoz":     "float32",
    "hostgal_photoz_err": "float32",
    "distmod":            "float32",
    "mwebv":              "float32",
    "target":             "int16",
    "true_target":        "int16",
}

# Configurar argumentos de línea de comandos
parser = argparse.ArgumentParser(description="Une, limpia y guarda training_set + metadata")
parser.add_argument("--obs", required=True, help="Ruta al archivo CSV de observaciones")
parser.add_argument("--meta", required=True, help="Ruta al archivo CSV de metadatos")
args = parser.parse_args()

print(f"→ Leyendo CSV train desde {args.obs} y {args.meta}...")
obs = pd.read_csv(args.obs, dtype=dtype_obs)
meta = pd.read_csv(args.meta, dtype=dtype_meta)

print(f"    · {len(obs):,} filas en {Path(args.obs).name}")
print(f"    · {len(meta):,} filas en {Path(args.meta).name}")


# Merge sobre el id del objeto
df = obs.merge(meta, on="object_id", how="left", copy=False)
print(f"    · {len(df):,} filas tras merge")

# Eliminar columnas innecesarias
#   - flux_err == 0:    fotometría no pudo estimar la incertidumbre de esa medida
#   - flux NaN:         un hueco en la serie: nubes, satélites, fallos de CCD… no hay valor de brillo.
# mask_bad = (df["flux_err"] == 0) | (df["flux"].isna())
# n_bad = mask_bad.sum()
# df = df[~mask_bad].reset_index(drop=True)
# print(f"    · Filtradas {n_bad:,} filas con flux_err==0 o flux NaN")

# delta_t = tiempo transcurrido desde la primera observación del objeto hasta la actual
df["delta_t"] = df.groupby("object_id")["mjd"].transform(lambda x: x - x.min())

# Guardar datos limpios
df.to_pickle(OUT)
print(f"✔ Guardado {len(df):,} filas en {OUT}")
