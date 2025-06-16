"""
Une, limpia y guarda training_set(+metadata) ⇒ clean.pkl (pickle en streaming)

Este script procesa los CSV de observaciones en dos fases:
  1) Calcula, por chunk, el mjd mínimo por objeto.
  2) Por chunk, lee de nuevo, filtra, calcula delta_t, une metadatos y hace 
     pickle.dump() de cada trozo en el mismo archivo .pkl.
"""

from pathlib import Path
import pandas as pd
import pickle
import argparse

pd.options.mode.chained_assignment = None

BASE = Path(__file__).resolve().parents[2]
RAW = BASE / "data" / "raw"
OUT = RAW / "challenge_train.pkl"

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
}

parser = argparse.ArgumentParser(description="Une, limpia y guarda training_set + metadata en .pkl")
parser.add_argument("--obs", required=True, nargs='+', help="Rutas a los CSV de observaciones")
parser.add_argument("--meta", required=True, help="Ruta al CSV de metadata")
parser.add_argument("--chunksize", type=int, default=1_000_000, help="Filas por chunk")
args = parser.parse_args()

# Calcular mjd mínimo por object_id
print("→ Fase 1: cálculo de mjd mínimo por objeto")
min_mjd = {}

for fn in args.obs:
    print(f"   Procesando (fase 1) {Path(fn).name} ...")
    for chunk in pd.read_csv(fn, dtype=dtype_obs,
                             usecols=["object_id", "mjd"],
                             chunksize=args.chunksize):
        grp = chunk.groupby("object_id", sort=False)["mjd"].min()
        for obj, m in grp.items():
            # guardamos el mínimo global
            if obj in min_mjd:
                if m < min_mjd[obj]:
                    min_mjd[obj] = m
            else:
                min_mjd[obj] = m

min_mjd = pd.Series(min_mjd, name="min_mjd")
print(f"   → Objetos únicos: {len(min_mjd):,}")

# Leer metadata y añadir min_mjd como columna
print(f"→ Leyendo metadatos desde {Path(args.meta).name} ...")
meta = pd.read_csv(args.meta, dtype=dtype_meta)
meta = meta.set_index("object_id").join(min_mjd, how="left").reset_index()
print(f"   → Metadatos tras merge: {len(meta):,} filas")

# Procesar, filtrar y volcar por chunks en pickle
print(f"→ Fase 2: limpieza por chunks y escritura en {OUT.name}")
with open(OUT, "wb") as fout:
    for fn in args.obs:
        print(f"   Procesando (fase 2) {Path(fn).name} ...")
        for chunk in pd.read_csv(fn, dtype=dtype_obs, chunksize=args.chunksize):
            # Unir metadata y min_mjd
            chunk = chunk.merge(meta, on="object_id", how="left", copy=False)

            # Filtrado temprano
            mask_bad = (chunk["flux_err"] == 0) | (chunk["flux"].isna())
            chunk = chunk.loc[~mask_bad]

            # Calcular delta_t
            chunk["delta_t"] = (chunk["mjd"] - chunk["min_mjd"]).astype("float32")

            # Eliminar auxiliar
            chunk.drop(columns=["min_mjd"], inplace=True)

            # Dump del trozo limpio
            pickle.dump(chunk, fout, protocol=pickle.HIGHEST_PROTOCOL)

print(f"✔ Proceso completado. Datos limpios en {OUT}")
