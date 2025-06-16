import pandas as pd
import numpy as np
import json
from pathlib import Path
from tqdm import tqdm

def truncate_series(flux: np.ndarray,
                      flux_err: np.ndarray,
                      mask: np.ndarray,
                      T_max: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Trunca los arrays flux, flux_err y mask a T_max pasos temporales.
    Si alguna serie es más larga que T_max, la recorta; si es más corta, la deja intacta.
    """
    # Recorta hasta T_max
    flux_trunc = flux[:T_max]
    flux_err_trunc = flux_err[:T_max]
    mask_trunc = mask[:T_max]
    return flux_trunc, flux_err_trunc, mask_trunc

BASE = Path(__file__).resolve().parents[2]
RAW = BASE / "data" / "raw"
PROC = BASE / "data" / "processed" / "npz"
df = pd.read_pickle(RAW / "challenge_train_df.pkl")

bands = ["u", "g", "r", "i", "z", "y"]
total_objects = df['object_id'].nunique()

for i, (obj_id, subdf) in enumerate(tqdm(df.groupby("object_id"), total=total_objects, desc="Processing objects")):
    subdf = subdf.copy()

    # --- Tabla -> | indice = delta_t | columnas = bandas | valores = flux
    subdf["band"] = subdf["passband"].map(dict(enumerate(bands)))

    flux = subdf.pivot(index="delta_t", columns="band", values="flux")
    flux_err = subdf.pivot(index="delta_t", columns="band", values="flux_err")

    # --- Reindex para dejar los mismos delta_t en todas las bandas
    all_delta_t = np.sort(subdf["delta_t"].unique())
    flux = flux.reindex(all_delta_t)
    flux_err = flux_err.reindex(all_delta_t)

    # --- Usar mascara para marcar los valores NaN como 0 
    mask = (~flux.isna()).astype("uint8")

    flux = flux.fillna(0).values
    flux_err = flux_err.fillna(0).values
    mask = mask.fillna(0).values

    # --- Truncar series a T_max 
    # T_max = 200
    # flux, flux_err, mask = truncate_series(flux, flux_err, mask, T_max)

    # --- Metadatos a guardar
    meta = subdf.iloc[0][["hostgal_photoz", "hostgal_photoz_err", "distmod", "mwebv"]].to_numpy()
    extra_meta = subdf.iloc[0][["ra", "decl", "ddf_bool", "hostgal_specz"]].to_dict()
    
    # --- Guardar datos comprimidos en formato npz
    out_dir = PROC / f"{obj_id//10000:05d}"  # sub-folder cada 10k IDs
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # (Posibilidad de guardalos como HDF5 en el futuro)
    np.savez_compressed(
        out_dir / f"{obj_id}.npz",

        flux=flux,
        flux_err=flux_err,
        mask=mask,
        photo_z=meta[0],
        photo_z_err=meta[1],
        distmod=meta[2],
        mwebv=meta[3],

        ra=extra_meta["ra"],
        decl=extra_meta["decl"],
        ddf_bool=extra_meta["ddf_bool"],
        hostgal_specz=extra_meta["hostgal_specz"]
    )

    # Estructura de tensores guardada:
    # - flux: array 2D (n_timestamps, n_bands) con valores de flujo
    # - flux_err: array 2D (n_timestamps, n_bands) con errores de flujo  
    # - mask: array 2D (n_timestamps, n_bands) con máscara binaria (1=válido, 0=NaN)
    # - photo_z: scalar con redshift fotométrico de la galaxia anfitriona
    # - distmod: scalar con módulo de distancia
    # - mwebv: scalar con extinción por polvo galáctico
    # Resto de metadatos