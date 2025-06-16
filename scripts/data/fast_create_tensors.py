import pandas as pd
import numpy as np
import json
from pathlib import Path
from tqdm import tqdm
import multiprocessing as mp
from functools import partial

def process_object(obj_data, bands, proc_dir):
    """Procesa un objeto individual - función para paralelizar"""
    obj_id, subdf = obj_data
    
    subdf = subdf.reset_index(drop=True)
    
    band_map = {i: band for i, band in enumerate(bands)}
    subdf["band"] = subdf["passband"].map(band_map)

    all_delta_t = np.sort(subdf["delta_t"].unique())
    
    flux = subdf.pivot_table(index="delta_t", columns="band", values="flux", fill_value=0)
    flux_err = subdf.pivot_table(index="delta_t", columns="band", values="flux_err", fill_value=0)
    
    if not flux.index.equals(pd.Index(all_delta_t)):
        flux = flux.reindex(all_delta_t, fill_value=0)
        flux_err = flux_err.reindex(all_delta_t, fill_value=0)

    mask = (~subdf.pivot_table(index="delta_t", columns="band", values="flux").isna()).astype("uint8")
    mask = mask.reindex(all_delta_t, fill_value=0)

    flux_array = flux.values
    flux_err_array = flux_err.values
    mask_array = mask.values

    first_row = subdf.iloc[0]
    meta = first_row[["hostgal_photoz", "hostgal_photoz_err", "distmod", "mwebv"]].values
    
    out_dir = proc_dir / f"{obj_id//10000:05d}"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    np.savez_compressed(
        out_dir / f"{obj_id}.npz",
        flux=flux_array.astype(np.float32),  
        flux_err=flux_err_array.astype(np.float32),
        mask=mask_array,
        photo_z=np.float32(meta[0]),
        photo_z_err=np.float32(meta[1]),
        distmod=np.float32(meta[2]),
        mwebv=np.float32(meta[3]),
        ra=np.float32(first_row["ra"]),
        decl=np.float32(first_row["decl"]),
        ddf_bool=bool(first_row["ddf_bool"]),
        hostgal_specz=np.float32(first_row["hostgal_specz"])
    )

def main():

    print("Iniciando procesamiento de datos...")

    BASE = Path(__file__).resolve().parents[2]
    RAW = BASE / "data" / "raw"
    PROC = BASE / "data" / "processed" / "npz"
    
    columns_needed = ["object_id", "passband", "delta_t", "flux", "flux_err", 
                     "hostgal_photoz", "hostgal_photoz_err", "distmod", "mwebv",
                     "ra", "decl", "ddf_bool", "hostgal_specz"]
    
    print("Cargando datos...")
    df = pd.read_pickle(RAW / "challenge_train_df.pkl")[columns_needed]
    
    bands = ["u", "g", "r", "i", "z", "y"]
    total_objects = df['object_id'].nunique()
    
    print(f"Procesando {total_objects} objetos...")
    
    n_cores = mp.cpu_count() - 1  
    
    process_func = partial(process_object, bands=bands, proc_dir=PROC)
    
    grouped_data = list(df.groupby("object_id"))
    
    with mp.Pool(n_cores) as pool:
        list(tqdm(
            pool.imap(process_func, grouped_data),
            total=total_objects,
            desc="Processing objects"
        ))

if __name__ == "__main__":
    main()