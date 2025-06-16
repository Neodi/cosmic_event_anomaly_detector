import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split


# Definición de rutas
BASE = Path(__file__).resolve().parents[2]
RAW_DATA = BASE / "data" / "raw" / "challenge_train_df.pkl"
OUT_DIR = BASE / "data" / "processed"
TRAIN_NPY = OUT_DIR / "train.npy"
VAL_NPY = OUT_DIR / "val.npy"
TEST_NPY = OUT_DIR / "test.npy"

# Parametros split
RANDOM_STATE = 42
TEST_SIZE = 0.2
VAL_RATIO = 0.5

def main():

    df = pd.read_pickle(RAW_DATA)[["object_id", "true_target"]].drop_duplicates()

    print(f"Numero de apariciones por clase en true_target:")
    print(df["true_target"].value_counts().sort_index())
    print(f"Total de objetos: {len(df)}")

    special_test_df = df[df["true_target"] >= 99]
    remaining_df = df[df["true_target"] < 99]

    # 1º Split: Train | (Val+Test)
    train_df, temp_df = train_test_split(
        remaining_df,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=remaining_df["true_target"]
    )

    # 2º Split: Val | Test
    val_df, test_df = train_test_split(
        temp_df,
        test_size=VAL_RATIO,
        random_state=RANDOM_STATE,
        stratify=temp_df["true_target"]
    )

    test_df = pd.concat([test_df, special_test_df], ignore_index=True)

    # Imprimir número de elementos por clase en cada split
    print("Train split - Elementos por clase:")
    print(train_df["true_target"].value_counts().sort_index())
    print(f"Total train: {len(train_df)}")
    print()

    print("Validation split - Elementos por clase:")
    print(val_df["true_target"].value_counts().sort_index())
    print(f"Total validation: {len(val_df)}")
    print()

    print("Test split - Elementos por clase:")
    print(test_df["true_target"].value_counts().sort_index())
    print(f"Total test: {len(test_df)}")
    print()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Guardar los splits en formato numpy
    np.save(TRAIN_NPY, train_df["object_id"].to_numpy())
    np.save(VAL_NPY, val_df["object_id"].to_numpy())
    np.save(TEST_NPY, test_df["object_id"].to_numpy())

    print(f"Guardados {len(train_df)} ids en {TRAIN_NPY.name}")
    print(f"Guardados {len(val_df)} ids en {VAL_NPY.name}")
    print(f"Guardados {len(test_df)} ids en {TEST_NPY.name}")


if __name__ == "__main__":
    main()