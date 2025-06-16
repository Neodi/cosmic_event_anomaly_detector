from pathlib import Path
import json
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.metrics import (
    classification_report, 
    confusion_matrix, 
    roc_curve, 
    auc,
    precision_recall_curve, 
    average_precision_score
)

BASE = Path(__file__).resolve().parents[1]
DATA_PROC = BASE / "data" / "processed"
FIG_DIR = BASE / "reports" / "figures" / "metrics" / datetime.now().strftime("%Y%m%d_%H%M%S")
FIG_DIR.mkdir(exist_ok=True, parents=True)

VAL_CSV = DATA_PROC / "val_table_final.csv"
TEST_CSV = DATA_PROC / "test_table_final.csv"
VAL_JSON = DATA_PROC / "metrics_val.json"
TEST_JSON = DATA_PROC / "metrics_test.json"

def metrics_binary(df, y_col, pred_col, score_col=None, label="val"):
    """Devuelve dict con report, confusión y curvas ROC/PR (si score)."""
    y_true = df[y_col].values
    y_pred = df[pred_col].values

    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_true, y_pred).tolist()
    print(cm)
    out = {"classification_report": report, "conf_matrix": cm}

    if score_col is not None and score_col in df.columns:
        score = df[score_col].values
        fpr, tpr, _ = roc_curve(y_true, score)
        prec, rec, _ = precision_recall_curve(y_true, score)

        out["roc_auc"] = auc(fpr, tpr)
        out["pr_auc"] = average_precision_score(y_true, score)

        # ROC
        plt.figure()
        plt.plot(fpr, tpr, label=f"AUC={out['roc_auc']:.3f}")
        plt.plot([0, 1], [0, 1], "--", c="gray")
        plt.title(f"ROC ({label})")
        plt.xlabel("FPR")
        plt.ylabel("TPR")
        plt.legend()
        plt.savefig(FIG_DIR / f"roc_{label}.png")
        plt.close()

        # PR
        plt.figure()
        plt.plot(rec, prec, label=f"AUC={out['pr_auc']:.3f}")
        plt.title(f"PR ({label})")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.legend()
        plt.savefig(FIG_DIR / f"pr_{label}.png")
        plt.close()

    return out


def main():
    # Validación
    if VAL_CSV.exists():
        val_df = pd.read_csv(VAL_CSV)
        # Ajustar RARE_SET según Las clases raras
        RARE_SET = {64}
        val_df["y_true"] = val_df["true_target"].isin(RARE_SET).astype(int)

        val_metrics = metrics_binary(
            val_df,
            y_col="y_true",
            pred_col="is_event",
            score_col="chi2_pp",
            label="val",
        )
        json.dump(val_metrics, open(VAL_JSON, "w"), indent=4)
        print(f"✅ Métricas val guardadas en {VAL_JSON}")
    else:
        print(f"⚠️  No existe {VAL_CSV}. Salteando validación.")

    # Test
    if TEST_CSV.exists():
        test_df = pd.read_csv(TEST_CSV)
        if "true_target" in test_df.columns:
            # Si en test el valor 99 indica evento desconocido
            test_df["y_true"] = (test_df["true_target"] >= 99).astype(int)

            test_metrics = metrics_binary(
                test_df,
                y_col="y_true",
                pred_col="is_event",
                score_col="chi2_pp",
                label="test",
            )
            json.dump(test_metrics, open(TEST_JSON, "w"), indent=4)
            print(f"✅ Métricas test guardadas en {TEST_JSON}")
        else:
            print(f"ℹ️  {TEST_CSV} no contiene 'true_target'. No se calculan métricas de test.")
    else:
        print(f"⚠️  No existe {TEST_CSV}. Salteando test.")


if __name__ == '__main__':
    main()
