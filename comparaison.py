# -*- coding: utf-8 -*-
"""Etape Comparaison sur validation + explication detaillee de McNemar."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chi2

RACINE = Path(__file__).resolve().parent
DIR_SELECTION = RACINE / "selection"


def mcnemar_pvalue(pred_a: np.ndarray, pred_b: np.ndarray, y_true: np.ndarray) -> float:
    """Test de McNemar avec correction de continuité (Yates)."""
    correct_a = pred_a == y_true
    correct_b = pred_b == y_true
    n01 = (~correct_a & correct_b).sum()   # B juste, A faux
    n10 = (correct_a & ~correct_b).sum()   # A juste, B faux
    total = n01 + n10
    if total == 0:
        return 1.0
    stat = (abs(n01 - n10) - 1) ** 2 / total
    return float(1 - chi2.cdf(stat, df=1))


def main() -> None:
    results_path = DIR_SELECTION / "raffinage_resultats_validation.csv"
    preds_path = DIR_SELECTION / "raffinage_predictions_validation.csv"

    if not results_path.exists() or not preds_path.exists():
        raise FileNotFoundError("Exécuter d'abord selection.py.")

    df_res = pd.read_csv(results_path)
    df_preds = pd.read_csv(preds_path)

    y_true = df_preds["y_true"].values
    algo_keys = [c.replace("pred_", "") for c in df_preds.columns if c.startswith("pred_")]

    # --- Tableau métriques ---
    print("\n" + "=" * 80)
    print("COMPARAISON DES MODÈLES - jeu de validation")
    print("=" * 80)
    cols_affich = [c for c in ["label", "mode", "val_roc_auc", "val_f1", "val_accuracy", "val_precision", "val_recall"] if c in df_res.columns]
    print(df_res[cols_affich].sort_values("val_roc_auc", ascending=False).to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    # --- Test de McNemar ---
    print("\n" + "=" * 80)
    print("TEST DE McNEMAR - comparaison par paires de modeles")
    print("=" * 80)
    print("Comment la comparaison est faite:")
    print("- On compare 2 modeles A et B sur EXACTEMENT les memes lignes de validation.")
    print("- n01 = nb lignes ou A est faux et B est juste.")
    print("- n10 = nb lignes ou A est juste et B est faux.")
    print("- Si n01 et n10 sont proches: performances comparables.")
    print("- Si n01 et n10 sont tres differents: un modele domine l'autre.")
    print("- H0: n01 == n10 (pas de difference significative).")
    print("- p < 0.05 => difference statistiquement significative.")
    print("")

    w = 12
    header = f"{'':>{w}}" + "".join(f"{k:>{w}}" for k in algo_keys)
    print(header)

    for k1 in algo_keys:
        row_str = f"{k1:>{w}}"
        pred1 = df_preds[f"pred_{k1}"].values
        for k2 in algo_keys:
            if k1 == k2:
                row_str += f"{'-':>{w}}"
            else:
                pred2 = df_preds[f"pred_{k2}"].values
                p = mcnemar_pvalue(pred1, pred2, y_true)
                marker = "*" if p < 0.05 else " "
                row_str += f"{p:.3f}{marker:>{w - 5}}"
        print(row_str)

    print("\nDETAIL PAR PAIRE (n01, n10, p-value):")
    for i, k1 in enumerate(algo_keys):
        pred1 = df_preds[f"pred_{k1}"].values
        for k2 in algo_keys[i + 1:]:
            pred2 = df_preds[f"pred_{k2}"].values
            correct_1 = pred1 == y_true
            correct_2 = pred2 == y_true
            n01 = int((~correct_1 & correct_2).sum())
            n10 = int((correct_1 & ~correct_2).sum())
            p = mcnemar_pvalue(pred1, pred2, y_true)
            verdict = "SIGNIFICATIF" if p < 0.05 else "non significatif"
            print(f"- {k1} vs {k2}: n01={n01}, n10={n10}, p={p:.4f} -> {verdict}")

    # --- Recommandation ---
    best_row = df_res.sort_values("val_roc_auc", ascending=False).iloc[0]
    print(f"\nMeilleur modèle (AUC validation) : {best_row.get('label', best_row.get('key', '?'))} "
          f"- AUC={best_row['val_roc_auc']:.4f}  F1={best_row['val_f1']:.4f}")
    print("-> Lancer evaluation.py pour l'evaluation finale sur test.")


if __name__ == "__main__":
    main()
