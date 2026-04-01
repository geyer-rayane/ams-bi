# -*- coding: utf-8 -*-
"""
Comparaison des modèles sur le jeu de validation.

Charge les sorties de selection.py :
  - selection/resultats_validation.csv  -> métriques par algorithme
  - selection/predictions_validation.csv -> prédictions binaires par algorithme

Analyses :
  1. Tableau de métriques trié par ROC AUC.
  2. Test de McNemar (significativité statistique entre paires de modèles).
     H0 : les deux classifieurs font le même nombre d'erreurs.
     p < 0.05 -> différence significative.
  3. Matrice de p-valeurs entre toutes les paires d'algorithmes.

Aucune sortie fichier - résultats uniquement en terminal.
-> Pour l'évaluation finale sur test : evaluation.py
"""

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
    results_path = DIR_SELECTION / "resultats_validation.csv"
    preds_path = DIR_SELECTION / "predictions_validation.csv"

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
    print("TEST DE McNEMAR (p-valeurs) - * = différence significative (p < 0.05)")
    print("=" * 80)

    w = 10
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

    print("\n* p < 0.05 -> les deux modèles diffèrent significativement")

    # --- Recommandation ---
    best_row = df_res.sort_values("val_roc_auc", ascending=False).iloc[0]
    print(f"\nMeilleur modèle (AUC validation) : {best_row.get('label', best_row.get('key', '?'))} "
          f"- AUC={best_row['val_roc_auc']:.4f}  F1={best_row['val_f1']:.4f}")
    print("-> Lancer evaluation.py pour l'evaluation finale sur le jeu de test.")


if __name__ == "__main__":
    main()
