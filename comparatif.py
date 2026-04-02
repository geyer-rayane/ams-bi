# -*- coding: utf-8 -*-
"""
Comparaison des modeles (etape 'comparaison des modeles').

Objectif:
- Lire les resultats de raffinage sur validation (un modele "meilleur" par algorithme)
- Trier et afficher un tableau selon une metrque de performance (roc_auc, accuracy, f1, precision, recall)

Ce fichier ne fait pas de tests statistiques (McNemar) : c'est le role de comparaison.py.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

RACINE = Path(__file__).resolve().parent
DIR_SELECTION = RACINE / "selection"


def main() -> None:
    parser = argparse.ArgumentParser(description="Comparaison des modeles sur validation (tri par une metrique).")
    parser.add_argument(
        "--metric",
        default="roc_auc",
        choices=["roc_auc", "accuracy", "f1", "precision", "recall"],
        help="Metrique de tri et affichage.",
    )
    args = parser.parse_args()

    path = DIR_SELECTION / "raffinage_resultats_validation.csv"
    if not path.exists():
        raise FileNotFoundError(f"Manquant: {path}. Executer d'abord selection/apprentissage/raffinage.")

    df = pd.read_csv(path, low_memory=False)

    col_map = {
        "roc_auc": "val_roc_auc",
        "accuracy": "val_accuracy",
        "f1": "val_f1",
        "precision": "val_precision",
        "recall": "val_recall",
    }
    sort_col = col_map[args.metric]

    cols_show = ["label", "mode", "key", "params", "val_roc_auc", "val_accuracy", "val_f1", "val_precision", "val_recall"]
    cols_show = [c for c in cols_show if c in df.columns]
    df2 = df[cols_show].sort_values(sort_col, ascending=False)

    print("")
    print("=" * 86)
    print(f"COMPARATIF DES MODELES (validation) - tri par: {args.metric}")
    print("=" * 86)

    # Format compact en terminal
    show_cols = [c for c in cols_show if c not in {"params"}]
    params_present = "params" in cols_show

    if params_present:
        show_cols = ["label", "mode", "key", "params", sort_col]
        if "val_f1" in df2.columns:
            show_cols += ["val_f1"]
        if "val_accuracy" in df2.columns:
            show_cols += ["val_accuracy"]
        if "val_precision" in df2.columns:
            show_cols += ["val_precision"]
        if "val_recall" in df2.columns:
            show_cols += ["val_recall"]

    # Reorder et afficher
    df2 = df2.sort_values(sort_col, ascending=False)
    print(df2[show_cols].to_string(index=False))

    print("")
    best = df2.iloc[0]
    best_label = best.get("label", best.get("key", "?"))
    best_score = best.get(sort_col, None)
    print(f"Meilleur modele selon {args.metric}: {best_label} (score={best_score})")


if __name__ == "__main__":
    main()

