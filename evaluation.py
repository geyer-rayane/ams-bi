# -*- coding: utf-8 -*-
"""Etape Evaluation finale sur jeu de test (apres raffinage)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score

from raffinage import build_model, extraire_X_y

RACINE = Path(__file__).resolve().parent
DIR_SPLIT = RACINE / "decoupage"
DIR_SELECTION = RACINE / "selection"
TARGET = "cible_churn"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", default="best", help="Clé algo (ex. rf) ou 'best' pour auto.")
    args = parser.parse_args()

    config_path = DIR_SELECTION / "raffinage_meilleurs_modeles.json"
    if not config_path.exists():
        raise FileNotFoundError("Exécuter d'abord selection.py.")

    with open(config_path, encoding="utf-8") as f:
        configs = json.load(f)

    algo_key = configs["_best_overall"] if args.algo == "best" else args.algo
    if algo_key not in configs:
        raise ValueError(f"Cle '{algo_key}' absente de raffinage_meilleurs_modeles.json.")

    best = configs[algo_key]
    mode = best["mode"]

    # --- Chargement des données ---
    train_df = pd.read_csv(DIR_SPLIT / "jeu_apprentissage.csv", low_memory=False)
    val_df   = pd.read_csv(DIR_SPLIT / "jeu_validation.csv",   low_memory=False)
    test_df  = pd.read_csv(DIR_SPLIT / "jeu_test.csv",         low_memory=False)

    trainval_df = pd.concat([train_df, val_df], ignore_index=True)

    X_tv,   y_tv   = extraire_X_y(trainval_df, mode)
    X_test, y_test = extraire_X_y(test_df,     mode)

    # --- Ré-entraînement sur train + val ---
    model = build_model(algo_key, best["params"])
    model.fit(X_tv, y_tv)

    # --- Évaluation sur test ---
    y_true = y_test.to_numpy()
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    auc  = float(roc_auc_score(y_true, y_proba))
    f1   = float(f1_score(y_true, y_pred, zero_division=0))
    acc  = float(accuracy_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec  = float(recall_score(y_true, y_pred, zero_division=0))

    print("\n" + "=" * 60)
    print("ÉVALUATION FINALE - jeu de test")
    print("=" * 60)
    print(f"Algorithme    : {best.get('label', algo_key)}  (mode={mode})")
    print(f"Hyperparamètres : {best['params']}")
    print(f"Train+Val     : {len(trainval_df)} lignes  |  Test : {len(test_df)} lignes")
    print("-" * 60)
    print(f"ROC AUC   : {auc:.4f}")
    print(f"F1-score  : {f1:.4f}")
    print(f"Accuracy  : {acc:.4f}")
    print(f"Précision : {prec:.4f}")
    print(f"Rappel    : {rec:.4f}")
    print("-" * 60)
    print("Rapport de classification :")
    print(classification_report(y_true, y_pred, target_names=["non-churn", "churn"], zero_division=0))
    print("Matrice de confusion :")
    cm = confusion_matrix(y_true, y_pred)
    print(f"  VN={cm[0,0]:5d}  FP={cm[0,1]:5d}")
    print(f"  FN={cm[1,0]:5d}  VP={cm[1,1]:5d}")

    # --- Sauvegarde résultat test ---
    schema_test = {
        "algorithme": algo_key,
        "label": best.get("label", algo_key),
        "mode": mode,
        "params": best["params"],
        "val_roc_auc": best.get("val_roc_auc"),
        "test": {"roc_auc": auc, "f1": f1, "accuracy": acc, "precision": prec, "recall": rec},
    }
    (DIR_SELECTION / "evaluation_test_finale.json").write_text(
        json.dumps(schema_test, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print("- Resultats sauvegardes -> selection/evaluation_test_finale.json")
    print("- Etape suivante -> interpretation.py")


if __name__ == "__main__":
    main()
