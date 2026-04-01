# -*- coding: utf-8 -*-
"""
Évaluation finale sur le jeu de test.

Charge la configuration du meilleur modèle depuis selection/meilleurs_par_algo.json,
ré-entraîne sur train + validation combinés, puis évalue sur test.

Usage :
  python evaluation.py               # utilise _best_overall
  python evaluation.py --algo rf     # spécifie l'algorithme (key de CATALOG)

Interprétation :
  - Arbre / Forêt : importances de features (top 15).
  - Régression logistique : coefficients (top 15 en valeur absolue).
  - Autres : aucune (boîte noire).

-> Cette étape ne doit être exécutée QU'UNE SEULE FOIS sur le modèle définitif.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

# Réutilise CATALOG et les helpers de selection.py
from selection import CATALOG, build_model, extraire_X_y

RACINE = Path(__file__).resolve().parent
DIR_SPLIT = RACINE / "decoupage"
DIR_SELECTION = RACINE / "selection"
TARGET = "cible_churn"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", default="best", help="Clé algo (ex. rf) ou 'best' pour auto.")
    args = parser.parse_args()

    config_path = DIR_SELECTION / "meilleurs_par_algo.json"
    if not config_path.exists():
        raise FileNotFoundError("Exécuter d'abord selection.py.")

    with open(config_path, encoding="utf-8") as f:
        configs = json.load(f)

    algo_key = configs["_best_overall"] if args.algo == "best" else args.algo
    if algo_key not in configs:
        raise ValueError(f"Clé '{algo_key}' absente de meilleurs_par_algo.json.")

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
    algo_cfg = next((c for c in CATALOG if c["key"] == algo_key), None)
    if algo_cfg is None:
        raise ValueError(f"Algorithme '{algo_key}' non trouvé dans CATALOG.")

    model = build_model(algo_cfg, best["params"])
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

    # --- Interprétation ---
    print("\n" + "=" * 60)
    print("INTERPRÉTATION DU MODÈLE")
    print("=" * 60)

    # Récupère le modèle sous-jacent (si CalibratedClassifierCV)
    base = model
    if hasattr(model, "estimator"):
        base = model.estimator  # CalibratedClassifierCV

    feature_names = list(X_tv.columns)

    if hasattr(base, "feature_importances_"):
        importances = base.feature_importances_
        idx = np.argsort(importances)[::-1][:15]
        print("Importances de features (top 15) :")
        for rank, i in enumerate(idx, 1):
            print(f"  {rank:2d}. {feature_names[i]:<35} {importances[i]:.4f}")

    elif hasattr(base, "coef_"):
        coefs = base.coef_.ravel() if base.coef_.ndim > 1 else base.coef_
        idx = np.argsort(np.abs(coefs))[::-1][:15]
        print("Coefficients (top 15 en valeur absolue) :")
        for rank, i in enumerate(idx, 1):
            print(f"  {rank:2d}. {feature_names[i]:<35} {coefs[i]:+.4f}")

    elif hasattr(model, "calibrated_classifiers_"):
        # CalibratedClassifierCV - moyenne des estimateurs calibrés
        all_coefs = []
        for cc in model.calibrated_classifiers_:
            inner = cc.estimator
            if hasattr(inner, "coef_"):
                all_coefs.append(inner.coef_.ravel())
            elif hasattr(inner, "feature_importances_"):
                all_coefs.append(inner.feature_importances_)
        if all_coefs:
            avg = np.mean(all_coefs, axis=0)
            idx = np.argsort(np.abs(avg))[::-1][:15]
            print("Importance moyenne des features (calibré, top 15) :")
            for rank, i in enumerate(idx, 1):
                print(f"  {rank:2d}. {feature_names[i]:<35} {avg[i]:+.4f}")
        else:
            print("Interprétation non disponible pour cet algorithme.")
    else:
        print("Interprétation non disponible pour cet algorithme.")

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
    print(f"\nResultats sauvegardes -> selection/evaluation_test_finale.json")


if __name__ == "__main__":
    main()
