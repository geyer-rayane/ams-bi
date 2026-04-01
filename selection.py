# -*- coding: utf-8 -*-
"""
Sélection + Apprentissage + Raffinage (grid search).

Étapes couvertes :
  Sélection   : catalogue des algorithmes et grilles d'hyperparamètres.
  Apprentissage : entraînement sur jeu d'apprentissage.
  Raffinage   : grid search sur validation, sélection du meilleur par algorithme.

Algorithmes : k-NN, Arbre, Forêt, GaussianNB, BernoulliNB, CategoricalNB,
              Régression logistique, SVM (linéaire calibré).

Modes de features :
  "z"      → z_* (StandardScaler) + sit_*/mot_* (one-hot)   — pour la plupart des algos
  "binary" → sit_*/mot_* uniquement                          — BernoulliNB
  "cat"    → cat_* (encodage ordinal)                        — CategoricalNB

Options :
  --quick   : grilles réduites (test rapide)
  --smote   : utilise jeu_apprentissage_smote.csv

Sorties : selection/resultats_validation.csv
          selection/meilleurs_par_algo.json
          selection/predictions_validation.csv
→ Comparaison détaillée : comparaison.py
→ Évaluation sur test   : evaluation.py
"""

from __future__ import annotations

import argparse
import json
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.naive_bayes import BernoulliNB, CategoricalNB, GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier

RACINE = Path(__file__).resolve().parent
DIR_SPLIT = RACINE / "decoupage"
DIR_SORTIE = RACINE / "selection"
TARGET = "cible_churn"
RANDOM_STATE = 42

# ---------------------------------------------------------------------------
# Catalogue des algorithmes
# ---------------------------------------------------------------------------
CATALOG: list[dict[str, Any]] = [
    {
        "key": "knn", "label": "k-NN", "mode": "z",
        "cls": KNeighborsClassifier, "fixed": {},
        "grid": {"n_neighbors": [3, 5, 7, 11], "weights": ["uniform", "distance"]},
        "grid_quick": {"n_neighbors": [5], "weights": ["uniform"]},
    },
    {
        "key": "tree", "label": "Arbre de décision", "mode": "z",
        "cls": DecisionTreeClassifier, "fixed": {"random_state": RANDOM_STATE},
        "grid": {"max_depth": [None, 5, 10], "criterion": ["gini", "entropy"], "min_samples_leaf": [1, 5]},
        "grid_quick": {"max_depth": [5], "criterion": ["gini"], "min_samples_leaf": [1]},
    },
    {
        "key": "rf", "label": "Forêt aléatoire", "mode": "z",
        "cls": RandomForestClassifier, "fixed": {"random_state": RANDOM_STATE, "n_jobs": -1},
        "grid": {
            "n_estimators": [100, 300], "max_depth": [None, 5, 10],
            "max_features": ["sqrt"], "class_weight": [None, "balanced"],
        },
        "grid_quick": {"n_estimators": [100], "max_depth": [None], "max_features": ["sqrt"], "class_weight": [None]},
    },
    {
        "key": "gnb", "label": "Naive Bayes gaussien", "mode": "z",
        "cls": GaussianNB, "fixed": {},
        "grid": {"var_smoothing": [1e-9, 1e-7, 1e-5]},
        "grid_quick": {"var_smoothing": [1e-9]},
    },
    {
        "key": "bnb", "label": "Naive Bayes Bernoulli", "mode": "binary",
        "cls": BernoulliNB, "fixed": {},
        "grid": {"alpha": [0.1, 0.5, 1.0]},
        "grid_quick": {"alpha": [1.0]},
    },
    {
        "key": "cnb", "label": "Naive Bayes catégoriel", "mode": "cat",
        "cls": CategoricalNB, "fixed": {},
        "grid": {"alpha": [0.1, 0.5, 1.0]},
        "grid_quick": {"alpha": [1.0]},
    },
    {
        "key": "logreg", "label": "Régression logistique", "mode": "z",
        "cls": LogisticRegression, "fixed": {"max_iter": 5000, "random_state": RANDOM_STATE},
        "grid": {"C": [0.01, 0.1, 1.0, 10.0], "class_weight": [None, "balanced"]},
        "grid_quick": {"C": [1.0], "class_weight": [None]},
    },
    {
        "key": "svm", "label": "SVM (linéaire calibré)", "mode": "z",
        "cls": LinearSVC, "fixed": {"max_iter": 3000, "random_state": RANDOM_STATE},
        "grid": {"C": [0.1, 1.0, 10.0], "class_weight": [None, "balanced"]},
        "grid_quick": {"C": [1.0], "class_weight": [None]},
    },
]

# ---------------------------------------------------------------------------
# Utilitaires
# ---------------------------------------------------------------------------

def extraire_X_y(df: pd.DataFrame, mode: str) -> tuple[pd.DataFrame, pd.Series]:
    y = df[TARGET].astype(int)
    if mode == "z":
        cols = sorted(c for c in df.columns if c.startswith(("z_", "sit_", "mot_")))
    elif mode == "binary":
        cols = sorted(c for c in df.columns if c.startswith(("sit_", "mot_")))
    elif mode == "cat":
        cols = sorted(c for c in df.columns if c.startswith("cat_"))
    else:
        raise ValueError(f"Mode inconnu : {mode}")
    if not cols:
        raise ValueError(f"Aucune colonne trouvée pour mode='{mode}'")
    X = df[cols].copy()
    if mode == "cat":
        X = X.clip(lower=0).astype(int)
    return X, y


def grid_product(param_grid: dict) -> list[dict]:
    keys = list(param_grid.keys())
    return [dict(zip(keys, vals)) for vals in product(*param_grid.values())]


def eval_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray) -> dict[str, float]:
    return {
        "roc_auc":   float(roc_auc_score(y_true, y_proba)),
        "f1":        float(f1_score(y_true, y_pred, zero_division=0)),
        "accuracy":  float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall":    float(recall_score(y_true, y_pred, zero_division=0)),
    }


def build_model(cfg: dict, params: dict) -> Any:
    full = {**cfg["fixed"], **params}
    if cfg["cls"] is LinearSVC:
        return CalibratedClassifierCV(LinearSVC(**full), cv=3)
    return cfg["cls"](**full)


def run_grid_search(
    cfg: dict, X_tr: pd.DataFrame, y_tr: pd.Series,
    X_val: pd.DataFrame, y_val: pd.Series, quick: bool
) -> tuple[dict, np.ndarray, np.ndarray]:
    """Retourne (best_result_dict, y_pred_best, y_proba_best)."""
    param_grid = cfg["grid_quick"] if quick else cfg["grid"]
    y_val_np = y_val.to_numpy()
    best_auc, best_result, best_pred, best_proba = -1.0, None, None, None

    for params in grid_product(param_grid):
        model = build_model(cfg, params)
        model.fit(X_tr, y_tr)
        y_pred = model.predict(X_val)
        y_proba = model.predict_proba(X_val)[:, 1]
        m = eval_metrics(y_val_np, y_pred, y_proba)
        if m["roc_auc"] > best_auc:
            best_auc = m["roc_auc"]
            best_result = {
                "key": cfg["key"], "label": cfg["label"], "mode": cfg["mode"],
                "params": params,
                **{f"val_{k}": v for k, v in m.items()},
            }
            best_pred, best_proba = y_pred.copy(), y_proba.copy()

    return best_result, best_pred, best_proba  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="Grilles réduites (test rapide).")
    parser.add_argument("--smote", action="store_true", help="Utilise jeu_apprentissage_smote.csv.")
    args = parser.parse_args()

    DIR_SORTIE.mkdir(parents=True, exist_ok=True)

    train_file = "jeu_apprentissage_smote.csv" if args.smote else "jeu_apprentissage.csv"
    train_df = pd.read_csv(DIR_SPLIT / train_file, low_memory=False)
    val_df   = pd.read_csv(DIR_SPLIT / "jeu_validation.csv", low_memory=False)

    print(f"Train : {len(train_df)} lignes ({'SMOTE' if args.smote else 'original'})  |  Val : {len(val_df)} lignes")
    print(f"Mode  : {'quick' if args.quick else 'complet'}\n")

    all_best: dict[str, dict] = {}
    all_preds: dict[str, np.ndarray] = {}
    all_probas: dict[str, np.ndarray] = {}

    for cfg in CATALOG:
        try:
            X_tr, y_tr = extraire_X_y(train_df, cfg["mode"])
            X_val, y_val = extraire_X_y(val_df, cfg["mode"])
        except ValueError as e:
            print(f"  [{cfg['key']:8s}] ignoré — {e}")
            continue

        print(f"  [{cfg['key']:8s}] {cfg['label']:<28}", end="", flush=True)
        best, pred, proba = run_grid_search(cfg, X_tr, y_tr, X_val, y_val, args.quick)

        all_best[cfg["key"]] = best
        all_preds[cfg["key"]] = pred
        all_probas[cfg["key"]] = proba
        print(f"  AUC={best['val_roc_auc']:.4f}  F1={best['val_f1']:.4f}  Acc={best['val_accuracy']:.4f}")

    # --- Résumé trié ---
    df_res = pd.DataFrame(list(all_best.values())).sort_values("val_roc_auc", ascending=False)
    best_overall_key = df_res.iloc[0]["key"]

    print("\n" + "=" * 78)
    print(f"{'Algorithme':<28} {'Mode':<8} {'AUC':>7} {'F1':>7} {'Acc':>7} {'Prec':>7} {'Rec':>7}")
    print("-" * 78)
    for row in df_res.itertuples():
        marker = " <-- best" if row.key == best_overall_key else ""
        print(
            f"{row.label:<28} {row.mode:<8} "
            f"{row.val_roc_auc:7.4f} {row.val_f1:7.4f} {row.val_accuracy:7.4f} "
            f"{row.val_precision:7.4f} {row.val_recall:7.4f}{marker}"
        )
    print("=" * 78)
    print(f"\nMeilleur (val AUC) : {best_overall_key} -> lancer evaluation.py pour le test final")

    # --- Sauvegarde ---
    df_res.to_csv(DIR_SORTIE / "resultats_validation.csv", index=False, encoding="utf-8-sig")

    save_best = {k: v for k, v in all_best.items()}
    save_best["_best_overall"] = best_overall_key
    (DIR_SORTIE / "meilleurs_par_algo.json").write_text(
        json.dumps(save_best, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    y_true_val = val_df[TARGET].astype(int).values
    preds_df = pd.DataFrame({"ID": val_df["ID"].values, "y_true": y_true_val})
    for key, pred in all_preds.items():
        preds_df[f"pred_{key}"] = pred
        preds_df[f"proba_{key}"] = all_probas[key]
    preds_df.to_csv(DIR_SORTIE / "predictions_validation.csv", index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    main()
