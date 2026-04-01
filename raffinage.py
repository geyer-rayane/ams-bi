# -*- coding: utf-8 -*-
"""Etape Raffinage: recherche des meilleurs hyperparametres sur validation."""

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
from sklearn.naive_bayes import CategoricalNB
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier

RANDOM_STATE = 42
RACINE = Path(__file__).resolve().parent
DIR_SPLIT = RACINE / "decoupage"
DIR_SELECTION = RACINE / "selection"


def extraire_X_y(df: pd.DataFrame, mode: str) -> tuple[pd.DataFrame, pd.Series]:
    y = df["cible_churn"].astype(int)
    if mode == "z":
        cols = sorted(c for c in df.columns if c.startswith(("z_", "sit_", "mot_")))
    else:
        cols = sorted(c for c in df.columns if c.startswith("cat_"))
    X = df[cols].copy()
    if mode == "cat":
        X = X.clip(lower=0).astype(int)
    return X, y


def build_model(key: str, params: dict[str, Any]):
    if key == "tree":
        return DecisionTreeClassifier(random_state=RANDOM_STATE, **params)
    if key == "rf":
        return RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1, **params)
    if key == "cnb":
        return CategoricalNB(**params)
    if key == "logreg":
        return LogisticRegression(max_iter=5000, random_state=RANDOM_STATE, **params)
    if key == "svm":
        return CalibratedClassifierCV(LinearSVC(max_iter=3000, random_state=RANDOM_STATE, **params), cv=3)
    raise ValueError(f"Algo inconnu: {key}")


def grid_product(param_grid: dict[str, list[Any]]) -> list[dict[str, Any]]:
    keys = list(param_grid.keys())
    return [dict(zip(keys, vals)) for vals in product(*param_grid.values())]


def evaluer(y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray) -> dict[str, float]:
    return {
        "roc_auc": float(roc_auc_score(y_true, y_proba)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--smote", action="store_true")
    args = parser.parse_args()

    plan = json.loads((DIR_SELECTION / "plan_selection.json").read_text(encoding="utf-8"))
    train_file = "jeu_apprentissage_smote.csv" if args.smote else "jeu_apprentissage.csv"
    train_df = pd.read_csv(DIR_SPLIT / train_file, low_memory=False)
    val_df = pd.read_csv(DIR_SPLIT / "jeu_validation.csv", low_memory=False)

    rows: list[dict[str, Any]] = []
    best_models: dict[str, Any] = {}
    pred_val = pd.DataFrame({"ID": val_df["ID"].values, "y_true": val_df["cible_churn"].astype(int).values})

    print("RAFFINAGE")
    print("- Principe: pour chaque algorithme, test de toutes les combinaisons de grille sur validation")
    print("- Critere de choix: ROC AUC (validation)")

    for algo in plan["algorithmes"]:
        Xtr, ytr = extraire_X_y(train_df, algo["mode"])
        Xv, yv = extraire_X_y(val_df, algo["mode"])
        grid = algo["grille_quick"] if args.quick else algo["grille"]
        combos = grid_product(grid)
        best_auc = -1.0
        best = None
        best_pred = None
        best_proba = None

        for p in combos:
            model = build_model(algo["key"], p)
            model.fit(Xtr, ytr)
            pred = model.predict(Xv)
            proba = model.predict_proba(Xv)[:, 1]
            met = evaluer(yv.to_numpy(), pred, proba)
            if met["roc_auc"] > best_auc:
                best_auc = met["roc_auc"]
                best = {"key": algo["key"], "label": algo["label"], "mode": algo["mode"], "params": p, **{f"val_{k}": v for k, v in met.items()}}
                best_pred = pred
                best_proba = proba

        assert best is not None and best_pred is not None and best_proba is not None
        rows.append(best)
        best_models[algo["key"]] = best
        pred_val[f"pred_{algo['key']}"] = best_pred
        pred_val[f"proba_{algo['key']}"] = best_proba
        print(f"  - {algo['key']}: {len(combos)} combinaisons teste -> AUC={best['val_roc_auc']:.4f}, F1={best['val_f1']:.4f}")

    df = pd.DataFrame(rows).sort_values("val_roc_auc", ascending=False)
    best_overall = df.iloc[0]["key"]
    best_models["_best_overall"] = best_overall

    df.to_csv(DIR_SELECTION / "raffinage_resultats_validation.csv", index=False, encoding="utf-8-sig")
    pred_val.to_csv(DIR_SELECTION / "raffinage_predictions_validation.csv", index=False, encoding="utf-8-sig")
    (DIR_SELECTION / "raffinage_meilleurs_modeles.json").write_text(json.dumps(best_models, indent=2, ensure_ascii=False), encoding="utf-8")

    print("- Resultat principal: selection/raffinage_resultats_validation.csv")
    print(f"- Meilleur modele global (validation): {best_overall}")


if __name__ == "__main__":
    main()
