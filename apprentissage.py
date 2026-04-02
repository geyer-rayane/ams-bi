# -*- coding: utf-8 -*-
"""Etape Apprentissage: premier passage (sans raffinage complet)."""

from __future__ import annotations

import argparse
import json
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
        cols = sorted(c for c in df.columns if c.startswith(("z_", "sit_")))
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
    modeles: dict[str, Any] = {}
    print("APPRENTISSAGE")
    print(f"- Train: {len(train_df)} ({'SMOTE' if args.smote else 'original'})")
    print(f"- Validation: {len(val_df)}")
    print("- Methode: 1ere passe (1 jeu d'hyperparametres / algo)")

    for algo in plan["algorithmes"]:
        grid = algo["grille_quick"] if args.quick else algo["grille"]
        params = {k: v[0] for k, v in grid.items()}
        Xtr, ytr = extraire_X_y(train_df, algo["mode"])
        Xv, yv = extraire_X_y(val_df, algo["mode"])
        m = build_model(algo["key"], params)
        m.fit(Xtr, ytr)
        pred = m.predict(Xv)
        proba = m.predict_proba(Xv)[:, 1]
        met = evaluer(yv.to_numpy(), pred, proba)
        row = {"key": algo["key"], "label": algo["label"], "mode": algo["mode"], "params": json.dumps(params), **met}
        rows.append(row)
        modeles[algo["key"]] = {"label": algo["label"], "mode": algo["mode"], "params": params}
        print(f"  - {algo['key']}: AUC={met['roc_auc']:.4f} F1={met['f1']:.4f}")

    df = pd.DataFrame(rows).sort_values("roc_auc", ascending=False)
    df.to_csv(DIR_SELECTION / "apprentissage_validation.csv", index=False, encoding="utf-8-sig")
    (DIR_SELECTION / "apprentissage_modeles.json").write_text(json.dumps(modeles, indent=2, ensure_ascii=False), encoding="utf-8")
    print("- Resultat: selection/apprentissage_validation.csv")


if __name__ == "__main__":
    main()
