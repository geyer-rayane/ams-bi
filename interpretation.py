# -*- coding: utf-8 -*-
"""Etape Interpretation: attributs les plus importants du meilleur modele."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from raffinage import build_model, extraire_X_y

RACINE = Path(__file__).resolve().parent
DIR_SPLIT = RACINE / "decoupage"
DIR_SELECTION = RACINE / "selection"


def main() -> None:
    cfg = json.loads((DIR_SELECTION / "raffinage_meilleurs_modeles.json").read_text(encoding="utf-8"))
    best_key = cfg["_best_overall"]
    best = cfg[best_key]

    train = pd.read_csv(DIR_SPLIT / "jeu_apprentissage.csv", low_memory=False)
    val = pd.read_csv(DIR_SPLIT / "jeu_validation.csv", low_memory=False)
    trainval = pd.concat([train, val], ignore_index=True)

    X, y = extraire_X_y(trainval, best["mode"])
    model = build_model(best_key, best["params"])
    model.fit(X, y)

    base = model
    if hasattr(model, "estimator"):
        base = model.estimator

    print("INTERPRETATION")
    print(f"- Modele: {best_key} ({best['label']})")
    print(f"- Features utilisees: {X.shape[1]}")

    if hasattr(base, "feature_importances_"):
        vals = base.feature_importances_
        idx = np.argsort(vals)[::-1][:15]
        print("- Top 15 importances:")
        top = []
        for r, i in enumerate(idx, 1):
            print(f"  {r:2d}. {X.columns[i]:<30} {vals[i]:.6f}")
            top.append({"rank": r, "feature": X.columns[i], "value": float(vals[i])})
    elif hasattr(base, "coef_"):
        vals = base.coef_.ravel() if base.coef_.ndim > 1 else base.coef_
        idx = np.argsort(np.abs(vals))[::-1][:15]
        print("- Top 15 coefficients absolus:")
        top = []
        for r, i in enumerate(idx, 1):
            print(f"  {r:2d}. {X.columns[i]:<30} {vals[i]:+.6f}")
            top.append({"rank": r, "feature": X.columns[i], "value": float(vals[i])})
    elif hasattr(model, "calibrated_classifiers_"):
        arr = []
        for cc in model.calibrated_classifiers_:
            inner = cc.estimator
            if hasattr(inner, "coef_"):
                arr.append(inner.coef_.ravel())
        if arr:
            vals = np.mean(arr, axis=0)
            idx = np.argsort(np.abs(vals))[::-1][:15]
            print("- Top 15 coefficients moyens (SVM calibre):")
            top = []
            for r, i in enumerate(idx, 1):
                print(f"  {r:2d}. {X.columns[i]:<30} {vals[i]:+.6f}")
                top.append({"rank": r, "feature": X.columns[i], "value": float(vals[i])})
        else:
            top = []
            print("- Interpretation indisponible pour ce modele.")
    else:
        top = []
        print("- Interpretation indisponible pour ce modele.")

    out = {"best_model": best_key, "label": best["label"], "mode": best["mode"], "top_features": top}
    (DIR_SELECTION / "interpretation_top_features.json").write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print("- Fichier: selection/interpretation_top_features.json")


if __name__ == "__main__":
    main()
