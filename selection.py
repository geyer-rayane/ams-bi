# -*- coding: utf-8 -*-
"""
Etape Selection.

But:
- fixer les algorithmes a comparer;
- fixer les grilles d'hyperparametres;
- fixer le mode de pretraitement par algorithme.

NB: Naive Bayes retenu = CategoricalNB uniquement.
"""

from __future__ import annotations

import json
from pathlib import Path


RACINE = Path(__file__).resolve().parent
DIR_SORTIE = RACINE / "selection"


def main() -> None:
    DIR_SORTIE.mkdir(parents=True, exist_ok=True)

    plan = {
        "target": "cible_churn",
        "algorithmes": [
            {
                "key": "knn",
                "label": "k-NN",
                "mode": "z",
                "grille": {"n_neighbors": [3, 5, 7, 11], "weights": ["uniform", "distance"]},
                "grille_quick": {"n_neighbors": [5], "weights": ["uniform"]},
            },
            {
                "key": "tree",
                "label": "Arbre de decision",
                "mode": "z",
                "grille": {"max_depth": [None, 5, 10], "criterion": ["gini", "entropy"], "min_samples_leaf": [1, 5]},
                "grille_quick": {"max_depth": [5], "criterion": ["gini"], "min_samples_leaf": [1]},
            },
            {
                "key": "rf",
                "label": "Foret aleatoire",
                "mode": "z",
                "grille": {"n_estimators": [100, 300], "max_depth": [None, 5, 10], "max_features": ["sqrt"], "class_weight": [None, "balanced"]},
                "grille_quick": {"n_estimators": [100], "max_depth": [None], "max_features": ["sqrt"], "class_weight": [None]},
            },
            {
                "key": "cnb",
                "label": "Naive Bayes categoriel",
                "mode": "cat",
                "grille": {"alpha": [0.1, 0.5, 1.0]},
                "grille_quick": {"alpha": [1.0]},
            },
            {
                "key": "logreg",
                "label": "Regression logistique",
                "mode": "z",
                "grille": {"C": [0.01, 0.1, 1.0, 10.0], "class_weight": [None, "balanced"]},
                "grille_quick": {"C": [1.0], "class_weight": [None]},
            },
            {
                "key": "svm",
                "label": "SVM lineaire calibre",
                "mode": "z",
                "grille": {"C": [0.1, 1.0, 10.0], "class_weight": [None, "balanced"]},
                "grille_quick": {"C": [1.0], "class_weight": [None]},
            },
        ],
    }

    out = DIR_SORTIE / "plan_selection.json"
    out.write_text(json.dumps(plan, indent=2, ensure_ascii=False), encoding="utf-8")

    print("SELECTION")
    print(f"- Algorithmes retenus: {len(plan['algorithmes'])}")
    for a in plan["algorithmes"]:
        print(f"  - {a['key']}: {a['label']} (mode={a['mode']})")
    print("- Naive Bayes gaussien: retire")
    print("- Naive Bayes bernoulli: retire")
    print(f"- Fichier: {out}")


if __name__ == "__main__":
    main()
