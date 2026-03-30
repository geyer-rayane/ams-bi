# -*- coding: utf-8 -*-
"""
Sélection et classification (train / validation / test fixés).

Algorithmes exigés :
  - k plus proches voisins (k-NN)
  - arbre de décision
  - forêt d'arbres décisionnel (Random Forest)
  - classifieur bayésien naïf (Naive Bayes, variante Gaussienne)
  - régression logistique

Règle d'intégrité :
  - les instances utilisées pour train/validation/test viennent de `decoupage/`
    et restent donc comparables entre algorithmes.
  - le jeu `test` sert uniquement à l'évaluation finale du meilleur modèle
    choisi sur `validation` (sinon biais).

Prétraitement possible via recodage :
  - choix d'une représentation du bloc numérique normalisé : `z_*` vs `mm_*`
  - toujours conserver les indicatrices one-hot : `sit_*` et `mot_*`
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier


RACINE = Path(__file__).resolve().parent
DIR_SPLIT = RACINE / "decoupage"
DIR_SORTIE = RACINE / "selection"

TARGET_COL = "cible_churn"
ID_COL = "ID"

RANDOM_STATE = 42


@dataclass(frozen=True)
class Candidate:
    algorithm: str
    feature_mode: str
    hyperparams: dict[str, Any]
    val_metrics: dict[str, float]


def charger_split(nom_fichier: str) -> pd.DataFrame:
    p = DIR_SPLIT / nom_fichier
    if not p.exists():
        raise FileNotFoundError(f"Manquant : {p}. Exécuter `decoupage.py` d'abord.")
    return pd.read_csv(p, low_memory=False)


def colonnes_features(df: pd.DataFrame) -> dict[str, list[str]]:
    cols = list(df.columns)
    z_cols = [c for c in cols if c.startswith("z_")]
    mm_cols = [c for c in cols if c.startswith("mm_")]
    indicator_cols = [c for c in cols if c.startswith("sit_") or c.startswith("mot_")]

    # Sanity checks (protection contre une matrice mal formée)
    if not z_cols and not mm_cols:
        raise ValueError("Aucune colonne `z_*` ou `mm_*` trouvée dans le CSV découpé.")
    if TARGET_COL not in cols:
        raise ValueError(f"Colonne cible absente : `{TARGET_COL}`.")
    if ID_COL not in cols:
        raise ValueError(f"Colonne ID absente : `{ID_COL}`.")

    return {
        "z": sorted(z_cols),
        "mm": sorted(mm_cols),
        "indicator": sorted(indicator_cols),
    }


def extraire_X_y(df: pd.DataFrame, feature_mode: str) -> tuple[pd.DataFrame, pd.Series]:
    blocks = colonnes_features(df)
    indicator_cols = blocks["indicator"]
    if feature_mode == "z":
        selected = blocks["z"] + indicator_cols
    elif feature_mode == "mm":
        selected = blocks["mm"] + indicator_cols
    else:
        raise ValueError(f"feature_mode inconnu : {feature_mode}")

    X = df[selected]
    y = df[TARGET_COL].astype(int)
    return X, y


def eval_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray) -> dict[str, float]:
    # AUC sur la classe 1 (démission)
    auc = float(roc_auc_score(y_true, y_proba))
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "roc_auc": auc,
        "f1": float(f1_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
    }


def grid_product(param_grid: dict[str, list[Any]]) -> list[dict[str, Any]]:
    keys = list(param_grid.keys())
    values = [param_grid[k] for k in keys]
    out: list[dict[str, Any]] = []
    for combo in product(*values):
        out.append({k: combo[i] for i, k in enumerate(keys)})
    return out


def build_model(algorithm: str, hyperparams: dict[str, Any]) -> Any:
    if algorithm == "knn":
        return KNeighborsClassifier(**hyperparams)
    if algorithm == "tree":
        return DecisionTreeClassifier(**hyperparams, random_state=RANDOM_STATE)
    if algorithm == "rf":
        return RandomForestClassifier(**hyperparams, random_state=RANDOM_STATE, n_jobs=-1)
    if algorithm == "naive_bayes":
        return GaussianNB(**hyperparams)
    if algorithm == "logreg":
        return LogisticRegression(**hyperparams, max_iter=5000, random_state=RANDOM_STATE)
    raise ValueError(f"Algorithme inconnu : {algorithm}")


def recherche_hyperparametres(
    algorithm: str,
    feature_mode: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    candidates_params: list[dict[str, Any]],
) -> Candidate:
    best: Candidate | None = None

    yv = y_val.to_numpy()
    for params in candidates_params:
        model = build_model(algorithm, params)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        y_proba = model.predict_proba(X_val)[:, 1]
        metrics = eval_metrics(yv, y_pred, y_proba)

        if best is None or metrics["roc_auc"] > best.val_metrics["roc_auc"]:
            best = Candidate(
                algorithm=algorithm,
                feature_mode=feature_mode,
                hyperparams=params,
                val_metrics=metrics,
            )

    assert best is not None
    return best


def concat_train_val(train_df: pd.DataFrame, val_df: pd.DataFrame) -> pd.DataFrame:
    # On conserve uniquement l'union pour ré-entraîner le meilleur modèle avant d'évaluer sur test.
    return pd.concat([train_df, val_df], ignore_index=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="Réduit le nombre de combinaisons (debug rapide).")
    args = parser.parse_args()

    DIR_SORTIE.mkdir(parents=True, exist_ok=True)

    train_df = charger_split("jeu_apprentissage.csv")
    val_df = charger_split("jeu_validation.csv")
    test_df = charger_split("jeu_test.csv")

    # --- Grilles d'hyperparamètres (raisonnables pour un rendu de projet) ---
    if args.quick:
        grid = {
            "knn": {"n_neighbors": [5, 11], "weights": ["uniform", "distance"], "p": [2]},
            "tree": {"max_depth": [None, 10], "criterion": ["gini"], "min_samples_leaf": [1, 5]},
            "rf": {"n_estimators": [150], "max_depth": [None, 10], "max_features": ["sqrt"], "min_samples_leaf": [1, 5]},
            "naive_bayes": {"var_smoothing": [1e-9, 1e-7]},
            "logreg": {"C": [0.1, 1.0], "class_weight": [None, "balanced"]},
        }
    else:
        grid = {
            "knn": {"n_neighbors": [3, 5, 7, 11], "weights": ["uniform", "distance"], "p": [2]},
            "tree": {"max_depth": [None, 5, 10, 15], "criterion": ["gini", "entropy"], "min_samples_leaf": [1, 5]},
            "rf": {
                "n_estimators": [100, 300],
                "max_depth": [None, 5, 10],
                "max_features": ["sqrt", "log2"],
                "min_samples_leaf": [1, 5],
            },
            "naive_bayes": {"var_smoothing": [1e-9, 1e-8, 1e-7, 1e-6]},
            "logreg": {"C": [0.01, 0.1, 1.0, 10.0], "class_weight": [None, "balanced"]},
        }

    # --- Recherche sur validation pour (algo, z/mm) ---
    algorithms = [
        ("knn", "kNN (k plus proches voisins)"),
        ("tree", "Arbre de décision"),
        ("rf", "Forêt d'arbres décisionnels"),
        ("naive_bayes", "Naive Bayes (gaussien)"),
        ("logreg", "Régression logistique"),
    ]

    feature_modes = ["z", "mm"]

    all_candidates: list[dict[str, Any]] = []
    best_overall: Candidate | None = None

    for algo_key, algo_label in algorithms:
        best_algo: Candidate | None = None

        for feature_mode in feature_modes:
            X_train, y_train = extraire_X_y(train_df, feature_mode)
            X_val, y_val = extraire_X_y(val_df, feature_mode)
            params_list = grid_product(grid[algo_key])

            best_here = recherche_hyperparametres(
                algorithm=algo_key,
                feature_mode=feature_mode,
                X_train=X_train,
                y_train=y_train,
                X_val=X_val,
                y_val=y_val,
                candidates_params=params_list,
            )

            all_candidates.append(
                {
                    "algorithm": algo_key,
                    "algorithm_label": algo_label,
                    "feature_mode": feature_mode,
                    "hyperparams": json.dumps(best_here.hyperparams, ensure_ascii=False),
                    **{f"val_{k}": v for k, v in best_here.val_metrics.items()},
                }
            )

            if best_algo is None or best_here.val_metrics["roc_auc"] > best_algo.val_metrics["roc_auc"]:
                best_algo = best_here

        assert best_algo is not None

        if best_overall is None or best_algo.val_metrics["roc_auc"] > best_overall.val_metrics["roc_auc"]:
            best_overall = best_algo

    assert best_overall is not None

    # --- Ré-entraînement du meilleur candidat sur train+val, puis test final ---
    train_val_df = concat_train_val(train_df, val_df)
    X_trainval, y_trainval = extraire_X_y(train_val_df, best_overall.feature_mode)
    X_test, y_test = extraire_X_y(test_df, best_overall.feature_mode)

    final_model = build_model(best_overall.algorithm, best_overall.hyperparams)
    final_model.fit(X_trainval, y_trainval)

    y_test_np = y_test.to_numpy()
    y_pred_test = final_model.predict(X_test)
    y_proba_test = final_model.predict_proba(X_test)[:, 1]
    test_metrics = eval_metrics(y_test_np, y_pred_test, y_proba_test)

    schema = {
        "cible": TARGET_COL,
        "feature_mode": best_overall.feature_mode,
        "algorithme": best_overall.algorithm,
        "hyperparams": best_overall.hyperparams,
        "selection_sur": {
            "jeu": "validation",
            "critere": "roc_auc",
            "meilleure_valeurs": best_overall.val_metrics,
        },
        "evaluation_finale_sur": {
            "jeu": "test",
            "mesures": test_metrics,
        },
        "random_state": RANDOM_STATE,
    }

    # --- Écritures ---
    df_candidates = pd.DataFrame(all_candidates).sort_values(by="val_roc_auc", ascending=False)
    df_candidates.to_csv(DIR_SORTIE / "candidats_validation.csv", index=False, encoding="utf-8-sig")
    (DIR_SORTIE / "meilleur_modele_test_final.json").write_text(json.dumps(schema, indent=2, ensure_ascii=False), encoding="utf-8")

    # Prédictions (utile pour un tableau/annexes)
    test_out = pd.DataFrame(
        {
            ID_COL: test_df[ID_COL].values,
            "y_true": y_test_np,
            "y_pred": y_pred_test,
            "y_proba_1": y_proba_test,
        }
    )
    test_out.to_csv(DIR_SORTIE / "predictions_test_meilleur_modele.csv", index=False, encoding="utf-8-sig")

    # Synthèse texte
    lines = [
        "=== Sélection (classification) ===",
        f"Critère : ROC AUC sur la validation",
        "",
        f"Meilleur modèle (validation) : algorithm={best_overall.algorithm} feature_mode={best_overall.feature_mode}",
        f"Hyperparams : {json.dumps(best_overall.hyperparams, ensure_ascii=False)}",
        f"Val metrics : {best_overall.val_metrics}",
        "",
        "Évaluation finale sur test :",
        f"{test_metrics}",
        "",
        f"Fichiers : {str(DIR_SORTIE / 'candidats_validation.csv')}, {str(DIR_SORTIE / 'meilleur_modele_test_final.json')}, {str(DIR_SORTIE / 'predictions_test_meilleur_modele.csv')}",
    ]
    (DIR_SORTIE / "synthese_selection.txt").write_text("\n".join(lines), encoding="utf-8")

    print("OK -", DIR_SORTIE)
    print("  Meilleur modèle :", best_overall.algorithm, best_overall.feature_mode)
    print("  Val ROC AUC :", best_overall.val_metrics["roc_auc"])
    print("  Test ROC AUC :", test_metrics["roc_auc"])


if __name__ == "__main__":
    main()

