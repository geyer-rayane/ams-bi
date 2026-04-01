# -*- coding: utf-8 -*-
"""
Prétraitement : réduction de dimension et sélection de variables (avant modèles de fouille).

Entrée : recodage/union_matrice_modele.csv (union table1+table2, mode principal).
Fallback : recodage/table2_matrice_modele.csv si l'union n'existe pas.

Méthodes :
  - ACP / PCA : réduction de dimension, variance expliquée ;
    les composantes sont des combinaisons linéaires → interprétabilité réduite.
  - SelectKBest (ANOVA F) : garde k variables les plus liées à la cible (noms préservés).

Sorties : dossier pretraitement/
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif

RACINE = Path(__file__).resolve().parent
_MATRICE_UNION = RACINE / "recodage" / "union_matrice_modele.csv"
_MATRICE_T2 = RACINE / "recodage" / "table2_matrice_modele.csv"
MATRICE = _MATRICE_UNION if _MATRICE_UNION.exists() else _MATRICE_T2
SORTIE = RACINE / "pretraitement"


def charger_matrice(chemin: Path) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    df = pd.read_csv(chemin, low_memory=False)
    y = df["cible_churn"]
    ids = df["ID"]
    cols_meta = [c for c in ("ID", "cible_churn", "source", "id_fichier") if c in df.columns]
    X = df.drop(columns=cols_meta)
    return X, y, ids


def main() -> None:
    SORTIE.mkdir(parents=True, exist_ok=True)
    matrice = _MATRICE_UNION if _MATRICE_UNION.exists() else _MATRICE_T2
    if not matrice.exists():
        raise FileNotFoundError(
            "Matrice absente. Exécuter d'abord concatenation.py puis recodage.py."
        )

    X, y, ids = charger_matrice(matrice)

    # --- PCA : variance cumulée 95 % ---
    pca_95 = PCA(n_components=0.95, random_state=42)
    Z_95 = pca_95.fit_transform(X)
    n95 = pca_95.n_components_

    cols_95 = [f"PC{i+1}" for i in range(n95)]
    out_95 = pd.DataFrame(Z_95, columns=cols_95)
    out_95.insert(0, "ID", ids.values)
    out_95["cible_churn"] = y.values
    out_95.to_csv(SORTIE / "matrice_pca_95var.csv", index=False, encoding="utf-8-sig")

    # --- PCA : 10 composantes (fixe, comparaison / viz) ---
    k_fix = min(10, X.shape[1])
    pca_10 = PCA(n_components=k_fix, random_state=42)
    Z_10 = pca_10.fit_transform(X)
    cols_10 = [f"PC{i+1}" for i in range(k_fix)]
    out_10 = pd.DataFrame(Z_10, columns=cols_10)
    out_10.insert(0, "ID", ids.values)
    out_10["cible_churn"] = y.values
    out_10.to_csv(SORTIE / "matrice_pca_10composantes.csv", index=False, encoding="utf-8-sig")

    var_exp = pd.DataFrame(
        {
            "composante": [f"PC{i+1}" for i in range(len(pca_95.explained_variance_ratio_))],
            "variance_expliquee": pca_95.explained_variance_ratio_,
            "variance_cumulee": np.cumsum(pca_95.explained_variance_ratio_),
        }
    )
    var_exp.to_csv(SORTIE / "pca_variance_expliquee.csv", index=False, encoding="utf-8-sig")

    # Composantes : contribution des variables originales (loadings)
    loadings = pd.DataFrame(
        pca_95.components_.T,
        index=X.columns,
        columns=[f"PC{i+1}" for i in range(n95)],
    )
    loadings.to_csv(SORTIE / "pca_loadings_95var.csv", encoding="utf-8-sig")

    # --- Sélection de variables (univariée) ---
    k_best = min(10, X.shape[1])
    skb = SelectKBest(score_func=f_classif, k=k_best)
    skb.fit(X, y)
    support = skb.get_support()
    cols_sel = X.columns[support].tolist()
    X_sel = X.loc[:, support].reset_index(drop=True)
    out_sel = pd.concat([ids.reset_index(drop=True), X_sel, y.reset_index(drop=True)], axis=1)
    out_sel.columns = ["ID"] + cols_sel + ["cible_churn"]
    out_sel.to_csv(SORTIE / "matrice_selectkbest_f10.csv", index=False, encoding="utf-8-sig")

    scores = pd.DataFrame(
        {"variable": X.columns, "score_f": skb.scores_, "selectionnee": support}
    ).sort_values("score_f", ascending=False)
    scores.to_csv(SORTIE / "selectkbest_scores.csv", index=False, encoding="utf-8-sig")

    schema = {
        "entree": str(matrice.relative_to(RACINE)),
        "n_lignes": int(len(X)),
        "n_features_init": int(X.shape[1]),
        "pca_95pct": {
            "n_composantes": int(n95),
            "variance_totale_expliquee": float(np.sum(pca_95.explained_variance_ratio_)),
        },
        "pca_10composantes": {
            "n_composantes": int(k_fix),
            "variance_totale_expliquee": float(np.sum(pca_10.explained_variance_ratio_)),
        },
        "selectkbest": {"k": int(k_best), "methode": "f_classif (ANOVA F entre classes)"},
        "interpretation": "PCA = axes orthogonaux, combinaisons linéaires des variables ; interprétation métier souvent plus difficile qu'avec variables brutes. SelectKBest conserve des noms de variables explicites.",
    }
    (SORTIE / "schema_pretraitement.json").write_text(
        json.dumps(schema, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    out_95.head(200).to_csv(SORTIE / "echantillon_pca_95var_200lignes.csv", index=False, encoding="utf-8-sig")

    txt = [
        "=== Prétraitement ===",
        f"Entrée : {matrice.name}",
        f"Lignes={len(X)}, features={X.shape[1]}",
        f"PCA 95% variance -> {n95} composantes",
        f"PCA {k_fix} composantes (fixe)",
        f"SelectKBest -> {k_best} variables",
        "",
        "Fichiers : matrice_pca_95var.csv, matrice_pca_10composantes.csv,",
        "pca_variance_expliquee.csv, pca_loadings_95var.csv,",
        "matrice_selectkbest_f10.csv, selectkbest_scores.csv, schema_pretraitement.json",
    ]
    (SORTIE / "synthese_pretraitement.txt").write_text("\n".join(txt), encoding="utf-8")

    print("OK —", SORTIE)


if __name__ == "__main__":
    main()
