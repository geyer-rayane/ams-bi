# -*- coding: utf-8 -*-
"""
Prétraitement : réduction de dimension par ACP.

Entrée : recodage/union_matrice_modele.csv (priorité) ou table2_matrice_modele.csv.
Méthode unique : PCA à 95 % de variance expliquée.

Le nombre de composantes retenu est affiché en terminal et dépend de la matrice d'entrée.
Les loadings (contributions des variables originales) sont exportés pour l'interprétabilité.

Sorties : dossier pretraitement/
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

RACINE = Path(__file__).resolve().parent
_MATRICE_UNION = RACINE / "recodage" / "union_matrice_modele.csv"
_MATRICE_T2 = RACINE / "recodage" / "table2_matrice_modele.csv"
SORTIE = RACINE / "pretraitement"

COLS_META = {"ID", "cible_churn", "source", "id_fichier"}


def charger_matrice(chemin: Path) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    df = pd.read_csv(chemin, low_memory=False)
    y = df["cible_churn"]
    ids = df["ID"]
    X = df.drop(columns=[c for c in COLS_META if c in df.columns])
    return X, y, ids


def main() -> None:
    SORTIE.mkdir(parents=True, exist_ok=True)
    matrice = _MATRICE_UNION if _MATRICE_UNION.exists() else _MATRICE_T2
    if not matrice.exists():
        raise FileNotFoundError("Exécuter d'abord concatenation.py puis recodage.py.")

    X, y, ids = charger_matrice(matrice)

    pca = PCA(n_components=0.95, random_state=42)
    Z = pca.fit_transform(X)
    n_comp = pca.n_components_
    var_totale = float(np.sum(pca.explained_variance_ratio_))

    cols = [f"PC{i+1}" for i in range(n_comp)]
    out = pd.DataFrame(Z, columns=cols)
    out.insert(0, "ID", ids.values)
    out["cible_churn"] = y.values
    out.to_csv(SORTIE / "matrice_pca_95var.csv", index=False, encoding="utf-8-sig")

    pd.DataFrame(
        {
            "composante": cols,
            "variance_expliquee": pca.explained_variance_ratio_,
            "variance_cumulee": np.cumsum(pca.explained_variance_ratio_),
        }
    ).to_csv(SORTIE / "pca_variance_expliquee.csv", index=False, encoding="utf-8-sig")

    pd.DataFrame(
        pca.components_.T,
        index=X.columns,
        columns=cols,
    ).to_csv(SORTIE / "pca_loadings.csv", encoding="utf-8-sig")

    schema = {
        "entree": str(matrice.relative_to(RACINE)),
        "n_lignes": int(len(X)),
        "n_features_init": int(X.shape[1]),
        "pca_95pct": {
            "n_composantes": int(n_comp),
            "variance_totale_expliquee": round(var_totale, 6),
        },
    }
    (SORTIE / "schema_pretraitement.json").write_text(
        json.dumps(schema, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print(f"Prétraitement OK — {matrice.name}")
    print(f"  Lignes={len(X)}  features initiales={X.shape[1]}")
    print(f"  PCA 95% variance -> {n_comp} composantes principales (variance expliquee : {var_totale:.4f})")


if __name__ == "__main__":
    main()
