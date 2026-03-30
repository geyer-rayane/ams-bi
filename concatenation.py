# -*- coding: utf-8 -*-
"""
Phase concaténation : empilement vertical sur colonnes communes (sans clé métier commune).

Implémentation principale :
  - concaténation des lignes de table1 et table2 sur les seules colonnes communes ;
  - colonne « source » (origine du fichier) ;
  - identifiants uniques régénérés pour éviter les collisions d'ID entre fichiers ;
  - export d'un extrait dans concatenation/ pour contrôle (pas d'export massif du jeu complet).

Descriptifs optionnels : moyennes / médianes par table sur les numériques communes
(sans test d'hypothèse — simple lecture exploratoire).

Sorties : dossier concatenation/
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

RACINE = Path(__file__).resolve().parent
DATA = RACINE / "data"
SORTIE = RACINE / "concatenation"

SENTINELLE_NON_DEM = "31/12/1900"


def charger() -> tuple[pd.DataFrame, pd.DataFrame]:
    t1 = pd.read_csv(DATA / "table1.csv", low_memory=False)
    t2 = pd.read_csv(DATA / "table2.csv", low_memory=False)
    return t1, t2


def colonnes_communes(t1: pd.DataFrame, t2: pd.DataFrame) -> list[str]:
    return sorted(set(t1.columns) & set(t2.columns))


def cible_table2(df: pd.DataFrame) -> pd.Series:
    """1 = démission observée (date réelle), 0 = sentinelle non-démission."""
    d = df["DTDEM"].astype(str).str.strip()
    return (d != SENTINELLE_NON_DEM).astype(int)


def union_verticale(t1: pd.DataFrame, t2: pd.DataFrame) -> pd.DataFrame:
    """
    Concaténation verticale sur les seules colonnes communes + indicateur de source.
    Conservation de l'ID d'origine dans id_fichier ; nouveaux ID globaux uniques.
    """
    communes = colonnes_communes(t1, t2)
    a = t1[communes].copy()
    b = t2[communes].copy()
    a["source"] = "table1"
    b["source"] = "table2"
    a["id_fichier"] = a["ID"]
    b["id_fichier"] = b["ID"]
    a["ID"] = range(1, len(a) + 1)
    b["ID"] = range(len(a) + 1, len(a) + len(b) + 1)
    return pd.concat([a, b], axis=0, ignore_index=True)


def union_avec_labels_churn(t1: pd.DataFrame, t2: pd.DataFrame) -> pd.DataFrame:
    """Union + label explicite (diagnostic du déséquilibre si fusion pour apprentissage naïf)."""
    u = union_verticale(t1, t2)
    labels = []
    for _, row in u.iterrows():
        if row["source"] == "table2":
            d = str(row["DTDEM"]).strip()
            labels.append(1 if d != SENTINELLE_NON_DEM else 0)
        else:
            labels.append(1)
    u["label_churn_obs"] = labels
    return u


def descriptifs_numeriques_sans_test(
    t1: pd.DataFrame, t2: pd.DataFrame, cols: list[str]
) -> pd.DataFrame:
    """Médiane et moyenne uniquement (pas de test statistique)."""
    rows = []
    for c in cols:
        if c not in t1.columns or c not in t2.columns:
            continue
        s1 = pd.to_numeric(t1[c], errors="coerce").dropna()
        s2 = pd.to_numeric(t2[c], errors="coerce").dropna()
        if len(s1) == 0 or len(s2) == 0:
            continue
        rows.append(
            {
                "variable": c,
                "n_table1": int(len(s1)),
                "n_table2": int(len(s2)),
                "median_table1": float(s1.median()),
                "median_table2": float(s2.median()),
                "mean_table1": float(s1.mean()),
                "mean_table2": float(s2.mean()),
            }
        )
    return pd.DataFrame(rows)


def resume_strategies(t1: pd.DataFrame, t2: pd.DataFrame, u: pd.DataFrame, uc: pd.DataFrame) -> dict:
    y2 = cible_table2(t2)
    return {
        "n_table1": len(t1),
        "n_table2": len(t2),
        "n_union": len(u),
        "colonnes_communes": colonnes_communes(t1, t2),
        "n_colonnes_communes": len(colonnes_communes(t1, t2)),
        "colonnes_solo_table1": sorted(set(t1.columns) - set(t2.columns)),
        "colonnes_solo_table2": sorted(set(t2.columns) - set(t1.columns)),
        "table2_taux_demission": float(y2.mean()),
        "table2_n_demission": int(y2.sum()),
        "table2_n_non_dem": int((y2 == 0).sum()),
        "union_n_label_1": int((uc["label_churn_obs"] == 1).sum()),
        "union_n_label_0": int((uc["label_churn_obs"] == 0).sum()),
        "note": "Sans test de comparaison de distributions, les écarts descriptifs restent indicatifs. Table1 = uniquement démissionnaires.",
    }


def main() -> None:
    SORTIE.mkdir(parents=True, exist_ok=True)
    t1, t2 = charger()
    communes = colonnes_communes(t1, t2)

    u = union_verticale(t1, t2)
    uc = union_avec_labels_churn(t1, t2)

    num_communes = [c for c in communes if c != "ID" and pd.api.types.is_numeric_dtype(t1[c])]
    desc = descriptifs_numeriques_sans_test(t1, t2, num_communes)
    desc.to_csv(SORTIE / "descriptifs_variables_communes.csv", index=False, encoding="utf-8-sig")

    res = resume_strategies(t1, t2, u, uc)
    (SORTIE / "resume_strategies.json").write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")

    u.head(500).to_csv(SORTIE / "echantillon_union_500lignes.csv", index=False, encoding="utf-8-sig")

    lignes = [
        "=== Concaténation : synthèse ===",
        f"Colonnes communes ({len(communes)}) : {', '.join(communes)}",
        f"Lignes table1={len(t1)}, table2={len(t2)}, union={len(u)}",
        f"Table2 taux démission (DTDEM != {SENTINELLE_NON_DEM}) : {res['table2_taux_demission']:.4f}",
        "",
        "Fichiers : resume_strategies.json, echantillon_union_500lignes.csv,",
        "descriptifs_variables_communes.csv (moyennes/médianes, sans test statistique).",
        "",
        "Les IDs d'origine sont dans id_fichier ; ID = identifiant unique dans l'union.",
    ]
    (SORTIE / "synthese_concatenation.txt").write_text("\n".join(lignes), encoding="utf-8")

    # Ancien fichier au nom trompeur : supprimer s'il existe pour éviter confusion
    ancien = SORTIE / "comparaison_distributions_numeriques.csv"
    if ancien.exists():
        ancien.unlink()

    print("OK — sorties dans", SORTIE)


if __name__ == "__main__":
    main()
