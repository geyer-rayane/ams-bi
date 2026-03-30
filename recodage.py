# -*- coding: utf-8 -*-
"""
Recodage des données pour la fouille (classification churn) — table2 comme base principale.

Opérations typiques :
  - cible binaire à partir de DTDEM ;
  - traitement des sentinelles ;
  - dérivation de variables (âge, ancienneté à la date d'extraction 2007) ;
  - discrétisation (quantiles sur MTREV, etc.) ;
  - normalisation (StandardScaler, MinMaxScaler) sur blocs numériques ;
  - numérisation catégorielle (one-hot via pandas).

Sorties : dossier recodage/ (CSV + schéma JSON). Tester l'effet des codages sur chaque algorithme
reste une étape ultérieure (comparaison empirique des modèles).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler

RACINE = Path(__file__).resolve().parent
DATA = RACINE / "data"
SORTIE = RACINE / "recodage"

DATE_EXTRACTION = pd.Timestamp("2007-01-01")
SENTINELLE_DTDEM = "31/12/1900"
SENTINELLE_NAISS = "0000-00-00"


def charger_table2() -> pd.DataFrame:
    return pd.read_csv(DATA / "table2.csv", low_memory=False)


def cible_churn(df: pd.DataFrame) -> pd.Series:
    d = df["DTDEM"].astype(str).str.strip()
    return (d != SENTINELLE_DTDEM).astype(int)


def derivations_temporelles(df: pd.DataFrame) -> pd.DataFrame:
    """Âge approximatif au 01/01/2007 et ancienneté d'adhésion (années)."""
    out = df.copy()
    nais = out["DTNAIS"].astype(str).str.strip()
    masque_nais = nais.eq(SENTINELLE_NAISS) | nais.isna()
    dt_nais = pd.to_datetime(nais.replace(SENTINELLE_NAISS, np.nan), errors="coerce")
    out["age_2007"] = (DATE_EXTRACTION - dt_nais).dt.days / 365.25
    out.loc[masque_nais | dt_nais.isna(), "age_2007"] = np.nan

    dt_adh = pd.to_datetime(out["DTADH"], dayfirst=True, errors="coerce")
    out["anciennete_adh_ans"] = (DATE_EXTRACTION - dt_adh).dt.days / 365.25
    return out


def discretiser_mtrev(series: pd.Series, q: int = 5) -> pd.Series:
    """Discrétisation par quantiles (labels ordonnés)."""
    s = pd.to_numeric(series, errors="coerce")
    try:
        return pd.qcut(s, q=q, duplicates="drop", labels=False)
    except ValueError:
        return pd.Series(np.nan, index=series.index)


def preparer_features_brutes(df: pd.DataFrame) -> pd.DataFrame:
    d = derivations_temporelles(df)
    d["cible_churn"] = cible_churn(df)
    # Qualitatifs : chaîne stable pour get_dummies
    d["CDSITFAM"] = d["CDSITFAM"].astype(str)
    d["CDMOTDEM"] = d["CDMOTDEM"].fillna("__MANQUANT__").astype(str)
    d["MTREV_discrete"] = discretiser_mtrev(d["MTREV"])
    return d


def matrice_numerique_normalisee(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Colonnes numériques brutes (hors ID, hors cible) + normalisation.
    Deux variantes : Z-score (StandardScaler) et [0,1] (MinMaxScaler).
    """
    cols_num = [
        "CDSEXE",
        "MTREV",
        "NBENF",
        "CDTMT",
        "CDCATCL",
        "BPADH",
        "age_2007",
        "anciennete_adh_ans",
    ]
    X = df[cols_num].copy()
    meta = {"colonnes": cols_num, "n": len(X)}
    # Remplacer inf / garder NaN pour que le scaler ignore ou imputer - ici imputation médiane simple
    for c in cols_num:
        med = X[c].median()
        X[c] = X[c].fillna(med)

    z = StandardScaler().fit_transform(X)
    mm = MinMaxScaler().fit_transform(X)

    df_z = pd.DataFrame(z, columns=[f"z_{c}" for c in cols_num], index=df.index)
    df_mm = pd.DataFrame(mm, columns=[f"mm_{c}" for c in cols_num], index=df.index)
    return pd.concat([df_z, df_mm], axis=1), meta


def matrice_one_hot(df: pd.DataFrame) -> pd.DataFrame:
    """Numérisation catégorielle (one-hot) pour CDSITFAM et CDMOTDEM."""
    dummies = pd.get_dummies(
        df[["CDSITFAM", "CDMOTDEM"]],
        columns=["CDSITFAM", "CDMOTDEM"],
        prefix=["sit", "mot"],
        dtype=int,
    )
    return dummies


def main() -> None:
    SORTIE.mkdir(parents=True, exist_ok=True)
    raw = charger_table2()
    prep = preparer_features_brutes(raw)

    # Jeux exportés
    colonnes_export_brut = [
        "ID",
        "CDSEXE",
        "MTREV",
        "NBENF",
        "CDSITFAM",
        "CDTMT",
        "CDMOTDEM",
        "CDCATCL",
        "BPADH",
        "DTDEM",
        "age_2007",
        "anciennete_adh_ans",
        "MTREV_discrete",
        "cible_churn",
    ]
    prep[colonnes_export_brut].to_csv(
        SORTIE / "table2_recodage_brut_derive.csv", index=False, encoding="utf-8-sig"
    )

    num_norm, meta_num = matrice_numerique_normalisee(prep)
    dummies = matrice_one_hot(prep)

    combine = pd.concat(
        [
            prep[["ID", "cible_churn"]],
            num_norm,
            dummies,
        ],
        axis=1,
    )
    combine.to_csv(SORTIE / "table2_matrice_modele.csv", index=False, encoding="utf-8-sig")

    combine.head(300).to_csv(
        SORTIE / "echantillon_matrice_modele_300lignes.csv", index=False, encoding="utf-8-sig"
    )

    schema = {
        "date_reference_extraction": str(DATE_EXTRACTION.date()),
        "cible": "cible_churn (1 = démission, DTDEM != 31/12/1900)",
        "sentinelles": {
            "DTNAIS": f"{SENTINELLE_NAISS} -> age_2007 NaN puis imputation médiane pour scaler",
            "DTDEM": f"{SENTINELLE_DTDEM} -> cible 0",
        },
        "derivations": ["age_2007 depuis DTNAIS", "anciennete_adh_ans depuis DTADH", "MTREV_discrete quantiles"],
        "normalisation": {
            "z_*": "StandardScaler (moyenne 0, écart-type 1) sur bloc numérique",
            "mm_*": "MinMaxScaler [0,1] sur le même bloc",
        },
        "categoriel": "one-hot (sit_*, mot_*) ; CDMOTDEM manquant recodé __MANQUANT__",
        "algorithmes": {
            "arbres_SVM_naive_bayes": "souvent mieux avec variables peu transformées ou discrétisées ; normalisation moins critique pour arbres",
            "regression_logistique_reseaux": "normalisation / one-hot recommandées",
            "knn": "normalisation indispensable pour distances",
        },
        "dimensions": {"n_lignes": len(combine), "n_colonnes": combine.shape[1]},
    }
    (SORTIE / "schema_recodage.json").write_text(
        json.dumps(schema, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    txt = [
        "=== Recodage (table2) ===",
        f"Lignes : {len(prep)}, matrice modèle : {combine.shape[1]} colonnes.",
        "Fichiers : table2_recodage_brut_derive.csv, table2_matrice_modele.csv,",
        "echantillon_matrice_modele_300lignes.csv, schema_recodage.json",
        "",
        "Effet du codage : à comparer empiriquement selon l'algorithme (voir schema_recodage.json).",
    ]
    (SORTIE / "synthese_recodage.txt").write_text("\n".join(txt), encoding="utf-8")

    print("OK —", SORTIE)


if __name__ == "__main__":
    main()
