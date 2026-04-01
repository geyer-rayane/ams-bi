# -*- coding: utf-8 -*-
"""
Recodage des données pour la fouille (classification churn).

Deux modes :
  - Union complète (mode principal) : table1 + table2 fusionnées via concatenation.py.
    Conserve tous les individus, y compris les démissionnaires historiques de table1.
    Cible : cible_churn posée dans union_complete.csv.
    Sortie : union_matrice_modele.csv (entrée de pretraitement.py et decoupage.py).
  - Table2 seule (mode secondaire, backward-compat) : sociétaires courants uniquement.

Opérations :
  - dérivation d'âge et d'ancienneté selon la source (DTNAIS pour table2, AGEAD pour table1) ;
  - discrétisation quantiles MTREV ;
  - normalisation StandardScaler + MinMaxScaler ;
  - one-hot CDSITFAM et CDMOTDEM.

Sorties : dossier recodage/
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
UNION_CSV = RACINE / "concatenation" / "union_complete.csv"

DATE_EXTRACTION = pd.Timestamp("2007-01-01")
SENTINELLE_DTDEM = "31/12/1900"
SENTINELLE_NAISS = "0000-00-00"


def charger_table2() -> pd.DataFrame:
    return pd.read_csv(DATA / "table2.csv", low_memory=False)


def charger_union() -> pd.DataFrame:
    """Charge l'union complète produite par concatenation.py."""
    if not UNION_CSV.exists():
        raise FileNotFoundError(
            f"Exécuter d'abord concatenation.py. Manquant : {UNION_CSV}"
        )
    return pd.read_csv(UNION_CSV, low_memory=False)


def cible_churn(df: pd.DataFrame) -> pd.Series:
    d = df["DTDEM"].astype(str).str.strip()
    return (d != SENTINELLE_DTDEM).astype(int)


def derivations_temporelles(df: pd.DataFrame) -> pd.DataFrame:
    """Âge approximatif au 01/01/2007 et ancienneté d'adhésion (années) — table2 uniquement."""
    out = df.copy()
    nais = out["DTNAIS"].astype(str).str.strip()
    masque_nais = nais.eq(SENTINELLE_NAISS) | nais.isna()
    dt_nais = pd.to_datetime(nais.replace(SENTINELLE_NAISS, np.nan), errors="coerce")
    out["age_ref"] = (DATE_EXTRACTION - dt_nais).dt.days / 365.25
    out.loc[masque_nais | dt_nais.isna(), "age_ref"] = np.nan

    dt_adh = pd.to_datetime(out["DTADH"], dayfirst=True, errors="coerce")
    out["anciennete_adh_ans"] = (DATE_EXTRACTION - dt_adh).dt.days / 365.25
    return out


def derivations_temporelles_union(df: pd.DataFrame) -> pd.DataFrame:
    """
    Dérive âge et ancienneté pour l'union complète (table1 + table2).

    Stratégie d'âge :
      - source=table2 : (DATE_EXTRACTION - DTNAIS) / 365.25 (comme avant) ;
      - source=table1 : AGEAD (âge à l'adhésion) + anciennete_adh_ans donne l'âge
        approximatif à la date de référence. NaN si AGEAD indisponible.
    """
    out = df.copy()

    dt_adh = pd.to_datetime(out["DTADH"], dayfirst=True, errors="coerce")
    out["anciennete_adh_ans"] = (DATE_EXTRACTION - dt_adh).dt.days / 365.25

    age = pd.Series(np.nan, index=out.index, dtype=float)

    masque_t2 = out["source"] == "table2"
    if "DTNAIS" in out.columns:
        nais = out.loc[masque_t2, "DTNAIS"].astype(str).str.strip()
        masque_nais = nais.eq(SENTINELLE_NAISS) | nais.isna()
        dt_nais = pd.to_datetime(nais.replace(SENTINELLE_NAISS, np.nan), errors="coerce")
        age_t2 = (DATE_EXTRACTION - dt_nais).dt.days / 365.25
        age_t2[masque_nais] = np.nan
        age.loc[masque_t2] = age_t2.values

    masque_t1 = out["source"] == "table1"
    if "AGEAD" in out.columns:
        agead = pd.to_numeric(out.loc[masque_t1, "AGEAD"], errors="coerce")
        anc_t1 = out.loc[masque_t1, "anciennete_adh_ans"]
        age.loc[masque_t1] = (agead + anc_t1).values

    out["age_ref"] = age
    return out


def discretiser_mtrev(series: pd.Series, q: int = 5) -> pd.Series:
    """Discrétisation par quantiles (labels ordonnés)."""
    s = pd.to_numeric(series, errors="coerce")
    try:
        return pd.qcut(s, q=q, duplicates="drop", labels=False)
    except ValueError:
        return pd.Series(np.nan, index=series.index)


def preparer_features_brutes(df: pd.DataFrame) -> pd.DataFrame:
    """Mode table2 uniquement."""
    d = derivations_temporelles(df)
    d["cible_churn"] = cible_churn(df)
    d["CDSITFAM"] = d["CDSITFAM"].astype(str)
    d["CDMOTDEM"] = d["CDMOTDEM"].fillna("__MANQUANT__").astype(str)
    d["MTREV_discrete"] = discretiser_mtrev(d["MTREV"])
    return d


def preparer_features_union(df: pd.DataFrame) -> pd.DataFrame:
    """
    Mode union complète.

    La colonne cible_churn est déjà présente dans union_complete.csv.
    CDMOTDEM est disponible pour tous les churners de table1 ; NaN pour non-churners
    de table2 — recodé __MANQUANT__ pour one-hot.
    """
    d = derivations_temporelles_union(df)
    d["CDSITFAM"] = d["CDSITFAM"].astype(str)
    d["CDMOTDEM"] = d["CDMOTDEM"].fillna("__MANQUANT__").astype(str)
    d["MTREV_discrete"] = discretiser_mtrev(d["MTREV"])
    return d


def matrice_numerique_normalisee(
    df: pd.DataFrame, cols_num: list[str] | None = None
) -> tuple[pd.DataFrame, dict]:
    """
    Colonnes numériques brutes (hors ID, hors cible) + normalisation.
    Deux variantes : Z-score (StandardScaler) et [0,1] (MinMaxScaler).
    Imputation médiane sur les NaN avant scaling.
    """
    if cols_num is None:
        cols_num = [
            "CDSEXE",
            "MTREV",
            "NBENF",
            "CDTMT",
            "CDCATCL",
            "BPADH",
            "age_ref",
            "anciennete_adh_ans",
        ]
    cols_num = [c for c in cols_num if c in df.columns]
    X = df[cols_num].copy()
    meta = {"colonnes": cols_num, "n": len(X)}
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


def _construire_matrice(prep: pd.DataFrame, nom: str, sortie: Path) -> pd.DataFrame:
    """Construit et exporte la matrice modèle (normalisation + one-hot) pour un DataFrame préparé."""
    num_norm, meta_num = matrice_numerique_normalisee(prep)
    dummies = matrice_one_hot(prep)
    combine = pd.concat([prep[["ID", "cible_churn"]], num_norm, dummies], axis=1)
    combine.to_csv(sortie / f"{nom}_matrice_modele.csv", index=False, encoding="utf-8-sig")
    combine.head(300).to_csv(
        sortie / f"echantillon_{nom}_matrice_300lignes.csv", index=False, encoding="utf-8-sig"
    )
    return combine


def main() -> None:
    SORTIE.mkdir(parents=True, exist_ok=True)

    # --- Mode union complète (pipeline principal) ---
    raw_union = charger_union()
    prep_union = preparer_features_union(raw_union)

    cols_export_union = [
        c for c in [
            "ID", "source", "id_fichier", "CDSEXE", "MTREV", "NBENF", "CDSITFAM",
            "CDTMT", "CDMOTDEM", "CDCATCL", "BPADH", "DTDEM",
            "age_ref", "anciennete_adh_ans", "MTREV_discrete", "cible_churn",
        ]
        if c in prep_union.columns
    ]
    prep_union[cols_export_union].to_csv(
        SORTIE / "union_recodage_brut_derive.csv", index=False, encoding="utf-8-sig"
    )
    combine_union = _construire_matrice(prep_union, "union", SORTIE)

    # --- Mode table2 (backward-compat) ---
    raw_t2 = charger_table2()
    prep_t2 = preparer_features_brutes(raw_t2)

    cols_export_t2 = [
        c for c in [
            "ID", "CDSEXE", "MTREV", "NBENF", "CDSITFAM", "CDTMT",
            "CDMOTDEM", "CDCATCL", "BPADH", "DTDEM",
            "age_ref", "anciennete_adh_ans", "MTREV_discrete", "cible_churn",
        ]
        if c in prep_t2.columns
    ]
    prep_t2[cols_export_t2].to_csv(
        SORTIE / "table2_recodage_brut_derive.csv", index=False, encoding="utf-8-sig"
    )
    combine_t2 = _construire_matrice(prep_t2, "table2", SORTIE)
    # Alias pour compatibilité descendante (pretraitement.py / decoupage.py anciens)
    combine_t2.to_csv(SORTIE / "table2_matrice_modele.csv", index=False, encoding="utf-8-sig")

    schema = {
        "date_reference_extraction": str(DATE_EXTRACTION.date()),
        "cible": "cible_churn (1 = démission)",
        "sentinelles": {
            "DTNAIS": f"{SENTINELLE_NAISS} → age_ref NaN puis imputation médiane",
            "DTDEM": f"{SENTINELLE_DTDEM} → cible 0",
        },
        "derivations": [
            "age_ref (DTNAIS pour table2, AGEAD+anciennete pour table1)",
            "anciennete_adh_ans depuis DTADH",
            "MTREV_discrete quantiles",
        ],
        "normalisation": {
            "z_*": "StandardScaler (moyenne 0, écart-type 1)",
            "mm_*": "MinMaxScaler [0,1]",
        },
        "categoriel": "one-hot (sit_*, mot_*) ; CDMOTDEM manquant → __MANQUANT__",
        "union": {
            "n_lignes": int(len(combine_union)),
            "n_colonnes": int(combine_union.shape[1]),
            "n_churn_1": int((combine_union["cible_churn"] == 1).sum()),
            "n_churn_0": int((combine_union["cible_churn"] == 0).sum()),
            "taux_churn": round(float(combine_union["cible_churn"].mean()), 6),
        },
        "table2": {
            "n_lignes": int(len(combine_t2)),
            "n_colonnes": int(combine_t2.shape[1]),
        },
        "fichier_principal_pipeline": "union_matrice_modele.csv",
    }
    (SORTIE / "schema_recodage.json").write_text(
        json.dumps(schema, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    txt = [
        "=== Recodage ===",
        f"Union : {len(combine_union)} lignes, {combine_union.shape[1]} colonnes — matrice principale (pipeline ML).",
        f"Table2 seule : {len(combine_t2)} lignes — conservée pour compatibilité.",
        "",
        "Fichiers principaux : union_matrice_modele.csv, union_recodage_brut_derive.csv",
        "Fichiers table2 : table2_matrice_modele.csv, table2_recodage_brut_derive.csv",
        "Schéma : schema_recodage.json",
    ]
    (SORTIE / "synthese_recodage.txt").write_text("\n".join(txt), encoding="utf-8")

    print("OK —", SORTIE)
    print(f"  union_matrice_modele.csv  : {len(combine_union)} lignes, {combine_union.shape[1]} cols")
    print(f"  table2_matrice_modele.csv : {len(combine_t2)} lignes, {combine_t2.shape[1]} cols")


if __name__ == "__main__":
    main()
