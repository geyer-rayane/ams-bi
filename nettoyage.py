# -*- coding: utf-8 -*-
"""
Analyse de nettoyage et de préparation des données — indépendante de exploration.py.

Objectifs :
  - quantifier valeurs manquantes et sentinelles « aberrantes » au sens métier ;
  - repérer redondances et attributs potentiellement superflus ;
  - proposer une typologie (numérique réel vs ordinal vs catégoriel) ;
  - esquisser des traitements possibles sans appliquer ici de pipeline de modélisation.

Sorties : dossier nettoyage/ (CSV + synthèse textuelle).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

RACINE = Path(__file__).resolve().parent
DATA = RACINE / "data"
SORTIE = RACINE / "nettoyage"

# Sentinelles connues (énoncé + inspection)
SENTINELLE_DTNAIS_ABSENT = "0000-00-00"
SENTINELLE_DTDEM_NON_DEM = "31/12/1900"


def charger_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    t1 = pd.read_csv(DATA / "table1.csv", low_memory=False)
    t2 = pd.read_csv(DATA / "table2.csv", low_memory=False)
    return t1, t2


def rapport_manquants(df: pd.DataFrame, nom: str) -> pd.DataFrame:
    """Compte NaN/pandas ; pour objets, compte aussi chaînes vides après strip."""
    lignes = []
    n = len(df)
    for c in df.columns:
        s = df[c]
        na = s.isna().sum()
        if s.dtype == object:
            vide = (s.astype(str).str.strip() == "").sum() - s.isna().sum()
            vide = max(0, int(vide))
        else:
            vide = 0
        manq = int(na + vide)
        lignes.append(
            {
                "table": nom,
                "colonne": c,
                "n_lignes": n,
                "manquants_pandas": int(na),
                "chaines_vides": vide,
                "total_manquants": manq,
                "pct": round(100.0 * manq / n, 4) if n else 0.0,
            }
        )
    return pd.DataFrame(lignes)


def analyse_sentinelles_table2(df: pd.DataFrame) -> dict:
    """Valeurs explicitement non informatives selon l'énoncé / parse."""
    out: dict = {}
    if "DTNAIS" in df.columns:
        s = df["DTNAIS"].astype(str).str.strip()
        out["DTNAIS_0000_00_00"] = int((s == SENTINELLE_DTNAIS_ABSENT).sum())
        out["DTNAIS_autres"] = int(len(s) - out["DTNAIS_0000_00_00"] - s.isna().sum())
    if "DTDEM" in df.columns:
        s = df["DTDEM"].astype(str).str.strip()
        out["DTDEM_31_12_1900_non_dem"] = int((s == SENTINELLE_DTDEM_NON_DEM).sum())
        out["DTDEM_autres"] = int((s != SENTINELLE_DTDEM_NON_DEM).sum() - s.isna().sum())
    if "CDMOTDEM" in df.columns:
        s = df["CDMOTDEM"]
        na = s.isna()
        vide = (~na) & (s.astype(str).str.strip().isin(["", "nan"]))
        out["CDMOTDEM_vide_ou_na"] = int(na.sum() + vide.sum())
    return out


def analyse_sentinelles_table1(df: pd.DataFrame) -> dict:
    """Table1 : démissionnaires ; pas de 31/12/1900 attendu comme « non démission »."""
    out: dict = {}
    if "RANGADH" in df.columns:
        out["RANGADH_manquants"] = int(df["RANGADH"].isna().sum())
        out["RANGADH_pct"] = round(100 * out["RANGADH_manquants"] / len(df), 4)
    return out


def outliers_iqr(df: pd.DataFrame, col: str, k: float = 1.5) -> pd.Series:
    """Masque booléen : valeurs hors [Q1 - k*IQR, Q3 + k*IQR] sur colonne numérique."""
    s = pd.to_numeric(df[col], errors="coerce")
    q1, q3 = s.quantile(0.25), s.quantile(0.75)
    iqr = q3 - q1
    low, high = q1 - k * iqr, q3 + k * iqr
    return (s < low) | (s > high)


def resume_outliers_numeriques(df: pd.DataFrame, cols: list[str], nom: str) -> pd.DataFrame:
    rows = []
    for c in cols:
        if c not in df.columns:
            continue
        if not pd.api.types.is_numeric_dtype(df[c]):
            continue
        mask = outliers_iqr(df, c)
        s = pd.to_numeric(df[c], errors="coerce")
        rows.append(
            {
                "table": nom,
                "colonne": c,
                "n": len(df),
                "outliers_iqr_1_5": int(mask.sum()),
                "pct_outliers": round(100 * mask.mean(), 4) if len(df) else 0.0,
                "min": float(s.min()),
                "max": float(s.max()),
            }
        )
    return pd.DataFrame(rows)


def typologie_attributs() -> pd.DataFrame:
    """
    Nature sémantique recommandée (hors choix pandas int/float/object).
    'intervalle' = écart entre deux valeurs interprétable (âge, années, revenu).
    'ordinal' = codes ordonnés ou quasi (tranches).
    'nominal' = catégories sans ordre naturel.
    'identifiant' = clé technique.
    'derive' = redondant avec d'autres champs si bien calculés.
    """
    rows = [
        # table1 + table2 communs
        ("ID", "identifiant", "Exclure du modèle ou clé de jointure interne."),
        ("CDSEXE", "nominal_ordinal", "Codes sous-classes ; plutôt nominal ou ordinal faible — pas une échelle de rapport."),
        ("MTREV", "intervalle_ratio", "Revenu : continu ; nombreux zéros et queue lourde → winsorisation / log ou binning."),
        ("NBENF", "intervalle_discret", "Compte entier ; 0 légitime."),
        ("CDSITFAM", "nominal", "Situation familiale."),
        ("DTADH", "date_brute", "Parser en date ; forte cardinalité en chaîne."),
        ("CDTMT", "nominal", "Statut sociétaire (code)."),
        ("CDCATCL", "nominal", "Type client."),
        ("BPADH", "inconnu", "Signification inconnue (énoncé) — à clarifier avant usage."),
        # table1
        ("CDDEM", "nominal", "Code démission ; quasi constant dans table1."),
        ("DTDEM", "date_brute", "Date de démission."),
        ("ANNEEDEM", "ordinal_intervalle", "Année : peut servir de feature temporelle ; liée à DTDEM."),
        ("CDMOTDEM", "nominal", "Motif démission."),
        ("AGEAD", "intervalle", "Âge à l'adhésion — variable de prédiction naturelle."),
        ("RANGAGEAD", "ordinal", "Tranche d'âge à l'adhésion — redondante avec AGEAD si bins cohérents."),
        ("AGEDEM", "intervalle", "Âge à la démission."),
        ("RANGAGEDEM", "ordinal", "Tranche âge démission — redondante avec AGEDEM."),
        ("RANGDEM", "ordinal", "Code année démission format texte — redondant avec ANNEEDEM/DTDEM."),
        ("ADH", "intervalle", "Ancienneté en années."),
        ("RANGADH", "ordinal", "Tranche ancienneté — redondante avec ADH ; manquants partiels."),
        # table2
        ("DTNAIS", "date_brute", "Date naissance ; sentinelles 0000-00-00 = manquant déclaré."),
        ("DTDEM", "date_ou_sentinelle", "31/12/1900 = non-démissionnaire (pas une date réelle)."),
    ]
    return pd.DataFrame(rows, columns=["colonne", "nature", "commentaire"])


def redondances_explicites() -> pd.DataFrame:
    """Décisions justifiées par l'analyse exploratoire (corrélations / structure)."""
    rows = [
        (
            "AGEAD vs RANGAGEAD",
            "forte dépendance (tranches dérivées de l'âge) ; garder l'âge numérique ou les tranches, pas les deux sans régularisation.",
        ),
        (
            "AGEDEM vs RANGAGEDEM",
            "idem.",
        ),
        (
            "ADH vs RANGADH",
            "idem ; RANGADH a des manquants en table1.",
        ),
        (
            "ANNEEDEM vs DTDEM / RANGDEM",
            "information temporelle dupliquée sous plusieurs formats.",
        ),
        (
            "table1 uniquement démissionnaires",
            "population différente de table2 ; ne pas fusionner naïvement pour un score sur sociétaires actuels.",
        ),
    ]
    return pd.DataFrame(rows, columns=["paire_ou_fait", "recommandation"])


def justification_filtrage_instances(t2: pd.DataFrame) -> str:
    """
    Répond à : faut-il écarter des instances non liées au problème ?

    Position prudente : ne pas supprimer massivement des lignes sans cible ou sans définition
    opérationnelle du problème. Les sentinelles (0000-00-00, 31/12/1900) sont des *codes*,
    pas nécessairement des erreurs à effacer.

    Suppression uniquement si :
      - doublons exacts sur clé métier (ici pas de clé stable inter-tables) ;
      - ligne entièrement vide de prédicteurs (rare).

    Exclure *toute* la table1 pour prédire sur table2 serait une exclusion de *population*,
    pas de « mauvaises » lignes : table1 sert plutôt historique / recalibrage.
    """
    n = len(t2)
    entierement_na = t2.isna().all(axis=1).sum()
    lignes = [
        "=== Filtrage d'instances ===",
        f"Table2 lignes : {n}",
        f"Lignes avec toutes les colonnes NaN (pandas) : {int(entierement_na)} — généralement 0.",
        "",
        "Recommandation : ne pas supprimer les lignes avec DTNAIS=0000-00-00 ni DTDEM=31/12/1900 :",
        "  ce sont des non-informations codées, à traiter comme données manquantes ou catégorie « inconnu ».",
        "",
        "Exclure des individus « non liés au problème » se justifie seulement après définition formelle",
        "de la population cible (ex. sociétaires actifs à la date d'extraction 2007) et de la variable",
        "à prédire ; ce périmètre n'est pas tranché dans ce script.",
    ]
    return "\n".join(lignes)


def ecrire_synthese_textuelle(
    manq1: pd.DataFrame,
    manq2: pd.DataFrame,
    sent2: dict,
    sent1: dict,
    outl1: pd.DataFrame,
    outl2: pd.DataFrame,
    chemin: Path,
) -> None:
    """Réponses structurées aux questions du sujet (résumé)."""
    lines = [
        "=== NETTOYAGE / PRÉPARATION — SYNTHÈSE (générée par nettoyage.py) ===",
        "",
        "1) Ces données nécessitent-elles un nettoyage ?",
        "   Oui, au sens « préparation pour modélisation » : parser les dates, encoder les catégories,",
        "   traiter les sentinelles comme manquants ou modalités dédiées, et gérer les outliers",
        "   sur MTREV (et variables financières) si le modèle l'exige.",
        "",
        "2) Faut-il écarter des instances non liées au problème ?",
        "   Pas de suppression massive recommandée ici : les codes aberrants connus sont des sentinelles.",
        "   Filtrer des lignes n'est pertinent qu'après définition de la population cible et de la cible Y.",
        "",
        "3) Sur quoi se baser pour décider ?",
        "   Dictionnaire de variables, taux de manquants, corrélation/ redondance (exploration),",
        "   et contraintes du modèle (ex. arbres vs régression linéaire).",
        "",
        "4) Valeurs manquantes (aperçu table1 RANGADH, table2 CDMOTDEM pour non-démissionnaires) :",
        f"   table1 RANGADH manquants : {sent1.get('RANGADH_manquants', 'N/A')} ({sent1.get('RANGADH_pct', '')} %).",
        f"   table2 CDMOTDEM vides/NA : {sent2.get('CDMOTDEM_vide_ou_na', 'N/A')} (motif absent si pas démission).",
        "",
        "5) Valeurs « aberrantes » au sens énoncé / métier :",
        f"   DTNAIS = 0000-00-00 : {sent2.get('DTNAIS_0000_00_00', 'N/A')} (table2).",
        f"   DTDEM = 31/12/1900 (non démissionnaire) : {sent2.get('DTDEM_31_12_1900_non_dem', 'N/A')} (table2).",
        "",
        "6) Attributs redondants : tranches RANG* vs âges/années numériques ; ANNEEDEM vs dates.",
        "",
        "7) Attributs superflus pour scoring : ID ; possiblement doublons informationnels (RANG* si AGE* gardé).",
        "   BPADH : superflu tant que la définition métier est inconnue.",
        "",
        "8) Entiers lus par pandas : souvent catégoriels (CDSEXE, CDTMT) ou ordinaux — ne pas assimiler",
        "   à une échelle de rapport sans vérification.",
        "",
        "9) Traitements possibles : imputation / catégorie 'manquant', one-hot ou cibles de catégories,",
        "   dates → timestamp ou année-mois, agrégation des modalités rares, winsorisation ou log(MTREV+1).",
        "",
        "10) Outliers (règle IQR 1.5) : indicatif seulement ; pour comptages à masse en 0 (NBENF) ou",
        "    variables à queue lourde (MTREV), le pourcentage d'« outliers » peut être artificiellement élevé.",
        "",
        "Fichiers CSV détaillés : manquants_*.csv, outliers_iqr_*.csv, sentinelles_table2.json",
    ]
    chemin.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    SORTIE.mkdir(parents=True, exist_ok=True)
    t1, t2 = charger_tables()

    manq1 = rapport_manquants(t1, "table1")
    manq2 = rapport_manquants(t2, "table2")
    manq1.to_csv(SORTIE / "manquants_table1.csv", index=False, encoding="utf-8-sig")
    manq2.to_csv(SORTIE / "manquants_table2.csv", index=False, encoding="utf-8-sig")

    sent2 = analyse_sentinelles_table2(t2)
    sent1 = analyse_sentinelles_table1(t1)
    (SORTIE / "sentinelles_table2.json").write_text(json.dumps(sent2, indent=2), encoding="utf-8")
    (SORTIE / "sentinelles_table1.json").write_text(json.dumps(sent1, indent=2), encoding="utf-8")

    num_cols_t1 = [c for c in t1.columns if pd.api.types.is_numeric_dtype(t1[c]) and c != "ID"]
    num_cols_t2 = [c for c in t2.columns if pd.api.types.is_numeric_dtype(t2[c]) and c != "ID"]
    outl1 = resume_outliers_numeriques(t1, num_cols_t1, "table1")
    outl2 = resume_outliers_numeriques(t2, num_cols_t2, "table2")
    outl1.to_csv(SORTIE / "outliers_iqr_table1.csv", index=False, encoding="utf-8-sig")
    outl2.to_csv(SORTIE / "outliers_iqr_table2.csv", index=False, encoding="utf-8-sig")

    typologie_attributs().to_csv(SORTIE / "typologie_attributs.csv", index=False, encoding="utf-8-sig")
    redondances_explicites().to_csv(SORTIE / "redondances.csv", index=False, encoding="utf-8-sig")

    filtre_txt = justification_filtrage_instances(t2)
    (SORTIE / "justification_filtrage_instances.txt").write_text(filtre_txt, encoding="utf-8")

    ecrire_synthese_textuelle(manq1, manq2, sent2, sent1, outl1, outl2, SORTIE / "synthese_questions_sujet.txt")

    print(f"Analyse terminée. Sorties dans : {SORTIE}")
    print("Fichiers principaux : synthese_questions_sujet.txt, manquants_*.csv, sentinelles_*.json")


if __name__ == "__main__":
    main()
