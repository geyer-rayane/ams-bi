"""
Exploration univariée et bivariée des tables table1 et table2.
Aucun nettoyage : données lues telles quelles.
Sorties : exploration/table1/{univariate,bivariate}/ et exploration/table2/{univariate,bivariate}/
"""

from __future__ import annotations

import itertools
from pathlib import Path

import pandas as pd

from analyse import (
    association_qual_qual,
    association_quant_qual,
    correlation_quant_quant,
    resume_qualitatif,
    resume_quantitatif,
)
from graphique import (
    barres_contingence,
    boites_a_moustaches,
    diagramme_barres,
    heatmap_correlations,
    histogramme,
    nuage_points,
)

RACINE = Path(__file__).resolve().parent
DATA = RACINE / "data"
EXPLORATION = RACINE / "exploration"


def _nom_fichier(col: str) -> str:
    return "".join(c if c not in r'\/:*?"<>|' else "_" for c in col)


def _paire_nom(c1: str, c2: str) -> str:
    a, b = sorted([_nom_fichier(c1), _nom_fichier(c2)])
    return f"{a}__{b}"


def colonnes_hors_id(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c != "ID"]


def partition_quant_qual(df: pd.DataFrame, colonnes: list[str]) -> tuple[list[str], list[str]]:
    quant, qual = [], []
    for c in colonnes:
        if pd.api.types.is_numeric_dtype(df[c]):
            quant.append(c)
        else:
            qual.append(c)
    return quant, qual


def executer_univarie_table(df: pd.DataFrame, nom_table: str, base_sortie: Path) -> pd.DataFrame:
    """Résumés + graphiques univariés pour chaque colonne (sauf ID)."""
    uni = base_sortie / "univariate"
    uni.mkdir(parents=True, exist_ok=True)
    cols = colonnes_hors_id(df)
    quant, qual = partition_quant_qual(df, cols)
    lignes = []
    for c in cols:
        if c in quant:
            stats = resume_quantitatif(df[c])
            stats["colonne"] = c
            stats["type"] = "quantitatif"
            lignes.append(stats)
            histogramme(df[c], uni / f"{_nom_fichier(c)}_histogramme.png")
        else:
            stats = resume_qualitatif(df[c])
            stats["colonne"] = c
            stats["type"] = "qualitatif"
            lignes.append(stats)
            diagramme_barres(df[c], uni / f"{_nom_fichier(c)}_barres.png")
    out = pd.DataFrame(lignes)
    out.to_csv(uni / f"resume_univarie_{nom_table}.csv", index=False, encoding="utf-8-sig")
    return out


def executer_bivarie_table(df: pd.DataFrame, nom_table: str, base_sortie: Path) -> pd.DataFrame:
    """Toutes les paires de colonnes (hors ID) : tests via analyse, figures via graphique."""
    bi = base_sortie / "bivariate"
    bi.mkdir(parents=True, exist_ok=True)
    cols = colonnes_hors_id(df)
    quant, qual = partition_quant_qual(df, cols)

    if len(quant) >= 2:
        heatmap_correlations(
            df,
            quant,
            bi / f"heatmap_pearson_{nom_table}.png",
            methode="pearson",
            titre=f"{nom_table} — Pearson",
        )
        heatmap_correlations(
            df,
            quant,
            bi / f"heatmap_spearman_{nom_table}.png",
            methode="spearman",
            titre=f"{nom_table} — Spearman",
        )

    lignes_bivar = []
    for c1, c2 in itertools.combinations(cols, 2):
        q1 = c1 in quant
        q2 = c2 in quant
        base = _paire_nom(c1, c2)

        if q1 and q2:
            mes = correlation_quant_quant(df[c1], df[c2])
            mes["var1"] = c1
            mes["var2"] = c2
            mes["cas"] = "quantitatif_quantitatif"
            lignes_bivar.append(mes)
            nuage_points(df[c1], df[c2], bi / f"{base}_nuage.png")

        elif not q1 and not q2:
            mes = association_qual_qual(df[c1], df[c2])
            mes["var1"] = c1
            mes["var2"] = c2
            mes["cas"] = "qualitatif_qualitatif"
            lignes_bivar.append(mes)
            barres_contingence(df[c1], df[c2], bi / f"{base}_contingence.png")

        else:
            if q1:
                col_q, col_g = c1, c2
            else:
                col_q, col_g = c2, c1
            mes = association_quant_qual(df[col_q], df[col_g])
            mes["var1"] = col_q
            mes["var2"] = col_g
            mes["cas"] = "quantitatif_qualitatif"
            lignes_bivar.append(mes)
            boites_a_moustaches(df[col_q], df[col_g], bi / f"{base}_boites.png")

    bdf = pd.DataFrame(lignes_bivar)
    bdf.to_csv(bi / f"resume_bivarie_{nom_table}.csv", index=False, encoding="utf-8-sig")
    return bdf


def main() -> None:
    EXPLORATION.mkdir(parents=True, exist_ok=True)
    t1_path = DATA / "table1.csv"
    t2_path = DATA / "table2.csv"
    df1 = pd.read_csv(t1_path, low_memory=False)
    df2 = pd.read_csv(t2_path, low_memory=False)

    out1 = EXPLORATION / "table1"
    out2 = EXPLORATION / "table2"
    out1.mkdir(parents=True, exist_ok=True)
    out2.mkdir(parents=True, exist_ok=True)

    executer_univarie_table(df1, "table1", out1)
    executer_univarie_table(df2, "table2", out2)
    executer_bivarie_table(df1, "table1", out1)
    executer_bivarie_table(df2, "table2", out2)


if __name__ == "__main__":
    main()
