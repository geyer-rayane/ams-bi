"""Graphiques bivariés."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def nuage_points(s1: pd.Series, s2: pd.Series, chemin: str | Path, titre: str | None = None) -> None:
    """Nuage de points (quantitatif × quantitatif)."""
    chemin = Path(chemin)
    chemin.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame({"x": s1, "y": s2}).dropna()
    fig, ax = plt.subplots(figsize=(7, 6))
    if len(df) == 0:
        ax.text(0.5, 0.5, "Aucune paire complète", ha="center", va="center")
    else:
        ax.scatter(df["x"], df["y"], alpha=0.25, s=8, edgecolors="none")
        ax.set_xlabel(s1.name)
        ax.set_ylabel(s2.name)
    ax.set_title(titre or f"{s1.name} vs {s2.name}")
    fig.tight_layout()
    fig.savefig(chemin, dpi=120)
    plt.close(fig)


def heatmap_correlations(
    df: pd.DataFrame,
    colonnes: list[str],
    chemin: str | Path,
    methode: str = "pearson",
    titre: str | None = None,
) -> None:
    """Matrice de corrélations (Pearson ou Spearman) entre colonnes numériques."""
    chemin = Path(chemin)
    chemin.parent.mkdir(parents=True, exist_ok=True)
    sub = df[colonnes]
    if len(colonnes) < 2:
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.text(0.5, 0.5, "Au moins 2 colonnes requises", ha="center", va="center")
        fig.savefig(chemin, dpi=120)
        plt.close(fig)
        return
    corr = sub.corr(method=methode, numeric_only=True)
    fig, ax = plt.subplots(figsize=(max(8, 0.5 * len(colonnes)), max(7, 0.45 * len(colonnes))))
    sns.heatmap(corr, ax=ax, cmap="RdBu_r", center=0, vmin=-1, vmax=1, square=True, annot=len(colonnes) <= 16)
    ax.set_title(titre or f"Corrélations ({methode})")
    fig.tight_layout()
    fig.savefig(chemin, dpi=120)
    plt.close(fig)


def boites_a_moustaches(quant: pd.Series, qual: pd.Series, chemin: str | Path, titre: str | None = None) -> None:
    """Boîtes à moustaches du quantitatif par modalité du qualitatif."""
    chemin = Path(chemin)
    chemin.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame({"q": quant, "g": qual.astype(str)}).dropna()
    fig, ax = plt.subplots(figsize=(10, 5))
    if len(df) < 2:
        ax.text(0.5, 0.5, "Données insuffisantes", ha="center", va="center")
    else:
        cats = df["g"].unique()
        if len(cats) > 35:
            top = df["g"].value_counts().head(35).index
            df = df[df["g"].isin(top)]
        data = [df.loc[df["g"] == c, "q"].values for c in df["g"].unique()]
        labels = [str(x)[:20] for x in df["g"].unique()]
        ax.boxplot(data, labels=labels, showfliers=False)
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", fontsize=7)
        ax.set_ylabel(str(quant.name))
        ax.set_xlabel(str(qual.name))
    ax.set_title(titre or f"{quant.name} par {qual.name}")
    fig.tight_layout()
    fig.savefig(chemin, dpi=120)
    plt.close(fig)


def barres_contingence(s1: pd.Series, s2: pd.Series, chemin: str | Path, titre: str | None = None) -> None:
    """Barres empilées approximant la contingence (effectifs par croisement)."""
    chemin = Path(chemin)
    chemin.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame({"a": s1.astype(str), "b": s2.astype(str)}).dropna()
    fig, ax = plt.subplots(figsize=(10, 5))
    if len(df) < 2:
        ax.text(0.5, 0.5, "Données insuffisantes", ha="center", va="center")
    else:
        ct = pd.crosstab(df["a"], df["b"])
        if ct.shape[0] > 25:
            ct = ct.loc[ct.sum(axis=1).nlargest(25).index]
        if ct.shape[1] > 15:
            ct = ct[ct.sum(axis=0).nlargest(15).index]
        ct.plot(kind="bar", stacked=True, ax=ax, legend=True, width=0.85)
        ax.set_xlabel(s1.name)
        ax.set_ylabel("Effectif")
        ax.legend(title=s2.name, bbox_to_anchor=(1.02, 1), fontsize=7)
    ax.set_title(titre or f"{s1.name} × {s2.name}")
    fig.tight_layout()
    fig.savefig(chemin, dpi=120, bbox_inches="tight")
    plt.close(fig)
