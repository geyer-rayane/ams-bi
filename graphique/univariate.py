"""Graphiques univariés."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def histogramme(series: pd.Series, chemin: str | Path, titre: str | None = None) -> None:
    """Histogramme pour une variable quantitative."""
    chemin = Path(chemin)
    chemin.parent.mkdir(parents=True, exist_ok=True)
    s = series.dropna()
    fig, ax = plt.subplots(figsize=(8, 4.5))
    if len(s) == 0:
        ax.text(0.5, 0.5, "Aucune valeur non manquante", ha="center", va="center")
    else:
        ax.hist(s.astype(float), bins=min(50, max(10, int(np.sqrt(len(s))))), edgecolor="black", alpha=0.85)
        ax.set_xlabel(series.name)
        ax.set_ylabel("Effectif")
    ax.set_title(titre or f"Distribution — {series.name}")
    fig.tight_layout()
    fig.savefig(chemin, dpi=120)
    plt.close(fig)


def diagramme_barres(series: pd.Series, chemin: str | Path, titre: str | None = None, max_modalites: int = 40) -> None:
    """Diagramme en barres des effectifs (qualitatif ou codes en chaîne)."""
    chemin = Path(chemin)
    chemin.parent.mkdir(parents=True, exist_ok=True)
    vc = series.dropna().astype(str).value_counts()
    fig, ax = plt.subplots(figsize=(10, max(4.5, min(0.35 * len(vc), 14))))
    if len(vc) == 0:
        ax.text(0.5, 0.5, "Aucune valeur non manquante", ha="center", va="center")
    else:
        plot_vc = vc.head(max_modalites)
        ax.barh(range(len(plot_vc)), plot_vc.values[::-1])
        ax.set_yticks(range(len(plot_vc)))
        ax.set_yticklabels(plot_vc.index[::-1], fontsize=8)
        ax.set_xlabel("Effectif")
    ax.set_title(titre or f"Effectifs — {series.name}")
    fig.tight_layout()
    fig.savefig(chemin, dpi=120)
    plt.close(fig)
