"""Résumés statistiques univariés (sans modification des données)."""

from __future__ import annotations

import numpy as np
import pandas as pd


def resume_quantitatif(series: pd.Series) -> dict:
    """Moyenne, médiane, écart-type, variance, min, max, quartiles (effectifs NaN ignorés)."""
    s = series
    vc = s.dropna()
    n = len(s)
    n_valid = len(vc)
    out: dict = {
        "n": n,
        "n_non_null": int(n_valid),
        "mean": float(vc.mean()) if n_valid else np.nan,
        "median": float(vc.median()) if n_valid else np.nan,
        "std": float(vc.std(ddof=1)) if n_valid > 1 else np.nan,
        "var": float(vc.var(ddof=1)) if n_valid > 1 else np.nan,
        "min": float(vc.min()) if n_valid else np.nan,
        "max": float(vc.max()) if n_valid else np.nan,
    }
    if n_valid:
        qs = vc.quantile([0.25, 0.5, 0.75])
        out["q25"] = float(qs[0.25])
        out["q50"] = float(qs[0.5])
        out["q75"] = float(qs[0.75])
        out["iqr"] = float(qs[0.75] - qs[0.25])
    else:
        out["q25"] = out["q50"] = out["q75"] = out["iqr"] = np.nan
    return out


def resume_qualitatif(series: pd.Series) -> dict:
    """Effectifs, modalité la plus fréquente, nombre de modalités distinctes."""
    s = series
    vc = s.dropna()
    n = len(s)
    n_valid = len(vc)
    counts = vc.value_counts(dropna=False)
    mode_val = counts.index[0] if len(counts) else np.nan
    mode_count = int(counts.iloc[0]) if len(counts) else 0
    return {
        "n": n,
        "n_non_null": int(n_valid),
        "n_modalites": int(vc.nunique(dropna=True)),
        "mode": str(mode_val) if pd.notna(mode_val) else np.nan,
        "effectif_mode": mode_count,
    }
