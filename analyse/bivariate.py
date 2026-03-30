"""Tests et mesures bivariées (pairwise, NaN exclus par test)."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def correlation_quant_quant(s1: pd.Series, s2: pd.Series) -> dict:
    """
    Corrélations Pearson, Spearman, Kendall + p-valeurs bilatérales (H0: pas de corrélation).
    """
    df = pd.DataFrame({"a": s1, "b": s2}).dropna()
    n = len(df)
    out: dict = {"n_apparies": int(n)}
    if n < 2:
        for name in ("pearson_r", "pearson_p", "spearman_rho", "spearman_p", "kendall_tau", "kendall_p"):
            out[name] = np.nan
        return out

    r, p = stats.pearsonr(df["a"], df["b"])
    out["pearson_r"] = float(r)
    out["pearson_p"] = float(p)

    rho, p_sp = stats.spearmanr(df["a"], df["b"])
    out["spearman_rho"] = float(rho)
    out["spearman_p"] = float(p_sp)

    tau, p_kd = stats.kendalltau(df["a"], df["b"])
    out["kendall_tau"] = float(tau) if tau == tau else np.nan
    out["kendall_p"] = float(p_kd) if p_kd == p_kd else np.nan
    return out


def association_qual_qual(s1: pd.Series, s2: pd.Series) -> dict:
    """
    Table de contingence, khi-deux d'indépendance, Cramér V.
    """
    df = pd.DataFrame({"a": s1.astype(str), "b": s2.astype(str)}).dropna()
    n = len(df)
    out: dict = {"n_apparies": int(n)}
    if n < 2:
        out["chi2"] = out["chi2_p"] = out["cramers_v"] = out["dof"] = np.nan
        return out

    tab = pd.crosstab(df["a"], df["b"])
    if tab.size == 0:
        out["chi2"] = out["chi2_p"] = out["cramers_v"] = out["dof"] = np.nan
        return out

    try:
        chi2, p, dof, _expected = stats.chi2_contingency(tab)
    except ValueError:
        out["chi2"] = out["chi2_p"] = out["cramers_v"] = out["dof"] = np.nan
        return out

    out["chi2"] = float(chi2)
    out["chi2_p"] = float(p)
    out["dof"] = int(dof)
    n_tot = float(tab.values.sum())
    r, k = tab.shape
    denom = n_tot * (min(r, k) - 1)
    cramers = np.sqrt(chi2 / denom) if denom > 0 and min(r, k) > 1 else np.nan
    out["cramers_v"] = float(cramers) if cramers == cramers else np.nan
    return out


def _epsilon_squared_kruskal(h_stat: float, n: int, k: int) -> float:
    """Epsilon-squared (effet) pour Kruskal-Wallis (borné à [0, 1])."""
    if n <= k or k < 2:
        return np.nan
    val = (h_stat - k + 1) / (n - k)
    if val != val:
        return np.nan
    return float(max(0.0, min(1.0, val)))


def association_quant_qual(quant: pd.Series, qual: pd.Series) -> dict:
    """
    Comparaison du quantitatif entre groupes du qualitatif :
    Kruskal-Wallis ; ANOVA à un facteur ; epsilon² (Kruskal).
    """
    df = pd.DataFrame({"q": quant, "g": qual.astype(str)}).dropna()
    n = len(df)
    out: dict = {"n_apparies": int(n)}
    groups = [grp["q"].values for _, grp in df.groupby("g", sort=False)]
    k = len(groups)
    if n < 3 or k < 2:
        out["kruskal_h"] = out["kruskal_p"] = out["epsilon2_kruskal"] = np.nan
        out["anova_f"] = out["anova_p"] = np.nan
        return out

    nonempty = [g for g in groups if len(g) > 0]
    if len(nonempty) < 2:
        out["kruskal_h"] = out["kruskal_p"] = out["epsilon2_kruskal"] = np.nan
        out["anova_f"] = out["anova_p"] = np.nan
        return out

    h_stat, p_kw = stats.kruskal(*nonempty)
    out["kruskal_h"] = float(h_stat)
    out["kruskal_p"] = float(p_kw)
    out["epsilon2_kruskal"] = _epsilon_squared_kruskal(float(h_stat), n, k)

    try:
        f_stat, p_an = stats.f_oneway(*nonempty)
        out["anova_f"] = float(f_stat)
        out["anova_p"] = float(p_an)
    except Exception:
        out["anova_f"] = out["anova_p"] = np.nan

    return out
