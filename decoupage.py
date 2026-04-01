# -*- coding: utf-8 -*-
"""
Découpage train / validation / test + rééquilibrage des classes sur le jeu d'apprentissage.

Entrée : recodage/union_matrice_modele.csv (mode principal, table1+table2).
Fallback : recodage/table2_matrice_modele.csv.

Répartition : 70 % / 15 % / 15 % stratifiée sur cible_churn.

Rééquilibrage :
  SMOTE (Synthetic Minority Over-sampling Technique) appliqué UNIQUEMENT sur le jeu
  d'apprentissage, après le découpage — jamais sur validation ni test, pour conserver
  une estimation honnête de la performance sur données réelles.

  Deux sorties d'apprentissage :
    - jeu_apprentissage.csv          : train original (déséquilibré, préserve la distribution)
    - jeu_apprentissage_smote.csv    : train rééquilibré par SMOTE (pour classifieurs sensibles)

Sorties : dossier decoupage/
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split

RACINE = Path(__file__).resolve().parent
_MATRICE_UNION = RACINE / "recodage" / "union_matrice_modele.csv"
_MATRICE_T2 = RACINE / "recodage" / "table2_matrice_modele.csv"

TEST_SIZE = 0.15
VAL_SIZE_DANS_RESTE = 0.15 / (1 - 0.15)  # ≈ 0.1765 → 15 % du total
SORTIE = RACINE / "decoupage"

COLS_META = ("ID", "cible_churn", "source", "id_fichier")


def _resoudre_matrice() -> Path:
    if _MATRICE_UNION.exists():
        return _MATRICE_UNION
    if _MATRICE_T2.exists():
        return _MATRICE_T2
    raise FileNotFoundError(
        "Aucune matrice trouvée. Exécuter concatenation.py puis recodage.py."
    )


def _stats_partition(nom: str, d: pd.DataFrame, n_total: int) -> dict:
    yd = d["cible_churn"]
    return {
        "nom": nom,
        "n": int(len(d)),
        "pct_total": round(100 * len(d) / n_total, 2),
        "n_churn_1": int(yd.sum()),
        "n_churn_0": int((yd == 0).sum()),
        "taux_churn": round(float(yd.mean()), 6),
    }


def main() -> None:
    SORTIE.mkdir(parents=True, exist_ok=True)
    matrice = _resoudre_matrice()

    df = pd.read_csv(matrice, low_memory=False)
    y = df["cible_churn"]
    n_total = len(df)

    # --- Découpage stratifié ---
    train_val, test, y_tv, y_test = train_test_split(
        df, y, test_size=TEST_SIZE, stratify=y, random_state=42
    )
    train, val, y_train, y_val = train_test_split(
        train_val, y_tv,
        test_size=VAL_SIZE_DANS_RESTE,
        stratify=y_tv,
        random_state=42,
    )

    train.to_csv(SORTIE / "jeu_apprentissage.csv", index=False, encoding="utf-8-sig")
    val.to_csv(SORTIE / "jeu_validation.csv", index=False, encoding="utf-8-sig")
    test.to_csv(SORTIE / "jeu_test.csv", index=False, encoding="utf-8-sig")

    # --- SMOTE sur jeu d'apprentissage uniquement ---
    feature_cols = [c for c in train.columns if c not in COLS_META]
    X_train = train[feature_cols]
    smote = SMOTE(random_state=42, k_neighbors=5)
    X_res, y_res = smote.fit_resample(X_train, y_train)

    train_smote = pd.DataFrame(X_res, columns=feature_cols)
    train_smote["cible_churn"] = y_res
    train_smote.to_csv(SORTIE / "jeu_apprentissage_smote.csv", index=False, encoding="utf-8-sig")

    # --- Résumé ---
    resume = {
        "fichier_source": str(matrice.relative_to(RACINE)),
        "n_total": n_total,
        "proportion_test": TEST_SIZE,
        "proportion_validation_sur_reste": VAL_SIZE_DANS_RESTE,
        "splits_originaux": [
            _stats_partition("apprentissage", train, n_total),
            _stats_partition("validation", val, n_total),
            _stats_partition("test", test, n_total),
        ],
        "smote": {
            "applique_sur": "jeu_apprentissage uniquement",
            "n_avant": int(len(train)),
            "n_apres": int(len(train_smote)),
            "n_churn_1_avant": int(y_train.sum()),
            "n_churn_1_apres": int((train_smote["cible_churn"] == 1).sum()),
            "n_churn_0_apres": int((train_smote["cible_churn"] == 0).sum()),
            "taux_churn_apres": round(float(train_smote["cible_churn"].mean()), 6),
            "note": (
                "Validation et test conservent la distribution réelle (déséquilibrée). "
                "SMOTE génère des exemples synthétiques de la classe minoritaire (churners) "
                "par interpolation dans l'espace des features. "
                "Utiliser jeu_apprentissage_smote.csv pour les classifieurs sensibles au déséquilibre ; "
                "jeu_apprentissage.csv pour les arbres ou classifieurs avec class_weight."
            ),
        },
        "stratification": "cible_churn",
        "random_state": 42,
    }
    (SORTIE / "schema_decoupage.json").write_text(
        json.dumps(resume, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    lignes = [
        "=== Découpage train / validation / test ===",
        f"Source : {matrice.name}",
        f"Total lignes : {n_total}",
        f"Apprentissage : {len(train)} ({100*len(train)/n_total:.1f} %)  taux churn={y_train.mean():.4f}",
        f"Validation    : {len(val)} ({100*len(val)/n_total:.1f} %)  taux churn={y_val.mean():.4f}",
        f"Test          : {len(test)} ({100*len(test)/n_total:.1f} %)  taux churn={y_test.mean():.4f}",
        "",
        "=== SMOTE (rééquilibrage classe minoritaire — train uniquement) ===",
        f"Train avant SMOTE : {len(train)} lignes  (churn=1 : {int(y_train.sum())}, churn=0 : {int((y_train==0).sum())})",
        f"Train après SMOTE : {len(train_smote)} lignes (churn=1 : {int((train_smote['cible_churn']==1).sum())}, churn=0 : {int((train_smote['cible_churn']==0).sum())})",
        "",
        "Fichiers :",
        "  jeu_apprentissage.csv         (original, déséquilibré)",
        "  jeu_apprentissage_smote.csv   (rééquilibré par SMOTE)",
        "  jeu_validation.csv            (distribution réelle)",
        "  jeu_test.csv                  (distribution réelle)",
        "  schema_decoupage.json",
    ]
    (SORTIE / "synthese_decoupage.txt").write_text("\n".join(lignes), encoding="utf-8")

    print("OK —", SORTIE)
    print(f"  train={len(train)}  val={len(val)}  test={len(test)}")
    print(f"  train SMOTE={len(train_smote)} (churn équilibré)")


if __name__ == "__main__":
    main()
