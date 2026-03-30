# -*- coding: utf-8 -*-
"""
Découpage train / validation / test sur données pré-traitées (matrice recodée).

- Apprentissage : construire les estimateurs du modèle.
- Validation : régler hyperparamètres, comparer modèles, choisir le « meilleur ».
- Test : estimation honnête de la généralisation (données non utilisées à l’entraînement ni au choix du modèle).

Répartition par défaut : 60 % / 20 % / 20 % (stratifié sur la cible pour préserver la prévalence).
Entrée : recodage/table2_matrice_modele.csv (exécuter recodage.py avant).

Sorties : dossier decoupage/
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

RACINE = Path(__file__).resolve().parent
MATRICE = RACINE / "recodage" / "table2_matrice_modele.csv"
SORTIE = RACINE / "decoupage"

# Proportions : test 20 %, puis validation = 25 % du reste (= 20 % du total), train = 60 %
TEST_SIZE = 0.20
VAL_SIZE_DANS_RESTE = 0.25  # 0.25 * 0.8 = 0.2


def main() -> None:
    SORTIE.mkdir(parents=True, exist_ok=True)
    if not MATRICE.exists():
        raise FileNotFoundError(f"Exécuter recodage.py d'abord. Manquant : {MATRICE}")

    df = pd.read_csv(MATRICE, low_memory=False)
    y = df["cible_churn"]

    train_val, test, y_train_val, y_test = train_test_split(
        df,
        y,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=42,
    )
    train, val, y_train, y_val = train_test_split(
        train_val,
        y_train_val,
        test_size=VAL_SIZE_DANS_RESTE,
        stratify=y_train_val,
        random_state=42,
    )

    train.to_csv(SORTIE / "jeu_apprentissage.csv", index=False, encoding="utf-8-sig")
    val.to_csv(SORTIE / "jeu_validation.csv", index=False, encoding="utf-8-sig")
    test.to_csv(SORTIE / "jeu_test.csv", index=False, encoding="utf-8-sig")

    def stats(nom: str, d: pd.DataFrame) -> dict:
        yd = d["cible_churn"]
        return {
            "nom": nom,
            "n": int(len(d)),
            "pct_total": round(100 * len(d) / len(df), 2),
            "n_churn_1": int(yd.sum()),
            "n_churn_0": int((yd == 0).sum()),
            "taux_churn": round(float(yd.mean()), 6),
        }

    resume = {
        "fichier_source": str(MATRICE.relative_to(RACINE)),
        "n_total": len(df),
        "proportion_test": TEST_SIZE,
        "proportion_validation_sur_restre": VAL_SIZE_DANS_RESTE,
        "proportion_apprentissage_effectif": round(len(train) / len(df), 4),
        "splits": [stats("apprentissage", train), stats("validation", val), stats("test", test)],
        "stratification": "cible_churn",
        "random_state": 42,
    }
    (SORTIE / "schema_decoupage.json").write_text(
        json.dumps(resume, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    lignes = [
        "=== Découpage train / validation / test ===",
        f"Source : {MATRICE.name}",
        f"Total lignes : {len(df)}",
        f"Apprentissage : {len(train)} ({100*len(train)/len(df):.1f} %)",
        f"Validation    : {len(val)} ({100*len(val)/len(df):.1f} %)",
        f"Test          : {len(test)} ({100*len(test)/len(df):.1f} %)",
        "",
        "Fichiers : jeu_apprentissage.csv, jeu_validation.csv, jeu_test.csv, schema_decoupage.json",
    ]
    (SORTIE / "synthese_decoupage.txt").write_text("\n".join(lignes), encoding="utf-8")

    print("OK —", SORTIE)
    print(f"  train={len(train)}  val={len(val)}  test={len(test)}")


if __name__ == "__main__":
    main()
