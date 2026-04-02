#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Orchestrateur du projet AMS-BI.

Lance les étapes .py dans l'ordre métier avec affichage terminal clair.
Arrête l'exécution à la première erreur.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path


RACINE = Path(__file__).resolve().parent

PIPELINE = [
    ("Exploration", "exploration.py", []),
    ("Nettoyage", "nettoyage.py", []),
    ("Concatenation", "concatenation.py", []),
    ("Recodage", "recodage.py", []),
    ("Pretraitement", "pretraitement.py", []),
    ("Decoupage", "decoupage.py", []),
    ("Selection", "selection.py", []),
    ("Apprentissage", "apprentissage.py", []),
    ("Raffinage", "raffinage.py", []),
    ("Comparaison", "comparaison.py", []),
    ("Comparatif", "comparatif.py", []),
    ("Evaluation", "evaluation.py", []),
    ("Interpretation", "interpretation.py", []),
]


def _banner(title: str) -> None:
    line = "=" * 86
    print(f"\n{line}\n{title}\n{line}")


def _run_step(index: int, total: int, label: str, script: str, extra_args: list[str]) -> None:
    path = RACINE / script
    if not path.exists():
        raise FileNotFoundError(f"Script introuvable: {path}")

    _banner(f"[{index}/{total}] {label} -> {script}")
    cmd = [sys.executable, str(path), *extra_args]
    print("Commande :", " ".join(cmd))
    print("-" * 86)
    start = time.time()
    proc = subprocess.run(cmd, cwd=RACINE, check=False)
    elapsed = time.time() - start
    print("-" * 86)

    if proc.returncode != 0:
        raise RuntimeError(f"Echec {script} (exit code={proc.returncode})")

    print(f"OK {script} ({elapsed:.1f}s)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Lance tout le pipeline AMS-BI dans l'ordre.")
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Passe --quick à selection.py pour réduire le temps de recherche.",
    )
    parser.add_argument(
        "--smote",
        action="store_true",
        help="Passe --smote à selection.py (train rééquilibré).",
    )
    args = parser.parse_args()

    # Injecter options quick/smote pour apprentissage et raffinage
    ordered = []
    for label, script, extra in PIPELINE:
        step_args = list(extra)
        if script in {"apprentissage.py", "raffinage.py"}:
            if args.quick:
                step_args.append("--quick")
            if args.smote:
                step_args.append("--smote")
        ordered.append((label, script, step_args))

    _banner("Démarrage pipeline AMS-BI")
    print(f"Répertoire: {RACINE}")
    print(f"Etapes: {len(ordered)}")
    if args.quick or args.smote:
        print(f"Options: quick={args.quick}, smote={args.smote}")

    t0 = time.time()
    for i, (label, script, step_args) in enumerate(ordered, start=1):
        _run_step(i, len(ordered), label, script, step_args)

    total = time.time() - t0
    _banner("Pipeline terminé avec succès")
    print(f"Durée totale: {total:.1f}s")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        _banner("Pipeline interrompu")
        print(f"Erreur: {exc}")
        sys.exit(1)
