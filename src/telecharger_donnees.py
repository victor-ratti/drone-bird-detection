# -*- coding: utf-8 -*-
"""Telecharge le jeu de donnees Roboflow en local, dans data/.

La cle API est lue dans le fichier .env a la racine du projet, jamais passee
en argument : un argument de ligne de commande finit dans l'historique du
shell. Le .env est couvert par le .gitignore.

Usage :
    python src/telecharger_donnees.py
"""

import os
import sys

RACINE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENV = os.path.join(RACINE, ".env")
DEST = os.path.join(RACINE, "data")

WORKSPACE = "myworkspace-0p4nk"
PROJET = "drone-bird-detection-3nl79"
VERSION = 3
FORMAT = "yolov11"


def lire_env(chemin):
    """Lecteur .env minimal, pour ne pas dependre de python-dotenv."""
    valeurs = {}
    if not os.path.exists(chemin):
        return valeurs
    with open(chemin, encoding="utf-8") as f:
        for ligne in f:
            ligne = ligne.strip()
            if not ligne or ligne.startswith("#") or "=" not in ligne:
                continue
            cle, _, val = ligne.partition("=")
            valeurs[cle.strip()] = val.strip().strip('"').strip("'")
    return valeurs


def main():
    cle = os.environ.get("ROBOFLOW_API_KEY") or lire_env(ENV).get("ROBOFLOW_API_KEY")
    if not cle:
        print("Cle API absente.", file=sys.stderr)
        print(f"Renseigne ROBOFLOW_API_KEY dans {ENV}", file=sys.stderr)
        return 1

    try:
        from roboflow import Roboflow
    except ImportError:
        print("Paquet manquant. Lance :", file=sys.stderr)
        print("  .\\.venv\\Scripts\\python.exe -m pip install roboflow", file=sys.stderr)
        return 1

    os.makedirs(DEST, exist_ok=True)
    cible = os.path.join(DEST, f"Drone-Bird-Detection-{VERSION}")
    if os.path.exists(os.path.join(cible, "data.yaml")):
        print(f"Deja present : {cible}")
        return 0

    print(f"Telechargement de {PROJET} v{VERSION} au format {FORMAT}")
    rf = Roboflow(api_key=cle)
    version = rf.workspace(WORKSPACE).project(PROJET).version(VERSION)
    jeu = version.download(FORMAT, location=cible)

    print()
    print("Telecharge dans :", jeu.location)
    import glob
    for split in ("train", "valid", "test"):
        n = len(glob.glob(os.path.join(jeu.location, split, "images", "*")))
        print(f"  {split:6} : {n} images")
    return 0


if __name__ == "__main__":
    sys.exit(main())
