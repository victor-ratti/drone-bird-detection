# -*- coding: utf-8 -*-
"""Mesure combien de la separation drone / oiseau tient a la seule taille.

Motivation. Le modele entraine ne confond jamais un drone avec un oiseau sur le
jeu de test. Avant de conclure qu'il a appris a les distinguer visuellement, il
faut ecarter l'explication triviale : les deux classes ont peut-etre des
tailles si differentes qu'un seuil suffit.

Methode. On entraine le classifieur le plus stupide possible, un seuil unique
sur le cote de la boite, en n'utilisant AUCUN pixel. Le seuil est choisi sur le
split d'entrainement, puis applique tel quel au split de test. Son exactitude
est une borne inferieure de ce que le jeu donne gratuitement.

Lecture. Si ce classifieur atteint 90 %, alors le reseau de neurones n'apporte
que les 10 points restants, et la performance annoncee mesure surtout un biais
du jeu de donnees.

Usage :
    python src/biais_taille.py data/Drone-Bird-Detection-3
"""

import argparse
import glob
import json
import os
import sys

import numpy as np
import yaml

TAILLE = 640


def charger(racine, split):
    """Rend (classes, cotes_px) de toutes les boites d'un split."""
    classes, cotes = [], []
    for chemin in glob.glob(os.path.join(racine, split, "labels", "*.txt")):
        with open(chemin, encoding="utf-8") as f:
            for ligne in f:
                p = ligne.split()
                if len(p) < 5:
                    continue
                classes.append(int(p[0]))
                cotes.append(max(float(p[3]), float(p[4])) * TAILLE)
    return np.array(classes), np.array(cotes)


def meilleur_seuil(y, cote):
    """Seuil qui maximise l'exactitude, cherche sur les centiles observes.

    Regle : cote >= seuil predit la classe majoritaire au-dessus du seuil.
    """
    candidats = np.unique(np.percentile(cote, np.arange(1, 100)))
    meilleur, exact_max, sens = None, -1.0, 1
    for s in candidats:
        for orientation in (1, 0):
            pred = np.where(cote >= s, orientation, 1 - orientation)
            e = (pred == y).mean()
            if e > exact_max:
                meilleur, exact_max, sens = s, e, orientation
    return meilleur, exact_max, sens


def evaluer(y, cote, seuil, sens):
    pred = np.where(cote >= seuil, sens, 1 - sens)
    exact = (pred == y).mean()
    par_classe = {}
    for c in np.unique(y):
        m = y == c
        par_classe[int(c)] = float((pred[m] == y[m]).mean())
    return float(exact), par_classe, pred


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("racine")
    ap.add_argument("--json")
    args = ap.parse_args()

    with open(os.path.join(args.racine, "data.yaml"), encoding="utf-8") as f:
        noms = yaml.safe_load(f)["names"]

    y_tr, c_tr = charger(args.racine, "train")
    y_te, c_te = charger(args.racine, "test")

    seuil, exact_tr, sens = meilleur_seuil(y_tr, c_tr)
    exact_te, par_classe, pred = evaluer(y_te, c_te, seuil, sens)

    grande = noms[sens]
    print("Classifieur a un seuil, aucun pixel utilise")
    print(f"  seuil appris sur train : cote >= {seuil:.0f} px  ->  {grande}")
    print(f"  exactitude sur train   : {100 * exact_tr:.1f} %")
    print(f"  exactitude sur test    : {100 * exact_te:.1f} %")
    print()
    for c, e in sorted(par_classe.items()):
        print(f"    {noms[c]:6} : {100 * e:5.1f} % de bien classes")
    print()

    # Sous-ensemble des petits objets : la ou le seuil ne peut plus rien.
    petit = c_te < 32
    if petit.sum():
        e_petit, pc_petit, _ = evaluer(y_te[petit], c_te[petit], seuil, sens)
        print(f"  Sur les {petit.sum()} petits objets du test (< 32 px) :")
        print(f"    exactitude du seuil  : {100 * e_petit:.1f} %")
        for c in np.unique(y_te[petit]):
            print(f"      {noms[c]:6} : {int((y_te[petit] == c).sum()):3d} objets, "
                  f"{100 * pc_petit[int(c)]:5.1f} % bien classes")
    print()

    ecart = None
    if len(np.unique(y_te)) == 2:
        a = np.median(c_te[y_te == 0])
        b = np.median(c_te[y_te == 1])
        ecart = round(float(max(a, b) / min(a, b)), 2)
        print(f"  Cote median {noms[0]} : {a:.0f} px")
        print(f"  Cote median {noms[1]} : {b:.0f} px")
        print(f"  Rapport d'echelle entre les deux classes : x{ecart}")

    if args.json:
        os.makedirs(os.path.dirname(args.json) or ".", exist_ok=True)
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump({"seuil_px": round(float(seuil), 1),
                       "classe_au_dessus": grande,
                       "exactitude_train": round(exact_tr, 4),
                       "exactitude_test": round(exact_te, 4),
                       "par_classe_test": {noms[c]: round(e, 4)
                                           for c, e in par_classe.items()},
                       "rapport_echelle": ecart}, f,
                      ensure_ascii=False, indent=2)
        print(f"\nDetail ecrit dans {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
