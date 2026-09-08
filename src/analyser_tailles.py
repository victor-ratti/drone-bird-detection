# -*- coding: utf-8 -*-
"""Distribution de la taille des objets annotes, par classe et par split.

Le durcissement de l'etape 4 repose sur l'existence de petits objets dans le
jeu. Cette analyse le verifie avant d'y investir du temps.

Convention COCO, appliquee a la taille d'entree du modele (640x640) :
    petit   : aire < 32^2 px     soit moins de 1024 px
    moyen   : 32^2 a 96^2 px
    grand   : aire > 96^2 px

Les etiquettes YOLO sont normalisees. L'aire est donc calculee dans le repere
640x640 apres letterbox, c'est-a-dire telle que le modele la voit, et non dans
la resolution d'origine qui varie d'une image a l'autre.

Usage :
    python src/analyser_tailles.py data/Drone-Bird-Detection-3
    python src/analyser_tailles.py data/Drone-Bird-Detection-3 --json resultats/tailles.json
"""

import argparse
import glob
import json
import os
import sys

import numpy as np
import yaml

TAILLE = 640
SEUIL_PETIT = 32 ** 2
SEUIL_MOYEN = 96 ** 2


def lire_annotations(racine, split):
    """Rend (classes, aires_px, cotes_px) pour toutes les boites d'un split."""
    dossier = os.path.join(racine, split, "labels")
    classes, aires, cotes = [], [], []
    for chemin in glob.glob(os.path.join(dossier, "*.txt")):
        with open(chemin, encoding="utf-8") as f:
            for ligne in f:
                p = ligne.split()
                if len(p) < 5:
                    continue
                c, _, _, bw, bh = int(p[0]), *map(float, p[1:5])
                # Les coordonnees sont normalisees : les multiplier par 640
                # donne la boite dans le repere d'entree du modele.
                w, h = bw * TAILLE, bh * TAILLE
                classes.append(c)
                aires.append(w * h)
                cotes.append(max(w, h))
    return np.array(classes), np.array(aires), np.array(cotes)


def categoriser(aires):
    petit = aires < SEUIL_PETIT
    grand = aires > SEUIL_MOYEN
    return petit, ~petit & ~grand, grand


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("racine", help="dossier du jeu, contenant data.yaml")
    ap.add_argument("--json")
    args = ap.parse_args()

    with open(os.path.join(args.racine, "data.yaml"), encoding="utf-8") as f:
        noms = yaml.safe_load(f)["names"]

    rapport = {"seuils_px": {"petit": SEUIL_PETIT, "moyen": SEUIL_MOYEN},
               "taille_entree": TAILLE, "splits": {}}

    for split in ("train", "valid", "test"):
        c, aires, cotes = lire_annotations(args.racine, split)
        if len(aires) == 0:
            continue
        petit, moyen, grand = categoriser(aires)
        print(f"=== {split} : {len(aires)} objets ===")
        print(f"  petit  (< 32 px de cote)  : {petit.sum():5d}  "
              f"{100 * petit.mean():5.1f} %")
        print(f"  moyen  (32 a 96 px)       : {moyen.sum():5d}  "
              f"{100 * moyen.mean():5.1f} %")
        print(f"  grand  (> 96 px de cote)  : {grand.sum():5d}  "
              f"{100 * grand.mean():5.1f} %")
        print(f"  cote median               : {np.median(cotes):.0f} px")
        print(f"  cote au 10e centile       : {np.percentile(cotes, 10):.0f} px")
        print(f"  cote minimal              : {cotes.min():.0f} px")

        detail = {"objets": int(len(aires)),
                  "petit": int(petit.sum()), "moyen": int(moyen.sum()),
                  "grand": int(grand.sum()),
                  "cote_median_px": round(float(np.median(cotes)), 1),
                  "cote_p10_px": round(float(np.percentile(cotes, 10)), 1),
                  "cote_min_px": round(float(cotes.min()), 1),
                  "par_classe": {}}

        for i, nom in enumerate(noms):
            m = c == i
            if not m.any():
                continue
            p, mo, g = categoriser(aires[m])
            print(f"    {nom:6} : {m.sum():5d} objets, "
                  f"{100 * p.mean():4.1f} % petits, "
                  f"cote median {np.median(cotes[m]):.0f} px")
            detail["par_classe"][nom] = {
                "objets": int(m.sum()), "petit": int(p.sum()),
                "moyen": int(mo.sum()), "grand": int(g.sum()),
                "cote_median_px": round(float(np.median(cotes[m])), 1)}
        print()
        rapport["splits"][split] = detail

    if args.json:
        os.makedirs(os.path.dirname(args.json) or ".", exist_ok=True)
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(rapport, f, ensure_ascii=False, indent=2)
        print(f"Detail ecrit dans {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
