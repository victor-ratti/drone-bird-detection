# -*- coding: utf-8 -*-
"""Quantification int8 d'un modele ONNX de detection.

Deux modes, volontairement les deux, parce que la comparaison est le resultat :

- `dynamique` : ne quantifie que les poids, les activations restent en flottant
  et sont converties a la volee. Aucune donnee necessaire. Concu pour les
  MatMul et les LSTM, donc peu efficace sur un reseau a dominante Conv comme
  YOLO. Sert de temoin.
- `statique` : quantifie poids ET activations. Demande un jeu de calibration
  pour observer la plage reelle des activations. C'est le mode utile sur un CNN.

Usage :
    python src/quantifier.py modeles/baseline_n_best.onnx --mode dynamique
    python src/quantifier.py modeles/baseline_n_best.onnx --mode statique \
        --calibration data/Drone-Bird-Detection-3/valid/images --n 200
"""

import argparse
import glob
import os
import sys

import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None

from onnxruntime.quantization import (
    CalibrationDataReader, QuantFormat, QuantType,
    quantize_dynamic, quantize_static,
)
from onnxruntime.quantization.shape_inference import quant_pre_process


def letterbox(img, taille=640):
    """Redimensionne en gardant le rapport d'aspect, complete en gris.

    Reproduit le pretraitement d'ultralytics. Une calibration faite sur des
    images etirees mesurerait les mauvaises plages d'activation.
    """
    h, w = img.shape[:2]
    r = min(taille / h, taille / w)
    nh, nw = int(round(h * r)), int(round(w * r))
    redim = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    toile = np.full((taille, taille, 3), 114, dtype=np.uint8)
    haut, gauche = (taille - nh) // 2, (taille - nw) // 2
    toile[haut:haut + nh, gauche:gauche + nw] = redim
    return toile


def preparer(chemin, taille=640):
    """Image disque -> tenseur NCHW float32 normalise, comme a l'entrainement."""
    img = cv2.imread(chemin)
    if img is None:
        return None
    img = letterbox(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), taille)
    x = img.astype(np.float32) / 255.0
    return np.expand_dims(x.transpose(2, 0, 1), 0)


class LecteurCalibration(CalibrationDataReader):
    """Fournit les images de calibration une par une a onnxruntime."""

    def __init__(self, dossier, nom_entree, n=200, taille=640, graine=0):
        motifs = ("*.jpg", "*.jpeg", "*.png", "*.bmp")
        fichiers = sorted(f for m in motifs
                          for f in glob.glob(os.path.join(dossier, m)))
        if not fichiers:
            raise SystemExit(f"Aucune image dans {dossier}")
        rng = np.random.default_rng(graine)
        if len(fichiers) > n:
            fichiers = [fichiers[i] for i in
                        rng.choice(len(fichiers), n, replace=False)]
        self.nom = nom_entree
        self.taille = taille
        self.fichiers = fichiers
        self.iter = None
        print(f"  calibration sur {len(fichiers)} images de {dossier}")

    def get_next(self):
        if self.iter is None:
            self.iter = iter(self.fichiers)
        for chemin in self.iter:
            x = preparer(chemin, self.taille)
            if x is not None:
                return {self.nom: x}
        return None

    def rewind(self):
        self.iter = None


def sortie_pour(entree, suffixe):
    base, _ = os.path.splitext(entree)
    return f"{base}_{suffixe}.onnx"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("modele")
    ap.add_argument("--mode", choices=["dynamique", "statique"], required=True)
    ap.add_argument("--calibration", help="dossier d'images, requis en statique")
    ap.add_argument("--n", type=int, default=200, help="images de calibration")
    ap.add_argument("--sortie")
    args = ap.parse_args()

    if not os.path.exists(args.modele):
        raise SystemExit(f"Modele introuvable : {args.modele}")

    sortie = args.sortie or sortie_pour(args.modele, f"int8_{args.mode}")

    # Recommande par onnxruntime avant toute quantification : fige les formes
    # et simplifie le graphe. Sans cette passe, la quantification statique
    # laisse des blocs non quantifies.
    prepare = sortie_pour(args.modele, "prep")
    print(f"Pre-traitement du graphe -> {os.path.basename(prepare)}")
    quant_pre_process(args.modele, prepare, skip_symbolic_shape=False)

    if args.mode == "dynamique":
        print("Quantification dynamique, poids seuls")
        quantize_dynamic(prepare, sortie, weight_type=QuantType.QInt8)
    else:
        if not args.calibration:
            raise SystemExit("--calibration est requis en mode statique")
        if cv2 is None:
            raise SystemExit("opencv est requis pour la calibration")
        import onnxruntime as ort
        nom_entree = ort.InferenceSession(
            prepare, providers=["CPUExecutionProvider"]).get_inputs()[0].name
        print("Quantification statique, poids et activations")
        lecteur = LecteurCalibration(args.calibration, nom_entree, args.n)
        quantize_static(
            prepare, sortie, lecteur,
            quant_format=QuantFormat.QDQ,
            activation_type=QuantType.QUInt8,
            weight_type=QuantType.QInt8,
            per_channel=True,
        )

    os.remove(prepare)
    avant = os.path.getsize(args.modele) / 1024 / 1024
    apres = os.path.getsize(sortie) / 1024 / 1024
    print()
    print(f"  {os.path.basename(args.modele):<36} {avant:6.2f} Mo")
    print(f"  {os.path.basename(sortie):<36} {apres:6.2f} Mo   "
          f"x{avant / apres:.2f} plus leger")
    return 0


if __name__ == "__main__":
    sys.exit(main())
