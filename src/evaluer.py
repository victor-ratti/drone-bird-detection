# -*- coding: utf-8 -*-
"""Evaluation d'un modele ONNX de detection, sans PyTorch.

Calcule l'AP50 par classe et le mAP50, avec filtrage possible sur la taille des
objets. Le filtrage par taille est le coeur de l'etape 4 : il permet de mesurer
separement la performance sur les petits objets, qui sont le cas difficile
operationnel du contre-drone.

Tout se passe dans le repere 640x640 apres letterbox, celui que le modele voit.
Les annotations sont transformees dans ce meme repere, ce qui evite un
letterbox inverse et ses erreurs d'arrondi.

Protocole d'appariement : une prediction est un vrai positif si elle recouvre
une annotation de la meme classe avec un IoU superieur ou egal au seuil, et si
cette annotation n'a pas deja ete appariee a une prediction de score superieur.
AP calculee par interpolation sur tous les points, convention COCO.

Usage :
    python src/evaluer.py modeles/baseline_n_best.onnx data/Drone-Bird-Detection-3
    python src/evaluer.py modeles/baseline_n_best.onnx data/... --cote-max 32
    python src/evaluer.py modeles/baseline_n_best.onnx data/... --split valid --json r.json
"""

import argparse
import glob
import json
import os
import sys

import cv2
import numpy as np
import onnxruntime as ort
import yaml

TAILLE = 640


def letterbox(img, taille=TAILLE):
    """Redimensionne en gardant le rapport d'aspect, complete en gris 114."""
    h, w = img.shape[:2]
    r = min(taille / h, taille / w)
    nh, nw = int(round(h * r)), int(round(w * r))
    toile = np.full((taille, taille, 3), 114, dtype=np.uint8)
    haut, gauche = (taille - nh) // 2, (taille - nw) // 2
    toile[haut:haut + nh, gauche:gauche + nw] = cv2.resize(
        img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    return toile, r, gauche, haut


def annotations_letterbox(chemin, w0, h0, r, dx, dy):
    """Etiquettes YOLO normalisees -> boites xyxy dans le repere 640."""
    boites, classes = [], []
    if not os.path.exists(chemin):
        return np.zeros((0, 4)), np.zeros(0, dtype=int)
    with open(chemin, encoding="utf-8") as f:
        for ligne in f:
            p = ligne.split()
            if len(p) < 5:
                continue
            c = int(p[0])
            cx, cy, bw, bh = (float(v) for v in p[1:5])
            cx, cy, bw, bh = cx * w0, cy * h0, bw * w0, bh * h0
            x1 = cx * r - bw * r / 2 + dx
            y1 = cy * r - bh * r / 2 + dy
            boites.append([x1, y1, x1 + bw * r, y1 + bh * r])
            classes.append(c)
    return np.array(boites, dtype=np.float32), np.array(classes, dtype=int)


def iou_matrice(a, b):
    """IoU entre deux jeux de boites xyxy. Rend (len(a), len(b))."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), dtype=np.float32)
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    aire_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    aire_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (aire_a[:, None] + aire_b[None, :] - inter + 1e-9)


def nms(boites, scores, seuil=0.45):
    """Suppression des non-maxima, par classe deja filtree."""
    ordre = scores.argsort()[::-1]
    gardes = []
    while len(ordre):
        i = ordre[0]
        gardes.append(i)
        if len(ordre) == 1:
            break
        ious = iou_matrice(boites[i:i + 1], boites[ordre[1:]])[0]
        ordre = ordre[1:][ious < seuil]
    return np.array(gardes, dtype=int)


def predire(session, nom_entree, img, conf=0.001, seuil_nms=0.45):
    """Rend (boites xyxy repere 640, scores, classes)."""
    x = np.expand_dims(
        (img.astype(np.float32) / 255.0).transpose(2, 0, 1), 0)
    brut = session.run(None, {nom_entree: x})[0]          # (1, 4+nc, N)
    brut = brut[0].T                                       # (N, 4+nc)
    xywh, scores_cls = brut[:, :4], brut[:, 4:]
    cls = scores_cls.argmax(1)
    sc = scores_cls.max(1)
    m = sc > conf
    xywh, sc, cls = xywh[m], sc[m], cls[m]
    if len(sc) == 0:
        return np.zeros((0, 4)), np.zeros(0), np.zeros(0, dtype=int)
    boites = np.stack([xywh[:, 0] - xywh[:, 2] / 2, xywh[:, 1] - xywh[:, 3] / 2,
                       xywh[:, 0] + xywh[:, 2] / 2, xywh[:, 1] + xywh[:, 3] / 2], 1)
    gardes = []
    for c in np.unique(cls):
        idx = np.where(cls == c)[0]
        gardes.extend(idx[nms(boites[idx], sc[idx], seuil_nms)])
    g = np.array(sorted(gardes), dtype=int)
    return boites[g], sc[g], cls[g]


def ap_par_classe(tp, scores, n_verite):
    """AP en interpolation sur tous les points, convention COCO."""
    if n_verite == 0:
        return float("nan")
    if len(scores) == 0:
        return 0.0
    ordre = scores.argsort()[::-1]
    tp = tp[ordre]
    cum_tp = np.cumsum(tp)
    cum_fp = np.cumsum(1 - tp)
    rappel = cum_tp / n_verite
    precision = cum_tp / np.maximum(cum_tp + cum_fp, 1e-9)
    # Enveloppe decroissante de la precision, puis integration sur le rappel.
    precision = np.maximum.accumulate(precision[::-1])[::-1]
    r = np.concatenate(([0.0], rappel))
    p = np.concatenate(([precision[0] if len(precision) else 0.0], precision))
    return float(np.sum(np.diff(r) * p[1:]))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("modele")
    ap.add_argument("racine")
    ap.add_argument("--split", default="test")
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--conf", type=float, default=0.001)
    ap.add_argument("--cote-max", type=float, default=None,
                    help="ne garde que les objets dont le cote est sous ce seuil, en px")
    ap.add_argument("--cote-min", type=float, default=None)
    ap.add_argument("--threads", type=int, default=10)
    ap.add_argument("--json")
    args = ap.parse_args()

    with open(os.path.join(args.racine, "data.yaml"), encoding="utf-8") as f:
        noms = yaml.safe_load(f)["names"]

    opts = ort.SessionOptions()
    opts.intra_op_num_threads = args.threads
    session = ort.InferenceSession(args.modele, opts,
                                   providers=["CPUExecutionProvider"])
    nom_entree = session.get_inputs()[0].name

    images = sorted(glob.glob(os.path.join(args.racine, args.split, "images", "*")))
    if not images:
        raise SystemExit(f"Aucune image dans {args.racine}/{args.split}/images")

    stats = {c: {"tp": [], "scores": [], "n": 0} for c in range(len(noms))}
    confusion = np.zeros((len(noms) + 1, len(noms) + 1), dtype=int)
    ignorees = 0

    for k, chemin in enumerate(images, 1):
        img0 = cv2.imread(chemin)
        if img0 is None:
            continue
        h0, w0 = img0.shape[:2]
        img, r, dx, dy = letterbox(cv2.cvtColor(img0, cv2.COLOR_BGR2RGB))
        etiq = os.path.join(args.racine, args.split, "labels",
                            os.path.splitext(os.path.basename(chemin))[0] + ".txt")
        gt_b, gt_c = annotations_letterbox(etiq, w0, h0, r, dx, dy)

        # Filtrage par taille, applique aux annotations dans le repere 640.
        if len(gt_b) and (args.cote_max or args.cote_min):
            cote = np.maximum(gt_b[:, 2] - gt_b[:, 0], gt_b[:, 3] - gt_b[:, 1])
            garde = np.ones(len(cote), dtype=bool)
            if args.cote_max:
                garde &= cote < args.cote_max
            if args.cote_min:
                garde &= cote >= args.cote_min
            ignorees += int((~garde).sum())
            gt_b, gt_c = gt_b[garde], gt_c[garde]

        pr_b, pr_s, pr_c = predire(session, nom_entree, img, args.conf)

        # Le meme filtre de taille doit s'appliquer aux predictions.
        # Ne filtrer que la verite ferait compter chaque detection d'un gros
        # objet comme un faux positif, puisque sa verite a ete retiree, et
        # ecraserait la precision pour une raison purement methodologique.
        if len(pr_b) and (args.cote_max or args.cote_min):
            cote_p = np.maximum(pr_b[:, 2] - pr_b[:, 0], pr_b[:, 3] - pr_b[:, 1])
            garde_p = np.ones(len(cote_p), dtype=bool)
            if args.cote_max:
                garde_p &= cote_p < args.cote_max
            if args.cote_min:
                garde_p &= cote_p >= args.cote_min
            pr_b, pr_s, pr_c = pr_b[garde_p], pr_s[garde_p], pr_c[garde_p]

        for c in range(len(noms)):
            stats[c]["n"] += int((gt_c == c).sum())

        # Appariement glouton par score decroissant, par classe.
        for c in range(len(noms)):
            ip = np.where(pr_c == c)[0]
            ig = np.where(gt_c == c)[0]
            if len(ip) == 0:
                continue
            ip = ip[pr_s[ip].argsort()[::-1]]
            pris = set()
            ious = iou_matrice(pr_b[ip], gt_b[ig]) if len(ig) else None
            for j, p in enumerate(ip):
                ok = 0
                if ious is not None and len(ig):
                    libres = [t for t in range(len(ig)) if t not in pris]
                    if libres:
                        meilleur = max(libres, key=lambda t: ious[j, t])
                        if ious[j, meilleur] >= args.iou:
                            pris.add(meilleur)
                            ok = 1
                stats[c]["tp"].append(ok)
                stats[c]["scores"].append(float(pr_s[p]))

        # Matrice de confusion, sur les predictions de confiance usuelle.
        fort = pr_s > 0.25
        pb, pc = pr_b[fort], pr_c[fort]
        if len(gt_b):
            ious = iou_matrice(gt_b, pb) if len(pb) else np.zeros((len(gt_b), 0))
            for t in range(len(gt_b)):
                if ious.shape[1] and ious[t].max() >= args.iou:
                    confusion[gt_c[t], pc[ious[t].argmax()]] += 1
                else:
                    confusion[gt_c[t], len(noms)] += 1

        if k % 200 == 0:
            print(f"  {k}/{len(images)} images", flush=True)

    print()
    titre = f"{os.path.basename(args.modele)} sur {args.split}"
    if args.cote_max or args.cote_min:
        borne = []
        if args.cote_min:
            borne.append(f"cote >= {args.cote_min:.0f} px")
        if args.cote_max:
            borne.append(f"cote < {args.cote_max:.0f} px")
        titre += "  [" + ", ".join(borne) + "]"
    print(titre)
    print("-" * len(titre))

    aps, detail = [], {}
    for c, nom in enumerate(noms):
        a = ap_par_classe(np.array(stats[c]["tp"]),
                          np.array(stats[c]["scores"]), stats[c]["n"])
        detail[nom] = {"ap50": None if np.isnan(a) else round(a, 4),
                       "objets": stats[c]["n"]}
        if not np.isnan(a):
            aps.append(a)
        val = "n/a" if np.isnan(a) else f"{a:.4f}"
        print(f"  {nom:8} AP50 = {val:>8}   ({stats[c]['n']} objets)")
    m = float(np.mean(aps)) if aps else float("nan")
    print(f"  {'mAP50':8}      = {m:.4f}")
    if ignorees:
        print(f"\n  {ignorees} annotations hors du filtre de taille, ignorees")

    print("\n  Matrice de confusion (verite en ligne, prediction en colonne)")
    entetes = noms + ["manque"]
    print("      " + "".join(f"{e:>10}" for e in entetes))
    for i, nom in enumerate(noms):
        print(f"  {nom:>4}" + "".join(f"{confusion[i, j]:>10}"
                                      for j in range(len(noms) + 1)))

    if args.json:
        os.makedirs(os.path.dirname(args.json) or ".", exist_ok=True)
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump({"modele": os.path.basename(args.modele),
                       "split": args.split, "iou": args.iou,
                       "cote_max": args.cote_max, "cote_min": args.cote_min,
                       "map50": None if np.isnan(m) else round(m, 4),
                       "par_classe": detail,
                       "confusion": confusion[:len(noms)].tolist()},
                      f, ensure_ascii=False, indent=2)
        print(f"\nDetail ecrit dans {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
