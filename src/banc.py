# -*- coding: utf-8 -*-
"""Banc de mesure de latence pour un modele ONNX de detection.

Mesure le temps d'inference pur, sur CPU, avec chauffe prealable et
statistiques de dispersion. La latence ne depend pas du contenu de l'image,
seulement de sa taille : une entree aleatoire donne la meme mesure qu'une
photo, ce qui evite de dependre du jeu de donnees pour cette metrique.

Usage :
    python src/banc.py modeles/baseline_n_best.onnx
    python src/banc.py modeles/*.onnx --runs 200 --threads 1
    python src/banc.py modeles/a.onnx modeles/b.onnx --json resultats/banc.json
"""

import argparse
import json
import os
import platform
import statistics
import sys
import time

import numpy as np
import onnxruntime as ort


def machine():
    """Description de la machine, a consigner avec toute mesure."""
    infos = {
        "processeur": platform.processor() or platform.machine(),
        "architecture": platform.machine(),
        "systeme": f"{platform.system()} {platform.release()}",
        "coeurs_logiques": os.cpu_count(),
        "onnxruntime": ort.__version__,
    }
    if platform.system() == "Windows":
        # platform.processor() ne rend qu'un identifiant generique sur ARM64.
        try:
            import subprocess
            nom = subprocess.check_output(
                ["powershell", "-NoProfile", "-Command",
                 "(Get-CimInstance Win32_Processor).Name"],
                text=True, stderr=subprocess.DEVNULL).strip()
            if nom:
                infos["processeur"] = nom
        except Exception:
            pass
    return infos


def mesurer(chemin, runs=100, chauffe=10, threads=None):
    """Mesure la latence d'inference d'un modele ONNX."""
    opts = ort.SessionOptions()
    if threads:
        opts.intra_op_num_threads = threads
        opts.inter_op_num_threads = 1

    session = ort.InferenceSession(chemin, opts, providers=["CPUExecutionProvider"])
    entree = session.get_inputs()[0]

    # Les dimensions dynamiques (None ou chaine) sont ramenees a 1 ou 640.
    forme = [d if isinstance(d, int) else (1 if i == 0 else 640)
             for i, d in enumerate(entree.shape)]
    lot = np.random.rand(*forme).astype(np.float32)
    alim = {entree.name: lot}

    for _ in range(chauffe):
        session.run(None, alim)

    temps = []
    for _ in range(runs):
        t0 = time.perf_counter()
        session.run(None, alim)
        temps.append((time.perf_counter() - t0) * 1000.0)

    temps.sort()
    return {
        "modele": os.path.basename(chemin),
        "taille_mo": round(os.path.getsize(chemin) / 1024 / 1024, 2),
        "forme_entree": forme,
        "runs": runs,
        "threads": threads or "auto",
        "latence_ms_median": round(statistics.median(temps), 2),
        "latence_ms_moyenne": round(statistics.fmean(temps), 2),
        "latence_ms_p90": round(temps[int(0.90 * len(temps))], 2),
        "latence_ms_min": round(temps[0], 2),
        "latence_ms_max": round(temps[-1], 2),
        "fps_median": round(1000.0 / statistics.median(temps), 1),
    }


def charge_cpu():
    """Charge CPU instantanee, a consigner : elle fausse toute mesure."""
    if platform.system() != "Windows":
        return None
    try:
        import subprocess
        v = subprocess.check_output(
            ["powershell", "-NoProfile", "-Command",
             "(Get-CimInstance Win32_Processor).LoadPercentage"],
            text=True, stderr=subprocess.DEVNULL).strip()
        return int(v)
    except Exception:
        return None


def balayer(chemin, liste, runs, chauffe, sortie_json=None):
    """Balaye le nombre de threads intra-op sur un modele.

    Sur un CPU partage avec d'autres applications, la latence n'est pas
    monotone en nombre de threads : au-dela d'un certain point la contention
    et le trafic de cache coutent plus que le parallelisme n'apporte. Ce
    balayage trouve le point de fonctionnement, qui doit ensuite etre fige
    pour que toute comparaison entre modeles ait un sens.
    """
    threads = [int(x) for x in liste.split(",") if x.strip()]
    infos = machine()
    print("Machine de mesure")
    for k, v in infos.items():
        print(f"  {k:16} : {v}")
    print(f"  {'charge CPU debut':16} : {charge_cpu()} %")
    print()
    print(f"Balayage sur {os.path.basename(chemin)}, {runs} passes par point")
    print()
    entete = f"{'Threads':>7} {'Median':>10} {'p90':>9} {'FPS':>8}"
    print(entete)
    print("-" * len(entete))

    lignes = []
    for t in threads:
        r = mesurer(chemin, runs, chauffe, t)
        lignes.append(r)
        print(f"{t:>7} {r['latence_ms_median']:>9.2f}m "
              f"{r['latence_ms_p90']:>8.2f}m {r['fps_median']:>8.1f}")

    meilleur = min(lignes, key=lambda r: r["latence_ms_median"])
    pire = max(lignes, key=lambda r: r["latence_ms_median"])
    print()
    print(f"Optimum  : {meilleur['threads']} threads, "
          f"{meilleur['latence_ms_median']} ms, {meilleur['fps_median']} FPS")
    print(f"Pire cas : {pire['threads']} threads, "
          f"{pire['latence_ms_median']} ms, soit "
          f"x{pire['latence_ms_median'] / meilleur['latence_ms_median']:.1f} plus lent")
    fin = charge_cpu()
    print(f"Charge CPU fin : {fin} %")

    if sortie_json:
        os.makedirs(os.path.dirname(sortie_json) or ".", exist_ok=True)
        with open(sortie_json, "w", encoding="utf-8") as f:
            json.dump({"machine": infos, "charge_cpu_fin": fin,
                       "modele": os.path.basename(chemin),
                       "balayage": lignes}, f, ensure_ascii=False, indent=2)
        print(f"Detail ecrit dans {sortie_json}")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("modeles", nargs="+", help="fichiers .onnx a comparer")
    ap.add_argument("--runs", type=int, default=100)
    ap.add_argument("--chauffe", type=int, default=10)
    ap.add_argument("--threads", type=int, default=None,
                    help="threads intra-op. Par defaut, choix d'onnxruntime")
    ap.add_argument("--json", help="ecrit le detail dans ce fichier")
    ap.add_argument("--balayage", metavar="N,N,...",
                    help="balaye plusieurs nombres de threads sur le premier modele")
    args = ap.parse_args()

    if args.balayage:
        return balayer(args.modeles[0], args.balayage, args.runs,
                       args.chauffe, args.json)

    infos = machine()
    print("Machine de mesure")
    for k, v in infos.items():
        print(f"  {k:16} : {v}")
    print()

    lignes = []
    for chemin in args.modeles:
        if not os.path.exists(chemin):
            print(f"  introuvable, ignore : {chemin}", file=sys.stderr)
            continue
        print(f"  mesure de {os.path.basename(chemin)} ...", end="", flush=True)
        lignes.append(mesurer(chemin, args.runs, args.chauffe, args.threads))
        print(" fait")
    print()

    if not lignes:
        return 1

    entete = f"{'Modele':<34} {'Mo':>6} {'Median':>9} {'p90':>8} {'FPS':>7}"
    print(entete)
    print("-" * len(entete))
    for r in lignes:
        print(f"{r['modele']:<34} {r['taille_mo']:>6} "
              f"{r['latence_ms_median']:>8.2f}m {r['latence_ms_p90']:>7.2f}m "
              f"{r['fps_median']:>7.1f}")

    if len(lignes) > 1:
        ref = lignes[0]
        print()
        print(f"Reference : {ref['modele']}")
        for r in lignes[1:]:
            gain = ref["latence_ms_median"] / r["latence_ms_median"]
            poids = ref["taille_mo"] / r["taille_mo"]
            print(f"  {r['modele']:<32} x{gain:.2f} plus rapide, "
                  f"x{poids:.2f} plus leger")

    if args.json:
        os.makedirs(os.path.dirname(args.json) or ".", exist_ok=True)
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump({"machine": infos, "mesures": lignes}, f,
                      ensure_ascii=False, indent=2)
        print(f"\nDetail ecrit dans {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
