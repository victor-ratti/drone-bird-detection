# Drone vs Bird : détection, suivi et compression pour l'edge

Détecter et distinguer drones et oiseaux dans des images, puis faire tenir le
modèle sur un ordinateur faible, du type de ceux qu'un drone embarque.

> Projet en cours. Les chiffres ci-dessous sont mis à jour au fil des étapes.

## Le problème

Un système de détection de drones doit répondre à deux contraintes en même
temps :

1. **Ne pas confondre un drone avec un oiseau.** Une fausse alerte sur chaque
   pigeon rend le système inutilisable.
2. **Tourner sur du matériel contraint.** Le calcul se fait à bord ou dans un
   boîtier sur le terrain, pas dans un centre de données.

Ce projet traite les deux, et mesure les deux.

## Résultats

### Détection

Étape 1 terminée le 2026-09-08. YOLOv11 nano, 50 époques, 1 h 33 sur un T4.

| Modèle | Jeu | mAP50 | mAP50-95 | Précision | Rappel |
|---|---|---|---|---|---|
| Référence publiée (YOLOv11n) | test | 0.979 | - | - | - |
| **`baseline_n`** | **test** | **0.9878** | **0.7505** | **0.968** | **0.966** |
| `baseline_n` | validation | 0.975 | 0.758 | 0.968 | 0.955 |

Par classe, sur le jeu de test :

| Classe | mAP50 | mAP50-95 | Précision | Rappel | Instances |
|---|---|---|---|---|---|
| Drone | 0.9928 | 0.738 | 0.984 | 0.982 | 444 |
| Bird | 0.9829 | 0.763 | 0.953 | 0.950 | 456 |

### Matrice de confusion, jeu de test

|  | Vrai Bird | Vrai Drone | Vrai fond |
|---|---|---|---|
| **Prédit Bird** | 0.96 | 0.00 | 0.66 |
| **Prédit Drone** | 0.00 | 0.98 | 0.34 |
| **Prédit fond** | 0.04 | 0.02 | - |

**Lecture, et c'est le résultat le plus intéressant de l'étape.**

- **Aucune confusion croisée entre drone et oiseau**, sur 900 instances. Pas un
  drone déclaré oiseau, pas un oiseau déclaré drone. La discrimination, qui est
  le problème opérationnel du contre-drone, n'est tout simplement pas posée par
  ce jeu de données. C'est désormais mesuré, plus supposé.
- **Le seul mode d'erreur restant est la fausse alerte sur fond vide**, et deux
  tiers de ces détections fantômes sont étiquetées Bird. Sur un système réel,
  c'est cette colonne qui déclenche des alertes pour rien.
- **4 % des oiseaux et 2 % des drones sont manqués.** Écart faible, cohérent
  avec le léger déficit de rappel de la classe Bird.
- **L'écart entre mAP50 (0.988) et mAP50-95 (0.751) reste le second signal.** Le
  modèle trouve les objets de façon fiable, mais place ses cadres
  approximativement dès qu'on exige un recouvrement strict. Défaut de
  localisation, pas de détection. Sur un système de contre-drone, la qualité de
  la boîte conditionne l'estimation de distance et la stabilité du suivi.

Conséquence directe sur la suite : inutile de travailler la discrimination sur
ce jeu, elle est déjà parfaite. L'étape 4 de durcissement devient obligatoire,
pas optionnelle.

Caractéristiques du modèle : 2 582 542 paramètres, 6.4 GFLOPs, 5.2 Mo en
PyTorch, 10.1 Mo en ONNX fp32. Inférence à 4.3 ms sur T4. La mesure qui compte,
sur CPU ARM, arrive à l'étape 3.

### Vitesse et taille

| Modèle | Format | Taille | Latence CPU (ms) | FPS |
|---|---|---|---|---|
| `baseline_n` (à remplir) | PyTorch | | | |
| `baseline_n` (à remplir) | ONNX | | | |
| `baseline_n` quantifié int8 (à remplir) | ONNX | | | |

Machine de mesure : Snapdragon X Elite X1E80100, 12 coeurs, Windows ARM64,
sans GPU dédié. Choix volontaire : cette architecture est proche de celle des
calculateurs embarqués sur drone, bien plus qu'une carte graphique de bureau.

## Données

[Drone-Bird-Detection](https://universe.roboflow.com/myworkspace-0p4nk/drone-bird-detection-3nl79),
version 3 `drone-bird-nonaugmented`, publiée sur Roboflow Universe sous licence
CC BY 4.0.

- 7737 images, réparties en 5418 entraînement / 1547 validation / 772 test
- Deux classes : `Bird`, `Drone`
- Aucune augmentation appliquée en amont, donc pas de fuite entre les jeux

```
@misc{ drone-bird-detection-3nl79_dataset,
  title = { Drone-Bird-Detection Dataset },
  type = { Open Source Dataset },
  author = { MyWorkspace },
  howpublished = { \url{ https://universe.roboflow.com/myworkspace-0p4nk/drone-bird-detection-3nl79 } },
  journal = { Roboflow Universe },
  publisher = { Roboflow },
  year = { 2024 },
}
```

**Limite connue du jeu.** La référence publiée atteint 0.979 de mAP50, ce qui
indique un jeu facile : les objets sont grands et nets dans l'image. La
discrimination drone / oiseau à longue distance, qui est le vrai problème
opérationnel, n'y est pas représentée. Traité à l'étape 4 en isolant un
sous-ensemble de petits objets et en mesurant dessus séparément.

## Étapes

- [x] **1. Détection.** YOLOv11 nano, mAP50 = 0.9878 sur le jeu de test. Fait le 2026-09-08.
- [ ] **2. Suivi.** Relier les détections entre images (ByteTrack), mesurer le
      taux de perte de piste sur des vidéos.
- [ ] **3. Compression.** Export ONNX, quantification int8, mesure de la
      latence CPU avant et après, à dégradation de précision mesurée.
- [ ] **4. Durcissement.** Isoler les petits objets, mesurer la chute de
      performance, analyser la confusion drone / oiseau restante.

## Organisation du dépôt

```
drone-bird-detection/
├── notebooks/     entraînement, tourne sur Google Colab (GPU gratuit)
├── src/           scripts de mesure, tournent en local sur CPU
├── modeles/       poids exportés, .pt et .onnx
├── resultats/     courbes, matrices de confusion, tableaux de mesure
└── data/          jeu de données, non versionné
```

L'entraînement se fait sur Colab faute de GPU NVIDIA en local. La mesure de
vitesse se fait en local, et c'est volontaire : voir la section Résultats.

## Reproduire

1. Ouvrir `notebooks/01_entrainement_colab.ipynb` dans Google Colab.
2. Activer le GPU T4 : `Exécution` > `Modifier le type d'exécution`.
3. Renseigner sa clé API Roboflow dans la cellule 3.
4. Exécuter les cellules de haut en bas.

## Ce que je ferais avec un mois de plus

À remplir en fin de projet.
