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

### Durcissement : où le 0.98 ment

Le chiffre de tête mesure surtout le cas facile. Trois mesures indépendantes le
montrent, détail dans `resultats/04_durcissement.md`.

**Les classes sont largement séparables par la taille.** Côté médian d'un
oiseau : 378 px. D'un drone : 66 px. Rapport 5.72. Un classifieur à un seuil
unique sur la taille de la boîte, **sans regarder un seul pixel**, atteint
76.1 % d'exactitude, contre 50.7 % au hasard. Un quart du travail est offert par
la statistique du jeu.

**La performance s'effondre sur les petits objets.** Annotations et prédictions
filtrées par la même bande de taille.

| Bande | Bird AP50 | Drone AP50 | mAP50 |
|---|---|---|---|
| Petits, < 32 px | **0.3824** (13 obj.) | 0.8634 (76) | **0.6229** |
| Moyens, 32 à 96 px | 0.8047 (95) | 0.9215 (233) | 0.8631 |
| Grands, >= 96 px | 0.9957 (348) | 0.9478 (135) | 0.9718 |
| Jeu complet | 0.9736 (456) | 0.9892 (444) | 0.9814 |

**Le mAP50 passe de 0.981 à 0.623 sur les petits objets, et la classe Bird tombe
à 0.382.** Les oiseaux du jeu étant presque toujours grands, le modèle n'a
jamais appris à reconnaître un petit oiseau.

**Et le jeu ne contient pas le cas difficile** : 13 petits oiseaux dans tout le
jeu de test. Le problème opérationnel du contre-drone, distinguer à distance un
petit objet volant d'un oiseau, n'est représenté ni en quantité ni en
difficulté.

Suite : rejouer les trois mêmes mesures sur Anti-UAV, qui contient des séquences
de petits objets à distance. Le pipeline complet se rejoue tel quel.

Caractéristiques du modèle : 2 582 542 paramètres, 6.4 GFLOPs, 5.2 Mo en
PyTorch, 10.1 Mo en ONNX fp32. Inférence à 4.3 ms sur T4. La mesure qui compte,
sur CPU ARM, arrive à l'étape 3.

### Vitesse et taille

Mesures du 2026-09-08, 10 threads intra-op, 4 passes indépendantes par modèle,
chacune dans un processus neuf, machine au repos. Protocole et pièges de mesure
dans `resultats/02_banc_protocole.md`, analyse dans `resultats/03_quantification.md`.

| Modèle | Format | Taille | Médiane | Dispersion | FPS |
|---|---|---|---|---|---|
| `baseline_n` | ONNX fp32 | 10.11 Mo | **25.29 ms** | x1.11 | **39.5** |
| `baseline_n` | ONNX int8 statique | 3.03 Mo | 39.83 ms | x1.13 | 25.1 |
| `baseline_n` | ONNX int8 dynamique | 2.85 Mo | 209.72 ms | x1.03 | 4.8 |

**Le modèle fp32 tient le temps réel sur CPU ARM sans accélérateur**, à 39.5
images par seconde.

**La quantification int8 n'accélère pas ce modèle sur cette cible, elle le
ralentit.** Statique : 1.57 fois plus lent. Dynamique : 8.3 fois. Le gain de
taille est en revanche réel, facteur 3.34.

La cause n'est pas le modèle mais le backend : le chemin fp32 d'onnxruntime
passe par des noyaux NEON optimisés pour ARM64, le chemin int8 de l'exécuteur
CPU par défaut n'a pas d'équivalent aussi abouti et retombe sur des
implémentations génériques.

**Conclusion transposable à un déploiement embarqué : le choix du backend
précède le choix du format de poids.** Quantifier avant de savoir ce que le
runtime cible sait exécuter est une perte de temps. Le gain existerait sur le
NPU via l'execution provider QNN, sur XNNPACK, ou sur une cible Jetson en
TensorRT.

Le nombre de threads compte autant que le modèle : 1 thread donne 331 ms, 10
threads en donnent 24. Sur le calculateur d'un drone, ce réglage est une
décision de déploiement, pas un détail.

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
- [x] **4. Durcissement.** mAP50 de 0.981 à 0.623 sur les objets de moins de
      32 px, Bird à 0.382. Biais d'échelle de 5.72 entre les classes, un seuil
      de taille seul atteint 76.1 %. Fait le 2026-09-08.
- [ ] **5. Rejouer sur Anti-UAV**, qui contient le cas difficile absent ici.

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
