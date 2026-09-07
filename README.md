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

| Modèle | Jeu | mAP50 | mAP50-95 | mAP50 Drone | mAP50 Bird |
|---|---|---|---|---|---|
| Référence publiée (YOLOv11n) | test | 0.979 | - | - | - |
| `baseline_n` (à remplir) | test | | | | |

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

- [ ] **1. Détection.** Entraîner un YOLOv11 nano, mesurer sur le jeu de test.
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
