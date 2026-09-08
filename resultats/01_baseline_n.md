# Étape 1, modèle de base `baseline_n`

Journal brut de l'entraînement du 2026-09-08. Sert de trace, le README porte la
synthèse.

## Conditions

| | |
|---|---|
| Modèle de départ | `yolo11n.pt`, pré-entraîné COCO |
| Jeu | Drone-Bird-Detection v3, 5418 / 1547 / 772 |
| Époques | 50, sans arrêt anticipé déclenché (`patience=15`) |
| Taille d'image | 640 |
| Lot | 32 |
| Optimiseur | AdamW, choisi automatiquement, lr0 = 0.001667, momentum = 0.9 |
| Matériel | Tesla T4 sur Google Colab |
| Durée | 1 h 33 |
| Ultralytics | 8.4.143, torch 2.11.0+cu128 |

Architecture : 181 couches, 2 590 230 paramètres à l'entraînement, 2 582 542
après fusion, 6.4 GFLOPs. 451 des 499 tenseurs pré-entraînés ont été transférés,
la tête de détection étant réapprise pour 2 classes au lieu de 80.

## Résultats

Jeu de test, 772 images, 900 instances :

| | mAP50 | mAP50-95 | Précision | Rappel |
|---|---|---|---|---|
| Toutes classes | 0.9878 | 0.7505 | 0.968 | 0.966 |
| Drone (444) | 0.9928 | 0.738 | 0.984 | 0.982 |
| Bird (456) | 0.9829 | 0.763 | 0.953 | 0.950 |

Jeu de validation, 1547 images, 1752 instances : mAP50 0.975, mAP50-95 0.758.

Référence publiée par l'auteur du jeu, même architecture : mAP50 0.979.

## Courbe d'apprentissage

| Époque | mAP50 validation |
|---|---|
| 1 | 0.594 |
| 5 | 0.852 |
| 10 | 0.916 |
| 20 | 0.960 |
| 30 | 0.968 |
| 40 | 0.973 |
| 50 | 0.975 |

Palier atteint vers l'époque 30. Les 20 dernières époques rapportent 0.7 point.
La désactivation de la mosaïque aux 10 dernières époques (`close_mosaic=10`)
fait chuter les pertes d'entraînement sans gain notable en validation.

## Ce que ces chiffres disent

**Le pipeline est correct.** Dépasser la référence publiée de 0.9 point avec la
même architecture confirme qu'il n'y a pas d'erreur de configuration, de
chemins ou de classes.

**Le jeu est saturé.** À 0.988 de mAP50, il n'y a plus rien à gagner sur cette
métrique. Continuer à optimiser la détection serait du temps perdu.

**Le vrai gisement est ailleurs, dans l'écart mAP50 / mAP50-95.** 0.988 contre
0.751 : le modèle voit les objets mais place ses cadres approximativement. Sur
un système de contre-drone réel, la qualité de la boîte conditionne l'estimation
de distance et le suivi. C'est mesurable et améliorable.

**L'oiseau reste la classe faible** sur les deux métriques de comptage, avec un
rappel de 0.950 contre 0.982 pour le drone. À croiser avec la matrice de
confusion pour savoir si les oiseaux manqués deviennent des faux négatifs ou
des drones déclarés. C'est la différence entre un capteur aveugle et un système
qui tire sur des pigeons.

## Suite

Ne pas chercher un meilleur mAP50. Passer à l'étape 3, la compression et la
mesure sur CPU ARM, qui est le résultat que personne n'a publié sur ce jeu.

## Fichiers

Archive `resultats_baseline_n.zip` récupérée depuis Colab, contenant les poids,
les courbes, les matrices de confusion et les sorties de validation sur le jeu
de test.
