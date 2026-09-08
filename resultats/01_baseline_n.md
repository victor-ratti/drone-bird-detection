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

**La discrimination drone / oiseau est déjà résolue, et c'est une mauvaise
nouvelle.** La matrice de confusion sur le jeu de test ne montre aucune
confusion croisée : zéro drone déclaré oiseau, zéro oiseau déclaré drone, sur
900 instances.

|  | Vrai Bird | Vrai Drone | Vrai fond |
|---|---|---|---|
| Prédit Bird | 0.96 | 0.00 | 0.66 |
| Prédit Drone | 0.00 | 0.98 | 0.34 |
| Prédit fond | 0.04 | 0.02 | - |

Le déficit de rappel de la classe Bird (0.950 contre 0.982) ne vient donc pas
d'oiseaux pris pour des drones, mais d'oiseaux purement manqués : 4 % passent en
fond, contre 2 % pour les drones.

Le seul mode d'erreur qui subsiste est la fausse alerte sur fond vide, et deux
tiers de ces détections fantômes portent l'étiquette Bird.

**Ce que ça implique.** L'hypothèse posée le 2026-09-07 en choisissant ce jeu
("à 0.979 de référence, le jeu est facile et la discrimination n'y est pas le
vrai défi") est maintenant vérifiée par la mesure. L'étape 4 de durcissement
n'est plus une option de confort, c'est la seule façon de rendre ce projet
intéressant.

## Suite

Ne pas chercher un meilleur mAP50, il est saturé.

1. **Étape 3, compression et mesure sur CPU ARM.** Le résultat que personne n'a
   publié sur ce jeu, et celui qui parle aux postes embarqués.
2. **Étape 4, durcissement, désormais obligatoire.** Isoler les instances de
   moins de 32 pixels et mesurer dessus séparément. Si la confusion croisée
   reste nulle même sur les petits objets, il faudra changer de jeu pour
   Anti-UAV, qui contient le cas difficile.

## Fichiers

Archive `resultats_baseline_n.zip` récupérée depuis Colab, contenant les poids,
les courbes, les matrices de confusion et les sorties de validation sur le jeu
de test.
