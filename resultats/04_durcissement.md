# Étape 4, durcissement : où le 0.98 ment

## Question posée

Le modèle annonce 0.9878 de mAP50 et ne confond jamais un drone avec un oiseau.
Avant de croire ce chiffre, deux hypothèses à écarter :

1. Les deux classes sont peut-être séparables par la taille seule, auquel cas le
   réseau n'apporte pas grand-chose.
2. Le cas opérationnel réel, l'objet petit et lointain, est peut-être absent ou
   mal traité, et noyé dans une moyenne dominée par les objets faciles.

Les deux se vérifient. Les deux sont vraies.

## 1. Les classes sont largement séparables par la taille

Distribution du côté des boîtes sur le jeu de test, dans le repère 640x640 que
le modèle voit :

| Classe | Côté médian | Part de petits objets |
|---|---|---|
| Bird | 378 px | 2.6 % |
| Drone | 66 px | 11.7 % |

**Rapport d'échelle entre les deux classes : 5.72.** Un oiseau de ce jeu remplit
l'image, un drone est un petit objet.

Test : un classifieur à un seuil unique sur le côté de la boîte, **n'utilisant
aucun pixel**, seuil appris sur le split d'entraînement et appliqué tel quel au
test.

| | Exactitude |
|---|---|
| Tirage à la classe majoritaire | 50.7 % |
| Seuil de taille seul, côté >= 152 px donne Bird | **76.1 %** |
| Le modèle | ~99 % |

Le biais existe et il est substantiel : 25 points au-dessus du hasard, sans
regarder l'image. Il n'explique pas tout, le réseau apporte bien les 23 points
restants. Mais toute annonce de performance sur ce jeu doit mentionner que le
quart du travail est offert par la statistique des tailles.

## 2. La performance s'effondre sur les petits objets

Évaluation par bande de taille, sur le jeu de test. Annotations **et**
prédictions filtrées par la même bande, faute de quoi les détections d'objets
hors bande comptent en faux positifs et écrasent la précision pour une raison
purement méthodologique.

| Bande | Bird AP50 | Drone AP50 | mAP50 |
|---|---|---|---|
| Petits, côté < 32 px | **0.3824** (13 objets) | 0.8634 (76) | **0.6229** |
| Moyens, 32 à 96 px | 0.8047 (95) | 0.9215 (233) | 0.8631 |
| Grands, côté >= 96 px | 0.9957 (348) | 0.9478 (135) | 0.9718 |
| Jeu complet | 0.9736 (456) | 0.9892 (444) | 0.9814 |

**Le mAP50 passe de 0.981 à 0.623 sur les petits objets. La classe Bird tombe à
0.382, soit 2.6 fois moins que sur le jeu complet.**

Le chiffre affiché en tête du README mesure donc principalement le cas facile.
Sur le cas qui compte, un objet lointain de moins de 32 pixels, le modèle est
médiocre et il l'était sans que rien ne le signale.

L'inversion entre les deux classes est cohérente avec le biais du point 1 : les
oiseaux du jeu étant presque toujours grands, le modèle n'a jamais appris à
reconnaître un petit oiseau. Sur les grands objets Bird atteint 0.9957, sur les
petits il tombe à 0.3824.

## 3. Le jeu ne contient pas le cas difficile

**13 petits oiseaux dans tout le jeu de test.** Le chiffre de 0.3824 est donc à
prendre comme un signal, pas comme une mesure fiable : l'intervalle de confiance
sur 13 objets est large.

C'est la conclusion la plus importante de l'étape. Le problème opérationnel du
contre-drone, distinguer à distance un petit objet volant d'un oiseau, **n'est
pas représenté dans ce jeu de données**. Ni en quantité, ni en difficulté.

## Ce que ça change

L'hypothèse posée le 2026-09-07 au moment de choisir le jeu, "à 0.979 de
référence, ce jeu est facile et la discrimination n'y est pas le vrai défi", est
maintenant établie par trois mesures indépendantes : le rapport d'échelle, le
classifieur à un seuil, et la chute par bande de taille.

Suite logique : passer à **Anti-UAV** (challenge CVPR, RGB et infrarouge), qui
contient des séquences de petits objets à distance, et refaire les mêmes trois
mesures dessus. Le pipeline complet, entraînement, export, banc, évaluateur par
bande de taille, est maintenant en place et se rejoue tel quel.

## Note de méthode

Le premier passage de cette évaluation donnait 0.0977 de mAP50 sur les petits
objets, un effondrement spectaculaire. C'était un défaut de protocole : seules
les annotations étaient filtrées par taille, pas les prédictions, si bien que
chaque détection correcte d'un gros objet comptait en faux positif. La matrice
de confusion, calculée autrement, montrait pourtant 12 oiseaux détectés sur 13,
ce qui a mis la puce à l'oreille.

Troisième fois dans ce projet qu'un résultat spectaculaire s'avère être un
artefact de mesure. Les deux précédents sont dans `02_banc_protocole.md`.

## Outils écrits pour cette étape

- `src/analyser_tailles.py` : distribution des tailles par classe et par split.
- `src/biais_taille.py` : classifieur à un seuil, borne inférieure de ce que le
  jeu donne gratuitement.
- `src/evaluer.py` : évaluation AP50 sans PyTorch, sur onnxruntime et numpy,
  avec filtrage par bande de taille. Calibré à 0.6 point de l'implémentation
  d'ultralytics sur le jeu complet, 0.9814 contre 0.9878.
