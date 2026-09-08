# Étape 3, banc de mesure : protocole, et l'erreur de méthode qui a failli passer

## L'erreur, parce qu'elle est plus instructive que le résultat

Premier balayage du nombre de threads, tous les points mesurés dans le même
processus Python, l'un après l'autre :

| Threads | Médiane (ms) |
|---|---|
| 1 | 323 |
| 4 | 82 |
| 6 | **65** |
| 12 | 527 |

Conclusion tirée sur le moment : la latence n'est pas monotone, le réglage par
défaut d'onnxruntime est 8 fois plus lent que l'optimum, et le point de
fonctionnement à 4 ou 6 threads correspond à un cluster de 4 coeurs du
Snapdragon, dont le L2 de 36 Mo se partage en 3 blocs de 12 Mo.

L'explication était cohérente, vérifiable sur la fiche technique du CPU, et
entièrement fausse.

**Contrôle qui l'a démolie.** Cinq passes indépendantes, chacune dans un
processus neuf :

| Threads | Passes | Médiane |
|---|---|---|
| 4 | 40.92 / 40.60 / 40.67 / 40.57 / 41.29 | 40.8 ms |
| 12 | 24.64 / 25.05 / 26.00 / 26.38 / 27.40 | 25.9 ms |

12 threads est **plus rapide**, et les deux réglages sont stables à 2 et 11 %
près. L'inverse de la conclusion précédente.

**Cause.** `onnxruntime` ne libère pas ses pools de threads à la destruction
d'une session. Enchaîner les mesures dans un même processus fait s'accumuler les
threads : au septième point du balayage, une quarantaine de threads survivants
se disputaient 12 coeurs. **Le banc mesurait sa propre fuite, pas le modèle.**

Correction appliquée dans `src/banc.py` : chaque point de mesure tourne
désormais dans un processus neuf, via `mesurer_isole()`.

Ce qu'il faut en retenir, et qui vaut au-delà de ce projet : une explication
plausible et documentée n'est pas une preuve. Le contrôle de reproductibilité
en conditions indépendantes n'est pas une formalité, c'est ce qui distingue une
mesure d'une impression.

## Résultats, après correction

Balayage propre, un processus par point, 60 passes chacun :

| Threads | Médiane (ms) | p90 (ms) | FPS |
|---|---|---|---|
| 1 | 330.92 | 420.10 | 3.0 |
| 2 | 142.28 | 145.96 | 7.0 |
| 4 | 40.52 | 82.77 | 24.7 |
| 6 | 31.48 | 32.24 | 31.8 |
| 8 | 25.89 | 26.34 | 38.6 |
| **10** | **23.93** | 24.63 | **41.8** |
| 12 | 24.42 | 26.13 | 41.0 |

Courbe monotone, saturation à partir de 10 threads, p90 collé à la médiane sur
tous les points au-dessus de 4. C'est la signature d'une mesure saine.

Le passage de 2 à 4 threads gagne un facteur 3,5 pour un doublement des
ressources. Superlinéaire, probablement l'effet du cluster de 4 coeurs et de son
L2 partagé. Cette fois l'hypothèse reste une hypothèse : elle n'est pas
nécessaire pour conclure, et elle n'a pas été testée.

**Point de fonctionnement retenu : 10 threads.** Toutes les mesures comparatives
du projet sont faites à ce réglage.

## fp32 contre int8 dynamique

10 threads, 100 passes, processus isolés :

| Modèle | Taille | Médiane | FPS |
|---|---|---|---|
| fp32 | 10.11 Mo | 25.29 ms | 39.5 |
| int8 dynamique | 2.85 Mo | 209.72 ms | 4.8 |

**La quantification dynamique divise la taille par 3,55 et la vitesse par 8,3.**

Valeurs consolidees sur 4 passes independantes. Une premiere mesure isolee
donnait 2016 ms, soit un facteur 84 : elle etait polluee, et n'a pas resiste au
controle de reproductibilite. Detail dans `03_quantification.md`.

Ce n'est pas une contre-performance marginale, c'est un piège. La quantification
dynamique insère des opérateurs `DynamicQuantizeLinear` et bascule les
convolutions sur des noyaux entiers qui n'ont pas d'implémentation optimisée sur
ARM64 : l'exécution retombe sur un chemin de référence lent, et s'y ajoute le
coût de requantifier les activations à chaque inférence.

Elle est conçue pour les MatMul et les LSTM, pas pour un réseau à dominante
convolutive comme YOLO. Conservée dans le dépôt comme témoin mesuré, parce que
« j'ai testé et voilà de combien ça rate » vaut mieux que « je ne l'ai pas
tentée ».

## Protocole retenu

1. Un processus neuf par point de mesure. Jamais deux modèles dans le même.
2. Nombre de threads figé explicitement. Ici 10.
3. Chauffe de 10 passes avant les 60 à 100 passes mesurées.
4. Médiane et p90 rapportés ensemble. Un p90 très supérieur à la médiane
   signale une mesure polluée, pas un modèle irrégulier.
5. Machine sur secteur, applications lourdes fermées, charge relevée.
6. Toute conclusion contre-intuitive est rejouée en conditions indépendantes
   avant d'être écrite.

## Suite

Quantification statique, avec calibration sur 200 images de validation. C'est
elle qui quantifie aussi les activations et qui active les noyaux entiers
optimisés. À mesurer sur les deux axes : latence, et mAP après quantification.
Une latence gagnée contre du mAP perdu n'est pas un gain, c'est un arbitrage à
chiffrer.
