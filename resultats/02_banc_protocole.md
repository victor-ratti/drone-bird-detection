# Étape 3, banc de mesure : protocole et premier constat

## Le constat qui conditionne tout le reste

Mesurée sur la même machine, avec le même modèle et le même script, la latence
du modèle fp32 est passée de **23,8 ms à 542 ms en quelques minutes**. Facteur 23,
sans qu'une seule ligne de code ait changé.

Deux causes, identifiées le 2026-09-08.

### 1. Le nombre de threads n'est pas monotone

Balayage sur `baseline_n_best.onnx`, 40 passes par point, machine à 43 % de
charge de fond :

| Threads | Médiane (ms) | p90 (ms) | FPS |
|---|---|---|---|
| 1 | 323.20 | 390.85 | 3.1 |
| 2 | 129.97 | 135.62 | 7.7 |
| 3 | 88.03 | 93.57 | 11.4 |
| 4 | 82.12 | 107.73 | 12.2 |
| **6** | **64.78** | 251.06 | **15.4** |
| 8 | 133.64 | 427.62 | 7.5 |
| 12 | 527.39 | 741.97 | 1.9 |

Le réglage par défaut d'onnxruntime, qui prend les 12 coeurs, est **8,1 fois
plus lent** que l'optimum à 6 threads. Au-delà du point de fonctionnement, la
contention et le trafic de cache coûtent plus que le parallélisme n'apporte.

C'est directement transposable à l'embarqué : sur le calculateur d'un drone, le
SoC fait tourner autre chose que le détecteur. Laisser la bibliothèque prendre
tous les coeurs est une erreur de déploiement, pas un détail de réglage.

### 2. La charge de fond domine la mesure

À 6 threads figés, deux exécutions séparées de quelques minutes ont donné
**64,78 ms puis 213,87 ms**. Le navigateur occupait 39 à 50 % du CPU.

Conséquence : **aucune mesure faite sur une machine chargée n'est publiable.**

## Protocole retenu

Toute mesure destinée au README doit respecter ces cinq points :

1. Navigateur et applications lourdes fermés. Charge CPU de fond sous 10 %.
2. Machine sur secteur, mode d'alimentation « Utilisation normale ».
3. Nombre de threads figé explicitement, jamais laissé à `auto`.
4. Chauffe de 10 passes avant les 100 passes mesurées.
5. Charge CPU relevée au début et à la fin, consignée avec le résultat.

La médiane et le p90 sont rapportés ensemble : un p90 très supérieur à la
médiane signale une mesure polluée, pas un modèle irrégulier.

## Mesures provisoires, non publiables en l'état

Machine chargée, à titre indicatif seulement.

| Modèle | Taille | Médiane à 6 threads |
|---|---|---|
| fp32 | 10.11 Mo | 64.78 ms puis 213.87 ms |
| int8 dynamique | 2.85 Mo | 189.92 ms |

La quantification dynamique divise la taille par 3,55 mais n'apporte
essentiellement rien en vitesse. Résultat attendu : elle ne quantifie que les
poids, les activations restant en flottant et converties à la volée. Elle est
conçue pour les MatMul et les LSTM, pas pour un réseau à dominante Conv comme
YOLO. Elle sert de témoin, la quantification statique est la vraie piste.

## Suite

1. Refaire le balayage machine au repos, pour figer le point de fonctionnement.
2. Mesurer fp32 et int8 dynamique dans ces conditions.
3. Quantification statique avec calibration sur 200 images de validation.
4. Vérifier la précision après quantification : une latence gagnée contre du
   mAP perdu n'est pas un gain, c'est un arbitrage à chiffrer.
