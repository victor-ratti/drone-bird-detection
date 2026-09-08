# Étape 3, quantification int8 : un résultat négatif, et il est net

## Le résultat

10 threads intra-op, 60 passes par mesure, 4 passes indépendantes par modèle,
chacune dans un processus neuf. Machine au repos.

| Modèle | Taille | Passes (ms) | Médiane | Dispersion | FPS |
|---|---|---|---|---|---|
| **fp32** | 10.11 Mo | 24.0 / 24.9 / 25.8 / 26.6 | **25.29 ms** | x1.11 | **39.5** |
| int8 statique | 3.03 Mo | 37.6 / 38.7 / 40.7 / 42.4 | 39.83 ms | x1.13 | 25.1 |
| int8 dynamique | 2.85 Mo | 207.1 / 210.5 / 212.9 / 208.4 | 209.72 ms | x1.03 | 4.8 |

**Sur ce CPU, avec ce runtime, la quantification int8 n'accélère rien. Elle
ralentit.** Le statique est 1,57 fois plus lent que le fp32, le dynamique 8,3
fois.

Le gain de taille, lui, est réel : 10.11 Mo vers 3.03 Mo, facteur 3,34.

## Pourquoi

La quantification ne rend un modèle plus rapide que si le runtime dispose de
noyaux entiers optimisés pour l'architecture cible. Ce n'est pas le cas ici.

- Le chemin **fp32** d'onnxruntime passe par MLAS, avec des noyaux NEON écrits
  et optimisés pour ARM64. C'est du code rapide.
- Le chemin **int8** de l'exécuteur CPU par défaut ne dispose pas d'un équivalent
  aussi abouti sur ARM64. Les convolutions quantifiées retombent sur des
  implémentations génériques, et le surcoût de quantification et
  déquantification des tenseurs s'ajoute à chaque couche.
- Le **dynamique** est le pire des deux mondes : il requantifie les activations
  à chaque inférence, et il vise les MatMul et les LSTM, pas les convolutions.

Autrement dit, le facteur limitant n'est pas le modèle, c'est le **backend
d'exécution**.

## Ce que ça implique pour un déploiement embarqué

C'est le point à retenir, et il est transposable tel quel à un calculateur de
drone.

1. **Quantifier avant de savoir ce que le runtime cible sait exécuter est une
   perte de temps.** Le choix du backend précède le choix du format de poids.
2. **Le gain de taille et le gain de vitesse sont deux problèmes distincts.**
   Ici on a divisé la taille par 3,34 tout en perdant 37 % de débit. Si la
   contrainte est la mémoire flash, c'est un bon échange. Si c'est la latence,
   c'est une régression.
3. **Le fp32 tient déjà le temps réel** à 39,5 images par seconde sur un CPU ARM
   sans accélérateur. Sur ce matériel, il n'y a pas de problème de vitesse à
   résoudre.

## Où le gain existerait vraiment

Trois pistes, non explorées, par ordre de rendement attendu :

- **Le NPU du Snapdragon**, via l'execution provider QNN d'onnxruntime. C'est
  précisément le matériel conçu pour l'inférence entière. Demande le paquet
  `onnxruntime-qnn`, un build séparé : les providers disponibles dans
  l'installation courante se limitent à `CPUExecutionProvider` et
  `AzureExecutionProvider`.
- **L'execution provider XNNPACK**, qui possède des noyaux int8 ARM optimisés.
- **Sur une cible NVIDIA Jetson**, TensorRT en int8, où le gain de quantification
  est bien documenté. C'est l'architecture réellement embarquée par la plupart
  des drones du secteur.

## Méthode

Ce résultat a failli être faux deux fois.

- Un premier balayage concluait à une latence non monotone et à un optimum à 6
  threads. Artefact : `onnxruntime` ne libère pas ses pools de threads, et le
  banc mesurait sa propre fuite. Voir `02_banc_protocole.md`.
- Une première mesure du modèle dynamique donnait 2016 ms, soit un facteur 84.
  Rejouée en 4 passes indépendantes, la valeur est de 210 ms, facteur 8,3. La
  première mesure était polluée.

Dans les deux cas, c'est le contrôle de reproductibilité qui a tranché, pas
l'explication la plus séduisante.

## Reste à faire

Mesurer le mAP du modèle statique quantifié. Une perte de précision viendrait
s'ajouter à la perte de vitesse, ce qui achèverait le dossier. Une précision
intacte rendrait le modèle intéressant pour une cible contrainte en mémoire,
pas en temps.
