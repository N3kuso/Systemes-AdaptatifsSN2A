# -*- coding: utf-8 -*-
"""
Systèmes adaptatifs :
    Script principal pour le TP1 sur le Controle Automatique de Gain (CAG)

@author: kooky
"""

import numpy as np
import matplotlib.pyplot as plt
import random
from FunctionsCAG import Normalize
########################################################
# P1 :
########################################################
n = 8192 # Longueur du signal d'entrée
n_bins = 64 # Nombre de point pour l'histogramme
time = np.arange(0,n)

### Génération  d'un signal
## Gaussien ##
gaussien = np.random.normal(0, 0.2, n)

plt.subplot(211)
# Affichage Signal Gaussien
plt.plot(time, gaussien)
plt.title("Signal gaussien")
plt.subplot(212)
# Affichage Histogramme Gaussien
plt.hist(gaussien, bins=n_bins)
plt.title("Histogramme gaussien")
plt.show()

## Binaire ##
binaire = np.random.choice((-1,1), n)

plt.subplot(211)
# Affichage Signal binaire
plt.plot(time, binaire)
plt.title("Signal Binaire")
plt.subplot(212)
# Affichage Histogramme Binaire
plt.hist(binaire, bins=n_bins)
plt.title("Histogramme Binaire")
plt.show()

## Uniforme ##
uniforme = np.random.uniform(size=n)

plt.subplot(211)
# Affichage Signal Uniforme
plt.plot(time, uniforme)
plt.title("Signal Uniforme")
plt.subplot(212)
# Affichage Histogramme Uniforme
plt.hist(uniforme, bins=n_bins)
plt.title("Histogramme Uniforme")
plt.show()

