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
from FunctionsCAG import PlotSignal
########################################################
# P1 :
########################################################
n = 16384 # Longueur du signal d'entrée
n_bins = 64 # Nombre de point pour l'histogramme
time = np.arange(0,n)

### Génération  d'un signal ###
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

### Génération des signaux normalisé ###
## Gaussien Normalisé ##
gaussien_norm = Normalize(gaussien)

plt.subplot(211)
# Affichage Signal Gaussien Norm
plt.plot(time, gaussien_norm)
plt.title("Signal gaussien Normalisé")
plt.subplot(212)
# Affichage Histogramme Gaussien Norm
plt.hist(gaussien_norm, bins=n_bins)
plt.title("Histogramme gaussien Normalisé")
plt.show()

## Binaire Normalisé ##
binaire_norm = Normalize(binaire)

plt.subplot(211)
# Affichage Signal binaire Norm
plt.plot(time, binaire_norm)
plt.title("Signal Binaire Normalisé ")
plt.subplot(212)
# Affichage Histogramme Binaire Norm
plt.hist(binaire_norm, bins=n_bins)
plt.title("Histogramme Binaire Normalisé")
plt.show()

## Uniforme Normalisé ##
uniforme_norm = Normalize(uniforme)

plt.subplot(211)
# Affichage Signal Uniforme
plt.plot(time, uniforme_norm)
plt.title("Signal Uniforme Normalisé")
plt.subplot(212)
# Affichage Histogramme Uniforme
plt.hist(uniforme_norm, bins=n_bins)
plt.title("Histogramme Uniforme Normalisé")
plt.show()

########################################################
# P2 :
########################################################
### utilisation du CAG ###
from FunctionsCAG import CAG

# Définition des variables sigma2 et mu
sigma2 = np.power(1,2)

# Valeur théorique de h2
h2_theorical = np.full(n, sigma2)

## Visualisation de l'évolution du pas d'adaptation pour binaire ##
mu = np.arange(0.00005, 0.005, 0.0005) # Plage de valeur de mu (binaire) 0 <-> 2

# Affichage du h2 théorique
plt.plot(time, h2_theorical, "r", label="h2 théorique")

# Boucle pour tester les différentes valeurs de mu
for i in mu :    
    h2_experimental, y_experimental = CAG(binaire_norm, i, sigma2)
    plt.plot(time, h2_experimental, '-', linewidth=1)

plt.legend()
plt.title("Evolution du coef h2 pour un signal binaire")
plt.show()

## Visualisation de l'évolution du pas d'adaptation pour gaussien ##
mu = np.arange(0.00005, 0.005, 0.0005) # Plage de valeur de mu (gaussien)

# Affichage du h2 théorique
plt.plot(time, h2_theorical, "r", label="h2 théorique")

# Boucle pour tester les différentes valeurs de mu
for i in mu :    
    h2_experimental, y_experimental = CAG(gaussien_norm, i, sigma2)
    plt.plot(time, h2_experimental, '-', linewidth=1)

plt.legend()
plt.title("Evolution du coef h2 pour un signal gaussien")
plt.show()

## Visualisation de l'évolution du pas d'adaptation pour uniforme ##
mu = np.arange(0.00005, 0.005, 0.0005) # Plage de valeur de mu (uniforme)

# Affichage du h2 théorique
plt.plot(time, h2_theorical, "r", label="h2 théorique")

# Boucle pour tester les différentes valeurs de mu
for i in mu :    
    h2_experimental, y_experimental = CAG(uniforme_norm, i, sigma2)
    plt.plot(time, h2_experimental, '-', linewidth=1)

plt.legend()
plt.title("Evolution du coef h2 pour un signal uniforme")
plt.show()

########################################################
# P3 :
########################################################
# Valeur d'alpha pour modifier la puissance du signal
alpha = 0.46 #round(random.random(),1)
print(f"Valeur de alpha : {alpha}")
# Instant où la puissance du signal doit changer
instant_change = round(n/2)

## Signal binaire ##
# Création d'un signal binaire non stationnaire
binaire_not_stationnary = binaire_norm
binaire_not_stationnary[instant_change:] = alpha * binaire_not_stationnary[instant_change:]

mu = 0.005 # valeur du pas d'adaptation

# Utilisation du CAG
h2_experimental, y_experimental = CAG(binaire_not_stationnary, mu, sigma2)

# Calcul de la nouvelle valeur théorique de h2
new_h2_theorical = np.full(n, np.power(sigma2/alpha,2))

# Affichage
plt.plot(time, h2_theorical, "r", label="h2 théorique")
plt.plot(time, new_h2_theorical, "r--", label="h2 théorique 2")
plt.plot(time, h2_experimental, '-', linewidth=1)
plt.legend()
plt.title("Evolution du coef h2 pour un signal binaire non stationnaire")
plt.show()

PlotSignal(time, binaire_not_stationnary, y_experimental)

## Signal gaussien ##
# Création d'un signal gaussien non stationnaire
gaussien_not_stationnary = gaussien_norm
gaussien_not_stationnary[instant_change:] = alpha * gaussien_not_stationnary[instant_change:]

mu = 0.005 # valeur du pas d'adaptation

# Utilisation du CAG
h2_experimental, y_experimental = CAG(gaussien_not_stationnary, mu, sigma2)

# Calcul de la nouvelle valeur théorique de h2
new_h2_theorical = np.full(n, np.power(sigma2/alpha,2))

plt.plot(time, h2_theorical, "r", label="h2 théorique")
plt.plot(time, new_h2_theorical, "r--", label="h2 théorique 2")
plt.plot(time, h2_experimental, '-', linewidth=1)
plt.legend()
plt.title("Evolution du coef h2 pour un signal gaussien non stationnaire")
plt.show()

PlotSignal(time, gaussien_not_stationnary, y_experimental)

## Signal uniforme ##
# Création d'un signal uniforme non stationnaire
uniforme_not_stationnary = uniforme_norm
uniforme_not_stationnary[instant_change:] = alpha * uniforme_not_stationnary[instant_change:]

mu = 0.005 # valeur du pas d'adaptation

# Utilisation du CAG
h2_experimental, y_experimental = CAG(uniforme_not_stationnary, mu, sigma2)

# Calcul de la nouvelle valeur théorique de h2
new_h2_theorical = np.full(n, np.power(sigma2/alpha,2))

plt.plot(time, h2_theorical, "r", label="h2 théorique")
plt.plot(time, new_h2_theorical, "r--", label="h2 théorique 2")
plt.plot(time, h2_experimental, '-', linewidth=1)
plt.legend()
plt.title("Evolution du coef h2 pour un signal uniforme non stationnaire")
plt.show()

PlotSignal(time, uniforme_not_stationnary, y_experimental)