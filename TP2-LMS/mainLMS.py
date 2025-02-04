# -*- coding: utf-8 -*-
"""
Syst. Adapt. : Script principal pour le LMS

@author: kooky
"""
import numpy as np
import matplotlib.pyplot as plt
import FunctionLMS

########################################################
# Simulation Numériques
########################################################
# Initialisation de valeur pour la génération des signaux
n = 1600 # Taille du signal
time = np.arange(0,n) # Génèration d'un vecteur temps
f = 1000 # Fréquence du signal

## Génération du signal Binaire ##
binary_signal = FunctionLMS.BinarySignal(n)
# Affichage
FunctionLMS.PlotSignal(time, binary_signal)

## Génération du signal Gaussien
gaussian_signal = FunctionLMS.GaussianSignal(n)
# Affichage
FunctionLMS.PlotSignal(time, gaussian_signal)

## Génération du signal Sinusoidale ##
sinusoidal_signal = FunctionLMS.SinusoidalSignal(f, n, time)
# Affichage
FunctionLMS.PlotSignal(time, sinusoidal_signal, title="Signal Sinusoîdal")