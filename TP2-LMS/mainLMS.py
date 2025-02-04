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

valid_choice = False

while not valid_choice:
    choice_signal = input("Choix du signal : 1 (Binaire) - 2 (Gaussien) - 3 (Sinusoïdal)")

    if choice_signal == "1":
        ## Génération du signal Binaire ##
        input_signal = FunctionLMS.BinarySignal(n)
        # Affichage
        FunctionLMS.PlotSignal(time, input_signal, title="Signal Binaire")
        valid_choice = True
    elif choice_signal == "2":
        ## Génération du signal Gaussien
        input_signal = FunctionLMS.GaussianSignal(n)
        # Affichage
        FunctionLMS.PlotSignal(time, input_signal, title="Signal Gaussien") 
        valid_choice = True
    elif choice_signal == "3":
        ## Génération du signal Sinusoidal ##
        input_signal = FunctionLMS.SinusoidalSignal(f, n, time)
        # Affichage
        FunctionLMS.PlotSignal(time, input_signal, title="Signal Sinusoïdal")
        valid_choice = True
    else:
        print("Choix invalide. Veuillez entrer 1, 2 ou 3.")

## Simulation du filtre inconnu ##
h_unknown_coeff = [1, 0.75, 0.5]

y_tild = np.convolve(input_signal, h_unknown_coeff)
print(y_tild)
print(len(y_tild))

plt.show()