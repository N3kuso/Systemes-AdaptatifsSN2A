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
n = 16384 # Taille du signal
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

########################################################
# Si1 : Contexte non-bruité
########################################################
# Génération du signal de sortie avec le filtre RIF
y_tild = np.convolve(input_signal, h_unknown_coeff, mode='same')
# print(y_tild)
# print(len(y_tild))

# Génération du signal d'observation
noise_e = 0 # Contexte non-bruité donc bruit e[n] = 0
a = y_tild + noise_e # Signal d'observation, ici égal à y_tild
# print(a)

# Identification du filtre RIF grâce au LMS
mu = 0.001 # Pas d'adaptation du filtre
N = 3 # Nombre coefficient du filtre
estimated_coef, estimated_y = FunctionLMS.Lms(input_signal, a, mu, N) # Utilisation de la fonction LMS

# print(estimated_coef.shape)

FunctionLMS.PlotSignal(time, a, title="a[n]")
FunctionLMS.PlotSignal(time, estimated_y, title="y[n]")

FunctionLMS.PlotCoefficientsEvolution(estimated_coef, h_unknown_coeff)

# for test_mu in [0.01, 0.005, 0.001, 0.0005]:
#     estimated_coef, estimated_y = FunctionLMS.Lms(input_signal, a, test_mu, N)
#     print(f"Test avec mu={test_mu}, derniers coefficients:", estimated_coef[:, -1])
    
plt.show()