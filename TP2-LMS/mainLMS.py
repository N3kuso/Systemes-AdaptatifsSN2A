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

# Génération du signal d'observation
noise_e = 0 # Contexte non-bruité donc bruit e[n] = 0
a = y_tild + noise_e # Signal d'observation, ici égal à y_tild

# Identification du filtre RIF grâce au LMS
mu = 0.001 # Pas d'adaptation du filtre
N = 3 # Nombre coefficient du filtre
estimated_coef, estimated_y = FunctionLMS.Lms(input_signal, a, mu, N) # Utilisation de la fonction LMS

# Affichage du signal d'entrée
FunctionLMS.PlotSignal(time, a, title="a[n]")
# Affichage du signal d'entrée filtrée par le filtre RIF
FunctionLMS.PlotSignal(time, estimated_y, title="y[n]")

# Affichage de l'évolution des coefficients
FunctionLMS.PlotCoefficientsEvolution(estimated_coef, h_unknown_coeff, title=f"Évolution des coefficients LMS, mu : {mu}, ordre {N}")

### Surestimation du nombre de coefficients de la fonction LMS ###
h_unknown_coeff = [1, 0.75, 0.5, 0, 0] # Ajout de 0 fictif pour compatibilité avec la fonction PlotCoefficientsEvolution
## Ordre 4
N = 4
estimated_coef, estimated_y = FunctionLMS.Lms(input_signal, a, mu, N) # Utilisation de la fonction LMS
print(estimated_coef.shape)
# Affichage de l'évolution des coefficients
FunctionLMS.PlotCoefficientsEvolution(estimated_coef, h_unknown_coeff, title=f"Évolution des coefficients LMS, mu : {mu}, ordre {N}")

## Ordre 5
N = 5
estimated_coef, estimated_y = FunctionLMS.Lms(input_signal, a, mu, N) # Utilisation de la fonction LMS
# Affichage de l'évolution des coefficients
FunctionLMS.PlotCoefficientsEvolution(estimated_coef, h_unknown_coeff, title=f"Évolution des coefficients LMS, mu : {mu}, ordre {N}")

### Visualisation avec différentes valeurs de pas d'adaptation
N = 3 # Ordre parfait du filtre RIF
h_unknown_coeff = [1, 0.75, 0.5] # Retour au bon nombre de coeff pour le RIF
for test_mu in [0.01, 0.005, 0.001, 0.0005]:
    estimated_coef, estimated_y = FunctionLMS.Lms(input_signal, a, test_mu, N)
    # Affichage de l'évolution des coefficients
    FunctionLMS.PlotCoefficientsEvolution(estimated_coef, h_unknown_coeff, title=f"Évolution des coefficients LMS, mu : {test_mu}, ordre {N}")
    
plt.show()