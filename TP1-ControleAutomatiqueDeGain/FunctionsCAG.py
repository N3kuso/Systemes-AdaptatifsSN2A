# -*- coding: utf-8 -*-
"""
Fichiers de Functions pour le TP1 CAG

@author: kooky
"""
import numpy as np
import matplotlib.pyplot as plt

def Normalize(X):
    # (Signal - moy) / std
    X_norm = (X - np.mean(X))/np.std(X)
    print(f"Moyenne après normalisation : {np.mean(X_norm)}")
    print(f"Écart-type après normalisation : {np.std(X_norm)}")
    print(f"Puissance après normalisation : {np.mean(np.power(X_norm,2)):.5f}")
    return X_norm

def CAG(signal, mu, sigma2):
    """
    Fonction permettant de calculer le Contrôle Automatique de Gain
    
    Paramètres :
        signal -> Signal d'entrée
        mu -> pas d'adaptation
        sigma -> Puissance voulue en sortie
    
    Output : 
        H -> Vecteur des h au cours du temps
        Y -> Vecteur du signal estimé de sortie
    """
    
    h = 0 # Initialisation du premier h à 0
    H = [0] # Initialisation du vecteur H pour visualiser l'évolution du h
    Y = [0] # Initialisation du vecteur Y pour le signal estimé
    
    # Boucle qui scrute chaque échantillons du signal
    for i in range(len(signal)-1):
        y = h * signal[i] # Estimation du y estimé
        tmp = np.power(h,2) + mu * (sigma2 - np.power(y,2)) # Calcul de h2
        h = np.sqrt(tmp) # Calcul H
        
        H.append(tmp) # On ajoute la valeur de tmp au vecteur H pour voir son évolution
        Y.append(y)
    
    return H,Y
        
def PlotSignal(t, s1, s2):
    plt.subplot(211)
    # Affichage Signal d'entrée
    plt.plot(t, s1)
    plt.title("Signal d'entrée")
    plt.subplot(212)
    # Affichage signal de sortie
    plt.plot(t, s2)
    plt.title("Signal de sortie")
    plt.show()
    
def Evolutionh2withmu(t, s1, mu, sigma2, title):
    n = len(t)
    # Valeur théorique de h2
    h2_theorical = np.full(n, sigma2)
    
    plt.plot(t, h2_theorical, "r", label="h2 théorique")
    
    h2_experimental, y_experimental = CAG(s1, mu, sigma2)
    plt.plot(t, h2_experimental, '-', linewidth=1, label="h2 pratique")
    plt.legend()
    plt.title(title)
    plt.show()
    print(f"{s1}")
    print(f" Moyenne h2 pratique {np.mean(h2_experimental[8192:])}")
    erreur = round(np.mean(np.abs(h2_experimental[8192:] - h2_theorical[8192:])),5) * 100
    print(f" Erreur {erreur} %")
    erreur_quadratique = np.mean(np.power(h2_experimental[8192:] - h2_theorical[8192:],2))
    print(f"Erreur quadratique : {erreur_quadratique}")