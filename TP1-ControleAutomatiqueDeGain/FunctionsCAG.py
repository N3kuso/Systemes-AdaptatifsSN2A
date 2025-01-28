# -*- coding: utf-8 -*-
"""
Fichiers de Functions pour le TP1 CAG

@author: kooky
"""
import numpy as np

def Normalize(X):
    # (Signal - moy) / std
    return (X - np.mean(X))/np.std(X)

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
        
    