# -*- coding: utf-8 -*-
"""
Syst. Adapt. : Librairies de fonctions pour le script LMS

@author: kooky
"""
import numpy as np

def Lms(x, a, mu, N):
    """
    Fonction qui implémente le LMS

    INPUT :
        x -> Vecteur signal d'entrée
        a -> Vecteur signal de référence
        mu -> Le pas d'adaptation
        N -> Nombre de coefficient du filtre
    OUTPUT : 
        Hm -> Matrice contenant l'historique des coefficients
        y -> Vecteur signal de sortie
    """
    
    M = len(x) # Longueur du signal
    h = np.zeros(N) # Initialisation des coefficients du filtre
    Hm = np.zeros((N,M)) # Matrice contenant l'historique
    Y = np.zeros(M) # Initialisation du vecteur signal de sortie
    
    for n in range(N, M):
        xx = x[n-M] # Vecteur contenant les N dernieres valeurs de x
        y = h.T * xx # Estimation du y
        h = h + mu * (a[n] - h.T * xx) * xx # Correction des coefficients
        Hm[:, n+N] = h # Ajout du coefficient à la matrice historique
        Y[n + M] = y # Ajout de y au vecteur Y
    
    return Hm, Y

def Normalize(x):
    """
    Fonction qui normalise un signal x

    Input :
        x -> Vecteur signal d'entrée
    Output :
        y -> Vecteur signal sortie
    """
    # Normalisation du signal : (Signal - moy) / std
    y = (x - np.mean(x)) / np.std(x)
    return y
    

def BinarySignal(n):
    """
    Fonction qui génère un signal binaire

    Input :
        n -> Taille du signal
    Output :
        y -> Vecteur signal binaire
    """
    y = np.random.choice((-1,1), n) # Génération du signal binaire

    return y

def GaussianSignal(n):
    """
    Fonction qui génère un signal gaussien de moyenne 0 et std 0.1

    Input :
        n -> Taille du signal
    Output :
        y -> Vecteur signal gaussien
    """
    y = np.random.normal(0, 0.1, n) # Génération du signal Gaussien

    return y

def SinusoidalSignal(f, n):
    """
    Fonction qui génère un signal sinusoidale à phase aléatoire

    Input :
        f -> Fréquence du signal
        n -> Taille du signal
    Output :
        y -> Vecteur signal sinusoidale
    """
    phi = np.random.uniform(0, 2* np.pi) # Génération d'une phase aléatoire

    y = np.sin(2 * np.pi * f * n + phi) # Génération du signal sinusoidal

    return y
