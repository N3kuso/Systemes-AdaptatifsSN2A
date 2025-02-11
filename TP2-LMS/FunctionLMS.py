# -*- coding: utf-8 -*-
"""
Syst. Adapt. : Librairies de fonctions pour le script LMS

@author: kooky
"""
import numpy as np
import matplotlib.pyplot as plt


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
    h = np.zeros(N) # Initialisation des coefficients du filtre (vecteur colonne)
    Hm = np.zeros((N,M-1)) # Matrice contenant l'historique
    Y = np.zeros(M) # Initialisation du vecteur signal de sortie
    
    for n in range(N, M-1):
        xx = x[n+1:n-N+1:-1] # Vecteur contenant les N dernieres valeurs de x
        
        # print(xx)
        # print(f"xx : {xx.shape}")

        # DEBUG Condition qui vérifie que la dimension de xx ne dépasse pas N
        if xx.shape[0] != N:
            raise ValueError(f"Erreur: xx.shape={xx.shape}, attendu={N}")

        Y[n] = np.dot(h.T,xx) # Estimation du y (Produit scalaire)
        # print(f"y : {Y[n]}")

        error = a[n] - Y[n] # Calcul de l'erreur (Scalaire)
        # print(f"erreur : {error}")

        h = h + (mu * error * xx) # Correction des coefficients
        # print(f" h: {h}")
        
        Hm[:, n] = h # Ajout du coefficient à la matrice historique
        #Y[n] = y # Ajout de y au vecteur Y
    
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
    # y = Normalize(y)
    return y

def SinusoidalSignal(f, n, t):
    """
    Fonction qui génère un signal sinusoidale à phase aléatoire

    Input :
        f -> Fréquence du signal
        n -> Taille du signal
        t -> Vecteur temps
    Output :
        y -> Vecteur signal sinusoidale
    """
    phi = np.random.uniform(0, 2* np.pi, n) # Génération d'un vecteur de phase aléatoire

    y = np.sin(2 * np.pi * f * t + phi) # Génération du signal sinusoidal

    return y

def PlotSignal(t, s1, title="Signal", xlabel="Temps", ylabel="Amplitude"):
    """
    Fonction qui affiche un signal à l'aide de Matplotlib
    
    Input : 
        t -> Vecteur temps
        s1 -> Vecteur signal
        title  : Titre du graphique
        xlabel -> Label de l'axe des abscisses
        ylabel -> Label de l'axe des ordonnées
    """

    plt.figure(figsize=(10, 4))
    plt.plot(t, s1, label=title)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid()
    #plt.show()

def PlotCoefficientsEvolution(Hm, h_theorique, title="Évolution des coefficients LMS"):
    """
    Fonction qui affiche l'évolution des coefficients estimés par LMS comparé avec les valeurs théoriques à obtenir.

    INPUT :
        Hm          -> Matrice contenant l'historique des coefficients estimés
        h_theorique -> Vecteur contenant les valeurs théoriques des coefficients
        title       -> Titre du graphique
    """
    
    N, M = Hm.shape  # Nombre de coefficients

    plt.figure(figsize=(10, 6))  # Taille de la figure
    
    for i in range(N):
        plt.plot(Hm[i, :], label=f"$h_{i}$ estimé", linestyle="-") # Evolution du coefficient
        plt.axhline(h_theorique[i], color="k", linestyle="--", label=f"$h_{i}$ théorique")  # Ligne horizontale

    plt.title(title)
    plt.xlabel("Itérations")
    plt.ylabel("Valeur des coefficients")
    plt.legend()
    plt.grid()
    # plt.show()
