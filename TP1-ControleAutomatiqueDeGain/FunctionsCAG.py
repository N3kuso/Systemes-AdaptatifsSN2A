# -*- coding: utf-8 -*-
"""
Fichiers de Functions pour le TP1 CAG

@author: kooky
"""
import numpy as np

def Normalize(X):
    # (Signal - moy) / std
    return (X - np.mean(X))/np.std(X)
    