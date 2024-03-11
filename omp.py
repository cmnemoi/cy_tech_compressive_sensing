import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.fftpack import dct
from numpy.linalg import norm
from tkinter import N
from numpy.core.fromnumeric import argmax
eps=0.0001
iterMax= 100

def OMP(x, D, eps, iterMax):
    n, k = np.shape(D)
    it = 0
    alpha = np.zeros((k, 1))  # Utilisez np.zeros pour initialiser alpha comme un vecteur colonne.
    R = x
    index = []
    A = np.empty((n, 0))
    ps = np.zeros(k)  # Assurez-vous que ps est correctement initialisé avant la boucle.
    while norm(R) > eps and it < iterMax:
        for j in range(k):
            ps[j] = np.abs(np.dot(D[:, j].T, R)) / norm(D[:, j])
        m = np.argmax(ps)
        index.append(m)
        A = D[:, index]
        alpha_temp = np.linalg.pinv(A) @ x  # Utilisez @ pour la multiplication de matrices.
        for i, idx in enumerate(index):
            alpha[idx] = alpha_temp[i]
        R = x - A @ alpha_temp
        it += 1
    return alpha, R, it, index