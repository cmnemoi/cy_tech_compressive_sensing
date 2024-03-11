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


def ksvd(X, D0, m):
    n, l = np.shape(X)
    _, k = np.shape(D0)
    A = np.zeros((k, l))  # matrice Lambda des représentations parcimonieuses
    D = D0  # initialisation du dictionnaire

    for _ in range(m):  # boucle principale, m=nombre d'itérations de KSVD
        # Représentation parcimonieuse des l signaux
        for p in range(l):
            alpha, R, it, index = OMP(X[:, p], D, eps, iterMax)
            A[:, p] = alpha.flatten()

        # Actualisation des atomes du dictionnaire, un à un (on a k atomes)
        for i in range(k):
            Ei = X - np.dot(D, A) + np.dot(D[:, i].reshape(-1, 1), A[i, :].reshape(1, -1))
            # Cas des atomes inutiles
            if np.all(A[i, :] == 0):
                v = np.random.rand(n)
                D[:, i] = v / norm(v)
            else:
                # Indices des composantes non nulles de A[i, :]
                Wi = np.nonzero(A[i, :])[0]
                if Wi.size > 0:
                    # Construction de la matrice Omega_i
                    Omega_i = np.zeros((l, Wi.size))
                    for q, idx in enumerate(Wi):
                        Omega_i[idx, q] = 1
                    # Calcul de l'erreur de reconstruction
                    EiR = Ei @ Omega_i
                    # Décomposition en valeurs singulières de EiR
                    U, S, V = np.linalg.svd(EiR, full_matrices=False)
                    # Actualisation de l'atome n°i
                    D[:, i] = U[:, 0]
                    A[i, Wi] = S[0] * V[0, :]

    return D, A
