
import numpy as np
from numpy.linalg import norm

from omp import OMP

def ksvd(X, D0, m, eps=10**-4, iterMax=100):
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
