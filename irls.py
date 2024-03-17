import numpy as np
from numpy.linalg import norm, inv

def irls(x, D, iterMax=100, eps=10**-4, p=3):
    n, k = np.shape(D)
    alpha = np.zeros(k)
    Q = np.zeros((k, k))
    it = 0
    # Initialisation de alpha
    alpha0 = D.T @ inv(D @ D.T) @ x
    test = True
    # Boucle principale
    while test and it < iterMax:
        # Construction de la matrice Q
        for i in range(k):
            z = (np.abs(alpha0[i]) ** 2 + eps) ** (p / 2 - 1)
            Q[i, i] = 1 / z
        # Calcul du nouvel alpha
        alpha = Q @ D.T @ inv(D @ Q @ D.T) @ x
        # Critère d'arrêt
        if norm(alpha - alpha0) < np.sqrt(eps) / 100 and eps < 10 ** (-8):
            test = False
        else:
            eps = eps / 10
            alpha0 = alpha
            it += 1
 
    return alpha, None, it, None

if __name__ == '__main__':
    D = np.array([
        [np.sqrt(2)/2, np.sqrt(3)/3, np.sqrt(6)/3, 2/3, -1/3],
        [-np.sqrt(2)/2, -np.sqrt(3)/3, -np.sqrt(6)/6, 2/3, -2/3],
        [0, -np.sqrt(3)/3, np.sqrt(6)/6, 1/3, 2/3]
    ])
    x = np.array([4/3 - np.sqrt(2)/2, 4/3 + np.sqrt(2)/2, 2/3]).T

    print(irls(x, D))