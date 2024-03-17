import numpy as np
from numpy.linalg import norm

def Cosamp(x,D,s=None,eps=10**-4,iterMax=100):
    # Estimation de s proposée dans le papier introduisant CoSaMP (D. Needell and J. A. Tropp, 2008)
    if s is None:
        s = int( ( len(D) / ( 2 * np.log(len(x)) ) ) )

    # Initialisation
    k = D.shape[1]
    alpha = np.zeros(k)
    residual = x
    it = 0
    index = []
    while it < iterMax and norm(residual) > eps:
        # Calcul de la contribution de chaque atome au résidu
        contributions = np.abs(D.T @ residual) / np.linalg.norm(D, ord=2, axis=0)
        # Récupération des indices des 2s plus grandes contributions
        indexes = np.argsort(contributions)[-2*s:]
        index.extend(indexes)
        index = list(set(index))
        # Mise à jour de alpha par moindres carrés
        A = D[:, index]
        alpha[index] = np.linalg.pinv(A) @ x
        # Mise à jour de r
        residual = x - A @ alpha[index]
        it += 1

    return alpha, residual, it, index

if __name__ == '__main__':
    D = np.array([
        [np.sqrt(2)/2, np.sqrt(3)/3, np.sqrt(6)/3, 2/3, -1/3],
        [-np.sqrt(2)/2, -np.sqrt(3)/3, -np.sqrt(6)/6, 2/3, -2/3],
        [0, -np.sqrt(3)/3, np.sqrt(6)/6, 1/3, 2/3]
    ])
    x = np.array([4/3 - np.sqrt(2)/2, 4/3 + np.sqrt(2)/2, 2/3]).T

    print(Cosamp(x, D))