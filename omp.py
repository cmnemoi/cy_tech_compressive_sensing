import numpy as np
from numpy.linalg import norm

def OMP(x, D, eps=10**-4, iterMax=100):
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

if __name__ == '__main__':
    D = np.array([
        [np.sqrt(2)/2, np.sqrt(3)/3, np.sqrt(6)/3, 2/3, -1/3],
        [-np.sqrt(2)/2, -np.sqrt(3)/3, -np.sqrt(6)/6, 2/3, -2/3],
        [0, -np.sqrt(3)/3, np.sqrt(6)/6, 1/3, 2/3]
    ])
    x = np.array([4/3 - np.sqrt(2)/2, 4/3 + np.sqrt(2)/2, 2/3]).T

    print(OMP(x, D))