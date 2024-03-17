"""
This file contains the implementation of the StOMP (Stagewise Orthogonal Matching Pursuit) algorithm.

Author: Charles-Meldhine Madi Mnemoi
"""

import numpy as np

def stomp(x: np.ndarray, D: np.ndarray, epsilon: float = 10**-4, max_iterations: int = 100, threshold_coefficient: float = 2):
    """
    StOMP (Stagewise Orthogonal Matching Pursuit) algorithm.
    This algorithm is used to compute the parcimonious signal representation of a given signal x, for a given dictionnary D.

    Args:
        x (np.ndarray): the signal to find the parcimonious representation of.
        D (np.ndarray): the dictionnary to use to compute the parcimonious representation of x.
        epsilon (float, optional): the precision of the algorithm. Defaults to 10**-4.
        max_iterations (int, optional): the maximum number of iterations of the algorithm. Defaults to 100.
        threshold_coefficient (float, optional): the coefficient to use to compute the threshold. Defaults to 2. 
    """
    k = np.shape(D)[1]
    alpha = np.zeros(k)
    residual = x
    nb_iterations = 0
    indexes = []
    while np.linalg.norm(residual, ord=2) > epsilon and nb_iterations < max_iterations:
        # Compute the contribution of each atom to the residual
        contributions = np.abs(D.T @ residual) / np.linalg.norm(D, ord=2, axis=0)
        # Compute the threshold vector
        threshold = threshold_coefficient * np.linalg.norm(residual, ord=2) / np.sqrt(k)
        # Find the indexes of the atoms which contribution is above the threshold
        indexes.extend(np.where(contributions > threshold)[0])
        indexes = list(set(indexes))
        # Compute the new alpha vector
        A = D[:, indexes]
        alpha[indexes] = np.linalg.pinv(A) @ x
        # Compute the new residual
        residual = x - A @ alpha[indexes]
        # Increment the number of iterations
        nb_iterations += 1

    return alpha, residual, nb_iterations, indexes

if __name__ == '__main__':
    D = np.array([
        [np.sqrt(2)/2, np.sqrt(3)/3, np.sqrt(6)/3, 2/3, -1/3],
        [-np.sqrt(2)/2, -np.sqrt(3)/3, -np.sqrt(6)/6, 2/3, -2/3],
        [0, -np.sqrt(3)/3, np.sqrt(6)/6, 1/3, 2/3]
    ])
    x = np.array([4/3 - np.sqrt(2)/2, 4/3 + np.sqrt(2)/2, 2/3]).T

    print(stomp(x, D))