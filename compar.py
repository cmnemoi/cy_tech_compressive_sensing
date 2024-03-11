"""
Script for comparing the performances of the different algorithms used to compute parcimonious signal representations.

It computes the Mean Squared Error of the OMP and CoSaMP algorithms on the test set, for a given dictionnary.
It prints the comparaison in the console and saves them in a CSV file.
"""

import numpy as np
import pandas as pd

from omp import OMP
from cosamp import Cosamp

ALGORITHMS = {
    "OMP": OMP,
    "CoSaMP": Cosamp
}

def get_algorithm_mse_on_test_set(algorithm: callable, test_set: pd.DataFrame, dictionnary: np.ndarray) -> float:
    """
    Compute the Mean Squared Error of the given algorithm on the test set, for a given dictionnary.

    Parameters:
    -----------
    algorithm (callable): the algorithm to use to compute the parcimonious signal representation (OMP or CoSaMP)
    test_set (pd.DataFrame): the test set on which to compute the Mean Squared Error.
    dictionnary (np.ndarray): the dictionnary to use to compute the parcimonious signal representation.

    Returns:
    --------
    float: the Mean Squared Error of the given algorithm on the test set.
    """
    mse = 0
    for _, test_vector in test_set.items():
        test_vector = test_vector.to_numpy()
        alpha, _, _, _ = algorithm(test_vector, dictionnary)
        mse += np.linalg.norm(test_vector - (dictionnary @ alpha).flatten()) ** 2

    return mse / len(test_set)

def load_data() -> tuple[np.ndarray, pd.DataFrame]:
    """
    Load the dictionary and the test set from the local disk.

    Returns:
    --------
    tuple: (dictionnary, test_set)
    """
    dictionnary = pd.read_csv("Dico-appris.csv").to_numpy()
    test_set = pd.read_excel("DonneesProjet.xlsx", sheet_name="vecteurs pour valideer")
    
    return dictionnary, test_set

def save_algorithms_comparaison_to_csv(comparaison: dict[str, int]) -> None:
    """
    Save the comparaison in a CSV file.

    Parameters:
    -----------
    comparaison (dict): the comparaison to save in the CSV file.
    """
    comparaison_dataframe = pd.DataFrame()
    comparaison_dataframe["Algorithm"] = comparaison.keys()
    comparaison_dataframe["Mean Squared Error"] = comparaison.values()

    comparaison_dataframe.to_csv("comparaison.csv", index=False)
    
if __name__ == '__main__':
    dictionnary, test_set = load_data()
    comparaison = {
        algorithm_name: get_algorithm_mse_on_test_set(algorithm, test_set, dictionnary) 
        for algorithm_name, algorithm in ALGORITHMS.items()
    }

    save_algorithms_comparaison_to_csv(comparaison)

    print("Comparaison des erreurs quadratiques moyennes des algorithmes sur les trois signaux de test:")
    print(comparaison)