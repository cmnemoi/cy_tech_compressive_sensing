"""
Script for comparing the performances of the different algorithms used to compute parcimonious signal representations.

It computes the Mean Squared Error of the OMP and CoSaMP algorithms on the test set, for a given dictionnary.
It prints the comparaison in the console and saves them in a CSV file.

Author: Charles-Meldhine Madi Mnemoi
"""

from collections import namedtuple

import numpy as np
import pandas as pd

from omp import OMP
from cosamp import Cosamp
from irls import irls
from stomp import stomp

ALGORITHMS = {
    "OMP": OMP,
    "CoSaMP": Cosamp,
    "IRLS": irls,
    "StOMP": stomp
}

AlgorithmMetrics = namedtuple("AlgorithmMetrics", ["mean_squared_error", "mean_non_null_components", "mean_nb_iterations"])

def get_algorithm_metrics_on_test_set(algorithm: callable, test_set: pd.DataFrame, dictionnary: np.ndarray) -> AlgorithmMetrics:
    """
    Compute algorithm metrics on the test set for a given dictionnary.

    Parameters:
    -----------
    algorithm (callable): the algorithm to use to compute the parcimonious signal representation (OMP or CoSaMP)
    test_set (pd.DataFrame): the test set on which to compute the Mean Squared Error.
    dictionnary (np.ndarray): the dictionnary to use to compute the parcimonious signal representation.

    Returns:
    --------
    AlgorithmMetrics: the algorithm metrics on the test set : Mean Squared Error and number of iterations.
    """
    mse = 0
    mean_not_null_components = 0
    mean_nb_iterations = 0
    for _, test_vector in test_set.items():
        test_vector = test_vector.to_numpy()
        alpha, _, mean_nb_iterations, _ = algorithm(test_vector, dictionnary)
        mse += np.linalg.norm(test_vector - (dictionnary @ alpha).flatten()) ** 2
        mean_not_null_components += np.count_nonzero(alpha)
        mean_nb_iterations += mean_nb_iterations

    return AlgorithmMetrics(mse / len(test_set), mean_not_null_components / len(test_set), mean_nb_iterations / len(test_set))

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

def save_algorithms_comparaison_to_csv(comparaison: dict[AlgorithmMetrics]) -> None:
    """
    Save the comparaison in a CSV file.

    Parameters:
    -----------
    comparaison (dict): the comparaison to save in the CSV file.
    """
    comparaison_dataframe = pd.DataFrame(index=pd.Index(ALGORITHMS.keys(), name="Algorithme"))
    comparaison_dataframe["Erreur quadratique moyenne"] = [metrics.mean_squared_error for metrics in comparaison.values()]
    comparaison_dataframe["Nombre moyen de composantes non nulles"] = [metrics.mean_non_null_components for metrics in comparaison.values()]
    comparaison_dataframe["Nombre d'itérations moyen"] = [metrics.mean_nb_iterations for metrics in comparaison.values()]
    comparaison_dataframe.to_csv("comparaison.csv")
    
if __name__ == '__main__':
    print("Chargement des données...")
    dictionnary, test_set = load_data()

    print("Comparaison des algorithmes...")
    comparaison = {
        algorithm_name: get_algorithm_metrics_on_test_set(algorithm, test_set, dictionnary) 
        for algorithm_name, algorithm in ALGORITHMS.items()
    }

    print("Sauvegarde de la comparaison dans un fichier CSV...")
    save_algorithms_comparaison_to_csv(comparaison)

    print("Comparaison des erreurs quadratiques moyennes des algorithmes sur les trois signaux de test:")
    print({algorithm_name: metrics.mean_squared_error for algorithm_name, metrics in comparaison.items()})