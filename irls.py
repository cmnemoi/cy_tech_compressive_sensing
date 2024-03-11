import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.fftpack import dct
from numpy.linalg import norm
from tkinter import N
from numpy.core.fromnumeric import argmax
from re import X
def irls(x,D,iterMax, eps,p):
    n,k=np.shape(D)
    alpha=np.eros(k)
    Q=np.zeros((k,k))
    it=0
    #initialisation de alpha
    alpha0=D.T@np.inv(D@D.T)@X
    test=True
    #Boucle principale
    while test and it<iterMax:
        #construction de la matrice Q
        for i in range(k):
            z=(np.abs(alpha0[i])**2+eps)**(p/2-1)
            Q[i,i]=1/z
            #calcul du nouvel alpha
            alpha=Q@D.T@np.inv(D@D@D.T)@x
            #critère d'arrêt:
            if norm(alpha-alpha0)<np.sqrt(eps)/100 and eps<10**(-8):
                test= False
            else:
                eps=eps/10
                alpha0=alpha
                it=+1

    return alpha, it