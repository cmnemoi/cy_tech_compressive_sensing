import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.fftpack import dct
from numpy.linalg import norm
from tkinter import N
from numpy.core.fromnumeric import argmax

def coher(phi,D):
  m,n=np.shape(phi)
  n,k=np.shape(D)
  #matrice des cosinus entre lignes de phi et colonne de D
  Co=np.zeros((m,k))
  for i in range(m):
    for j in range(k):
      z=np.dot(phi[i,:].T,D[:,j])/(norm(phi[i,:])*norm(D[:,j]))
#Calcul de la valeur absolue
      Co[i,j]=np.abs(z)
#obtention du max
    a=np.max(Co)
    c=a*np.sqrt(n)
    return c