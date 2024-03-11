import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.fftpack import dct
from numpy.linalg import norm
from tkinter import N
from numpy.core.fromnumeric import argmax

eps=0.0001
iterMax= 100
def StOMP(x,D,eps,iterMax, t):
    n,k=np.shape(D)
    it=0
    alpha=np.matrix(np.zeros(k)).T
    R=x
    index=[]
    A=np.empty((n,0))
    ps=np.zeros(k)
    while norm(R)>eps and it<iterMax:
        for j in range (k):
            ps[j]=np.abs(np.dot(D[:,j].T,R))/norm(D[:,j])
        #m=np.argmax(ps)
        #Calcul du seuil
        Seuil=t*norm(R)/np.sqrt(k)
        #sélection des atomes dont la contrib >Seuil
        m=np.where(ps > Seuil)[0]
        #ajout des nouveaux indices aux anciens
        V=set(index)|set(m)
        index=list(V)
       # index.append(index)
        #Matrice formée dess atomes sélectionnés
        A=D[:,index]
        #Application des moindres carrés
        alpha[index]=np.dot(np.linalg.pinv(A),x)
        #Actualisation du résidu
        R=x-np.dot(A,alpha[index])
        it=it+1
    return alpha, R, it, index