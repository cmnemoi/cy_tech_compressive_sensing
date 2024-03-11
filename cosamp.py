import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.fftpack import dct
from numpy.linalg import norm
from tkinter import N
from numpy.core.fromnumeric import argmax

eps=0.0001
iterMax= 100
def Cosamp(x,D,eps,iterMax,s):
    n,k=np.shape(D)
    it=0
    alpha=np.zeros(k)
    R=x
    index=[]
    A=np.empty((n,0))
    ps=np.zeros(k)
    while norm(R)>eps and it<iterMax:
        for j in range (k):
            ps[j]=np.abs(np.dot(D[:,j].T,R))/norm(D[:,j])
        #sélection des atomes dont la contrib est parmi les 2s plus grandes
        m=np.argpartition(ps,-2*s)[-2*s:]
        #ajout des nouveaux indices aux anciens
        V=set(index)|set(m)
        index=list(V)
        #index.append(m)
        #Application des moindres carré
        A=D[:,index]
        alpha[index]=np.dot(np.linalg.pinv(A),x)
        index=np.argpartition(np.abs(alpha),-s)[-s:]
        #on recommence les moindres carrés avec les s atomes retenus
        alpha=np.zeros(k)
        A=D[:,index]
        alpha[index]=np.dot(np.linalg.pinv(A),x)
        #actualisation du résidu
        R=x-np.dot(A,alpha[index])
        it=it+1
    return alpha, R, it, index