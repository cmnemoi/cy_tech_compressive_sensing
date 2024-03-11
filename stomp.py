import numpy as np
from numpy.linalg import norm

def StOMP(x,D,eps=10**-4,iterMax=100,t=0.8):
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

if __name__ == '__main__':
    D = np.array([
        [np.sqrt(2)/2, np.sqrt(3)/3, np.sqrt(6)/3, 2/3, -1/3],
        [-np.sqrt(2)/2, -np.sqrt(3)/3, -np.sqrt(6)/6, 2/3, -2/3],
        [0, -np.sqrt(3)/3, np.sqrt(6)/6, 1/3, 2/3]
    ])
    x = np.array([4/3 - np.sqrt(2)/2, 4/3 + np.sqrt(2)/2, 2/3]).T

    print(StOMP(x, D))