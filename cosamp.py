import numpy as np
from numpy.linalg import norm

def Cosamp(x,D,s=None,eps=10**-4,iterMax=100):
    # Estimation de s proposée dans le papier introduisant CoSaMP (D. Needell and J. A. Tropp, 2008)
    if s is None:
        s = int( ( len(D) / ( 2 * np.log(len(x)) ) ) )

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


if __name__ == '__main__':
    D = np.array([
        [np.sqrt(2)/2, np.sqrt(3)/3, np.sqrt(6)/3, 2/3, -1/3],
        [-np.sqrt(2)/2, -np.sqrt(3)/3, -np.sqrt(6)/6, 2/3, -2/3],
        [0, -np.sqrt(3)/3, np.sqrt(6)/6, 1/3, 2/3]
    ])
    x = np.array([4/3 - np.sqrt(2)/2, 4/3 + np.sqrt(2)/2, 2/3]).T

    print(Cosamp(x, D, len(x)-1))