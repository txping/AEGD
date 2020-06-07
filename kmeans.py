import numpy as np
import argparse
import random
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import load_iris
from numpy import linalg as LA

parser = argparse.ArgumentParser(description='K-means on Iris')
ap = parser.add_argument
ap('--lr', help='step size', type=float, default=[3.0,6.0], nargs='+')
ap('--c', help='AEGD constant', type=float, default=1.0)
ap('--maxit', help='max iters', type=int, default=40)
ap('--runs', help='experiments repeat times', type=int, default=100)
ap('--tol', help='error tol', type=float, default=1e-6)
args = parser.parse_args()

def knn_loss(centers, data):
    loss = 0
    for point in data:
        loss += min([LA.norm(point-center)**2 for center in centers])
    loss /= 2*len(data)
    return loss

def knn_grad(centers, data):
    grad = np.zeros_like(centers)
    for point in data:
        i = np.argmin([LA.norm(point-center) for center in centers])
        grad[i,:] += centers[i] - point
    grad /= len(data)
    return grad

def g(x,f,data,c):
    return np.sqrt(f(x,data)+c)


def dg(x,f,df,data,c):
    return df(x,data)/(2*np.sqrt(f(x,data)+c))


def gd(x,f,df,data,lr):
    x = x - lr*df(x,data)
    return f(x,data), x


def aegd(x,f,df,data,lr,dg,r,c):
    dgx = dg(x,f,df,data,c)
    r = r/(1+2*lr*dgx**2)
    x = x-2*lr*dgx*r
    return f(x,data), x, r


def runner(x0,f,df,data,lr=args.lr,c=args.c,maxit=args.maxit,tol=args.tol):

    r = dict(fss=[f(x0,data)], xss=[x0],
             fas=[f(x0,data)], xas=[x0], ras=[g(x0,f,data,c)])

    fss, xss = r['fss'], r['xss']
    fas, xas, ras = r['fas'], r['xas'], r['ras']

    for i in range(maxit):
        fs, xs = gd(xss[-1],f,df,data,lr[0])
        fa, xa, ra = aegd(xas[-1],f,df,data,lr[1],dg,ras[-1],c)
        fss.append(fs)
        xss.append(xs)
        fas.append(fa)
        xas.append(xa)
        ras.append(ra)

    return r


########### Iris
Iris = load_iris()
Y = Iris.data.tolist()
eml=[]
gdl=[]
aegdl=[]

for i in range(args.runs):
    y0 = np.array(random.choices(Y,k=3))
    EM = KMeans(n_clusters=3, init=y0, n_init=1, algorithm='full')
    EM.fit(Y)
    res = runner(y0,knn_loss,knn_grad,Y)

    print('[{}] em:{:4f} gd:{:4f} aegd:{:4f}'.format(i,
           EM.inertia_/300,res['fss'][-1],res['fas'][-1]))

    eml.append(EM.inertia_/300)
    gdl.append(res['fss'][-1])
    aegdl.append(res['fas'][-1])

histdata = [eml, gdl, aegdl]
titles = ['EM', 'GD-lr3', 'AEGD-lr6']

fig, axs = plt.subplots(1,3, figsize=(15,5))

for idx, ax in enumerate(axs):
    ax.hist(histdata[idx],bins=20)
    ax.set_title(titles[idx])
    ax.set_xlabel('error')
    ax.set_ylabel('#')
    ax.grid()
    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                  ax.get_xticklabels() + ax.get_yticklabels()):
                  item.set_fontsize(15)
plt.show()
