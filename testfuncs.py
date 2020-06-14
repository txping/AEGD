import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 15})

from aegd import AEGD

r"""
To reproduce the results for the quadratic function, run:
python testfuncs.py --func quad --n 100 --lr 0.9 0.1 0.003 12 --maxiter 5000 --tn 1000

To reproduce the results for the 2D Rosenbrock function, run:
python testfuncs.py --func rosen --lr 3e-4 2e-4 6e-3 4e-4 --maxiter 20000 --tn 5000
"""

parser = argparse.ArgumentParser(description='Performance testing functions')
parser.add_argument('--func', default='rosen', type=str, help='objective function',
                    choices=['quad','rosen'])
parser.add_argument('--n', default=100, type=int, help='dim. of the quadratic function')
parser.add_argument('--lr', default=[3e-4,2e-4,6e-3,4e-4], type=float, help='learning rate', nargs='+')
parser.add_argument('--m', default=0.9, type=float, help='momentum for GDM')
parser.add_argument('--c', default=1, type=float,  help='constant for AEGD')
parser.add_argument('--maxiter', default=20000, type=int, help='max. iter')
parser.add_argument('--tn', default=5000, type=int, help='print every tn iterations')


def quad(x):
    r = 0.
    for i in range(len(x)):
        coef = 1. if i%2 else 10.
        r += ((x[i])/coef)**2
    return r

def rosen(x):
    return (1-x[0])**2+100*(x[1]-x[0]**2)**2

def runner(f=rosen, x0=[-3.0,-4.0], optim='GD', lr=0.01, maxiter=20000,
           tn=1000, m=0.9, c=1):
    z = []
    p = torch.tensor(x0, requires_grad=True, dtype=torch.float)

    if optim == 'GD':
        optimizer = torch.optim.SGD([p], lr=lr, momentum=0)
    elif optim == 'GDM':
        optimizer = torch.optim.SGD([p], lr=lr, momentum=m)
    elif optim == 'Adam':
        optimizer = torch.optim.Adam([p], lr=lr)
    elif optim == 'AEGD':
        optimizer = AEGD([p], lr=lr, c=c)

    for i in range(maxiter):
        if optim == 'AEGD':
            def closure():
                optimizer.zero_grad()
                loss = f(p)
                loss.backward()
                return loss
            loss = optimizer.step(closure)
        else:
            optimizer.zero_grad()
            loss = f(p)
            loss.backward()
            optimizer.step()

        if i % tn == 0:
            print('[{}] {}-{}-f: {}'.format(i, f.__name__, optim, loss))

        z.append(loss.detach().item())

    return z

args = parser.parse_args()
if args.func == 'rosen':
    f = rosen
    x0 = [-3, -4]
    xscale = 'linear'
    xstart = 0
else:
    f = quad
    x0 = np.ones(args.n)
    xscale = 'log'
    xstart = 10

optims = ['GD', 'GDM', 'Adam', 'AEGD']
colors = ['g', 'b', 'k', 'r']
plt.figure(1, figsize=(8,6))
for i in range(4):
    fs = runner(f=f, x0=x0, optim=optims[i], lr=args.lr[i],
                maxiter=args.maxiter, tn=args.tn, m=args.m, c=args.c)
    plt.plot(fs, colors[i], lw=1, label='{}-lr{}'.format(optims[i], args.lr[i]))
plt.yscale('log')
plt.xscale(xscale)
plt.xlim([xstart,args.maxiter])
plt.xlabel('Iteration')
plt.ylabel('|f(x)-f(x*)|')
plt.grid()
plt.legend(loc='lower left')
plt.show()
