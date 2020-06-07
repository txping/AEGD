import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 15})

from aegd import AEGD

parser = argparse.ArgumentParser(description='Performance testing functions')
parser.add_argument('--f', default='quad', type=str, help='objective function',
                    choices=['quad','rosen'])

def quad(x):
    r = 0.
    for i in range(len(x)):
        coef = 1. if i%2 else 10.
        r += ((x[i])/coef)**2
    return r

def rosen(x):
    return (1-x[0])**2+100*(x[1]-x[0]**2)**2

def runner(f=rosen, x0=[-3.0,-4.0], optim='gd', lr=0.01, maxiter=2000, tn=1000):
    x = [x0[0]]
    y = [x0[1]]
    z = []

    p = torch.tensor(x0, requires_grad=True, dtype=torch.float)

    if optim == 'gd':
        optimizer = torch.optim.SGD([p], lr=lr, momentum=0)
    elif optim == 'gdm':
        optimizer = torch.optim.SGD([p], lr=lr, momentum=0.9)
    elif optim == 'adam':
        optimizer = torch.optim.Adam([p], lr=lr)
    elif optim == 'aegd':
        optimizer = AEGD([p], lr=lr)

    for i in range(maxiter):
        if optim == 'aegd':
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

        x.append(p.detach().numpy()[0])
        y.append(p.detach().numpy()[1])
        z.append(loss.detach().item())

    return x,y,z

args = parser.parse_args()

if args.f == 'quad':
    xqs, yqs, fqs = runner(f=quad, x0=np.ones(100), optim='gd', lr=0.9, maxiter=5000, tn=1000)
    xqm, yqm, fqm = runner(f=quad, x0=np.ones(100), optim='gdm', lr=0.1, maxiter=5000, tn=1000)
    xqd, yqd, fqd = runner(f=quad, x0=np.ones(100), optim='adam', lr=0.003, maxiter=5000, tn=1000)
    xqa, yqa, fqa = runner(f=quad, x0=np.ones(100), optim='aegd', lr=12, maxiter=5000, tn=1000)

    plt.figure(1, figsize=(8,6))
    plt.plot(fqs, 'g', lw=1, label='GD-lr0.9')
    plt.plot(fqm, 'b', lw=1, label='GDM-lr0.1')
    plt.plot(fqd, 'k', lw=1, label='Adam-lr0.003')
    plt.plot(fqa, 'r', lw=1, label='AEGD-lr12')

    plt.xscale('log')
    plt.yscale('log')
    plt.xlim([10,5000])
    plt.xlabel('Iteration')
    plt.ylabel('|f(x)-f(x*)|')
    plt.grid()
    plt.legend()
    plt.show()
else:
    assert args.f == 'rosen'
    xrs, yrs, frs = runner(f=rosen, x0=[-3,-4], optim='gd', lr=0.0003, maxiter=20000, tn=5000)
    xrm, yrm, frm = runner(f=rosen, x0=[-3,-4], optim='gdm', lr=0.0002, maxiter=20000, tn=5000)
    xrd, yrd, frd = runner(f=rosen, x0=[-3,-4], optim='adam', lr=0.006, maxiter=20000, tn=5000)
    xra, yra, fra = runner(f=rosen, x0=[-3,-4], optim='aegd', lr=0.0004, maxiter=20000, tn=5000)

    plt.figure(1, figsize=(8,6))
    plt.plot(frs, 'g', lw=1, label='GD-lr0.0003')
    plt.plot(frm, 'b', lw=1, label='GDM-lr0.0002')
    plt.plot(frd, 'k', lw=1, label='Adam-lr0.006')
    plt.plot(fra, 'r', lw=1, label='AEGD-lr0.0004')

    plt.yscale('log')
    plt.xlim([0,20000])
    plt.xlabel('Iteration')
    plt.ylabel('|f(x)-f(x*)|')
    plt.grid()
    plt.legend()
    plt.show()
