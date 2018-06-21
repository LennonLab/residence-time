from __future__ import division
import  matplotlib.pyplot as plt
import numpy as np
import os
import sys


def figplot(fig, x, y, xlab, ylab, n, label):
    fig.add_subplot(3, 3, n)

    plt.plot(x, y, lw=1, color='0.3')
    plt.tick_params(
    axis='both',       # changes apply to both axes
    which='both',      # both major and minor ticks are affected
    bottom='off',      # ticks along the bottom edge are off
    top='off',         # ticks along the top edge are off
    left='off',
    labelbottom='off',
    labelleft='off') # labels along the bottom edge are off
    plt.xlabel(xlab, fontsize=8)

    if n in [1,2,3,4,5]: plt.ylabel(ylab, fontsize=8)
    else: plt.ylabel(ylab, fontsize=8)
    #if n == 6: plt.xlim(-2,1)

    return fig



p, fr, _lw, w, sz, fs = 2, 0.2, 1.0, 1, 1, 6
mydir = os.path.expanduser('~/GitHub/residence-time2/')

xlab = r"$\tau$"
tau = np.arange(-2, 2, 0.01)
N = 4 - tau**2
Prod = 4 - tau**2
S = 4 - tau**2
E = tau**2
W = (tau-1)**2
Dorm = 1*tau

fig = plt.figure()

fig = figplot(fig, tau, N, xlab, r"$N$", 1, 'N')
fig = figplot(fig, tau, Prod, xlab, r"$P$", 2, 'ProdI')
fig = figplot(fig, tau, S, xlab, r"$S$", 3, 'S')
fig = figplot(fig, tau, E, xlab, r"$E$", 4, 'E')
fig = figplot(fig, tau, W, xlab, r"$\beta$", 5, 'W')
fig = figplot(fig, tau, Dorm, xlab, '%Dormant', 6, 'Dorm')


plt.subplots_adjust(wspace=0.3, hspace=0.4)
plt.savefig(mydir + 'Emergence/predictions/Predictions-Taxa.png', dpi=200, bbox_inches = "tight")
plt.close()
