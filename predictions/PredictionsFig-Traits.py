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

fig = plt.figure()

G = -1*tau
bmr = -1*tau
disp = -1*tau
resus = -1*tau
spec = 4 - tau**2
dorm = 1*tau

fig = figplot(fig, tau, G, xlab, 'Growth rate', 1, 'Growth')
fig = figplot(fig, tau, bmr, xlab, 'Active BMR', 2, 'BRM')
fig = figplot(fig, tau, disp, xlab, 'Active dispersal', 3, 'Disp')
fig = figplot(fig, tau, resus, xlab, 'Resuscitation rate', 4, 'Resus')
fig = figplot(fig, tau, spec, xlab, 'Specialization', 5, 'Spec')
fig = figplot(fig, tau, dorm, xlab, 'Reduction of BMR\nin dormancy', 6, 'reducBMR')

plt.subplots_adjust(wspace=0.3, hspace=0.4)
plt.savefig(mydir + 'Emergence/predictions/Predictions-Traits.png', dpi=200, bbox_inches = "tight")
plt.close()
