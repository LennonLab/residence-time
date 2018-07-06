from __future__ import division
import  matplotlib.pyplot as plt
import numpy as np
import os
import sys


def figplot(fig, x, y, xlab, ylab, n):
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
    plt.xlabel(xlab, fontsize=14)
    plt.ylabel(ylab, fontsize=12)

    return fig



p, fr, _lw, w, sz, fs = 2, 0.2, 1.0, 1, 1, 6
mydir = os.path.expanduser('~/GitHub/residence-time/')

xlab = r"$\tau$"
tau = np.arange(-2, 2, 0.01)


G = -1*tau
bmr = -1*tau
disp = -1*tau
resus = -1*tau
spec = 4 - tau**2
dorm = 1*tau

fig = plt.figure()
fig = figplot(fig, tau, G, xlab, 'Growth', 1)
plt.savefig(mydir + 'predictions/predictions/Growth.png', dpi=200, bbox_inches = "tight")
plt.close()

fig = plt.figure()
fig = figplot(fig, tau, bmr, xlab, 'Active '+r'$B$', 1)
plt.savefig(mydir + 'predictions/predictions/BMR.png', dpi=200, bbox_inches = "tight")
plt.close()

fig = plt.figure()
fig = figplot(fig, tau, disp, xlab, 'Active dispersal', 1)
plt.savefig(mydir + 'predictions/predictions/Dispersal.png', dpi=200, bbox_inches = "tight")
plt.close()

fig = plt.figure()
fig = figplot(fig, tau, resus, xlab, 'Resuscitation', 1)
plt.savefig(mydir + 'predictions/predictions/Resus.png', dpi=200, bbox_inches = "tight")
plt.close()

fig = plt.figure()
fig = figplot(fig, tau, spec, xlab, 'Specialization', 1)
plt.savefig(mydir + 'predictions/predictions/Spec.png', dpi=200, bbox_inches = "tight")
plt.close()

fig = plt.figure()
fig = figplot(fig, tau, dorm, xlab, 'Reduction of '+r'$B$'+'\nin dormancy', 1)
plt.savefig(mydir + 'predictions/predictions/ReducB.png', dpi=200, bbox_inches = "tight")
plt.close()
