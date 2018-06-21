from __future__ import division
import  matplotlib.pyplot as plt
import numpy as np
import os
import sys


def figplot(fig, x, y, xlab, ylab, n, label):
    fig.add_subplot(4, 4, n)

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
W = tau**2
Dorm = 1*tau

fig = plt.figure()

fig = figplot(fig, tau, N, xlab, r"$N$", 1, 'N')
fig = figplot(fig, tau, Prod, xlab, r"$P_{I}$", 2, 'ProdI')
fig = figplot(fig, tau, S, xlab, r"$S$", 3, 'S')
fig = figplot(fig, tau, E, xlab, r"$E$", 4, 'E')
fig = figplot(fig, tau, W, xlab, r"$\beta$", 5, 'W')
fig = figplot(fig, tau, Dorm, xlab, '%Dormant', 6, 'Dorm')


G = -1*tau
bmr = -1*tau
disp = -1*tau
resus = -1*tau
spec = 4 - tau**2
dorm = 1*tau

fig = figplot(fig, tau, G, xlab, 'Growth rate', 7, 'Growth')
fig = figplot(fig, tau, bmr, xlab, 'Active BMR', 8, 'BRM')
fig = figplot(fig, tau, disp, xlab, 'Active dispersal', 9, 'Disp')
fig = figplot(fig, tau, resus, xlab, 'Resuscitation rate', 10, 'Resus')
fig = figplot(fig, tau, spec, xlab, 'Specialization', 11, 'Spec')
fig = figplot(fig, tau, dorm, xlab, 'Reduction of BMR\nin dormancy', 12, 'reducBMR')


tau1 = np.arange(-2, 0, 0.01)
tau2 = np.arange(0, 2, 0.01)
bmr1 = 1*tau1
bmr2 = -1*tau2
tau = tau1.tolist() + tau2.tolist()
bmr = bmr1.tolist() + bmr2.tolist()

ax = fig.add_subplot(4, 4, 13)

plt.plot(tau, bmr, lw=1, color='0.3')
plt.tick_params(
axis='both',       # changes apply to both axes
which='both',      # both major and minor ticks are affected
top='off',         # ticks along the top edge are off
left='off',
labelleft='off') # labels along the bottom edge are off

plt.axvline(0, color='k', ls='--', lw = 0.5)
plt.xlabel(r"$BMR$ - $1/\tau$", fontsize=8)
plt.ylabel(r"$N$", fontsize=8)

labels = [item.get_text() for item in ax.get_xticklabels()]
#print labels
#sys.exit()

labels[1] = ''
labels[2] = '0'
labels[3] = ''
ax.set_xticklabels(labels, fontsize=8)


ax = fig.add_subplot(4, 4, 14)

plt.plot(tau, bmr, lw=1, color='0.3')
plt.tick_params(
axis='both',       # changes apply to both axes
which='both',      # both major and minor ticks are affected
top='off',         # ticks along the top edge are off
left='off',
labelleft='off') # labels along the bottom edge are off
plt.axvline(0, color='k', ls='--', lw = 0.5)
plt.xlabel(r"$BMR$ - $1/\tau$", fontsize=8)
plt.ylabel(r"$P$", fontsize=8)

labels = [item.get_text() for item in ax.get_xticklabels()]
labels[1] = ''
labels[2] = '0'
labels[3] = ''
ax.set_xticklabels(labels, fontsize=8)

plt.subplots_adjust(wspace=0.3, hspace=0.4)
plt.savefig(mydir + 'Emergence/predictions/Predictions.png', dpi=200, bbox_inches = "tight")
plt.close()
