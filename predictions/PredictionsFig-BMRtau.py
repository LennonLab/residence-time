from __future__ import division
import  matplotlib.pyplot as plt
import numpy as np
import os
import sys


p, fr, _lw, w, sz, fs = 2, 0.2, 1.0, 1, 1, 6
mydir = os.path.expanduser('~/GitHub/residence-time/')

xlab = r"$\tau$"
tau = np.arange(-2, 2, 0.01)



tau1 = np.arange(-2, 0, 0.01)
tau2 = np.arange(0, 2, 0.01)
bmr1 = 1*tau1
bmr2 = -1*tau2
tau = tau1.tolist() + tau2.tolist()
bmr = bmr1.tolist() + bmr2.tolist()

fig = plt.figure()
ax = fig.add_subplot(2, 2, 1)

plt.plot(tau, bmr, lw=1, color='0.3')
plt.tick_params(
axis='both',       # changes apply to both axes
which='both',      # both major and minor ticks are affected
top='off',         # ticks along the top edge are off
left='off',
labelleft='off') # labels along the bottom edge are off

xlab =  r"$log_{10}(T*\tau$)"

plt.axvline(0, color='k', ls='--', lw = 0.5)
plt.xlabel(xlab, fontsize=18)
plt.ylabel('Productivity', fontsize=18)

labels = [item.get_text() for item in ax.get_xticklabels()]
#print labels
#sys.exit()

labels[1] = ''
labels[3] = '0'
labels[4] = ''
ax.set_xticklabels(labels, fontsize=14)

plt.subplots_adjust(wspace=0.3, hspace=0.4)
plt.savefig(mydir + 'predictions/predictions/P-equiv.png', dpi=200, bbox_inches = "tight")
plt.close()


fig = plt.figure()
ax = fig.add_subplot(2, 2, 1)

plt.plot(tau, bmr, lw=1, color='0.3')
plt.tick_params(
axis='both',       # changes apply to both axes
which='both',      # both major and minor ticks are affected
top='off',         # ticks along the top edge are off
left='off',
labelleft='off') # labels along the bottom edge are off

xlab =  r"$log_{10}(T*\tau$)"

plt.axvline(0, color='k', ls='--', lw = 0.5)
plt.xlabel(xlab, fontsize=18)
plt.ylabel('Active 'r"$N$", fontsize=18)

labels = [item.get_text() for item in ax.get_xticklabels()]
#print labels
#sys.exit()

labels[1] = ''
labels[3] = '0'
labels[4] = ''
ax.set_xticklabels(labels, fontsize=14)

plt.subplots_adjust(wspace=0.3, hspace=0.4)
plt.savefig(mydir + 'predictions/predictions/N-equiv.png', dpi=200, bbox_inches = "tight")
plt.close()
