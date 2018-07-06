from __future__ import division
import  matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import sys
from scipy import stats

p, fr, _lw, w, fs, sz = 2, 0.75, 0.5, 1, 4, 3

mydir = os.path.expanduser('~/GitHub/residence-time')
tools = os.path.expanduser(mydir + "/tools")
df = pd.read_csv(mydir + '/Model/results/data/SimData.csv')


def figplot(x, y, xlab, ylab, fig, n):
    fig.add_subplot(2, 2, n)
    plt.scatter(x, y, color='0.3', s=sz, linewidths=0.25, edgecolor='w')
    plt.tick_params(axis='both', labelsize=4)
    plt.xlabel(xlab, fontsize=8)
    plt.ylabel(ylab, fontsize=8)

    plt.plot([-5, 1], [-5, 1], lw=0.5, color='k', ls='--')
    plt.xlim(-5, 0)
    plt.ylim(-2.6, 0.3)
    return fig


simList = list(set(df['sim'].tolist()))
dils = []
us1 = []
us2 = []

for sim in simList:
    df2 = df[df['sim'] == sim]
    P = df2['ind.production']
    PS = df2['ind.production']/df2['active.species.richness']
    N = df2['active.total.abundance']
    NS = df2['active.total.abundance']/df2['active.species.richness']
    N0 = np.log10(N[:-1])
    N1 = np.log10(N[1:])
    NS0 = np.log10(NS[:-1])
    NS1 = np.log10(NS[1:])

    mls = [len(N0), len(N1), len(N), len(P)]

    m, b, r, p, std_err = stats.linregress(N0, N1)
    us1.append(b)

    m, b, r, p, std_err = stats.linregress(NS0, NS1)
    us2.append(b)

    dil = np.mean(df2['Q']/df2['V'])
    dils.append(dil)


fig = plt.figure()
xlab = r"$1/\tau$"
fig = figplot(np.log10(dils), np.log10(us1), xlab, r"$\mu$"+', Community-level', fig, 1)
fig = figplot(np.log10(dils), np.log10(us2), xlab, r"$\mu$"+', Avg. among species', fig, 2)

plt.subplots_adjust(wspace=0.4, hspace=0.4)
plt.savefig(mydir + '/Model/results/figures/Supplement/SupFig6.png', dpi=200, bbox_inches = "tight")
plt.close()
