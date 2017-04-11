from __future__ import division
import  matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from random import randint
import os
import statsmodels.api as sm


def assigncolor(xs):
    cDict = {}
    clrs = []
    for x in xs:
        if x not in cDict:
            r1 = lambda: randint(0,255)
            r2 = lambda: randint(0,255)
            r3 = lambda: randint(0,255)
            cDict[x] = '#%02X%02X%02X' % (r1(),r2(),r3())

        clrs.append(cDict[x])
    return clrs


def figplot(clrs, x, y, xlab, ylab, fig, n):
    fig.add_subplot(3, 3, n)
    #plt.scatter(x, y, lw=0.5, color=clrs, s = 4)
    plt.scatter(x, y, s = sz, color='0.7', linewidths=0.1, edgecolor='w')
    lowess = sm.nonparametric.lowess(y, x, frac=fr)
    x, y = lowess[:, 0], lowess[:, 1]
    plt.plot(x, y, lw=_lw, color='k')
    plt.tick_params(axis='both', labelsize=fs)
    plt.xlabel(xlab, fontsize=fs+3)
    plt.ylabel(ylab, fontsize=fs+3)
    return fig


p, fr, _lw, w, sz, fs = 2, 0.2, 1.5, 1, 5, 6
mydir = os.path.expanduser('~/GitHub/residence-time/')
df = pd.read_csv(mydir + 'Emergence/results/simulated_data/SimData.csv')
df2 = pd.DataFrame({'length' : df['length'].groupby(df['sim']).mean()})
df2['R'] = df['res.inflow'].groupby(df['sim']).mean()
df2['sim'] = df['sim'].groupby(df['sim']).mean()
df2['flow'] = df['flow.rate'].groupby(df['sim']).mean()
df2['tau'] = np.log10(df2['length']**p/df2['flow'])

df2['N'] = np.log10(df['total.abundance'].groupby(df['sim']).mean())
df2['Prod'] = np.log10(df['ind.production'].groupby(df['sim']).mean())
df2['S'] = df['species.richness'].groupby(df['sim']).mean()
df2['E'] = df['simpson.e'].groupby(df['sim']).mean()
df2['W'] = df['Whittakers.turnover'].groupby(df['sim']).mean()
df2['Dorm'] = df['Percent.Dormant'].groupby(df['sim']).mean()

clrs = assigncolor(df2['R'])
df2['clrs'] = clrs

fig = plt.figure()
xlab = r"$log_{10}$"+'(' + r"$\tau$" +')'

fig = figplot(df2['clrs'], df2['tau'], df2['N'], xlab, r"$N$", fig, 1)
fig = figplot(df2['clrs'], df2['tau'], df2['Prod'], xlab, 'Productivity', fig, 2)
fig = figplot(df2['clrs'], df2['tau'], df2['S'], xlab, r"$S$", fig, 4)
fig = figplot(df2['clrs'], df2['tau'], df2['E'], xlab, 'Evenness', fig, 5)
ylab = r"$log_{10}$"+'(' + r"$\beta$" +')'
fig = figplot(df2['clrs'], df2['tau'], np.log10(df2['W']), xlab, ylab, fig, 7)
fig = figplot(df2['clrs'], df2['tau'], df2['Dorm'], xlab, '%Dormant', fig, 8)

plt.subplots_adjust(wspace=0.4, hspace=0.4)
plt.savefig(mydir + 'Emergence/results/figures/Taxa_vs_Tau.png', dpi=200, bbox_inches = "tight")
