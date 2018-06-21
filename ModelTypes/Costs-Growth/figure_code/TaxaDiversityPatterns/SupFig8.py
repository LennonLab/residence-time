from __future__ import division
import  matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import statsmodels.api as sm


mydir = os.path.expanduser('~/GitHub/residence-time2/Emergence')
tools = os.path.expanduser(mydir + "/tools")

df = pd.read_csv(mydir + '/ModelTypes/Costs-Growth/results/simulated_data/SimData.csv')
#df = df[df['total.abundance'] > 1]
#df = df[df['species.richness'] > 1]


def assigncolor(xs):
    cDict = {}
    clrs = []
    for x in xs:
        if x not in cDict:
            if x <= 0: c = 'r'
            elif x <= 0.5: c = 'Orange'
            elif x <= 1: c = 'y'
            elif x <= 1.5: c = 'Lime'
            elif x <= 2: c = 'Green'
            elif x <= 2.5: c = 'DodgerBlue'
            elif x <= 3: c = 'Plum'
            clrs.append(c)

    return clrs


def figplot(clrs, x, y, xlab, ylab, fig, n):
    p, fr, _lw, w, sz, fs = 2, 0.75, 1, 1, 0.5, 6

    fig.add_subplot(1, 1, n)
    plt.scatter(x, y, lw=0.5, color=clrs, s=sz, linewidths=0.0, edgecolor=None)
    lowess = sm.nonparametric.lowess(y, x, frac=fr)
    x, y = lowess[:, 0], lowess[:, 1]
    plt.plot(x, y, lw=_lw, color='k')
    plt.tick_params(axis='both', labelsize=8)
    plt.xlabel(xlab, fontsize=14)
    plt.ylabel(ylab, fontsize=14)

    if n == 1:
        plt.plot([0, 5], [0, 5], lw=0.5, color='k')
    return fig


df2 = pd.DataFrame({'V' : df['V'].groupby(df['sim']).mean()})
df2['sim'] = df['sim'].groupby(df['sim']).mean()
df2['Q'] = df['Q'].groupby(df['sim']).mean()
df2['tau'] = np.log10(df2['V']/df2['Q'])
df2['age'] = np.log10(df['avg.age'].groupby(df['sim']).mean())
df2 = df2.replace([np.inf, -np.inf], np.nan).dropna()

clrs = assigncolor(np.log10(df2['V']))
df2['clrs'] = clrs

fig = plt.figure()
xlab = r"$log_{10}$"+'(' + r"$\tau$" +')'

fig = figplot(df2['clrs'], df2['tau'], df2['age'], xlab, r"$log_{10}$"+'(' + r"$age$" +')', fig, 1)

plt.subplots_adjust(wspace=0.5, hspace=0.45)
plt.savefig(mydir + '/ModelTypes/Costs-Growth/results/figures/Supplement/SupFig8.png', dpi=200, bbox_inches = "tight")
plt.close()
