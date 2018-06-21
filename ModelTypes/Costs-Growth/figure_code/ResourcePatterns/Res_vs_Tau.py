from __future__ import division
import  matplotlib.pyplot as plt
import statsmodels.api as sm
import pandas as pd
import numpy as np
import os

p, fr, _lw, w, sz, fs = 2, 0.5, 0.5, 1, 0.1, 6

mydir = os.path.expanduser('~/GitHub/residence-time2/Emergence')
tools = os.path.expanduser(mydir + "/tools")

df = pd.read_csv(mydir + '/ModelTypes/Costs-Growth/results/simulated_data/SimData.csv')


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
    fig.add_subplot(3, 3, n)
    plt.scatter(x, y, color=clrs, s=sz, linewidths=0.0, edgecolor=None)
    lowess = sm.nonparametric.lowess(y, x, frac=fr)
    x, y = lowess[:, 0], lowess[:, 1]
    plt.plot(x, y, lw=_lw, color='k')
    plt.tick_params(axis='both', labelsize=6)
    plt.xlabel(xlab, fontsize=8)
    plt.ylabel(ylab, fontsize=8)

    if n == 2: plt.ylim(-1, 1)
    return fig


df2 = pd.DataFrame({'V' : df['V'].groupby(df['sim']).mean()})
df2['Q'] = df['Q'].groupby(df['sim']).mean()

df2['tau'] = np.log10((df2['V'])/df2['Q'])
df2['R'] = np.log10(df['total.res'].groupby(df['sim']).mean()/df2['V'])
df2['RDens'] = np.log10(df['resource.concentration'].groupby(df['sim']).mean())
df2['Rrich'] = df['resource.richness'].groupby(df['sim']).mean()

df2 = df2.replace([np.inf, -np.inf], np.nan).dropna()
clrs = assigncolor(np.log10(df2['V']))
df2['clrs'] = clrs

xlab = r"$log_{10}$"+'(' + r"$\tau$" +')'
fig = plt.figure()

ylab = r"$log_{10}$"+ '(R density)'
fig = figplot(clrs, df2['tau'], df2['RDens'], xlab, ylab, fig, 1)

ylab = r"$log_{10}$"+ '(Total R)'
fig = figplot(clrs, df2['tau'], df2['R'], xlab, ylab, fig, 2)

ylab = r"$log_{10}$"+ '(R richness)'
fig = figplot(clrs, df2['tau'], df2['Rrich'], xlab, ylab, fig, 3)

plt.subplots_adjust(wspace=0.45, hspace=0.4)
plt.savefig(mydir + '/ModelTypes/Costs-Growth/results/figures/resources_tau.png', dpi=200, bbox_inches = "tight")
plt.close()
