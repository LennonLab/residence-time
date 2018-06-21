from __future__ import division
import  matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import statsmodels.api as sm


p, fr, _lw, w, fs, sz = 2, 0.5, 0.5, 1, 6, 0.5

mydir = os.path.expanduser('~/GitHub/residence-time2/Emergence')
tools = os.path.expanduser(mydir + "/tools")


def xfrm(X, _max): return _max-np.array(X)

def assigncolor(xs, kind):
    clrs = []

    if kind == 'Q':
        for x in xs:
            if x >= 10**-0.5: c = 'r'
            elif x >= 10**-1: c = 'Orange'
            elif x >= 10**-1.5: c = 'Lime'
            elif x >= 10**-2: c = 'Green'
            elif x >= 10**-2.5: c = 'DodgerBlue'
            else: c = 'Plum'
            clrs.append(c)

    elif kind == 'V':
        for x in xs:
            if x <= 0.5: c = 'r'
            elif x <= 1: c = 'Orange'
            elif x <= 1.5: c = 'gold'
            elif x <= 2: c = 'Green'
            elif x <= 2.5: c = 'DodgerBlue'
            elif x <= 3: c = 'Plum'
            clrs.append(c)
    return clrs



def figplot(clrs, x, y, xlab, ylab, fig, n, norm='y'):
    fig.add_subplot(3, 3, n)
    plt.yscale('log')
    plt.xscale('log')
    plt.scatter(x, y, color=clrs, s=sz, linewidths=0.0, edgecolor=None)

    b = 100
    ci = 99

    x, y = (np.array(t) for t in zip(*sorted(zip(np.log10(x), np.log10(y)))))

    Xi = xfrm(x, max(x)*1.05)
    bins = np.linspace(np.min(Xi), np.max(Xi)+1, b)
    ii = np.digitize(Xi, bins)

    pcts = np.array([np.percentile(y[ii==i], ci) for i in range(1, len(bins)) if len(y[ii==i]) > 0])
    xran = np.array([np.mean(x[ii==i]) for i in range(1, len(bins)) if len(y[ii==i]) > 0])

    lowess = sm.nonparametric.lowess(pcts, xran, frac=fr)
    x, y = lowess[:, 0], lowess[:, 1]
    plt.plot(10**x, 10**y, lw=0.5, color='k')
    plt.tick_params(axis='both', labelsize=6)

    if n == 7: plt.ylim(0, 2)
    plt.xlabel(xlab, fontsize=8)
    plt.ylabel(ylab, fontsize=8)

    return fig



df = pd.read_csv(mydir + '/ModelTypes/Costs-Growth/results/simulated_data/SimData.csv')

df3 = pd.DataFrame({'V' : df['V'].groupby(df['sim']).mean()})
df3['Q'] = df['Q'].groupby(df['sim']).mean()
df3['tau'] = df3['V']/df3['Q']
df3['Prod'] = df['ind.production'].groupby(df['sim']).mean()
df3['clrs_V'] = assigncolor(np.log10(df3['V']), 'V')
df3['clrs_Q'] = assigncolor(df3['Q'], 'Q')
df3 = df3.replace([np.inf, -np.inf, 0], np.nan).dropna()


df = df[df['total.abundance'] > 0]
df2 = pd.DataFrame({'V' : df['V'].groupby(df['sim']).mean()})
df2['Q'] = df['Q'].groupby(df['sim']).mean()
df2['tau'] = df2['V']/df2['Q']
df2['N'] = df['total.abundance'].groupby(df['sim']).mean()
df2['S'] = df['species.richness'].groupby(df['sim']).mean()
df2['NS'] = df2['N']/df2['S']
df2['clrs_V'] = assigncolor(np.log10(df2['V']), 'V')
df2['clrs_Q'] = assigncolor(df2['Q'], 'Q')

df2 = df2.replace([np.inf, -np.inf, 0], np.nan).dropna()

fig = plt.figure()
xlab = r"$\tau$"
fig = figplot(df2['clrs_V'], df2['tau'], df2['N'], xlab, r"$N$", fig, 1)
fig = figplot(df3['clrs_V'], df3['tau'], df3['Prod'], xlab, r"$P$", fig, 2)
fig = figplot(df2['clrs_V'], df2['tau'], df2['NS'], xlab, r"$N/S$", fig, 3)
fig = figplot(df2['clrs_Q'], df2['tau'], df2['N'], xlab, r"$N$", fig, 4)
fig = figplot(df3['clrs_Q'], df3['tau'], df3['Prod'], xlab, r"$P$", fig, 5)
fig = figplot(df2['clrs_Q'], df2['tau'], df2['NS'], xlab, r"$N/S$", fig, 6)

plt.subplots_adjust(wspace=0.5, hspace=0.5)
plt.savefig(mydir + '/ModelTypes/Costs-Growth/results/figures/Supplement/SupFig9.png', dpi=400, bbox_inches = "tight")
plt.close()
