from __future__ import division
import  matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import statsmodels.api as sm

p, fr, _lw, w, fs, sz = 2, 0.75, 0.5, 1, 4, 0.1

mydir = os.path.expanduser('~/GitHub/residence-time2/Emergence')
tools = os.path.expanduser(mydir + "/tools")

def assigncolor(xs, kind):
    cDict = {}
    clrs = []
    if kind == 'Q':
        for x in xs:
            if x not in cDict:
                if x >= -0.5: c = 'r'
                elif x >= -1: c = 'Orange'
                elif x >= -1.5: c = 'Lime'
                elif x >= -2: c = 'Green'
                elif x >= -2.5: c = 'DodgerBlue'
                else: c = 'Plum'
                cDict[x] = c
            clrs.append(cDict[x])

    elif kind == 'V':
        for x in xs:
            if x not in cDict:

                if x <= 0.5: c = 'r'
                elif x <= 1: c = 'Orange'
                elif x <= 1.5: c = 'Lime'
                elif x <= 2: c = 'Green'
                elif x <= 2.5: c = 'DodgerBlue'
                else: c = 'Plum'
                cDict[x] = c
            clrs.append(cDict[x])
    return clrs



def figplot(clrs, x, y, xlab, ylab, fig, n):
    fig.add_subplot(4, 4, n)
    if n == 1: plt.text(.75, 5, 'Colored by ' + r'$V$', fontsize=9)
    elif n == 2: plt.text(.75, 5, 'Colored by ' + r'$Q$', fontsize=9)

    plt.scatter(x, y, lw=0.5, color=clrs, s=sz, linewidths=0.0, edgecolor=None)
    plt.tick_params(axis='both', labelsize=4)
    plt.xlabel(xlab, fontsize=7)

    if n in [1,5,9,13]: plt.ylabel(ylab, fontsize=7)
    else: plt.yticks([])

    return fig


df = pd.read_csv(mydir + '/ModelTypes/Costs-Growth/results/simulated_data/SimData.csv')

df3 = pd.DataFrame({'V' : df['V'].groupby(df['sim']).mean()})
df3['Q'] = df['Q'].groupby(df['sim']).mean()
df3['tau'] = np.log10(df3['V']/df3['Q'])
df3['Prod'] = df['ind.production'].groupby(df['sim']).mean()
df3['clrs_V'] = assigncolor(np.log10(df3['V']), 'V')
df3['clrs_Q'] = assigncolor(np.log10(df3['Q']), 'Q')
df3 = df3.replace([np.inf, -np.inf, 0], np.nan).dropna()

df = df[df['total.abundance'] > 0]
df2 = pd.DataFrame({'V' : df['V'].groupby(df['sim']).mean()})
df2['Q'] = df['Q'].groupby(df['sim']).mean()
df2['tau'] = np.log10(df2['V']/df2['Q'])
df2['N'] = np.log10(df['total.abundance'].groupby(df['sim']).mean())
df2['S'] = np.log10(df['species.richness'].groupby(df['sim']).mean())
df2['E'] = df['simpson.e'].groupby(df['sim']).mean()
df2['Dorm'] = np.log10(100*df['percent.dormant'].groupby(df['sim']).mean())
df2['W'] = np.log10(df['whittakers.turnover'].groupby(df['sim']).mean())
df2['clrs_V'] = assigncolor(np.log10(df2['V']), 'V')
df2['clrs_Q'] = assigncolor(np.log10(df2['Q']), 'Q')
df2 = df2.replace([np.inf, -np.inf, 0], np.nan).dropna()

fig = plt.figure()
xlab = r"$log_{10}$"+'(' + r"$\tau$" +')'

fig = figplot(df2['clrs_V'], df2['tau'], df2['N'], xlab, r"$log_{10}$"+'(' + r"$N$" +')', fig, 1)
fig = figplot(df2['clrs_Q'], df2['tau'], df2['N'], xlab, r"$log_{10}$"+'(' + r"$N$" +')', fig, 2)

fig = figplot(df3['clrs_V'], df3['tau'], df3['Prod'], xlab, r"$P$", fig, 5)
fig = figplot(df3['clrs_Q'], df3['tau'], df3['Prod'], xlab, r"$P$", fig, 6)

fig = figplot(df2['clrs_V'], df2['tau'], df2['S'], xlab, r"$log_{10}$"+'(' + r"$S$" +')', fig, 9)
fig = figplot(df2['clrs_Q'], df2['tau'], df2['S'], xlab, r"$log_{10}$"+'(' + r"$S$" +')', fig, 10)

fig = figplot(df2['clrs_V'], df2['tau'], df2['E'], xlab, 'Evenness', fig, 13)
fig = figplot(df2['clrs_Q'], df2['tau'], df2['E'], xlab, 'Evenness', fig, 14)

plt.subplots_adjust(wspace=0.1, hspace=0.5)
plt.savefig(mydir + '/ModelTypes/Costs-Growth/results/figures/Supplement/SupFig5.png', dpi=400, bbox_inches = "tight")
plt.close()
