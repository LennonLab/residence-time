from __future__ import division
import  matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import statsmodels.api as sm


mydir = os.path.expanduser('~/GitHub/residence-time')
tools = os.path.expanduser(mydir + "/tools")

df = pd.read_csv(mydir + '/Model/results/data/SimData.csv')


def figplot(x, y, xlab, ylab, fig, n):
    p, fr, _lw, w, sz, fs = 2, 0.75, 1, 1, 6, 6

    fig.add_subplot(1, 1, n)
    plt.scatter(x, y, lw=0.5, color='0.3', s=sz, linewidths=0.5, edgecolor='w')
    lowess = sm.nonparametric.lowess(y, x, frac=fr)
    x, y = lowess[:, 0], lowess[:, 1]

    plt.plot(x, y, lw=_lw, color='k')
    plt.tick_params(axis='both', labelsize=12)
    plt.xlabel(xlab, fontsize=16)
    plt.ylabel(ylab, fontsize=16)

    if n == 1:
        plt.plot([0, 5], [0, 5], lw=0.5, color='k')
    return fig


df2 = pd.DataFrame({'V' : df['V'].groupby(df['sim']).mean()})
df2['sim'] = df['sim'].groupby(df['sim']).mean()
df2['Q'] = df['Q'].groupby(df['sim']).mean()
df2['tau'] = np.log10(df2['V']/df2['Q'])
df2['age'] = np.log10(df['avg.age'].groupby(df['sim']).mean())
df2 = df2.replace([np.inf, -np.inf], np.nan).dropna()

fig = plt.figure()
xlab = r"$log_{10}$"+'(' + r"$\tau$" +')'

fig = figplot(df2['tau'], df2['age'], xlab, r"$log_{10}$"+'(' + r"$age$" +')', fig, 1)

plt.subplots_adjust(wspace=0.5, hspace=0.45)
plt.savefig(mydir + '/Model/results/figures/Supplement/SupFig8.png', dpi=200, bbox_inches = "tight")
plt.close()
