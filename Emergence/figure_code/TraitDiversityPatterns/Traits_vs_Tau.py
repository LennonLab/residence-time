from __future__ import division
import  matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import statsmodels.api as sm


def assigncolor(xs):
    cDict = {}
    clrs = []
    for x in xs:
        if x not in cDict:
            if x < 10: c = 'r'
            elif x < 20: c = 'OrangeRed'
            elif x < 30: c = 'Orange'
            elif x < 40: c = 'Yellow'
            elif x < 50: c = 'Lime'
            elif x < 60: c = 'Green'
            elif x < 70: c = 'Cyan'
            elif x < 80: c = 'Blue'
            else: c = 'DarkViolet'
            cDict[x] = c

        clrs.append(cDict[x])
    return clrs



def figplot(clrs, x, y, xlab, ylab, fig, n):
    fig.add_subplot(3, 3, n)
    plt.scatter(x, y, lw=0.5, color=clrs, s=sz, linewidths=0.1, edgecolor='w')
    lowess = sm.nonparametric.lowess(y, x, frac=fr)
    x, y = lowess[:, 0], lowess[:, 1]
    plt.plot(x, y, lw=_lw, color='k')
    plt.tick_params(axis='both', labelsize=8)
    plt.xlabel(xlab, fontsize=10)
    plt.ylabel(ylab, fontsize=8)
    return fig


p, fr, _lw, w, sz = 2, 0.2, 1.5, 1, 20
mydir = os.path.expanduser('~/GitHub/residence-time/')
df = pd.read_csv(mydir + 'Emergence/results/simulated_data/SimData.csv')
df2 = pd.DataFrame({'length' : df['length'].groupby(df['sim']).mean()})
df2['sim'] = df['sim'].groupby(df['sim']).mean()
df2['R'] = df['res.inflow'].groupby(df['sim']).mean()
df2['flow'] = df['flow.rate'].groupby(df['sim']).mean()
df2['tau'] = np.log10(df2['length']**2/df2['flow'])
df2['Dorm'] = df['Percent.Dormant'].groupby(df['sim']).mean()

state = 'all'
df2['Grow'] = np.log10(df[state+'.avg.per.capita.growth'].groupby(df['sim']).mean())
df2['Maint'] = np.log10(df[state+'.avg.per.capita.maint'].groupby(df['sim']).mean())
df2['Disp'] = np.log10(df[state+'.avg.per.capita.active.dispersal'].groupby(df['sim']).mean())
df2['Eff'] = np.log10(df[state+'.avg.per.capita.efficiency'].groupby(df['sim']).max())
df2['RPF'] = np.log10(df[state+'.avg.per.capita.rpf'].groupby(df['sim']).mean())
df2['MF'] = np.log10(df[state+'.avg.per.capita.mf'].groupby(df['sim']).max())

clrs = assigncolor(df2['R'])
df2['clrs'] = clrs

xlab = r"$log_{10}$"+'(' + r"$\tau$" +')'
fig = plt.figure()

ylab = 'Growth rate'
fig = figplot(df2['clrs'], df2['tau'], df2['Grow'], xlab, ylab, fig, 1)

ylab = 'Maintenance energy'
fig = figplot(df2['clrs'], df2['tau'], df2['Maint'], xlab, ylab, fig, 2)

ylab = 'Active disperal rate'
fig = figplot(df2['clrs'], df2['tau'], df2['Disp'], xlab, ylab, fig, 4)

ylab = 'Random resuscitation\nfrom dormancy'
fig = figplot(df2['clrs'], df2['tau'], df2['RPF'], xlab, ylab, fig, 5)

ylab = 'Resource specialization'
fig = figplot(df2['clrs'], df2['tau'], df2['Eff'], xlab, ylab, fig, 7)

ylab = 'Decrease of maintenance\nenergy when dormant'
fig = figplot(df2['clrs'], df2['tau'], df2['MF'], xlab, ylab, fig, 8)

plt.subplots_adjust(wspace=0.5, hspace=0.4)
plt.savefig(mydir + 'Emergence/results/figures/Traits_vs_Tau.png', dpi=200, bbox_inches = "tight")
