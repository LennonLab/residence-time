from __future__ import division
import  matplotlib.pyplot as plt
import statsmodels.api as sm
import pandas as pd
import numpy as np
import os
#import sys

mydir = os.path.expanduser('~/GitHub/residence-time/Emergence')
tools = os.path.expanduser(mydir + "/tools")

df1 = pd.read_csv(mydir + '/results/simulated_data/Mason-SimData.csv')
df2 = pd.read_csv(mydir + '/results/simulated_data/Karst-SimData.csv')
df3 = pd.read_csv(mydir + '/results/simulated_data/BigRed2-SimData.csv')

frames = [df1, df2, df3]
df = pd.concat(frames)

def assigncolor(xs):
    cDict = {}
    clrs = []
    for x in xs:
        if x not in cDict:
            if x < 10: c = 'r'
            elif x < 20: c = 'OrangeRed'
            elif x < 30: c = 'Orange'
            elif x < 40: c = 'Gold'
            elif x < 50: c = 'Lime'
            elif x < 60: c = 'Green'
            elif x < 70: c = 'Cyan'
            elif x < 80: c = 'Blue'
            elif x < 90: c = 'Plum'
            else: c = 'Darkviolet'
            cDict[x] = c

        clrs.append(cDict[x])
    return clrs


def figplot(clrs, x, y, xlab, ylab, fig, n):
    fig.add_subplot(1, 1, n)
    plt.scatter(x, y, lw=0.5, color=clrs, s=sz, linewidths=0.1, edgecolor='w')
    lowess = sm.nonparametric.lowess(y, x, frac=fr)
    x, y = lowess[:, 0], lowess[:, 1]
    plt.plot(x, y, lw=_lw, color='k')
    plt.tick_params(axis='both', labelsize=12)
    plt.xlabel(xlab, fontsize=16)
    plt.ylabel(ylab, fontsize=16)
    return fig

p, fr, _lw, w, sz, fs = 2, 0.2, 2, 1, 20, 6

df2 = pd.DataFrame({'area' : df['area'].groupby(df['sim']).mean()})
df2['flow'] = df['flow.rate'].groupby(df['sim']).mean()
df2['tau'] = np.log10((df2['area'])/df2['flow'])
df2['R'] = np.log10(df['total.res'].groupby(df['sim']).mean()/df2['area'])
df2['RDens'] = np.log10(df['resource.concentration'].groupby(df['sim']).mean())

clrs = assigncolor(df2['area'])
df2['clrs'] = clrs

#### plot figure ###############################################################
xlab = r"$log_{10}$"+'(' + r"$\tau$" +')'
fig = plt.figure()

#ylab = r"$log_{10}$"+ '(' + r"$Total resource$" + ')'
#fig = figplot(clrs, df2['tau'], df2['R'], xlab, ylab, fig, 1)

ylab = r"$log_{10}$"+ '(' + r"$Resource density$" + ')'
fig = figplot(clrs, df2['tau'], df2['RDens'], xlab, ylab, fig, 1)

#### Final Format and Save #####################################################
plt.subplots_adjust(wspace=0.4, hspace=0.4)
plt.savefig(mydir + '/results/figures/resources_tau.png', dpi=600, bbox_inches = "tight")
plt.close()
