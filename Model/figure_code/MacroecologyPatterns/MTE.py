from __future__ import division
import  matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
from scipy import stats

mydir = os.path.expanduser('~/GitHub/residence-time')
tools = os.path.expanduser(mydir + "/tools")


def assigncolor(xs):
    cDict = {}
    clrs = []
    for x in xs:
        if x not in cDict:
            if x < 1: c = 'r'
            elif x < 2: c = 'Orange'
            elif x < 3: c = 'Gold'
            elif x < 4: c = 'Green'
            elif x < 5: c = 'Blue'
            else: c = 'DarkViolet'
            cDict[x] = c

        clrs.append(cDict[x])
    return clrs


def figplot(clrs, x, y, xlab, ylab, fig, n):
    fig.add_subplot(2, 2, n)
    plt.yscale('log')
    plt.xscale('log')

    plt.scatter(x, y, color=clrs, s = sz, linewidths=0.25, edgecolor='w')
    m, b, r, p, std_err = stats.linregress(np.log10(x), np.log10(y))
    plt.plot(np.arange(min(x), max(x), 0.1), 10**b * np.arange(min(x), max(x), 0.1)**m,
        ls='-', color='k', lw=0.5, label = 'slope = '+str(round(m,2)))

    plt.xlabel(xlab, fontsize=fs)
    plt.ylabel(ylab, fontsize=fs)
    plt.tick_params(axis='both', labelsize=fs-2)
    plt.legend(loc=4, fontsize=fs-2, frameon=False)
    return fig


minS = 1.5
fs = 10
p = 1
_lw = 0.5
w = 1
sz = 3
a = 0.8

df = pd.read_csv(mydir + '/Model/results/data/SimData.csv')
df = df[df['total.abundance'] > 0]

df['aM1'] = df['active.avg.per.capita.maint'] * (1-df['percent.dormant'])
df['dM1'] = df['dormant.avg.per.capita.maint'] * df['percent.dormant']
df['a_size1'] = df['active.avg.per.capita.size'] * (1-df['percent.dormant'])
df['d_size'] = df['dormant.avg.per.capita.size'] * df['percent.dormant']
df['M'] = df['avg.per.capita.maint']
df['aM'] = df['active.avg.per.capita.maint']
df['dM'] = df['dormant.avg.per.capita.maint']
df['tau'] = np.log10(df['V']/df['Q'])


df2 = pd.DataFrame({'tau' : df['tau'].groupby(df['sim']).mean()})
df2['S'] = df['species.richness'].groupby(df['sim']).mean()
df2['aM'] = df['aM'].groupby(df['sim']).mean()
df2['aM1'] = df['aM1'].groupby(df['sim']).mean()
df2['dM'] = df['dM'].groupby(df['sim']).mean()
df2['dM1'] = df['dM1'].groupby(df['sim']).mean()
df2['M'] = df['M'].groupby(df['sim']).mean()

df2['a_size'] = df['active.avg.per.capita.size'].groupby(df['sim']).mean()
df2['a_size1'] = df['a_size1'].groupby(df['sim']).mean()
df2['d_size'] = df['d_size'].groupby(df['sim']).mean()
df2['size'] = df['avg.per.capita.size'].groupby(df['sim']).mean()
df2['clrs'] = assigncolor(df2['tau'])
df2 = df2[df2['S'] > minS]



fig = plt.figure()


fig.add_subplot(2, 2, 1)
x = df2['a_size']
y = df2['aM']
df3 = pd.DataFrame({'x' : x})
df3['y'] = y
df3['clrs'] = assigncolor(df2['tau'])
df3 = df3.replace([np.inf, -np.inf], np.nan).dropna()
x = df3['x'].tolist()
y = df3['y'].tolist()
clrs = df3['clrs'].tolist()
xlab = 'Active body size'
ylab = 'Active ' + r'$BMR$'
fig = figplot(clrs, x, y, xlab, ylab, fig, 1)


fig.add_subplot(2, 2, 2)
x = df2['d_size']
y = df2['dM']
df3 = pd.DataFrame({'x' : x})
df3['y'] = y
df3['clrs'] = assigncolor(df2['tau'])
df3 = df3.replace([np.inf, -np.inf], np.nan).dropna()
x = df3['x'].tolist()
y = df3['y'].tolist()
clrs = df3['clrs'].tolist()
xlab = 'Dormant body size'
ylab = 'Dormant ' + r'$BMR$'
fig = figplot(clrs, x, y, xlab, ylab, fig, 2)



fig.add_subplot(2, 2, 3)
size = df2['a_size1'] + df2['d_size']
M = df2['aM1'] + df2['dM']
x = list(size)
y = list(M)
df3 = pd.DataFrame({'x' : x})
df3['y'] = y
df3['clrs'] = assigncolor(df2['tau'])
df3 = df3.replace([np.inf, -np.inf], np.nan).dropna()
x = df3['x'].tolist()
y = df3['y'].tolist()
clrs = df3['clrs'].tolist()
xlab = 'Body size, weighted'
ylab = r'$BMR$'+', weighted'
fig = figplot(clrs, x, y, xlab, ylab, fig, 3)


fig.add_subplot(2, 2, 4)
x = df2['size']
y = df2['M']
df3 = pd.DataFrame({'x' : x})
df3['y'] = y
df3['clrs'] = assigncolor(df2['tau'])
df3 = df3.replace([np.inf, -np.inf], np.nan).dropna()
x = df3['x'].tolist()
y = df3['y'].tolist()
clrs = df3['clrs'].tolist()
xlab = 'Body size, unweighted'
ylab = r'$BMR$'+', unweighted'
fig = figplot(clrs, x, y, xlab, ylab, fig, 4)


#### Final Format and Save #####################################################
plt.subplots_adjust(wspace=0.5, hspace=0.5)
plt.savefig(mydir + '/Model/results/figures/Supplement/SupFig4.png', dpi=400, bbox_inches = "tight")
plt.close()
