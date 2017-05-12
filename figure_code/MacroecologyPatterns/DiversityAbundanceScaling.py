from __future__ import division
import  matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import sys
from scipy import stats
import statsmodels.formula.api as smf
from statsmodels.stats.outliers_influence import summary_table

mydir = os.path.expanduser('~/GitHub/residence-time/Emergence')
sys.path.append(mydir+'/tools')
mydir2 = os.path.expanduser("~/")


def xfrm(X, _max): return -np.log10(_max - np.array(X))

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
    y2 = list(y)
    x2 = list(x)

    d = pd.DataFrame({'x': list(x2)})
    d['y'] = list(y2)
    f = smf.ols('y ~ x', d).fit()

    m, b, r, p, std_err = stats.linregress(x2, y2)

    st, data, ss2 = summary_table(f, alpha=0.05)
    fitted = data[:,2]
    mean_ci_low, mean_ci_upp = data[:,4:6].T
    ci_low, ci_upp = data[:,6:8].T

    x2, y2, fitted, ci_low, ci_upp = zip(*sorted(zip(x2, y2, fitted, ci_low, ci_upp)))

    if n == 1: lbl = r'$rarity$'+ ' = '+str(round(10**b,1))+'*'+r'$N$'+'$^{'+str(round(m,2))+'}$'
    elif n == 2: lbl = r'$Nmax$'+ ' = '+str(round(10**b,1))+'*'+r'$N$'+'$^{'+str(round(m,2))+'}$'
    elif n == 3: lbl = r'$Ev$'+ ' = '+str(round(10**b,1))+'*'+r'$N$'+'$^{'+str(round(m,2))+'}$'
    elif n == 4: lbl = r'$S$'+ ' = '+str(round(10**b,1))+'*'+r'$N$'+'$^{'+str(round(m,2))+'}$'

    plt.scatter(x2, y2, color = clrs, alpha= 1, s = 10, linewidths=0.1, edgecolor='w')

    plt.fill_between(x2, ci_upp, ci_low, color='0.5', lw=0.1, alpha=0.1)
    plt.plot(x2, fitted,  color='k', ls='--', lw=1.0, alpha=0.9, label = lbl)
    if n == 3: plt.legend(loc=1, fontsize=8, frameon=False)
    else: plt.legend(loc=2, fontsize=8, frameon=False)

    plt.xlabel(xlab, fontsize=12)
    plt.ylabel(ylab, fontsize=12)
    plt.tick_params(axis='both', labelsize=8)
    plt.xlim(0.9*min(x2), 1.1*max(x2))
    plt.ylim(min(ci_low), max(ci_upp))
    return fig


df1 = pd.read_csv(mydir + '/results/simulated_data/Mason-SimData.csv')
df2 = pd.read_csv(mydir + '/results/simulated_data/Karst-SimData.csv')
df3 = pd.read_csv(mydir + '/results/simulated_data/BigRed2-SimData.csv')

frames = [df1, df2, df3]
df = pd.concat(frames)

df2 = pd.DataFrame({'area' : df['area'].groupby(df['sim']).mean()})
df2['flow'] = df['flow.rate'].groupby(df['sim']).mean()
df2['tau'] = np.log10(df2['area']/df2['flow'])

df2['N'] = np.log10(df['total.abundance'].groupby(df['sim']).max())
df2['D'] = np.log10(df['N.max'].groupby(df['sim']).median())
df2['S'] = np.log10(df['species.richness'].groupby(df['sim']).max())
df2['E'] = np.log10(df['simpson.e'].groupby(df['sim']).median())
df2['R'] = np.log10(df['logmod.skew'].groupby(df['sim']).max())

df2 = df2.replace([np.inf, -np.inf], np.nan).dropna()
clrs = assigncolor(df2['tau'])
df2['clrs'] = clrs

fig = plt.figure()

xlab = '$log$'+r'$_{10}$'+'($N$)'
ylab = 'Rarity, '+r'$log_{10}$'
fig = figplot(df2['clrs'], df2['N'], df2['R'], xlab, ylab, fig, 1)

xlab = '$log$'+r'$_{10}$'+'($N$)'
ylab = 'Dominance, '+r'$log_{10}$'
fig = figplot(df2['clrs'], df2['N'], df2['D'], xlab, ylab, fig, 2)

xlab = '$log$'+r'$_{10}$'+'($N$)'
ylab = 'Evenness, ' +r'$log_{10}$'
fig = figplot(df2['clrs'], df2['N'], df2['E'], xlab, ylab, fig, 3)

xlab = '$log$'+r'$_{10}$'+'($N$)'
ylab = 'Richness, ' +r'$log_{10}$'
fig = figplot(df2['clrs'], df2['N'], df2['S'], xlab, ylab, fig, 4)


#### Final Format and Save #####################################################
plt.subplots_adjust(wspace=0.4, hspace=0.4)
plt.savefig(mydir + '/results/figures/DiversityAbundanceScaling.png', dpi=200, bbox_inches = "tight")
plt.close()
