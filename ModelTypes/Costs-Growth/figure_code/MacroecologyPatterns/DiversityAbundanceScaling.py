from __future__ import division
import  matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import sys
from scipy import stats
import statsmodels.formula.api as smf
from statsmodels.stats.outliers_influence import summary_table


p, fr, _lw, w, fs, sz = 2, 0.75, 0.5, 1, 6, 0.1
smin = False

mydir = os.path.expanduser('~/GitHub/residence-time2/Emergence')
sys.path.append(mydir+'/tools')
mydir2 = os.path.expanduser("~/")


def assigncolor(xs):
    cDict = {}
    clrs = []
    for x in xs:
        if x not in cDict:
            if x <= 10**1: c = 'r'
            elif x <= 10**2: c = 'Orange'
            elif x <= 10**3: c = 'Green'
            elif x <= 10**4: c = 'DodgerBlue'
            elif x <= 10**5: c = 'Plum'
            else: c = 'Purple'
            cDict[x] = c
        clrs.append(cDict[x])
    return clrs


def figplot(clrs, x, y, xlab, ylab, fig, n):

    fig.add_subplot(2, 2, n)
    plt.xscale('log')
    if n == 1: plt.yscale('log', subsy=[1, 2])
    plt.yscale('log')
    plt.minorticks_off()

    d = pd.DataFrame({'x': np.log10(x)})
    d['y'] = np.log10(y)
    f = smf.ols('y ~ x', d).fit()

    m, b, r, p, std_err = stats.linregress(np.log10(x), np.log10(y))
    st, data, ss2 = summary_table(f, alpha=0.05)
    fitted = data[:,2]
    mean_ci_low, mean_ci_upp = data[:,4:6].T
    ci_low, ci_upp = data[:,6:8].T

    x, y, fitted, ci_low, ci_upp, clrs = zip(*sorted(zip(x, y, fitted, ci_low, ci_upp, clrs)))

    x = np.array(x)
    y = np.array(y)
    fitted = 10**np.array(fitted)
    ci_low = 10**np.array(ci_low)
    ci_upp = 10**np.array(ci_upp)

    if n == 1: lbl = r'$rarity$'+ ' = '+str(round(10**b,1))+'*'+r'$N$'+'$^{'+str(round(m,2))+'}$'
    elif n == 2: lbl = r'$Nmax$'+ ' = '+str(round(10**b,1))+'*'+r'$N$'+'$^{'+str(round(m,2))+'}$'
    elif n == 3: lbl = r'$Ev$'+ ' = '+str(round(10**b,1))+'*'+r'$N$'+'$^{'+str(round(m,2))+'}$'
    elif n == 4: lbl = r'$S$'+ ' = '+str(round(10**b,1))+'*'+r'$N$'+'$^{'+str(round(m,2))+'}$'

    plt.scatter(x, y, s = sz, color=clrs, linewidths=0.0, edgecolor=None)
    plt.fill_between(x, ci_upp, ci_low, color='0.5', lw=0.1, alpha=0.2)
    plt.plot(x, fitted,  color='k', ls='--', lw=0.5, label = lbl)

    if n == 3: plt.legend(loc=3, fontsize=8, frameon=False)
    else: plt.legend(loc=2, fontsize=8, frameon=False)

    plt.xlabel(xlab, fontsize=10)
    plt.ylabel(ylab, fontsize=10)
    plt.tick_params(axis='both', labelsize=6)
    if n in [2, 4]: plt.ylim(min(y), max(y))
    elif n == 1: plt.ylim(min(ci_low), max(ci_upp))
    elif n == 3: plt.ylim(0.1, 1.1)

    return fig


df = pd.read_csv(mydir + '/ModelTypes/Costs-Growth/results/simulated_data/SimData.csv')
df = df[df['total.abundance'] > 0]

df2 = pd.DataFrame({'V' : df['V'].groupby(df['sim']).mean()})
df2['Q'] = df['Q'].groupby(df['sim']).mean()
df2['tau'] = df2['V']/df2['Q']

df2['N'] = df['total.abundance'].groupby(df['sim']).mean()
df2['D'] = df['N.max'].groupby(df['sim']).mean()
df2['S'] = df['species.richness'].groupby(df['sim']).mean()
df2['E'] = df['simpson.e'].groupby(df['sim']).mean()
df2['R'] = df['logmod.skew'].groupby(df['sim']).mean()
df2['R'] = df2['R'] + 0.2

if smin: df2 = df2[df2['S'] > 1]
df2 = df2.replace([np.inf, -np.inf], np.nan).dropna()

#print min(df2['R'])
#sys.exit()

clrs = assigncolor(df2['tau'])
df2['clrs'] = clrs

fig = plt.figure()

xlab = '$N$'
ylab = 'Rarity'
fig = figplot(df2['clrs'], df2['N'], df2['R'], xlab, ylab, fig, 1)

xlab = '$N$'
ylab = 'Dominance'
fig = figplot(df2['clrs'], df2['N'], df2['D'], xlab, ylab, fig, 2)

xlab = '$N$'
ylab = 'Evenness'
fig = figplot(df2['clrs'], df2['N'], df2['E'], xlab, ylab, fig, 3)

xlab = '$N$'
ylab = 'Richness'
fig = figplot(df2['clrs'], df2['N'], df2['S'], xlab, ylab, fig, 4)

plt.subplots_adjust(wspace=0.4, hspace=0.4)
plt.savefig(mydir + '/ModelTypes/Costs-Growth/results/figures/DiversityAbundanceScaling.png', dpi=400, bbox_inches = "tight")
plt.savefig(mydir + '/ModelTypes/Costs-Growth/results/figures/Supplement/SupFig3.png', dpi=400, bbox_inches = "tight")
plt.close()


#sys.exit()


df = pd.read_csv(mydir + '/ModelTypes/Costs-Growth/results/simulated_data/SimData.csv')
df = df[df['active.total.abundance'] > 0]

df2 = pd.DataFrame({'V' : df['V'].groupby(df['sim']).mean()})
df2['Q'] = df['Q'].groupby(df['sim']).mean()
df2['tau'] = df2['V']/df2['Q']

df2['N'] = df['active.total.abundance'].groupby(df['sim']).mean()
df2['D'] = df['active.N.max'].groupby(df['sim']).mean()
df2['S'] = df['active.species.richness'].groupby(df['sim']).mean()
df2['E'] = df['active.simpson.e'].groupby(df['sim']).mean()
df2['R'] = df['active.logmod.skew'].groupby(df['sim']).mean()
df2['R'] = df2['R'] + 0.5

if smin: df2 = df2[df2['S'] > 1]
df2 = df2.replace([np.inf, -np.inf], np.nan).dropna()

clrs = assigncolor(df2['tau'])
df2['clrs'] = clrs

fig = plt.figure()

xlab = '$N$'
ylab = 'Rarity'
fig = figplot(df2['clrs'], df2['N'], df2['R'], xlab, ylab, fig, 1)

xlab = '$N$'
ylab = 'Dominance'
fig = figplot(df2['clrs'], df2['N'], df2['D'], xlab, ylab, fig, 2)

xlab = '$N$'
ylab = 'Evenness'
fig = figplot(df2['clrs'], df2['N'], df2['E'], xlab, ylab, fig, 3)

xlab = '$N$'
ylab = 'Richness'
fig = figplot(df2['clrs'], df2['N'], df2['S'], xlab, ylab, fig, 4)

plt.subplots_adjust(wspace=0.4, hspace=0.4)
plt.savefig(mydir + '/ModelTypes/Costs-Growth/results/figures/DiversityAbundanceScaling-Active.png', dpi=400, bbox_inches = "tight")
plt.close()



df = pd.read_csv(mydir + '/ModelTypes/Costs-Growth/results/simulated_data/SimData.csv')
df = df[df['dormant.total.abundance'] > 0]

df2 = pd.DataFrame({'V' : df['V'].groupby(df['sim']).mean()})
df2['Q'] = df['Q'].groupby(df['sim']).mean()
df2['tau'] = df2['V']/df2['Q']

df2['N'] = df['dormant.total.abundance'].groupby(df['sim']).mean()
df2['D'] = df['dormant.N.max'].groupby(df['sim']).mean()
df2['S'] = df['dormant.species.richness'].groupby(df['sim']).mean()
df2['E'] = df['dormant.simpson.e'].groupby(df['sim']).mean()
df2['R'] = df['dormant.logmod.skew'].groupby(df['sim']).mean()
df2['R'] = df2['R'] + 0.5

if smin: df2 = df2[df2['S'] > 1]
df2 = df2.replace([np.inf, -np.inf], np.nan).dropna()

clrs = assigncolor(df2['tau'])
df2['clrs'] = clrs

fig = plt.figure()

xlab = '$N$'
ylab = 'Rarity'
fig = figplot(df2['clrs'], df2['N'], df2['R'], xlab, ylab, fig, 1)

xlab = '$N$'
ylab = 'Dominance'
fig = figplot(df2['clrs'], df2['N'], df2['D'], xlab, ylab, fig, 2)

xlab = '$N$'
ylab = 'Evenness'
fig = figplot(df2['clrs'], df2['N'], df2['E'], xlab, ylab, fig, 3)

xlab = '$N$'
ylab = 'Richness'
fig = figplot(df2['clrs'], df2['N'], df2['S'], xlab, ylab, fig, 4)

plt.subplots_adjust(wspace=0.4, hspace=0.4)
plt.savefig(mydir + '/ModelTypes/Costs-Growth/results/figures/DiversityAbundanceScaling-Dormant.png', dpi=400, bbox_inches = "tight")
plt.close()
