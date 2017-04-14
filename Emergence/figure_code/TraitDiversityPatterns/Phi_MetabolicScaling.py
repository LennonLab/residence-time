from __future__ import division
import  matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import statsmodels.formula.api as smf
from statsmodels.stats.outliers_influence import summary_table



def xfrm(X, _max): return -np.log10(_max - np.array(X))


def assigncolor(xs):
    cDict = {}
    clrs = []
    for x in xs:
        if x not in cDict:
            if x < 1: c = 'r'
            elif x < 2: c = 'OrangeRed'
            elif x < 3: c = 'Orange'
            elif x < 4: c = 'Yellow'
            elif x < 5: c = 'Lime'
            elif x < 6: c = 'Green'
            elif x < 7: c = 'Cyan'
            elif x < 8: c = 'Blue'
            else: c = 'DarkViolet'
            cDict[x] = c

        clrs.append(cDict[x])
    return clrs



def figplot(clrs, x, y, xlab, ylab, fig, n):
    '''main figure plotting function'''

    fig.add_subplot(1, 1, n)
    x = np.log10(x)
    y = np.log10(y)
    y2 = list(y)
    x2 = list(x)

    d = pd.DataFrame({'size': list(x2)})
    d['rate'] = list(y2)
    f = smf.ols('rate ~ size', d).fit()

    coef = f.params[1]
    st, data, ss2 = summary_table(f, alpha=0.05)
    fitted = data[:,2]
    mean_ci_low, mean_ci_upp = data[:,4:6].T
    ci_low, ci_upp = data[:,6:8].T
    x2, y2, fitted, ci_low, ci_upp = zip(*sorted(zip(x2, y2, fitted, ci_low, ci_upp)))

    plt.scatter(x2, y2, color = clrs, alpha= 1 , s = 50, linewidths=0.5, edgecolor='w')
    plt.fill_between(x2, ci_upp, ci_low, color='0.5', lw=0.1, alpha=0.1)
    plt.plot(x2, fitted,  color='k', ls='-', lw=2.0, alpha=0.9, label = '$z$ = '+str(round(coef, 2)))
    plt.xlabel(xlab, fontsize=18)
    plt.ylabel(ylab, fontsize=18)
    plt.tick_params(axis='both', labelsize=14)
    plt.xlim(0.9*min(x2), 1.1*max(x2))
    plt.ylim(min(ci_low), max(ci_upp))
    plt.legend(loc=2, fontsize=18, frameon=False)
    return fig



mydir = os.path.expanduser('~/GitHub/residence-time/Emergence')
tools = os.path.expanduser(mydir + "/tools")

df = pd.read_csv(mydir + '/results/simulated_data/SimData.csv')

df2 = pd.DataFrame({'length' : df['length'].groupby(df['sim']).mean()})
df2['R'] = df['res.inflow'].groupby(df['sim']).mean()
df2['flow'] = df['flow.rate'].groupby(df['sim']).mean()
df2['tau'] = np.log10(df2['length']**2/df2['flow'])

state = 'all'
df2['Biomass'] = df[state+'.biomass'].groupby(df['sim']).mean()
df2['size'] = df[state+'.size'].groupby(df['sim']).mean()
df2['G'] = df[state+'.avg.per.capita.growth'].groupby(df['sim']).mean()
df2['M'] = df[state+'.avg.per.capita.maint'].groupby(df['sim']).mean()
df2['D'] = df[state+'.avg.per.capita.active.dispersal'].groupby(df['sim']).mean()
df2['E'] = df[state+'.avg.per.capita.efficiency'].groupby(df['sim']).mean()
df2['RPF'] = df[state+'.avg.per.capita.rpf'].groupby(df['sim']).mean()
df2['MF'] = df[state+'.avg.per.capita.mf'].groupby(df['sim']).mean()

#df2['phi'] = df2['size'] * df2['G'] * df2['D'] * df2['E'] * df2['RPF'] * df2['MF'] * (1/df2['M'])
df2['phi'] = df2['size'] * df2['G'] * df2['D'] * df2['E'] * df2['RPF'] * df2['MF'] * (1/df2['M'])

df2['MSB'] = df2['phi']/df2['size']
df2['Pdens'] = (df2['Biomass'])/(df2['length']**2)

#df2 = df2.replace([np.inf, -np.inf, 0], np.nan).dropna()
clrs = assigncolor(df2['tau'])
df2['clrs'] = clrs

fig = plt.figure()

xlab = r"$log_{10}$"+'(Body size)'
ylab = r"$log_{10}$"+'('+r'$\phi$' + ')'
fig = figplot(df2['clrs'], df2['size'], df2['phi'], xlab, ylab, fig, 1)

plt.subplots_adjust(wspace=0.4, hspace=0.4)
plt.savefig(mydir + '/results/figures/Phi_MetabolicScaling.png', dpi=200, bbox_inches = "tight")
