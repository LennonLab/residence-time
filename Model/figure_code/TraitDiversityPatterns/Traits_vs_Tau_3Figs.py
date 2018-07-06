from __future__ import division
import  matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import statsmodels.api as sm
import sys


p, fr, _lw, w, fs, sz = 2, 0.5, 0.5, 1, 6, 0.5
minS = 1.75

mydir = os.path.expanduser('~/GitHub/residence-time2/Emergence')
tools = os.path.expanduser(mydir + "/tools")

d = pd.read_csv(mydir + '/ModelTypes/Costs-Growth/results/simulated_data/SimData.csv')
df = d[d['total.abundance'] > 0]

def assigncolor(xs, kind):
    cDict = {}
    clrs = []

    for x in xs:
        if x not in cDict:
            if x <= 0.5: c = 'r'
            elif x <= 1: c = 'Orange'
            elif x <= 1.5: c = 'gold'
            elif x <= 2: c = 'Green'
            elif x <= 2.5: c = 'DodgerBlue'
            elif x <= 3: c = 'Plum'
            clrs.append(c)
    return clrs



def figplot(clrs, x, y, xlab, ylab, fig, n, tp):
    fig.add_subplot(3, 3, n)

    plt.xscale('log')
    plt.yscale('log')
    plt.scatter(x, y, color=clrs, s=sz, linewidths=0.0, edgecolor=None)
    lowess = sm.nonparametric.lowess(np.log10(y), np.log10(x), frac=fr)
    x, y = lowess[:, 0], lowess[:, 1]
    plt.plot(10**x, 10**y, lw=_lw, color='k')


    plt.tick_params(axis='both', labelsize=5)
    plt.xlabel(xlab, fontsize=7)
    plt.ylabel(ylab, fontsize=7)

    plt.xlim(1, 10**5)
    if n == 2: plt.ylim(0.0003, 0.2)
    if n == 8: plt.ylim(4, 300)
    if n == 5: plt.ylim(0.002, 0.3)

    if tp == 'main':
        if n == 1: plt.text(2, 0.004, 'A', fontsize=7)
        if n == 2: plt.text(2, 0.0006, 'B', fontsize=7)
        if n == 4: plt.text(2, 0.005, 'C', fontsize=7)
        if n == 5: plt.text(2, 0.003, 'D', fontsize=7)
        if n == 7: plt.text(2, 8, 'E', fontsize=7)
        if n == 8: plt.text(2, 150, 'F', fontsize=7)

    return fig



dft = pd.DataFrame({'V' : df['V'].groupby(df['sim']).mean()})
dft['Q'] = df['Q'].groupby(df['sim']).mean()
dft['tau'] = dft['V']/dft['Q']

dft['S'] = df['species.richness'].groupby(df['sim']).mean()
dft['Rrich'] = df['resource.richness'].groupby(df['sim']).mean()
dft['G'] = df['growth'].groupby(df['sim']).mean()
dft['M'] = df['maint'].groupby(df['sim']).mean()
dft['D'] = df['dispersal'].groupby(df['sim']).mean()
dft['RF'] = df['rpf'].groupby(df['sim']).mean()
dft['MF'] = 1/df['mf'].groupby(df['sim']).mean()
dft['E'] = df['avg.per.capita.efficiency1e'].groupby(df['sim']).mean()

dft = dft[dft['S'] >= minS]
dft['clrs'] = assigncolor(np.log10(dft['V']), 'V')
#dft['clrs'] = assigncolor(dft['Q'], 'Q')
#dft = dft[dft['E'] > 0.001]

dft = dft.replace([np.inf, -np.inf, 0], np.nan).dropna()


xlab = r"$\tau$"
fig = plt.figure()
ylab = 'Growth rate'
fig = figplot(dft['clrs'], dft['tau'], dft['G'], xlab, ylab, fig, 1, tp='all')

ylab = r'$B$'
fig = figplot(dft['clrs'], dft['tau'], dft['M'], xlab, ylab, fig, 2, tp='all')

ylab = 'Disperal rate'
fig = figplot(dft['clrs'], dft['tau'], dft['D'], xlab, ylab, fig, 4, tp='all')

ylab = 'Resuscitation rate'
fig = figplot(dft['clrs'], dft['tau'], dft['RF'], xlab, ylab, fig, 5, tp='all')

ylab = 'Specialization'
fig = figplot(dft['clrs'], dft['tau'], dft['E'], xlab, ylab, fig, 7, tp='all')

ylab = '% Decrease of BMR\nin dormancy'
fig = figplot(dft['clrs'], dft['tau'], dft['MF'], xlab, ylab, fig, 8, tp='all')

plt.subplots_adjust(wspace=0.6, hspace=0.5)
plt.savefig(mydir + '/ModelTypes/Costs-Growth/results/figures/Traits_vs_Tau-All.png', dpi=200, bbox_inches = "tight")
plt.close()


#sys.exit()


df = d[d['active.total.abundance'] > 0]
dfa = pd.DataFrame({'V' : df['V'].groupby(df['sim']).mean()})
dfa['Q'] = df['Q'].groupby(df['sim']).mean()
dfa['tau'] = dfa['V']/dfa['Q']

dfa['S'] = df['active.species.richness'].groupby(df['sim']).mean()
dfa['Rrich'] = df['resource.richness'].groupby(df['sim']).mean()
dfa['G'] = df['active.growth'].groupby(df['sim']).mean()
dfa['M'] = df['active.maint'].groupby(df['sim']).mean()
dfa['D'] = df['active.dispersal'].groupby(df['sim']).mean()
dfa['RF'] = df['active.rpf'].groupby(df['sim']).mean()
dfa['MF'] = 1/df['active.mf'].groupby(df['sim']).mean()
dfa['E'] = df['active.avg.per.capita.efficiency1e'].groupby(df['sim']).mean()

dfa = dfa[dfa['S'] >= minS]
dfa['clrs'] = assigncolor(np.log10(dfa['V']), 'V')
#dfa['clrs'] = assigncolor(dfa['Q'], 'Q')
dfa = dfa[dfa['E'] > 0.001]
dfa = dfa.replace([np.inf, -np.inf, 0], np.nan).dropna()

xlab = r"$\tau$"
fig = plt.figure()
ylab = 'Growth rate'
fig = figplot(dfa['clrs'], dfa['tau'], dfa['G'], xlab, ylab, fig, 1, tp='act')

ylab = r'$B$'
fig = figplot(dfa['clrs'], dfa['tau'], dfa['M'], xlab, ylab, fig, 2, tp='act')

ylab = 'Disperal rate'
fig = figplot(dfa['clrs'], dfa['tau'], dfa['D'], xlab, ylab, fig, 4, tp='act')

ylab = 'Resuscitation rate'
fig = figplot(dfa['clrs'], dfa['tau'], dfa['RF'], xlab, ylab, fig, 5, tp='act')

ylab = 'Specialization'
fig = figplot(dfa['clrs'], dfa['tau'], dfa['E'], xlab, ylab, fig, 7, tp='act')

ylab = '% Decrease of BMR\nin dormancy'
fig = figplot(dfa['clrs'], dfa['tau'], dfa['MF'], xlab, ylab, fig, 8, tp='act')

plt.subplots_adjust(wspace=0.6, hspace=0.5)
plt.savefig(mydir + '/ModelTypes/Costs-Growth/results/figures/Traits_vs_Tau-Active.png', dpi=200, bbox_inches = "tight")
plt.close()




df = d[d['dormant.total.abundance'] > 0]
dfd = pd.DataFrame({'V' : df['V'].groupby(df['sim']).mean()})
dfd['Q'] = df['Q'].groupby(df['sim']).mean()
dfd['tau'] = dfd['V']/dfd['Q']

dfd['S'] = df['dormant.species.richness'].groupby(df['sim']).mean()
dfd['Rrich'] = df['resource.richness'].groupby(df['sim']).mean()
dfd['G'] = df['dormant.growth'].groupby(df['sim']).mean()
dfd['M'] = df['dormant.maint'].groupby(df['sim']).mean()
dfd['D'] = df['dormant.dispersal'].groupby(df['sim']).mean()
dfd['RF'] = df['dormant.rpf'].groupby(df['sim']).mean()
dfd['MF'] = 1/df['dormant.mf'].groupby(df['sim']).mean()
dfd['E'] = df['dormant.avg.per.capita.efficiency1e'].groupby(df['sim']).mean()

dfd = dfd[dfd['S'] >= minS]
dfd['clrs'] = assigncolor(np.log10(dfd['V']), 'V')
#dfd['clrs'] = assigncolor(dfd['Q'], 'Q')
dfd = dfd[dfd['E'] > 0.001]
dfd = dfd.replace([np.inf, -np.inf, 0], np.nan).dropna()

xlab = r"$\tau$"
fig = plt.figure()
ylab = 'Growth rate'
fig = figplot(dfd['clrs'], dfd['tau'], dfd['G'], xlab, ylab, fig, 1, tp='dor')

ylab = r'$B$'
fig = figplot(dfd['clrs'], dfd['tau'], dfd['M'], xlab, ylab, fig, 2, tp='dor')

ylab = 'Disperal rate'
fig = figplot(dfd['clrs'], dfd['tau'], dfd['D'], xlab, ylab, fig, 4, tp='dor')

ylab = 'Resuscitation rate'
fig = figplot(dfd['clrs'], dfd['tau'], dfd['RF'], xlab, ylab, fig, 5, tp='dor')

ylab = 'Specialization'
fig = figplot(dfd['clrs'], dfd['tau'], dfd['E'], xlab, ylab, fig, 7, tp='dor')

ylab = '% Decrease of BMR\nin dormancy'
fig = figplot(dfd['clrs'], dfd['tau'], dfd['MF'], xlab, ylab, fig, 8, tp='dor')

plt.subplots_adjust(wspace=0.6, hspace=0.5)
plt.savefig(mydir + '/ModelTypes/Costs-Growth/results/figures/Traits_vs_Tau-Dormant.png', dpi=200, bbox_inches = "tight")
plt.close()





xlab = r"$\tau$"
fig = plt.figure()
ylab = 'Growth rate'
fig = figplot(dfa['clrs'], dfa['tau'], dfa['G'], xlab, ylab, fig, 1, tp='main')

ylab = r'$B$'
fig = figplot(dft['clrs'], dft['tau'], dft['M'], xlab, ylab, fig, 2, tp='main')

ylab = 'Disperal rate'
fig = figplot(dfa['clrs'], dfa['tau'], dfa['D'], xlab, ylab, fig, 4, tp='main')

ylab = 'Resuscitation rate'
fig = figplot(dft['clrs'], dft['tau'], dft['RF'], xlab, ylab, fig, 5, tp='main')

ylab = 'Specialization'
fig = figplot(dft['clrs'], dft['tau'], dft['E'], xlab, ylab, fig, 7, tp='main')

ylab = '% Decrease of BMR\nin dormancy'
fig = figplot(dft['clrs'], dft['tau'], dft['MF'], xlab, ylab, fig, 8, tp='main')

plt.subplots_adjust(wspace=0.6, hspace=0.5)
plt.savefig(mydir + '/ModelTypes/Costs-Growth/results/figures/Traits_vs_Tau-Main.png', dpi=200, bbox_inches = "tight")
plt.close()





'''

df = d[d['total.abundance'] > 0]
dft = pd.DataFrame({'V' : df['V'].groupby(df['sim']).mean()})
dft['Q'] = df['Q'].groupby(df['sim']).mean()
dft['tau'] = dft['V']/dft['Q']

dft['S'] = df['species.richness'].groupby(df['sim']).mean()
dft['Rrich'] = df['resource.richness'].groupby(df['sim']).mean()
dft['G'] = df['avg.per.capita.growth'].groupby(df['sim']).mean()
dft['M'] = df['avg.per.capita.maint'].groupby(df['sim']).mean()
dft['D'] = df['avg.per.capita.dispersal'].groupby(df['sim']).mean()
dft['RF'] = df['avg.per.capita.rpf'].groupby(df['sim']).mean()
dft['MF'] = 1/df['avg.per.capita.mf'].groupby(df['sim']).mean()
dft['E'] = df['avg.per.capita.efficiency1e'].groupby(df['sim']).mean()

dft = dft[dft['S'] > minS]
dft['clrs'] = assigncolor(np.log10(dft['V']), 'V')
#dft['clrs'] = assigncolor(dft['Q'], 'Q')
dft = dft[dft['E'] > 0.001]
dft = dft.replace([np.inf, -np.inf], np.nan).dropna()


xlab = r"$\tau$"
fig = plt.figure()
ylab = 'Growth rate'
fig = figplot(dft['clrs'], dft['tau'], dft['G'], xlab, ylab, fig, 1)

ylab = 'BMR'
fig = figplot(dft['clrs'], dft['tau'], dft['M'], xlab, ylab, fig, 2)

ylab = 'Disperal rate'
fig = figplot(dft['clrs'], dft['tau'], dft['D'], xlab, ylab, fig, 4)

ylab = 'Resuscitation rate'
fig = figplot(dft['clrs'], dft['tau'], dft['RF'], xlab, ylab, fig, 5)

ylab = 'Specialization'
fig = figplot(dft['clrs'], dft['tau'], dft['E'], xlab, ylab, fig, 7)

ylab = '% Decrease of BMR\nin dormancy'
fig = figplot(dft['clrs'], dft['tau'], dft['MF'], xlab, ylab, fig, 8)

plt.subplots_adjust(wspace=0.6, hspace=0.5)
plt.savefig(mydir + '/ModelTypes/Costs-Growth/results/figures/Traits_vs_Tau-All_Weighted.png', dpi=200, bbox_inches = "tight")
plt.close()



df = d[d['active.total.abundance'] > 0]
dfa = pd.DataFrame({'V' : df['V'].groupby(df['sim']).mean()})
dfa['Q'] = df['Q'].groupby(df['sim']).mean()
dfa['tau'] = dfa['V']/dfa['Q']

dfa['S'] = df['active.species.richness'].groupby(df['sim']).mean()
dfa['Rrich'] = df['resource.richness'].groupby(df['sim']).mean()
dfa['G'] = df['active.avg.per.capita.growth'].groupby(df['sim']).mean()
dfa['M'] = df['active.avg.per.capita.maint'].groupby(df['sim']).mean()
dfa['D'] = df['active.avg.per.capita.dispersal'].groupby(df['sim']).mean()
dfa['RF'] = df['active.avg.per.capita.rpf'].groupby(df['sim']).mean()
dfa['MF'] = 1/df['active.avg.per.capita.mf'].groupby(df['sim']).mean()
dfa['E'] = df['active.avg.per.capita.efficiency1e'].groupby(df['sim']).mean()

dfa = dfa[dfa['S'] > minS]
dfa['clrs'] = assigncolor(np.log10(dfa['V']), 'V')
#dfa['clrs'] = assigncolor(dfa['Q'], 'Q')
dft = dft[dft['E'] > 0.001]
dfa = dfa.replace([np.inf, -np.inf], np.nan).dropna()

xlab = r"$\tau$"
fig = plt.figure()
ylab = 'Growth rate'
fig = figplot(dfa['clrs'], dfa['tau'], dfa['G'], xlab, ylab, fig, 1)

ylab = 'BMR'
fig = figplot(dfa['clrs'], dfa['tau'], dfa['M'], xlab, ylab, fig, 2)

ylab = 'Disperal rate'
fig = figplot(dfa['clrs'], dfa['tau'], dfa['D'], xlab, ylab, fig, 4)

ylab = 'Resuscitation rate'
fig = figplot(dfa['clrs'], dfa['tau'], dfa['RF'], xlab, ylab, fig, 5)

ylab = 'Specialization'
fig = figplot(dfa['clrs'], dfa['tau'], dfa['E'], xlab, ylab, fig, 7)

ylab = '% Decrease of BMR\nin dormancy'
fig = figplot(dfa['clrs'], dfa['tau'], dfa['MF'], xlab, ylab, fig, 8)

plt.subplots_adjust(wspace=0.6, hspace=0.5)
plt.savefig(mydir + '/ModelTypes/Costs-Growth/results/figures/Traits_vs_Tau-Active_Weighted.png', dpi=200, bbox_inches = "tight")
plt.close()




df = d[d['dormant.total.abundance'] > 0]
dfd = pd.DataFrame({'V' : df['V'].groupby(df['sim']).mean()})
dfd['Q'] = df['Q'].groupby(df['sim']).mean()
dfd['tau'] = dfd['V']/dfd['Q']

dfd['S'] = df['dormant.species.richness'].groupby(df['sim']).mean()
dfd['Rrich'] = df['resource.richness'].groupby(df['sim']).mean()
dfd['G'] = df['dormant.avg.per.capita.growth'].groupby(df['sim']).mean()
dfd['M'] = df['dormant.avg.per.capita.maint'].groupby(df['sim']).mean()
dfd['D'] = df['dormant.avg.per.capita.dispersal'].groupby(df['sim']).mean()
dfd['RF'] = df['dormant.avg.per.capita.rpf'].groupby(df['sim']).mean()
dfd['MF'] = 1/df['dormant.avg.per.capita.mf'].groupby(df['sim']).mean()
dfd['E'] = df['dormant.avg.per.capita.efficiency1e'].groupby(df['sim']).mean()

dfd = dfd[dfd['S'] > minS]
dfd['clrs'] = assigncolor(np.log10(dfd['V']), 'V')
#dfd['clrs'] = assigncolor(dfd['Q'], 'Q')
dft = dft[dft['E'] > 0.001]
dfd = dfd.replace([np.inf, -np.inf], np.nan).dropna()

xlab = r"$\tau$"
fig = plt.figure()
ylab = 'Growth rate'
fig = figplot(dfd['clrs'], dfd['tau'], dfd['G'], xlab, ylab, fig, 1)

ylab = 'BMR'
fig = figplot(dfd['clrs'], dfd['tau'], dfd['M'], xlab, ylab, fig, 2)

ylab = 'Disperal rate'
fig = figplot(dfd['clrs'], dfd['tau'], dfd['D'], xlab, ylab, fig, 4)

ylab = 'Resuscitation rate'
fig = figplot(dfd['clrs'], dfd['tau'], dfd['RF'], xlab, ylab, fig, 5)

ylab = 'Specialization'
fig = figplot(dfd['clrs'], dfd['tau'], dfd['E'], xlab, ylab, fig, 7)

ylab = '% Decrease of BMR\nin dormancy'
fig = figplot(dfd['clrs'], dfd['tau'], dfd['MF'], xlab, ylab, fig, 8)

plt.subplots_adjust(wspace=0.6, hspace=0.5)
plt.savefig(mydir + '/ModelTypes/Costs-Growth/results/figures/Traits_vs_Tau-Dormant_Weighted.png', dpi=200, bbox_inches = "tight")
plt.close()



xlab = r"$\tau$"
fig = plt.figure()
ylab = 'Growth rate'
fig = figplot(dfa['clrs'], dfa['tau'], dfa['G'], xlab, ylab, fig, 1)

ylab = 'BMR'
fig = figplot(dft['clrs'], dft['tau'], dft['M'], xlab, ylab, fig, 2)

ylab = 'Disperal rate'
fig = figplot(dfa['clrs'], dfa['tau'], dfa['D'], xlab, ylab, fig, 4)

ylab = 'Resuscitation rate'
fig = figplot(dft['clrs'], dft['tau'], dft['RF'], xlab, ylab, fig, 5)

ylab = 'Specialization'
fig = figplot(dfa['clrs'], dfa['tau'], dfa['E'], xlab, ylab, fig, 7)

ylab = '% Decrease of BMR\nin dormancy'
fig = figplot(dft['clrs'], dft['tau'], dft['MF'], xlab, ylab, fig, 8)

plt.subplots_adjust(wspace=0.6, hspace=0.5)
plt.savefig(mydir + '/ModelTypes/Costs-Growth/results/figures/Traits_vs_Tau-Main_Weighted.png', dpi=200, bbox_inches = "tight")
plt.close()
'''
