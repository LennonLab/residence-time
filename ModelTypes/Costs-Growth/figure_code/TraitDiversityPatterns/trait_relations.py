from __future__ import division
import  matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os


mydir = os.path.expanduser('~/GitHub/residence-time2/Emergence')
tools = os.path.expanduser(mydir + "/tools")


def xfrm(X, _max): return -np.log(_max-np.array(X))

def laplace_reg(X, loc, scale, const):
    laplace = const*(1/(2*scale)) * np.exp((-np.abs(X - loc))/scale)
    return laplace

def assigncolor(xs):
    cDict = {}
    clrs = []
    for x in xs:
        if x not in cDict:
            if x <= 1: c = 'r'
            elif x <= 2: c = 'Orange'
            elif x <= 3: c = 'Green'
            elif x <= 4: c = 'DodgerBlue'
            elif x <= 5: c = 'Plum'
            else: c = 'Purple'
            cDict[x] = c

        clrs.append(cDict[x])
    return clrs



df = pd.read_csv(mydir + '/ModelTypes/Costs-Growth/results/simulated_data/SimData.csv')
df = df[df['total.abundance'] > 0]

df2 = pd.DataFrame({'N' : df['active.total.abundance'].groupby(df['sim']).mean()})
df2['S'] = df['active.species.richness'].groupby(df['sim']).mean()
df2['P'] = df['ind.production'].groupby(df['sim']).mean()
df2['tau'] = df['V'].groupby(df['sim']).mean()/df['Q'].groupby(df['sim']).mean()

df2['repro.p'] = df['active.repro.p'].groupby(df['sim']).mean()
df2['D'] = df['active.avg.per.capita.dispersal'].groupby(df['sim']).mean()
df2['size'] = df['active.avg.per.capita.size'].groupby(df['sim']).mean()
df2['q'] = df['active.total.biomass'].groupby(df['sim']).mean()
df2['M'] = df['active.avg.per.capita.maint'].groupby(df['sim']).mean()
df2['G'] = df['active.avg.per.capita.growth'].groupby(df['sim']).mean()

clrs = assigncolor(np.log10(df2['tau']))

#### plot figures ###############################################################
fs = 6
fig = plt.figure()
sz = 0.5
w = 0.75
b = 1000
ci = 99

#### N vs. Tau #################################################################
fig.add_subplot(3, 3, 1)
x = np.log10(df2['G'])
y = np.log10(df2['repro.p'])

plt.scatter(x, y, s = sz, color=clrs, linewidths=0.0, edgecolor=None)
plt.ylabel('Repro P', fontsize=fs)
plt.xlabel('G', fontsize=fs)
plt.tick_params(axis='both', which='major', labelsize=fs-2)


#### production vs. Tau ########################################################
fig.add_subplot(3, 3, 2)
x = np.log10(df2['G'])
y = np.log10(df2['M'])

plt.scatter(x, y, s = sz, color=clrs, linewidths=0.0, edgecolor=None)
plt.ylabel('M', fontsize=fs)
plt.xlabel('G', fontsize=fs)
plt.tick_params(axis='both', which='major', labelsize=fs-2)


#### S vs. Tau #################################################################
fig.add_subplot(3, 3, 3)
x = np.log10(df2['G'])
y = np.log10(df2['D'])

plt.scatter(x, y, s = sz, color=clrs, linewidths=0.0, edgecolor=None)
plt.ylabel('D', fontsize=fs)
plt.xlabel('G', fontsize=fs)
plt.tick_params(axis='both', which='major', labelsize=fs-2)


#### Final Format and Save #####################################################
plt.subplots_adjust(wspace=0.4, hspace=0.4)
plt.savefig(mydir + '/ModelTypes/Costs-Growth/results/figures/trait-relations.png', dpi=200, bbox_inches = "tight")
plt.close()
