from __future__ import division
import  matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import sys
from scipy import stats

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
            if x < 1: c = 'r'
            elif x < 2: c = 'Orange'
            elif x < 3: c = 'Gold'
            elif x < 4: c = 'Green'
            elif x < 5: c = 'Blue'
            else: c = 'DarkViolet'
            cDict[x] = c

        clrs.append(cDict[x])
    return clrs



_lw = 2
sz = 50

df2 = pd.DataFrame({'area' : df['area'].groupby(df['sim']).mean()})
df2['flow'] = df['flow.rate'].groupby(df['sim']).mean()

print len(df2['flow'].tolist())
sys.exit()

df2['tau'] = np.log10(df2['area']/df2['flow'])

df2['N'] = df['total.abundance'].groupby(df['sim']).mean()
df2['NS'] = np.log10(df['avg.pop.size'].groupby(df['sim']).mean())
df2['var'] = np.log10(df['pop.var'].groupby(df['sim']).mean())

df2 = df2[df2['var'] > 1]

df2 = df2.replace([np.inf, -np.inf], np.nan).dropna()

clrs = assigncolor(df2['tau'])
df2['clrs'] = clrs

#### plot figure ###############################################################
fs = 18
fig = plt.figure()
fig.add_subplot(1, 1, 1)

Nlist = df2['NS'].tolist()
Vlist = df2['var'].tolist()

plt.scatter(Nlist, Vlist, color=df2['clrs'], s = sz, linewidths=0.5, edgecolor='w')
m, b, r, p, std_err = stats.linregress(Nlist, Vlist)
Nlist = np.array(Nlist)
plt.plot(Nlist, m*Nlist + b, '-', color='k', label='$z$ = '+str(round(m,2)), lw=_lw)
xlab = r"$log_{10}$"+'(mean)'
ylab = r"$log_{10}$"+'(variance)'
plt.xlabel(xlab, fontsize=fs)
plt.tick_params(axis='both', labelsize=fs-4)
plt.ylabel(ylab, fontsize=fs)
plt.legend(loc=2, fontsize=fs, frameon=False)

#### Final Format and Save #####################################################
plt.subplots_adjust(wspace=0.4, hspace=0.4)
plt.savefig(mydir + '/results/figures/TaylorsLaw.png', dpi=200, bbox_inches = "tight")
plt.close()
