from __future__ import division
import  matplotlib.pyplot as plt
import pandas as pd
from random import randint
import numpy as np
import os
from scipy import stats

mydir = os.path.expanduser('~/GitHub/residence-time/Emergence')
tools = os.path.expanduser(mydir + "/tools")


def assigncolor(xs):
    cDict = {}
    clrs = []
    for x in xs:
        if x not in cDict:
            r1 = lambda: randint(0,255)
            r2 = lambda: randint(0,255)
            r3 = lambda: randint(0,255)
            cDict[x] = '#%02X%02X%02X' % (r1(),r2(),r3())

        clrs.append(cDict[x])
    return clrs


_lw = 2
sz = 20

df = pd.read_csv(mydir + '/results/simulated_data/SimData.csv')

df2 = pd.DataFrame({'length' : df['length'].groupby(df['sim']).mean()})
df2['R'] = df['res.inflow'].groupby(df['sim']).mean()
df2['NS'] = np.log10(df['avg.pop.size'].groupby(df['sim']).mean())
df2['var'] = np.log10(df['pop.var'].groupby(df['sim']).mean())
df2 = df2.replace([np.inf, -np.inf], np.nan).dropna()
df2 = df2[df2['var'] > 0]
#df2 = df2[df2['NS'] > 0.5]
clrs = assigncolor(df2['R'])
df2['clrs'] = clrs

#### plot figure ###############################################################
fs = 14
fig = plt.figure()
fig.add_subplot(1, 1, 1)

Nlist = df2['NS'].tolist()
Vlist = df2['var'].tolist()

plt.scatter(Nlist, Vlist, lw=_lw, color=df2['clrs'], s = sz)
m, b, r, p, std_err = stats.linregress(Nlist, Vlist)
Nlist = np.array(Nlist)
plt.plot(Nlist, m*Nlist + b, '-', color='k', label='Taylor\'s Law, '+'$z$ = '+str(round(m,2)), lw=_lw)
xlab = r"$log_{10}$"+'(Pop mean)'
ylab = r"$log_{10}$"+'(variance)'
plt.xlabel(xlab, fontsize=fs)
plt.tick_params(axis='both', labelsize=fs-3)
plt.ylabel(ylab, fontsize=fs)
plt.legend(loc=2, fontsize=fs)

#### Final Format and Save #####################################################
plt.subplots_adjust(wspace=0.4, hspace=0.4)
plt.savefig(mydir + '/results/figures/TaylorsLaw.png', dpi=200, bbox_inches = "tight")
plt.close()
