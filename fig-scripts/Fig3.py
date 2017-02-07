from __future__ import division
import  matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import sys
import scipy as sc
from scipy import stats
from scipy.optimize import curve_fit
from math import pi

mydir = os.path.expanduser('~/GitHub/residence-time')
sys.path.append(mydir+'/tools')
mydir2 = os.path.expanduser("~/")

df = pd.read_csv(mydir + '/results/simulated_data/SimData.csv')

df2 = pd.DataFrame({'width' : df['width']})
df2['flow'] = df['flow.rate']
df2['tau'] = (df2['width']**3)/df2['flow']
df2['dil'] = 1/df2['tau']

df2['N'] = df['total.abundance']
df2['S'] = df['species.richness']
df2['Prod'] = df['ind.production']
df2['E'] = np.log10(df['simpson.e'])
df2['W'] = df['Whittakers.turnover']
df2['Dorm'] = df['Percent.Dormant']

df2['AvgG'] = df['avg.per.capita.growth']
df2['AvgDisp'] = df['avg.per.capita.active.dispersal']
df2['AvgRPF'] = df['avg.per.capita.RPF']
df2['AvgE'] = df['avg.per.capita.N.efficiency']
df2['AvgMaint'] = df['avg.per.capita.maint']
df2['MF'] = df['avg.per.capita.MF']/np.max(df['avg.per.capita.MF'])

E = 0.1

df2['P'] = (1/df2['AvgMaint']) * (1/df2['AvgRPF']) * df2['MF']
df2['G'] = (df2['AvgG'] * df2['AvgDisp']) *  E
df2['phi'] = df2['G'] * df2['P']

#df2['P'] = (1/df2['AvgMaint']) * (1-df2['AvgRPF']) * df2['MF']
#df2['G'] = (df2['AvgG'] * df2['AvgDisp'])
#df2['phi'] = df2['G'] / df2['P']

df2 = df2.replace([np.inf, -np.inf], np.nan).dropna()
df2['x'] = (df2['phi'] * df2['tau'])

#df2 = df2[df2['x'] < 50]
df2 = df2[df2['W'] != 1]
df2 = df2[df2['tau'] > 2.5]

df2 = df2.replace([np.inf, -np.inf], np.nan).dropna()
xs = df2['x'].tolist()

#### plot figure ###############################################################

#xlab = r"$log_{10}$"+'(' + r"$\tau$" +') * ' + r"$log_{10}$"+'(' + r"$\phi$" +')'
xlab =  r"$\tau$" +' * ' + r"$\phi$"
fs = 6 # fontsize
fig = plt.figure()

mct = 1
binz = 'log'
ps = 120
gd = 20
a = 0.1

xl = -0.5
xh = 2

#### N vs. Tau #################################################################
Vs = df2['N'].tolist()
maxv = max(Vs)
i = Vs.index(maxv)
imax = xs[i]
fig.add_subplot(3, 3, 1)

plt.hexbin(df2['x'], df2['N'], mincnt=mct, gridsize = gd, bins=binz, cmap=plt.cm.Greys_r)
plt.scatter(imax, maxv, lw = 1.5, s = ps, facecolors='none', edgecolors='r')
plt.axvline(1, color='r', lw = 2, ls = '-')

plt.ylabel(r"$N$", fontsize=fs+3)
plt.xlabel(xlab, fontsize=fs+3)
plt.tick_params(axis='both', which='major', labelsize=fs)


#### Prod vs. Tau #################################################################
Vs = df2['Prod'].tolist()
maxv = max(Vs)
i = Vs.index(maxv)
imax = xs[i]
fig.add_subplot(3, 3, 2)

plt.hexbin(df2['x'], df2['Prod'], mincnt=mct, gridsize = gd, bins=binz, cmap=plt.cm.Greys_r)
plt.scatter(imax, maxv, lw = 2, s = ps, facecolors='none', edgecolors='r')
plt.axvline(1, color='r', lw = 2, ls = '-')


plt.ylabel("Productivity", fontsize=fs+3)
plt.xlabel(xlab, fontsize=fs+3)
plt.tick_params(axis='both', which='major', labelsize=fs)


#### S vs. Tau #################################################################
Vs = df2['S'].tolist()
maxv = max(Vs)
i = Vs.index(maxv)
imax = xs[i]
fig.add_subplot(3, 3, 4)

plt.hexbin(df2['x'], df2['S'], mincnt=mct, gridsize = gd, bins=binz, cmap=plt.cm.Greys_r)
plt.scatter(imax, maxv, lw = 2, s = ps, facecolors='none', edgecolors='r')
plt.axvline(1, color='r', lw = 2, ls = '-')

plt.ylabel(r"$S$", fontsize=fs+3)
plt.xlabel(xlab, fontsize=fs+3)
plt.tick_params(axis='both', which='major', labelsize=fs)

#### E vs. Tau #################################################################
Vs = df2['E'].tolist()
maxv = min(Vs)
i = Vs.index(maxv)
imax = xs[i]
fig.add_subplot(3, 3, 5)

plt.hexbin(df2['x'], df2['E'], mincnt=mct, gridsize = gd, bins=binz, cmap=plt.cm.Greys_r)
plt.scatter(imax, maxv, lw = 1.5, s = ps, facecolors='none', edgecolors='r')
plt.axvline(1, color='r', lw = 2, ls = '-')

plt.ylabel(r"$E$", fontsize=fs+3)
plt.xlabel(xlab, fontsize=fs+3)
plt.tick_params(axis='both', which='major', labelsize=fs)

#### W vs. Tau #################################################################
Vs = df2['W'].tolist()
maxv = max(Vs)
i = Vs.index(maxv)
imax = xs[i]
fig.add_subplot(3, 3, 7)

plt.hexbin(df2['x'], df2['W'], mincnt=mct, gridsize = gd, bins=binz, cmap=plt.cm.Greys_r)
plt.scatter(imax, maxv, lw = 1.5, s = ps, facecolors='none', edgecolors='r')
plt.axvline(1, color='r', lw = 2, ls = '-')

plt.ylabel(r"$W$", fontsize=fs+3)
plt.xlabel(xlab, fontsize=fs+3)
plt.tick_params(axis='both', which='major', labelsize=fs)


#### Dorm vs. Tau #################################################################
Vs = df2['Dorm'].tolist()
maxv = min(Vs)
i = Vs.index(maxv)
imax = xs[i]
fig.add_subplot(3, 3, 8)

plt.hexbin(df2['x'], df2['Dorm'], mincnt=mct, gridsize = gd, bins=binz, cmap=plt.cm.Greys_r)
plt.scatter(imax, maxv, lw = 1.5, s = ps, facecolors='none', edgecolors='r')
plt.axvline(1, color='r', lw = 2, ls = '-')

plt.ylabel("%Dormant", fontsize=fs+3)
plt.xlabel(xlab, fontsize=fs+3)
plt.tick_params(axis='both', which='major', labelsize=fs)
#### Final Format and Save #####################################################
plt.subplots_adjust(wspace=0.4, hspace=0.4)
plt.savefig(mydir + '/results/figures/Fig3.png', dpi=200, bbox_inches = "tight")
plt.close()
#plt.show()
