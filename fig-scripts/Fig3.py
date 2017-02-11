from __future__ import division
import  matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import sys

mydir = os.path.expanduser('~/GitHub/residence-time')
sys.path.append(mydir+'/tools')
mydir2 = os.path.expanduser("~/")

df = pd.read_csv(mydir + '/results/simulated_data/SimData.csv')

df2 = pd.DataFrame({'width' : df['width'].groupby(df['ct']).mean()})
df2['flow'] = df['flow.rate'].groupby(df['ct']).mean()

df2['tau'] = np.log10(df2['width']**3/df2['flow'])
#df2['tau'] = np.log10(df2['flow'])
#df2['tau'] = np.log10(df2['width'])

df2['N'] = df['total.abundance'].groupby(df['ct']).mean()
df2['S'] = df['species.richness'].groupby(df['ct']).mean()
df2['Prod'] = df['ind.production'].groupby(df['ct']).mean()
df2['E'] = df['simpson.e'].groupby(df['ct']).mean()
df2['W'] = np.log10(df['Whittakers.turnover'].groupby(df['ct']).mean())
df2['Dorm'] = df['Percent.Dormant'].groupby(df['ct']).mean()


df2['G'] = df['active.avg.per.capita.growth'].groupby(df['ct']).mean()

#df2['AvgMaint'] = np.log10(df['active.avg.per.capita.maint']).groupby(df['ct']).mean()
df2['Maint'] = np.log10(df['dormant.avg.per.capita.maint']).groupby(df['ct']).mean()

df2['Disp'] = df['active.avg.per.capita.active.dispersal'].groupby(df['ct']).mean()
 
#df2['AvgRPF'] = df['active.avg.per.capita.rpf'].groupby(df['ct']).mean()
df2['RPF'] = df['dormant.avg.per.capita.rpf'].groupby(df['ct']).mean()

df2['Eff'] = df['active.avg.per.capita.efficiency'].groupby(df['ct']).mean()

df2['MF'] = df['active.avg.per.capita.mf'].groupby(df['ct']).mean()
#df2['AvgMF'] = df['dormant.avg.per.capita.mf'].groupby(df['ct']).mean()
E = 0.1

#df2['P'] = (1/df2['Maint']) * (1/df2['RPF']) * df2['MF']
#df2['G'] = (df2['G'] * df2['Disp']) *  E
#df2['phi'] = df2['G'] * df2['P']

df2['Ps'] = (1/df2['Maint']) * (1-df2['RPF']) * df2['MF']
df2['Gs'] = (df2['G'] * df2['Disp'])
df2['phi'] = df2['Gs'] / df2['Ps']

df2 = df2.replace([np.inf, -np.inf], np.nan).dropna()
df2['x'] = (df2['phi'] * df2['tau'])

#df2 = df2[df2['x'] < 50]
#df2 = df2[df2['W'] != 1]
#df2 = df2[df2['tau'] > 2.5]

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
maxv = min(Vs)
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
