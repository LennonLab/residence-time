from __future__ import division
import  matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import sys

mydir = os.path.expanduser('~/GitHub/residence-time')
df = pd.read_csv(mydir + '/Emergence/results/simulated_data/SimData.csv')

df2 = pd.DataFrame({'length' : df['length'].groupby(df['sim']).mean()})
df2['sim'] = df['sim'].groupby(df['sim']).mean()
df2['flow'] = df['flow.rate'].groupby(df['sim']).mean()
df2['tau'] = df2['length']**2/df2['flow']
df2['dil'] = 1/df2['tau']

df2['N'] = np.log10(df['total.abundance'].groupby(df['sim']).mean())
df2['Prod'] = np.log10(df['ind.production'].groupby(df['sim']).mean())
df2['S'] = np.log10(df['species.richness'].groupby(df['sim']).mean())
df2['E'] = df['simpson.e'].groupby(df['sim']).mean()
df2['W'] = df['Whittakers.turnover'].groupby(df['sim']).mean()
df2['Dorm'] = df['Percent.Dormant'].groupby(df['sim']).mean()

state = 'all'
df2['G'] = df[state+'.avg.per.capita.growth'].groupby(df['sim']).mean()
df2['M'] = df[state+'.avg.per.capita.maint'].groupby(df['sim']).mean()
df2['D'] = df[state+'.avg.per.capita.active.dispersal'].groupby(df['sim']).mean()
df2['E'] = df[state+'.avg.per.capita.efficiency'].groupby(df['sim']).mean()
df2['RPF'] = df[state+'.avg.per.capita.rpf'].groupby(df['sim']).mean()
df2['MF'] = df[state+'.avg.per.capita.mf'].groupby(df['sim']).mean()

#df2['phi'] = df2['G'] + df2['D'] + df2['E'] + df2['RPF'] - (df2['MF'] * df2['M'])
df2['phi'] = df2['G'] * df2['D'] * df2['E'] * df2['RPF'] * df2['MF'] * (1/df2['M'])

df2['x'] = np.log10(df2['phi']) / np.log10(df2['tau'])

#### plot figure ###############################################################

xlab =  r"$log(\tau)$" +'/' + r"$log(\phi)$"
fs = 6 # fontsize
fig = plt.figure()

gd = 20
binz = 'log'
mnct = 1
xl = -2
xh = 0
sz = 2
#### N vs. Tau #################################################################
fig.add_subplot(3, 3, 1)
plt.axvline(-1, color='0.6', lw = 2)
plt.scatter(df2['x'], df2['N'], s = sz, color='k')
plt.ylabel(r"$log$" + "(" + r"$N$" + ")", fontsize=fs+3)
plt.xlabel(xlab, fontsize=fs+3)
plt.tick_params(axis='both', which='major', labelsize=fs)
plt.xlim(xl, xh)

#### production vs. Tau ########################################################
#dat = dat.convert_objects(convert_numeric=True).dropna()
fig.add_subplot(3, 3, 2)
plt.axvline(-1, color='0.6', lw = 2)
plt.scatter(df2['x'], df2['Prod'], s = sz, color='k')
plt.ylabel(r"$log$" + "(" + r"$P$" + ")", fontsize=fs+3)
plt.xlabel(xlab, fontsize=fs+3)
plt.tick_params(axis='both', which='major', labelsize=fs)
plt.xlim(xl, xh)

#### S vs. Tau #################################################################
fig.add_subplot(3, 3, 3)
plt.axvline(-1, color='0.6', lw = 2)
plt.scatter(df2['x'], df2['S'], s = sz, color='k')
plt.ylabel(r"$log$" + "(" + r"$S$" + ")", fontsize=fs+3)
plt.xlabel(xlab, fontsize=fs+3)
plt.tick_params(axis='both', which='major', labelsize=fs)
plt.xlim(xl, xh)

#### Final Format and Save #####################################################
plt.subplots_adjust(wspace=0.4, hspace=0.4)
plt.savefig(mydir + '/Emergence/results/figures/Fig3.png', dpi=200, bbox_inches = "tight")
sys.exit()
