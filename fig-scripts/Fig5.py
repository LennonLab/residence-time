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

df2 = pd.DataFrame({'width' : df['width']}) #.groupby(df['sim']).mean()})
df2['flow'] = df['flow.rate'] #.groupby(df['sim']).mean()
df2['tau'] = np.log10(df2['width']/df2['flow'])

df2['N'] = df['total.abundance'] #.groupby(df['sim']).mean())
df2['S'] = df['species.richness'] #.groupby(df['sim']).mean()

df2['Prod'] = df['ind.production'] #.groupby(df['sim']).mean())
df2['E'] = np.log10(df['simpson.e']) #.groupby(df['sim']).mean()
df2['W'] = np.log10(df['Whittakers.turnover']) #.groupby(df['sim']).mean()
df2['Dorm'] = df['Percent.Dormant'] #.groupby(df['sim']).mean()

df2['Dil'] = np.log10(1/df2['tau'])

df2['AvgG'] = df['avg.per.capita.growth'] #.groupby(df['sim']).mean()
df2['AvgDisp'] = df['avg.per.capita.active.dispersal'] #.groupby(df['sim']).mean()
df2['AvgRPF'] = df['avg.per.capita.RPF'] #.groupby(df['sim']).mean()
df2['AvgE'] = df['avg.per.capita.N.efficiency'] #.groupby(df['sim']).mean()
df2['AvgMaint'] = df['avg.per.capita.maint'] #.groupby(df['sim']).mean()
df2['MF'] = df['avg.per.capita.MF']/np.mean(df['avg.per.capita.MF'])

# DON'T WORK
#df2['phi'] = np.log10(   (df2['AvgMaint'] * df2['AvgRPF'] * df2['MF']) / (df2['AvgG'] * df2['AvgDisp'] * df2['AvgE'])  )
#df2['phi'] = np.log10(    (df2['AvgG'] * df2['AvgDisp'] * df2['AvgE']) / (df2['AvgMaint'] * df2['AvgRPF'] * df2['MF']) )

df2['phi'] = (df2['AvgMaint'] * df2['AvgRPF'] * df2['MF']) / (df2['AvgG'] * df2['AvgDisp'] * df2['AvgE'])


df2['x'] = df2['phi']#*df2['Dil']
#df2 = df2[df2['x'] < 3.4]
#df2 = df2[df2['x'] > 0.1]

#### plot figure ###############################################################

#xlab = r"$log_{10}$"+'(' + r"$\tau$" +') * ' + r"$log_{10}$"+'(' + r"$\phi$" +')'
#xlab =  r"$\tau$" +' * ' + r"$\phi$"
fs = 6 # fontsize
fig = plt.figure()

gd = 20
binz = 'log'
mnct = 1

#xl = -0.25
#xh = 0.25
#### N vs. Tau #################################################################
fig.add_subplot(3, 3, 1)

plt.scatter(df2['phi'], df2['N'], color = 'k', alpha=0.01)
plt.ylabel(r"$log_{10}$" + "(" + r"$N$" + ")", fontsize=fs+3)
#plt.xlabel(xlab, fontsize=fs+3)
plt.tick_params(axis='both', which='major', labelsize=fs)
#plt.text(1.1, 2, 'A', color = 'y', fontweight='bold')
#plt.xlim(xl, xh)
#plt.axvline(1, color='k', lw = 2)

#### Final Format and Save #####################################################
plt.subplots_adjust(wspace=0.4, hspace=0.4)
plt.savefig(mydir + '/results/figures/Fig3.png', dpi=600, bbox_inches = "tight")
#plt.show()
plt.close()







xl = -0.25
xh = 0.25

#### production vs. Tau ########################################################
#dat = dat.convert_objects(convert_numeric=True).dropna()
fig.add_subplot(3, 3, 2)

plt.hexbin(df2['x'], df2['Prod'], mincnt=mnct, gridsize = gd, bins=binz, cmap=plt.cm.Greys)
plt.ylabel('Productivity', fontsize=fs+3)
plt.xlabel(xlab, fontsize=fs+3)
plt.tick_params(axis='both', which='major', labelsize=fs)
#plt.text(4.2, -0.25, 'B', color = 'y', fontweight='bold')
plt.xlim(xl, xh)
plt.axvline(1, color='k', lw = 2)

#### S vs. Tau #################################################################
fig.add_subplot(3, 3, 4)

plt.hexbin(df2['x'], df2['S'], mincnt=mnct, gridsize = gd, bins=binz, cmap=plt.cm.Greys)
plt.ylabel(r"$S$", fontsize=fs+3)
plt.xlabel(xlab, fontsize=fs+3)
plt.tick_params(axis='both', which='major', labelsize=fs)
#plt.text(1.1, 1.6, 'C', color = 'y', fontweight='bold')
plt.xlim(xl, xh)
plt.axvline(1, color='k', lw = 2)

#### E vs. Tau #################################################################
fig.add_subplot(3, 3, 5)

plt.hexbin(df2['x'], df2['E'], mincnt=mnct, gridsize = gd, bins=binz, cmap=plt.cm.Greys)
plt.ylabel(r"$log_{10}$"+'(Evenness)', fontsize=fs+3)
plt.xlabel(xlab, fontsize=fs+3)
plt.tick_params(axis='both', which='major', labelsize=fs)
#plt.text(4.7, 0.3, 'D', color = 'y', fontweight='bold')
plt.xlim(xl, xh)
plt.axvline(1, color='k', lw = 2)

#### W vs. Tau #################################################################
fig.add_subplot(3, 3, 7)

plt.hexbin(df2['x'], df2['W'], mincnt=mnct, gridsize = gd, bins=binz, cmap=plt.cm.Greys)
plt.ylabel(r"$log_{10}$"+'(' + r"$\beta$" +')', fontsize=fs+3)
#plt.ylabel(r"$\beta_{w}$", fontsize=fs+5)
plt.xlabel(xlab, fontsize=fs+3)
plt.tick_params(axis='both', which='major', labelsize=fs)
#plt.text(1.1, -3.0, 'E', color = 'y', fontweight='bold')
plt.xlim(xl, xh)
plt.axvline(1, color='k', lw = 2)

#### % dormancy vs. Tau ########################################################
fig.add_subplot(3, 3, 8)

plt.hexbin(df2['x'], df2['Dorm'], mincnt=mnct, gridsize = gd, bins=binz, cmap=plt.cm.Greys)
#plt.ylabel(r"$log_{10}$"+'(% Dormancy)', fontsize=fs+3)
plt.ylabel('% Dormant', fontsize=fs+3)
plt.xlabel(xlab, fontsize=fs+3)
plt.tick_params(axis='both', which='major', labelsize=fs)
#plt.text(4.7, 0.2, 'F', color = 'y', fontweight='bold')
plt.xlim(xl, xh)
plt.axvline(1, color='k', lw = 2)

#### Final Format and Save #####################################################
plt.subplots_adjust(wspace=0.4, hspace=0.4)
plt.savefig(mydir + '/results/figures/Fig3.png', dpi=200, bbox_inches = "tight")
#plt.show()
plt.close()
