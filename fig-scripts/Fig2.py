from __future__ import division
import  matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import sys


def xfrm(X, _max): return -np.log(_max-np.array(X))
#def ivrt(Y, _max): return _max-np.exp(-np.array(Y))


mydir = os.path.expanduser('~/GitHub/residence-time')
sys.path.append(mydir+'/tools')
mydir2 = os.path.expanduser("~/")

df = pd.read_csv(mydir + '/results/simulated_data/SimData.csv')

df2 = pd.DataFrame({'width' : df['width']})
df2['flow'] = df['flow.rate']
df2['tau'] = np.log10((df['height'] * df['length'] * df2['width'])/df2['flow'])

df2['AvgGrow'] = df['avg.per.capita.growth']
df2['AvgActDisp'] = df['avg.per.capita.active.dispersal']
df2['AvgMaint'] = df['avg.per.capita.maint']
df2['AvgRPF'] = df['avg.per.capita.RPF']
df2['AvgMF'] = df['avg.per.capita.MF']
df2['AvgEff'] = df['avg.per.capita.N.efficiency']

#### plot figure ###############################################################
xlab = r"$log_{10}$"+'(' + r"$\tau$" +')'
fs = 6 # fontsize
fig = plt.figure()

gd = 15
mnct = 1
binz = 'log'
trans = 1
mct = 1

#### AvgGrow vs. Tau #################################################################
fig.add_subplot(3, 3, 1)

#df3 = df2[df2['AvgGrow'] > 0.0]
#plt.hexbin(df3['tau'], df3['AvgGrow'], mincnt=mnct, gridsize = gd, bins=binz, cmap=plt.cm.Greys, alpha = trans)
plt.hexbin(df2['tau'], df2['AvgGrow'], mincnt=mct, gridsize = gd, bins=binz, cmap=plt.cm.jet, alpha = 1)

plt.ylabel('Specific growth rate', fontsize=fs+2)
plt.xlabel(xlab, fontsize=fs+3)
plt.tick_params(axis='both', which='major', labelsize=fs)
#plt.text(2, 0.15, 'A', color = 'y', fontweight='bold')
#plt.text(1.0, 1.05, 'Growth Syndrome', color = 'Crimson', fontsize = 10, fontweight='bold')

#### AvgActDisp vs. Tau #################################################################
fig.add_subplot(3, 3, 2)

#plt.hexbin(df2['tau'], df2['AvgMaint'], mincnt=mnct, gridsize = gd, bins=binz, cmap=plt.cm.Greys, alpha = trans)
plt.hexbin(df2['tau'], df2['AvgMaint'], mincnt=mct, gridsize = gd, bins=binz, cmap=plt.cm.jet, alpha = 1)

plt.ylabel('Maintenance energy, '+r"$log_{10}$", fontsize=fs+2)
plt.xlabel(xlab, fontsize=fs+3)
plt.tick_params(axis='both', which='major', labelsize=fs)
#plt.text(1.3, -4.6, 'C', color = 'y', fontweight='bold')
#plt.text(0.5, -0.38, 'Persistence Syndrome', color = 'Steelblue', fontsize = 10, fontweight='bold')

#### E vs. Tau #################################################################
fig.add_subplot(3, 3, 4)

#plt.hexbin(df2['tau'], df2['AvgActDisp'], mincnt=mnct, gridsize = gd, bins=binz, cmap=plt.cm.Greys, alpha = trans)
plt.hexbin(df2['tau'], df2['AvgActDisp'], mincnt=mct, gridsize = gd, bins=binz, cmap=plt.cm.jet, alpha = 1)

plt.ylabel('Active disperal rate', fontsize=fs+2)
plt.xlabel(xlab, fontsize=fs+3)
plt.tick_params(axis='both', which='major', labelsize=fs)
#plt.text(2, 0.1, 'B', color = 'y', fontweight='bold')


#### AvgEff vs. Tau #################################################################
fig.add_subplot(3, 3, 5)

#plt.hexbin(df2['tau'], df2['AvgRPF'], mincnt=0, gridsize = gd, bins=binz, cmap=plt.cm.Greys, alpha = trans)
plt.hexbin(df2['tau'], df2['AvgRPF'], mincnt=mct, gridsize = gd, bins=binz, cmap=plt.cm.jet, alpha = 1)

plt.ylabel('Random resuscitation\nfrom dormancy, ' + r"$log_{10}$", fontsize=fs+2)
plt.xlabel(xlab, fontsize=fs+3)
plt.tick_params(axis='both', which='major', labelsize=fs)
#plt.text(1.3, -2, 'E', color = 'y', fontweight='bold')


#### AvgRPF vs. Tau #################################################################
fig.add_subplot(3, 3, 7)

#plt.hexbin(df2['tau'], df2['AvgEff'], mincnt=mnct, gridsize = gd, bins=binz, cmap=plt.cm.Greys, alpha = trans)
plt.hexbin(df2['tau'], df2['AvgEff'], mincnt=mct, gridsize = gd, bins=binz, cmap=plt.cm.jet, alpha = 1)

plt.ylabel('Resource specialization', fontsize=fs+2)
plt.xlabel(xlab, fontsize=fs+3)
plt.tick_params(axis='both', which='major', labelsize=fs)
#plt.text(1.3, 0.099, 'D', color = 'y', fontweight='bold')


#### AvgRPF vs. Tau #################################################################
fig.add_subplot(3, 3, 8)

#plt.hexbin(df2['tau'], df2['AvgMF'], mincnt=mnct, gridsize = gd, bins=binz, cmap=plt.cm.Greys, alpha = trans)
plt.hexbin(df2['tau'], df2['AvgMF'], mincnt=mct, gridsize = gd, bins=binz, cmap=plt.cm.jet, alpha = 1)

plt.ylabel('Decrease of maintenance\nenergy when dormant, ' + r"$log_{10}$", fontsize=fs+2)
plt.xlabel(xlab, fontsize=fs+3)
plt.tick_params(axis='both', which='major', labelsize=fs)
#plt.text(2.1, 4, 'F', color = 'y', fontweight='bold')


#### Final Format and Save #####################################################
plt.subplots_adjust(wspace=0.5, hspace=0.5)
plt.savefig(mydir + '/results/figures/Fig2.png', dpi=200, bbox_inches = "tight")
plt.close()
