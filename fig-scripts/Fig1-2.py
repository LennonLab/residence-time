from __future__ import division
import  matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import sys


mydir = os.path.expanduser('~/GitHub/residence-time')
df = pd.read_csv(mydir + '/results/simulated_data/SimData.csv')

df2 = pd.DataFrame({'width' : df['width'].groupby(df['ct']).mean()})
df2['flow'] = df['flow.rate'].groupby(df['ct']).mean()

#df2['tau'] = df2['width']**3/df2['flow']
df2['tau'] = np.log10(df2['width']**1/df2['flow'])
#df2['tau'] = np.log10(df2['flow'])
#df2['tau'] = np.log10(df2['width']**3)

df2['N'] = df['total.abundance'].groupby(df['ct']).mean()
df2['S'] = df['species.richness'].groupby(df['ct']).mean()
df2['Prod'] = df['ind.production'].groupby(df['ct']).mean()
df2['E'] = df['simpson.e'].groupby(df['ct']).mean()
df2['W'] = np.log10(df['Whittakers.turnover'].groupby(df['ct']).mean())
df2['Dorm'] = df['Percent.Dormant'].groupby(df['ct']).mean()


df2['AvgGrow'] = df['active.avg.per.capita.growth'].groupby(df['ct']).mean()

df2['AvgMaint'] = np.log10(df['active.avg.per.capita.maint']).groupby(df['ct']).mean()
#df2['AvgMaint'] = np.log10(df['dormant.avg.per.capita.maint']).groupby(df['ct']).mean()

df2['AvgActDisp'] = df['active.avg.per.capita.active.dispersal'].groupby(df['ct']).mean()
 
df2['AvgRPF'] = df['active.avg.per.capita.rpf'].groupby(df['ct']).mean()
#df2['AvgRPF'] = df['dormant.avg.per.capita.rpf'].groupby(df['ct']).mean()

df2['AvgEff'] = df['active.avg.per.capita.efficiency'].groupby(df['ct']).mean()

df2['AvgMF'] = df['active.avg.per.capita.mf'].groupby(df['ct']).mean()
#df2['AvgMF'] = df['dormant.avg.per.capita.mf'].groupby(df['ct']).mean()


#print len(df2['N'])
#sys.exit()
#df2 = df2[df2['N'] > 1]
#df2 = df2[df2['AvgMF'] != 40]
#df2 = df2[df2['Dorm'] != 0]
#df2 = df2[df2['Dorm'] != 100]
#df2 = df2[df2['Prod'] < 15]

#### plot figure ###############################################################
xlab = r"$log_{10}$"+'(' + r"$\tau$" +')'
fs = 6 # fontsize
fig = plt.figure()

gd = 20
mct = 1
binz = 'log'

#### N vs. Tau #################################################################
fig.add_subplot(3, 3, 1)

plt.hexbin(df2['tau'], df2['N'], mincnt=mct, gridsize = gd, bins=binz, cmap=plt.cm.jet)
plt.ylabel(r"$log_{10}$"+'(' + r"$N$" + ')', fontsize=fs+3)
plt.xlabel(xlab, fontsize=fs+3)
plt.tick_params(axis='both', which='major', labelsize=fs)
#plt.xlim(2,6)
#### production vs. Tau ########################################################
fig.add_subplot(3, 3, 2)

plt.hexbin(df2['tau'], df2['Prod'], mincnt=mct, gridsize = gd, bins=binz, cmap=plt.cm.jet)
plt.ylabel(r"$log_{10}$"+'(' + r"$Productivity$" + ')', fontsize=fs+3)
plt.xlabel(xlab, fontsize=fs+3)
plt.tick_params(axis='both', which='major', labelsize=fs)
#plt.xlim(2,6)

#### S vs. Tau #################################################################
fig.add_subplot(3, 3, 4)

plt.hexbin(df2['tau'], df2['S'], mincnt=mct, gridsize = gd, bins=binz, cmap=plt.cm.jet)
plt.ylabel(r"$log_{10}$"+'(' + r"$S$" +')', fontsize=fs+3)
plt.xlabel(xlab, fontsize=fs+3)
plt.tick_params(axis='both', which='major', labelsize=fs)
#plt.xlim(2,6)

#### E vs. Tau #################################################################
fig.add_subplot(3, 3, 5)

plt.hexbin(df2['tau'], df2['E'], mincnt=mct, gridsize = gd, bins=binz, cmap=plt.cm.jet)
plt.ylabel(r"$log_{10}$"+'(' + r"$Evenness$" +')', fontsize=fs+3)
plt.xlabel(xlab, fontsize=fs+3)
plt.tick_params(axis='both', which='major', labelsize=fs)
#plt.xlim(2,6)

#### W vs. Tau #################################################################
ax5 = fig.add_subplot(3, 3, 7)

#df3 = df2[df2['W'] != -1]
plt.hexbin(df2['tau'], df2['W'], mincnt=mct, gridsize = gd, bins=binz, cmap=plt.cm.jet)
plt.ylabel(r"$log_{10}$"+'(' + r"$\beta$" +')', fontsize=fs+3)
plt.xlabel(xlab, fontsize=fs+3)
plt.tick_params(axis='both', which='major', labelsize=fs)
#plt.xlim(2,6)

#### dormancy vs. Tau ########################################################
fig.add_subplot(3, 3, 8)

plt.hexbin(df2['tau'], df2['Dorm'], mincnt=mct, gridsize = gd, bins=binz, cmap=plt.cm.jet)
plt.ylabel(r"$log_{10}$"+'(' + r"$Dormant$" +')', fontsize=fs+3)
plt.xlabel(xlab, fontsize=fs+3)
plt.tick_params(axis='both', which='major', labelsize=fs)
#plt.xlim(2,6)

#### Final Format and Save #####################################################
plt.subplots_adjust(wspace=0.4, hspace=0.4)
plt.savefig(mydir + '/results/figures/Fig1.png', dpi=200, bbox_inches = "tight")
plt.close()



#df2 = df2[df2['N'] > 0]
#### plot figure ###############################################################
xlab = r"$log_{10}$"+'(' + r"$\tau$" +')'
fs = 6 # fontsize
fig = plt.figure()


#### AvgGrow vs. Tau #################################################################
fig.add_subplot(3, 3, 1)

plt.hexbin(df2['tau'], df2['AvgGrow'], mincnt=mct, gridsize = gd, bins=binz, cmap=plt.cm.jet, alpha = 1)
plt.ylabel('Specific growth rate', fontsize=fs+2)
plt.xlabel(xlab, fontsize=fs+3)
plt.tick_params(axis='both', which='major', labelsize=fs)
#plt.xlim(2,6)

#### AvgActDisp vs. Tau #################################################################
fig.add_subplot(3, 3, 2)

plt.hexbin(df2['tau'], df2['AvgMaint'], mincnt=mct, gridsize = gd, bins=binz, cmap=plt.cm.jet, alpha = 1)
plt.ylabel('Maintenance energy, '+r"$log_{10}$", fontsize=fs+2)
plt.xlabel(xlab, fontsize=fs+3)
plt.tick_params(axis='both', which='major', labelsize=fs)
#plt.xlim(2,6)

#### E vs. Tau #################################################################
fig.add_subplot(3, 3, 4)

plt.hexbin(df2['tau'], df2['AvgActDisp'], mincnt=mct, gridsize = gd, bins=binz, cmap=plt.cm.jet, alpha = 1)
plt.ylabel('Active disperal rate', fontsize=fs+2)
plt.xlabel(xlab, fontsize=fs+3)
plt.tick_params(axis='both', which='major', labelsize=fs)
#plt.xlim(2,6)

#### AvgEff vs. Tau #################################################################
fig.add_subplot(3, 3, 5)

plt.hexbin(df2['tau'], df2['AvgRPF'], mincnt=mct, gridsize = gd, bins=binz, cmap=plt.cm.jet, alpha = 1)
plt.ylabel('Random resuscitation\nfrom dormancy, ' + r"$log_{10}$", fontsize=fs+2)
plt.xlabel(xlab, fontsize=fs+3)
plt.tick_params(axis='both', which='major', labelsize=fs)
#plt.xlim(2,6)

#### AvgRPF vs. Tau #################################################################
fig.add_subplot(3, 3, 7)

df3 = df2[df2['AvgEff'] < 0.13]
plt.hexbin(df2['tau'], df2['AvgEff'], mincnt=mct, gridsize = gd, bins=binz, cmap=plt.cm.jet, alpha = 1)
plt.ylabel('Resource specialization', fontsize=fs+2)
plt.xlabel(xlab, fontsize=fs+3)
plt.tick_params(axis='both', which='major', labelsize=fs)
#plt.xlim(2,6)
plt.ylim(0.0, 0.1)

#### AvgRPF vs. Tau #################################################################
fig.add_subplot(3, 3, 8)

df3 = df2[df2['AvgMF'] > 0]
plt.hexbin(df2['tau'], df2['AvgMF'], mincnt=mct, gridsize = gd, bins=binz, cmap=plt.cm.jet, alpha = 1)
plt.ylabel('Decrease of maintenance\nenergy when dormant, ' + r"$log_{10}$", fontsize=fs+2)
plt.xlabel(xlab, fontsize=fs+3)
plt.tick_params(axis='both', which='major', labelsize=fs)
#plt.xlim(2,6)


#### Final Format and Save #####################################################
plt.subplots_adjust(wspace=0.5, hspace=0.5)
plt.savefig(mydir + '/results/figures/Fig2.png', dpi=200, bbox_inches = "tight")
plt.close()
