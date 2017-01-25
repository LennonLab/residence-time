from __future__ import division
import  matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import random
import os
import sys


mydir = os.path.expanduser('~/GitHub/residence-time')
sys.path.append(mydir+'/tools')
mydir2 = os.path.expanduser("~/")

df = pd.read_csv(mydir + '/results/simulated_data/SimData.csv')
df2 = pd.DataFrame({'tau' : (df['width'])/df['flow.rate'].groupby(df['sim']).mean()})
 

df2['Dil'] = np.log10( 1.0/df2['tau'] )

df2['AvgG'] = df['avg.per.capita.growth'].groupby(df['sim']).mean()
df2['AvgDisp'] = df['avg.per.capita.active.dispersal'].groupby(df['sim']).mean()
df2['AvgRPF'] = df['avg.per.capita.RPF'].groupby(df['sim']).mean()


df2['AvgE'] = df['avg.per.capita.N.efficiency'].groupby(df['sim']).mean()
df2['AvgMaint'] = df['avg.per.capita.maint'].groupby(df['sim']).mean()
df2['MF'] = df['avg.per.capita.MF'].groupby(df['sim']).mean()/100

#df2['phi'] = np.log10(  (df2['AvgMaint'] * df2['AvgE'] * df2['MF'])  /  (df2['AvgG'] * df2['AvgDisp'] * df2['AvgRPF'])  )
df2['phi'] = np.log10(   (df2['AvgMaint'] * df2['AvgRPF'] * df2['MF']) / (df2['AvgG'] * df2['AvgDisp'] * df2['AvgE'])  )

df2 = df2[df2['phi'] > -5] 
df2 = df2[df2['phi'] < -1]

df2['N'] = df['total.abundance'].groupby(df['sim']).mean()
df2['P'] = df['ind.production'].groupby(df['sim']).mean()

#### plot figure ###############################################################
gd = 10
mnct = 1
fs = 6 # fontsize
fig = plt.figure()


N = df2['N'].tolist()
P = df2['P'].tolist()
phi = df2['phi'].tolist()
dil = df2['Dil'].tolist()

x_ends = [-5.1, -1.8]
y_ends = [0, 0]

N.extend(y_ends)
P.extend(y_ends)
phi.extend(x_ends)
dil.extend(x_ends)

#### N vs. Tau #################################################################
fig.add_subplot(2, 2, 1)

xlab = r"$log_{10}(\phi)$"
plt.hexbin(phi, N, mincnt=mnct, gridsize = gd, bins='log', cmap=plt.cm.jet)
plt.ylabel(r"$log_{10}$"+'(' + r"$N$" +')', fontsize=fs+5)
plt.xlabel(xlab, fontsize=fs+8)
plt.tick_params(axis='both', which='major', labelsize=fs)
plt.xlim(-5.0, -2.0)

#### S vs. Tau #################################################################
fig.add_subplot(2, 2, 2)

plt.hexbin(phi, P, mincnt=mnct, gridsize = gd, bins='log', cmap=plt.cm.jet)
plt.ylabel(r"$log_{10}$"+'(' + r"$Productivity$" +')', fontsize=fs+5)
plt.xlabel(xlab, fontsize=fs+8)
plt.tick_params(axis='both', which='major', labelsize=fs)
plt.xlim(-5.0, -2.0)


#### N vs. Tau #################################################################
fig.add_subplot(2, 2, 3)
xlab = r"$log_{10}(1/\tau)$"

plt.hexbin(dil, N, mincnt=mnct, gridsize = gd, bins='log', cmap=plt.cm.jet)
plt.ylabel(r"$log_{10}$"+'(' + r"$N$" +')', fontsize=fs+5)
plt.xlabel(xlab, fontsize=fs+8)
plt.tick_params(axis='both', which='major', labelsize=fs)
plt.xlim(-5.0, -2.0)

#### S vs. Tau #################################################################
fig.add_subplot(2, 2, 4)

plt.hexbin(dil, P, mincnt=mnct, gridsize = gd, bins='log', cmap=plt.cm.jet)
plt.ylabel(r"$log_{10}$"+'(' + r"$Productivity$" +')', fontsize=fs+5)
plt.xlabel(xlab, fontsize=fs+8)
plt.tick_params(axis='both', which='major', labelsize=fs)
plt.xlim(-5.0, -2.0)


#### Final Format and Save #####################################################
plt.subplots_adjust(wspace=0.4, hspace=0.4)
plt.savefig(mydir + '/results/figures/Fig4_1x2.png', dpi=600, bbox_inches = "tight")
#plt.show()
plt.close()
