from __future__ import division
import  matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import sys

mydir = os.path.expanduser('~/GitHub/residence-time')
sys.path.append(mydir+'/tools')
mydir2 = os.path.expanduser("~/")

dat = pd.read_csv(mydir + '/results/simulated_data/SimData.csv')
dat = dat.convert_objects(convert_numeric=True)

#dat = dat[np.isfinite(dat['avg.per.capita.growth'])]
#dat = dat[np.isfinite(dat['avg.per.capita.active.dispersal'])]
#dat = dat[np.isfinite(dat['avg.per.capita.maint'])]
#dat = dat[np.isfinite(dat['avg.per.capita.N.efficiency'])]
tau = np.log10(dat['width']/dat['flow.rate']).tolist()

AvgGrow = dat['avg.per.capita.growth'].tolist()
AvgActDisp = dat['avg.per.capita.active.dispersal'].tolist()
AvgMaint = dat['avg.per.capita.maint'].tolist()
#AvgEff = dat['avg.per.capita.N.efficiency'].tolist()
AvgEff = dat['avg.per.capita.MF'].tolist()

print dat.shape

gd = 30
mnct = 1

#### plot figure ###############################################################
xlab = r"$log_{10}$"+'(' + r"$\tau$" +')'
fs = 8 # fontsize
fig = plt.figure()

#### AvgGrow vs. Tau #################################################################
fig.add_subplot(2, 2, 1)

plt.hexbin(tau, AvgGrow, mincnt=mnct, gridsize = gd, bins='log', cmap=plt.cm.jet)
plt.ylabel('Specific growth rate', fontsize=fs+6)
plt.xlabel(xlab, fontsize=fs+6)
plt.tick_params(axis='both', which='major', labelsize=fs)

#### AvgActDisp vs. Tau #################################################################
fig.add_subplot(2, 2, 2)

plt.hexbin(tau, AvgActDisp, mincnt=mnct, gridsize = gd, bins='log', cmap=plt.cm.jet)
plt.ylabel('Active disperal rate', fontsize=fs+6)
plt.xlabel(xlab, fontsize=fs+6)
plt.tick_params(axis='both', which='major', labelsize=fs)

#### E vs. Tau #################################################################
fig.add_subplot(2, 2, 3)

plt.hexbin(tau, AvgMaint, mincnt=mnct, gridsize = gd, bins='log', cmap=plt.cm.jet)
plt.ylabel('Maintenance energy', fontsize=fs+6)
plt.xlabel(xlab, fontsize=fs+6)
plt.tick_params(axis='both', which='major', labelsize=fs)

#### AvgEff vs. Tau #################################################################
fig.add_subplot(2, 2, 4)

plt.hexbin(tau, AvgEff, mincnt=mnct, gridsize = gd, bins='log', cmap=plt.cm.jet)
plt.ylabel('Growth efficiency', fontsize=fs+6)
plt.xlabel(xlab, fontsize=fs+6)
plt.tick_params(axis='both', which='major', labelsize=fs)

#### Final Format and Save #####################################################
plt.subplots_adjust(wspace=0.4, hspace=0.4)
plt.savefig(mydir + '/results/figures/Fig3-heat.png', dpi=600, bbox_inches = "tight")
#plt.show()
plt.close()
