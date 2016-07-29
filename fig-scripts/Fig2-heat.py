from __future__ import division
import  matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import random
import os
import sys

import statsmodels.stats.api as sms
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.sandbox.regression.predstd import wls_prediction_std
from statsmodels.stats.outliers_influence import summary_table

mydir = os.path.expanduser('~/GitHub/residence-time')
sys.path.append(mydir+'/tools')
mydir2 = os.path.expanduser("~/")

dat = pd.read_csv(mydir + '/results/simulated_data/SimData.csv')

tau = np.log10(dat['width']/dat['flow.rate'])
N = np.log10(dat['total.abundance'])
S = np.log10(dat['species.richness'])

#### plot figure ###############################################################
gd = 30
mnct = 1

xlab = r"$log_{10}$"+'(' + r"$\tau$" +')'
fs = 8 # fontsize
fig = plt.figure()


#### N vs. Tau #################################################################
fig.add_subplot(2, 2, 1)
plt.hexbin(tau, N, mincnt=mnct, gridsize = gd, bins='log', cmap=plt.cm.jet)
plt.ylabel(r"$log_{10}$"+'(' + r"$N$" +')', fontsize=fs+6)
plt.xlabel(xlab, fontsize=fs+6)
#plt.ylim(0.0, 4.0)
plt.tick_params(axis='both', which='major', labelsize=fs)

#### S vs. Tau #################################################################
fig.add_subplot(2, 2, 2)
plt.hexbin(tau, S, mincnt=mnct, gridsize = gd, bins='log', cmap=plt.cm.jet)
plt.ylabel(r"$log_{10}$"+'(' + r"$S$" +')', fontsize=fs+6)
plt.xlabel(xlab, fontsize=fs+6)
#plt.ylim(0, 30)
plt.tick_params(axis='both', which='major', labelsize=fs)

#### E vs. Tau #################################################################
dat = dat.convert_objects(convert_numeric=True)#.dropna()
tau = np.log10(dat['width']/dat['flow.rate'])
E = dat['simpson.e']

fig.add_subplot(2, 2, 3)
plt.hexbin(tau, E, mincnt=mnct, gridsize = gd, bins='log', cmap=plt.cm.jet)
#plt.ylim(0.0, 1.0)
plt.ylabel('Evenness', fontsize=fs+6)
plt.xlabel(xlab, fontsize=fs+6)
plt.tick_params(axis='both', which='major', labelsize=fs)

#### W vs. Tau #################################################################
#dat = dat.convert_objects(convert_numeric=True).dropna()
tau = np.log10(dat['width']/dat['flow.rate'])
W = np.log10(dat['Whittakers.turnover'])

fig.add_subplot(2, 2, 4)
plt.hexbin(tau, W, mincnt=mnct, gridsize = gd, bins='log', cmap=plt.cm.jet)
plt.ylabel(r"$log_{10}$"+'(' + r"$\beta$" +')', fontsize=fs+6)
plt.xlabel(xlab, fontsize=fs+6)
plt.tick_params(axis='both', which='major', labelsize=fs)

#### Final Format and Save #####################################################
plt.subplots_adjust(wspace=0.4, hspace=0.4)
plt.savefig(mydir + '/results/figures/Fig2-heat.png', dpi=600, bbox_inches = "tight")
#plt.show()
