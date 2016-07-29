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
#dat.drop(dat.index[:1000], inplace=True)

dat = dat[dat['motion'] == '\'fluid\'']

dat = dat[np.isfinite(dat['total.abundance'])]
dat = dat[dat['total.abundance'] > 0]
dat = dat[dat['species.richness'] > 0]
dat = dat[dat['simpson.e'] < 1]
dat = dat[dat['Whittakers.turnover'] > 0]
#dat = dat[dat['barriers'] == 0]

tau = np.log10(dat['width']/dat['flow.rate']).tolist()
N = np.log10(dat['total.abundance']).tolist()

AvgGrow = dat['avg.per.capita.growth'].tolist()
AvgActDisp = dat['avg.per.capita.active.dispersal'].tolist()
AvgMaint = dat['avg.per.capita.maint'].tolist()
AvgNEff = dat['avg.per.capita.N.efficiency'].tolist()


#### plot figure ###############################################################
#xlab = r"$log_{10}$"+'(' + r"$\tau$" +')'
fs = 8 # fontsize
fig = plt.figure()

aboveAvgGrow = []
belowAvgGrow = []

aboveAvgActDisp = []
belowAvgActDisp = []

aboveAvgMaint = []
belowAvgMaint = []

aboveAvgNEff = []
belowAvgNEff = []

minTau = 2.5
maxTau = 4.3

for i, val in enumerate(tau):
    if val < minTau:
        belowAvgGrow.append(AvgGrow[i])
        belowAvgNEff.append(AvgNEff[i])
        belowAvgActDisp.append(AvgActDisp[i])
        belowAvgMaint.append(AvgMaint[i])


    elif val > maxTau:
        aboveAvgGrow.append(AvgGrow[i])
        aboveAvgNEff.append(AvgNEff[i])
        aboveAvgActDisp.append(AvgActDisp[i])
        aboveAvgMaint.append(AvgMaint[i])


print min(tau), max(tau)
print len(tau), len(belowAvgGrow), len(aboveAvgGrow)

ax = fig.add_subplot(2, 2, 1)
ax.set_title('average growth')
plt.boxplot([np.log10(aboveAvgGrow), np.log10(belowAvgGrow)], labels=['slow', 'fast'])
plt.tick_params(axis='both', which='major', labelsize=fs)
#plt.ylim(-2.0, -0.5)

ax = fig.add_subplot(2, 2, 2)
ax.set_title('avg act disp')
plt.boxplot([aboveAvgActDisp, belowAvgActDisp], labels=['slow', 'fast'])
plt.tick_params(axis='both', which='major', labelsize=fs)
#plt.ylim(0.0, 0.2)

ax = fig.add_subplot(2, 2, 3)
ax.set_title('avg maint')
plt.boxplot([aboveAvgMaint, belowAvgMaint], labels=['slow', 'fast'])
plt.tick_params(axis='both', which='major', labelsize=fs)
#plt.ylim(0.0, 0.001)

ax = fig.add_subplot(2, 2, 4)
ax.set_title('Efficiency')
plt.boxplot([aboveAvgNEff, belowAvgNEff], labels=['slow', 'fast'])
plt.tick_params(axis='both', which='major', labelsize=fs)
#plt.ylim(0.0, 0.2)

#plt.ylabel(r"$log_{10}$"+'(' + r"$N$" +')', fontsize=fs+6)
#plt.xlabel(xlab, fontsize=fs+6)
plt.tick_params(axis='both', which='major', labelsize=fs)

#### Final Format and Save #####################################################
plt.subplots_adjust(wspace=0.4, hspace=0.4)
plt.savefig(mydir + '/results/figures/Fig3-Tau.png', dpi=600, bbox_inches = "tight")
#plt.show()
