from __future__ import division
import  matplotlib.pyplot as plt
import pandas as pd
from random import randint
import numpy as np
import sys
import os

import statsmodels.api as sm
from scipy import interpolate
from scipy.interpolate import Rbf, InterpolatedUnivariateSpline

mydir = os.path.expanduser('~/GitHub/residence-time')
dat = pd.read_csv(mydir + '/results/simulated_data/SimData.csv')
#dat = dat[dat['total.abundance'] > 10]

#### plot figure ###############################################################
xlab = r"$log_{10}$"+'(' + r"$\tau$" +')'
fs = 6 # fontsize
fig = plt.figure()

_lw = 0.5
w = 1
p = 1
#### N vs. Tau #################################################################
ax1 = fig.add_subplot(3, 3, 1)
ax2 = fig.add_subplot(3, 3, 2)
ax3 = fig.add_subplot(3, 3, 4)
ax4 = fig.add_subplot(3, 3, 5)
ax5 = fig.add_subplot(3, 3, 7)
ax6 = fig.add_subplot(3, 3, 8)


sims = list(set(dat['sim'].tolist()))

for sim in sims:
    print 'sim:', sim

    df = dat[dat['sim'] == sim]
    df2 = pd.DataFrame({'width' : df['width'].groupby(df['ct']).mean()})

    df2['flow'] = df['flow.rate'].groupby(df['ct']).mean()
    df2['tau'] = np.log10(df2['width']**p/df2['flow'])

    df2['N'] = df['total.abundance'].groupby(df['ct']).mean()
    df2['Prod'] = df['ind.production'].groupby(df['ct']).mean()
    df2['S'] = df['species.richness'].groupby(df['ct']).mean()
    df2['E'] = df['simpson.e'].groupby(df['ct']).mean()
    df2['W'] = df['Whittakers.turnover'].groupby(df['ct']).mean()
    df2['Dorm'] = df['Percent.Dormant'].groupby(df['ct']).mean()

    df2 = df2.replace([np.inf, -np.inf], np.nan).dropna()
    r1 = lambda: randint(0,255)
    r2 = lambda: randint(0,255)
    r3 = lambda: randint(0,255)
    clr = '#%02X%02X%02X' % (r1(),r2(),r3())

    x = df2['tau'].tolist()
    y = df2['N'].tolist()
    ax1.plot(x, y, lw=_lw, color=clr)
    ax1.set_xlabel(xlab, fontsize=fs+3)
    ax1.tick_params(axis='both', labelsize=fs)
    ax1.set_ylabel(r"$log_{10}$"+'(' + r"$N$" + ')', fontsize=fs+3)

    x = df2['tau'].tolist()
    y = df2['Prod'].tolist()
    ax2.plot(x, y, lw=_lw, color=clr)
    ax2.set_xlabel(xlab, fontsize=fs+3)
    ax2.tick_params(axis='both', labelsize=fs)
    ax2.set_ylabel(r"$log_{10}$"+'(' + r"$Productivity$" + ')', fontsize=fs+3)

    x = df2['tau']
    y = df2['S']
    ax3.plot(x, y, lw=_lw, color=clr)
    ax3.set_xlabel(xlab, fontsize=fs+3)
    ax3.tick_params(axis='both', labelsize=fs)
    ax3.set_ylabel(r"$log_{10}$"+'(' + r"$S$" +')', fontsize=fs+3)


    x = df2['tau']
    y = df2['E']
    ax4.plot(x, y, lw=_lw, color=clr)
    ax4.set_xlabel(xlab, fontsize=fs+3)
    ax4.tick_params(axis='both', labelsize=fs)
    ax4.set_ylabel(r"$log_{10}$"+'(' + r"$Evenness$" +')', fontsize=fs+3)

    x = df2['tau']
    y = df2['W']
    ax5.plot(x, y, lw=_lw, color=clr)
    ax5.set_xlabel(xlab, fontsize=fs+3)
    ax5.tick_params(axis='both', labelsize=fs)
    ax5.set_ylabel(r"$log_{10}$"+'(' + r"$\beta$" +')', fontsize=fs+3)

    x = df2['tau']
    y = df2['Dorm']
    ax6.plot(x, y, lw=_lw, color=clr)
    ax6.set_xlabel(xlab, fontsize=fs+3)
    ax6.tick_params(axis='both', labelsize=fs)
    ax6.set_ylabel(r"$log_{10}$"+'(' + r"$Dormant$" +')', fontsize=fs+3)


#### Final Format and Save #####################################################
plt.subplots_adjust(wspace=0.4, hspace=0.4)
plt.savefig(mydir + '/results/figures/Fig1.png', dpi=200, bbox_inches = "tight")
plt.close()










#### plot figure ###############################################################
xlab = r"$log_{10}$"+'(' + r"$\tau$" +')'
fs = 6 # fontsize
fig = plt.figure()

#### N vs. Tau #################################################################
ax1 = fig.add_subplot(3, 3, 1)
ax2 = fig.add_subplot(3, 3, 2)
ax3 = fig.add_subplot(3, 3, 4)
ax4 = fig.add_subplot(3, 3, 5)
ax5 = fig.add_subplot(3, 3, 7)
ax6 = fig.add_subplot(3, 3, 8)


sims = list(set(dat['sim'].tolist()))

for sim in sims:
    print 'sim:', sim

    df = dat[dat['sim'] == sim]
    df2 = pd.DataFrame({'width' : df['width'].groupby(df['ct']).mean()})

    df2['flow'] = df['flow.rate'].groupby(df['ct']).mean()
    df2['tau'] = np.log10(df2['width']**p/df2['flow'])

    df2['Grow'] = df['active.avg.per.capita.growth'].groupby(df['ct']).mean()
    df2['Maint'] = np.log10(df['dormant.avg.per.capita.maint']).groupby(df['ct']).mean()
    df2['Disp'] = df['active.avg.per.capita.active.dispersal'].groupby(df['ct']).mean()
    df2['RPF'] = df['dormant.avg.per.capita.rpf'].groupby(df['ct']).mean()
    df2['Eff'] = df['active.avg.per.capita.efficiency'].groupby(df['ct']).mean()
    df2['MF'] = df['dormant.avg.per.capita.mf'].groupby(df['ct']).mean()


    df2 = df2.replace([np.inf, -np.inf], np.nan).dropna()
    r1 = lambda: randint(0,255)
    r2 = lambda: randint(0,255)
    r3 = lambda: randint(0,255)
    clr = '#%02X%02X%02X' % (r1(),r2(),r3())

    x = df2['tau'].tolist()
    y = df2['Grow'].tolist()
    ax1.plot(x, y, lw=_lw, color=clr)
    ax1.set_xlabel(xlab, fontsize=fs+3)
    ax1.tick_params(axis='both', labelsize=fs)
    ax1.set_ylabel(r"$log_{10}$"+'(' + r"$Growth rate$" + ')', fontsize=fs+3)

    x = df2['tau'].tolist()
    y = df2['Maint'].tolist()
    ax2.plot(x, y, lw=_lw, color=clr)
    ax2.set_xlabel(xlab, fontsize=fs+3)
    ax2.tick_params(axis='both', labelsize=fs)
    ax2.set_ylabel(r"$log_{10}$"+'(' + r"$Maintenance energy$" + ')', fontsize=fs+3)

    x = df2['tau']
    y = df2['Disp']
    ax3.plot(x, y, lw=_lw, color=clr)
    ax3.set_xlabel(xlab, fontsize=fs+3)
    ax3.tick_params(axis='both', labelsize=fs)
    ax3.set_ylabel('Active disperal rate', fontsize=fs+2)


    x = df2['tau']
    y = df2['RPF']
    ax4.plot(x, y, lw=_lw, color=clr)
    ax4.set_xlabel(xlab, fontsize=fs+3)
    ax4.tick_params(axis='both', labelsize=fs)
    ax4.set_ylabel('Random resuscitation\nfrom dormancy, ' + r"$log_{10}$", fontsize=fs+2)

    x = df2['tau']
    y = df2['Eff']
    ax5.plot(x, y, lw=_lw, color=clr)
    ax5.set_xlabel(xlab, fontsize=fs+3)
    ax5.tick_params(axis='both', labelsize=fs)
    ax5.set_ylabel('Resource specialization', fontsize=fs+2)

    x = df2['tau']
    y = df2['MF']
    ax6.plot(x, y, lw=_lw, color=clr)
    ax6.set_xlabel(xlab, fontsize=fs+3)
    ax6.tick_params(axis='both', labelsize=fs)
    ax6.set_ylabel('Decrease of maintenance\nenergy when dormant, ' + r"$log_{10}$", fontsize=fs+2)


#### Final Format and Save #####################################################
plt.subplots_adjust(wspace=0.4, hspace=0.4)
plt.savefig(mydir + '/results/figures/Fig2.png', dpi=200, bbox_inches = "tight")
plt.close()
