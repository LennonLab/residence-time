from __future__ import division
import  matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import sys


mydir = os.path.expanduser('~/GitHub/residence-time')
sys.path.append(mydir+'/tools')
mydir2 = os.path.expanduser("~/")


def plot_dat(fig, x, y, clrs = ['0.2', '0.2'], clim = [95, 75]):

    grain = 0.25
    xran = np.arange(0, 6, grain).tolist()

    binned = np.digitize(x, xran).tolist()
    bins = [list([]) for _ in xrange(len(xran))]

    for i, val in enumerate(binned):
        bins[val-1].append(y[i])

    for j, ci in enumerate(clim):

        pct5 = []
        pct50 = []
        pct95 = []
        xran = []
        for i, _bin in enumerate(bins):
            if len(_bin) > 0:
                pct5.append(np.percentile(_bin, 100 - ci))
                pct95.append(np.percentile(_bin, ci))
                pct50.append(np.percentile(_bin, 50))

                x = (i+1)*grain
                xran.append(x)

        plt.fill_between(xran, pct5, pct95, facecolor= clrs[j], alpha=0.4, lw=0.0)
        plt.plot(xran, pct50, ls='--', color = clrs[j], lw = 1)
    return fig


df = pd.read_csv(mydir + '/results/simulated_data/SimData.csv')

df2 = pd.DataFrame({'width' : df['width']}) #.groupby(df['sim']).mean()})
df2['flow'] = df['flow.rate'] #.groupby(df['sim']).mean()
df2['tau'] = np.log10((df['height'] * df['length'] * df2['width'])/df2['flow'])

df2['N'] = np.log10(df['total.abundance']) #.groupby(df['sim']).mean())
df2['S'] = np.log10(df['species.richness']) #.groupby(df['sim']).mean()

df2['Prod'] = np.log10(df['ind.production']+1) #.groupby(df['sim']).mean())
df2['E'] = df['simpson.e'] #.groupby(df['sim']).mean()
df2['W'] = df['Whittakers.turnover'] #.groupby(df['sim']).mean()
df2['Dorm'] = df['Percent.Dormant'] #.groupby(df['sim']).mean()


#### plot figure ###############################################################
xlab = r"$log_{10}$"+'(' + r"$\tau$" +')'
fs = 6 # fontsize
fig = plt.figure()

gd = 10
mnct = 0
mnct1 = 0
binz = 'log'
trans = 1

c = '0.3'

#### N vs. Tau #################################################################
fig.add_subplot(3, 3, 1)

fig = plot_dat(fig, df2['tau'], df2['N'])
plt.ylabel(r"$log_{10}$"+'(' + r"$N$" +')', fontsize=fs+3)
plt.xlabel(xlab, fontsize=fs+3)
plt.tick_params(axis='both', which='major', labelsize=fs)
#plt.text(1.1, 2, 'A', color = 'y', fontweight='bold')

#### Productivity vs. Tau #################################################################
fig.add_subplot(3, 3, 2)

fig = plot_dat(fig, df2['tau'], df2['Prod'])
plt.ylabel(r"$log_{10}$"+'(Productivity)', fontsize=fs+3)
plt.xlabel(xlab, fontsize=fs+3)
plt.tick_params(axis='both', which='major', labelsize=fs)
#plt.text(4.2, -0.25, 'B', color = 'y', fontweight='bold')

#### S vs. Tau #################################################################
fig.add_subplot(3, 3, 4)

fig = plot_dat(fig, df2['tau'], df2['S'])
plt.ylabel(r"$log_{10}$"+'(' + r"$S$" +')', fontsize=fs+3)
plt.xlabel(xlab, fontsize=fs+3)
plt.tick_params(axis='both', which='major', labelsize=fs)
#plt.text(1.1, 1.6, 'C', color = 'y', fontweight='bold')

#### Evenness vs. Tau #################################################################
fig.add_subplot(3, 3, 5)

fig = plot_dat(fig, df2['tau'], df2['E'])
plt.ylabel('Evenness', fontsize=fs+3)
plt.xlabel(xlab, fontsize=fs+3)
plt.tick_params(axis='both', which='major', labelsize=fs)
#plt.text(4.7, 0.3, 'D', color = 'y', fontweight='bold')

#### Beta diversity vs. Tau #################################################################
fig.add_subplot(3, 3, 7)

fig = plot_dat(fig, df2['tau'], df2['W'])
plt.ylabel(r"$log_{10}$"+'(' + r"$\beta$" +')', fontsize=fs+3)
plt.ylabel(r"$\beta_{w}$", fontsize=fs+5)
plt.xlabel(xlab, fontsize=fs+3)
plt.tick_params(axis='both', which='major', labelsize=fs)
#plt.text(1.1, -3.0, 'E', color = 'y', fontweight='bold')

#### % Dormant vs. Tau #################################################################
fig.add_subplot(3, 3, 8)

fig = plot_dat(fig, df2['tau'], df2['Dorm'])
#plt.ylabel(r"$log_{10}$"+'(% Dormancy)', fontsize=fs+3)
plt.ylabel('% Dormancy', fontsize=fs+3)
plt.xlabel(xlab, fontsize=fs+3)
plt.tick_params(axis='both', which='major', labelsize=fs)
#plt.text(4.7, 0.2, 'F', color = 'y', fontweight='bold')


#### Final Format and Save #####################################################
plt.subplots_adjust(wspace=0.4, hspace=0.4)
plt.savefig(mydir + '/results/figures/Fig1-Hulls.png', dpi=600, bbox_inches = "tight")
#plt.show()
plt.close()
