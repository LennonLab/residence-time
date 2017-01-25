from __future__ import division
import  matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import sys

mydir = os.path.expanduser('~/GitHub/residence-time')
sys.path.append(mydir+'/tools')
mydir2 = os.path.expanduser("~/")


def plot_dat(fig, x, y, clr, clims = [95, 75]):

    grain = 0.25
    xran = np.arange(0, 6, grain).tolist()

    binned = np.digitize(x, xran).tolist()
    bins = [list([]) for _ in xrange(len(xran))]

    for i, val in enumerate(binned):
        bins[val-1].append(y[i])

    for clim in clims:
        pct5 = []
        pct50 = []
        pct95 = []
        xran = []
        for i, _bin in enumerate(bins):
            if len(_bin) > 0:
                pct5.append(np.percentile(_bin, 100 - clim))
                pct95.append(np.percentile(_bin, clim))
                pct50.append(np.percentile(_bin, 50))

                x = (i+1)*grain
                xran.append(x)

        plt.fill_between(xran, pct5, pct95, facecolor= '0.2', alpha=0.3, lw=0.0)
        plt.plot(xran, pct50, ls='--', color = '0.3', lw = 1)
    return fig


df = pd.read_csv(mydir + '/results/simulated_data/SimData.csv')

df2 = pd.DataFrame({'width' : df['width']}) #.groupby(df['sim']).mean()})
df2['flow'] = df['flow.rate'] #.groupby(df['sim']).mean()
df2['tau'] = np.log10(df2['width']/df2['flow'])

df2['N'] = np.log10(df['total.abundance']) #.groupby(df['sim']).mean())
df2['S'] = np.log10(df['species.richness']) #.groupby(df['sim']).mean())
df2['AvgGrow'] = df['avg.per.capita.growth'] #.groupby(df['sim']).mean()
df2['AvgActDisp'] = df['avg.per.capita.active.dispersal'] #.groupby(df['sim']).mean()
df2['AvgMaint'] = df['avg.per.capita.maint'] #.groupby(df['sim']).mean())
df2['AvgRPF'] = df['avg.per.capita.RPF'] #.groupby(df['sim']).mean()
df2['AvgMF'] = df['avg.per.capita.MF'] #.groupby(df['sim']).mean()
df2['AvgEff'] = df['avg.per.capita.N.efficiency'] #.groupby(df['sim']).mean()

#### plot figure ###############################################################
xlab = r"$log_{10}$"+'(' + r"$\tau$" +')'
fs = 6 # fontsize
fig = plt.figure()

gd = 10
mnct = 0
mnct1 = 0
binz = 'log'
trans = 1

gc = 'r'
pc = 'b'

#### AvgGrow vs. Tau #################################################################
fig.add_subplot(3, 3, 1)

fig = plot_dat(fig, df2['tau'], df2['AvgGrow'], gc)
plt.ylabel('Growth rate', fontsize=fs+3)
plt.xlabel(xlab, fontsize=fs+3)
plt.tick_params(axis='both', which='major', labelsize=fs)

#plt.text(1.2, 1.005, 'Growth Syndrome', color = gc, fontsize = 10, fontweight='bold')


#### AvgActDisp vs. Tau #################################################################
fig.add_subplot(3, 3, 2)

fig = plot_dat(fig, df2['tau'], df2['AvgMaint'], pc)
plt.ylabel('Maintenance energy', fontsize=fs+3)
plt.xlabel(xlab, fontsize=fs+3)
plt.tick_params(axis='both', which='major', labelsize=fs)

#plt.text(0.7, -1.8, 'Persistence Syndrome', color = pc, fontsize = 10, fontweight='bold')


#### E vs. Tau #################################################################
fig.add_subplot(3, 3, 4)

fig = plot_dat(fig, df2['tau'], df2['AvgActDisp'], gc)
plt.ylabel('Active disperal', fontsize=fs+3)
plt.xlabel(xlab, fontsize=fs+3)
plt.tick_params(axis='both', which='major', labelsize=fs)

#### AvgEff vs. Tau #################################################################
fig.add_subplot(3, 3, 5)

fig = plot_dat(fig, df2['tau'], df2['AvgRPF'], pc)
plt.ylabel('Probability of\nrandom resuscitation', fontsize=fs+3)
plt.xlabel(xlab, fontsize=fs+3)
plt.tick_params(axis='both', which='major', labelsize=fs)

#### AvgRPF vs. Tau #################################################################
fig.add_subplot(3, 3, 7)

fig = plot_dat(fig, df2['tau'], df2['AvgEff'], gc)
plt.ylabel('Resource\nspecialization', fontsize=fs+3)
plt.xlabel(xlab, fontsize=fs+3)
plt.tick_params(axis='both', which='major', labelsize=fs)

#### AvgRPF vs. Tau #################################################################
fig.add_subplot(3, 3, 8)

fig = plot_dat(fig, df2['tau'], df2['AvgMF'], pc)
plt.ylabel('Maintenance energy\nreduction factor', fontsize=fs+3)
plt.xlabel(xlab, fontsize=fs+3)
plt.tick_params(axis='both', which='major', labelsize=fs)
#plt.text(2.1, 4, 'F', color = 'y', fontweight='bold')

#### Final Format and Save #####################################################
plt.subplots_adjust(wspace=0.45, hspace=0.4)
plt.savefig(mydir + '/results/figures/Fig2-Hulls.png', dpi=600, bbox_inches = "tight")
#plt.show()
#plt.close()
