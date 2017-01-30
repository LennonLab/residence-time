from __future__ import division
import  matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
#import sys
from scipy.optimize import curve_fit
from math import pi


def xfrm(X, _max): return -np.log(_max-np.array(X))
#def ivrt(Y, _max): return _max-np.exp(-np.array(Y))

def laplace_reg(X, loc, scale, const):
    laplace = const*(1/(2*scale)) * np.exp((-np.abs(X - loc))/scale)
    return laplace

def norm_reg(X, mu, std, const):
    norm = const * (1/(np.sqrt(2*(std**2)*pi))) * np.exp(-((X - mu)**2)/(2*(std**2)))
    return norm


def plot_curves(X, Y, fig):
    X, Y = (np.array(t) for t in zip(*sorted(zip(X, Y))))

    Xi = xfrm(X, max(X)*1.05)
    bins = np.linspace(np.min(Xi), np.max(Xi)+1, 200)
    ii = np.digitize(Xi, bins)

    pcts = np.array([np.percentile(Y[ii==i], 99.9) for i in range(1, len(bins)) if len(Y[ii==i]) > 0])
    xran = np.array([np.mean(X[ii==i]) for i in range(1, len(bins)) if len(Y[ii==i]) > 0])

    mu, std, const = 4, 1, 1
    popt, pcov = curve_fit(norm_reg, xran, pcts, [mu, std, const])
    model = norm_reg(xran, *popt)
    #plt.plot(xran, model, color='b', lw=2, label='Gaussian')

    loc, scale, const = 4, 1, 1
    popt, pcov = curve_fit(laplace_reg, xran, pcts, [loc, scale, const])
    model = laplace_reg(xran, *popt)
    plt.plot(xran, model, color='w', lw=3, label='Laplace')
    plt.plot(xran, model, color='k', lw=1, label='Laplace')

    return fig


mydir = os.path.expanduser('~/GitHub/residence-time')
df = pd.read_csv(mydir + '/results/simulated_data/SimData.csv')

df2 = pd.DataFrame({'width' : df['width']})
df2['flow'] = df['flow.rate']
df2['tau'] = np.log10((df['height'] * df['length'] * df2['width'])/df2['flow'])

df2['N'] = df['total.abundance']
df2['S'] = df['species.richness']
df2['Prod'] = df['ind.production']
df2['E'] = np.log10(df['simpson.e'])
df2['W'] = np.log10(df['Whittakers.turnover'])
df2['Dorm'] = df['Percent.Dormant']


#### plot figure ###############################################################
xlab = r"$log_{10}$"+'(' + r"$\tau$" +')'
fs = 6 # fontsize
fig = plt.figure()

gd = 25
mnct = 0
binz = 'log'
radius = 2

#### N vs. Tau #################################################################
fig.add_subplot(3, 3, 1)

plt.hexbin(df2['tau'], df2['N'], mincnt=mnct, gridsize = gd, bins=binz, cmap=plt.cm.Greys)
#fig = plot_curves(df2['tau'], df2['N'], fig)
plt.ylabel(r"$log_{10}$"+'(' + r"$N$" + ')', fontsize=fs+3)
plt.xlabel(xlab, fontsize=fs+3)
#plt.ylim(1,2000)
plt.tick_params(axis='both', which='major', labelsize=fs)
#plt.text(1.1, 2, 'A', color = 'y', fontweight='bold')

#### production vs. Tau ########################################################
fig.add_subplot(3, 3, 2)

plt.hexbin(df2['tau'], df2['Prod'], mincnt=mnct, gridsize = gd, bins=binz, cmap=plt.cm.Greys)
#fig = plot_curves(df2['tau'], df2['Prod'], fig)
plt.ylabel(r"$log_{10}$"+'(' + r"$Productivity$" + ')', fontsize=fs+3)
plt.xlabel(xlab, fontsize=fs+3)
plt.tick_params(axis='both', which='major', labelsize=fs)
#plt.text(4.2, -0.25, 'B', color = 'y', fontweight='bold')

#### S vs. Tau #################################################################
fig.add_subplot(3, 3, 4)

plt.hexbin(df2['tau'], df2['S'], mincnt=mnct, gridsize = gd, bins=binz, cmap=plt.cm.Greys)
#fig = plot_curves(df2['tau'], df2['S'], fig)
plt.ylabel(r"$log_{10}$"+'(' + r"$S$" +')', fontsize=fs+3)
plt.xlabel(xlab, fontsize=fs+3)
plt.tick_params(axis='both', which='major', labelsize=fs)
#plt.text(1.1, 1.6, 'C', color = 'y', fontweight='bold')

#### E vs. Tau #################################################################
fig.add_subplot(3, 3, 5)

plt.hexbin(df2['tau'], df2['E'], mincnt=mnct, gridsize = gd, bins=binz, cmap=plt.cm.Greys)
plt.ylabel(r"$log_{10}$"+'(' + r"$Evenness$" +')', fontsize=fs+3)
plt.xlabel(xlab, fontsize=fs+3)
plt.tick_params(axis='both', which='major', labelsize=fs)
#plt.text(4.7, 0.3, 'D', color = 'y', fontweight='bold')

#### W vs. Tau #################################################################
ax5 = fig.add_subplot(3, 3, 7)

#df3 = df2[df2['W'] < 0]
#df3 = df3[df3['W'] < 0.6]

plt.hexbin(df2['tau'], df2['W'], mincnt=mnct, gridsize = 15, bins=binz, cmap=plt.cm.Greys)
plt.ylabel(r"$log_{10}$"+'(' + r"$\beta$" +')', fontsize=fs+3)
plt.xlabel(xlab, fontsize=fs+3)
plt.tick_params(axis='both', which='major', labelsize=fs)
#plt.text(1.1, -3.0, 'E', color = 'y', fontweight='bold')

#### dormancy vs. Tau ########################################################
fig.add_subplot(3, 3, 8)

#df3 = df2[df2['Dorm'] < 0]
#df3 = df3[df3['Dorm'] != -2]

plt.hexbin(df2['tau'], df2['Dorm'], mincnt=mnct, gridsize = gd, bins=binz, cmap=plt.cm.Greys)
plt.ylabel(r"$log_{10}$"+'(' + r"$Dormant$" +')', fontsize=fs+3)
plt.xlabel(xlab, fontsize=fs+3)
plt.tick_params(axis='both', which='major', labelsize=fs)
#plt.text(4.7, 0.2, 'F', color = 'y', fontweight='bold')

#### Final Format and Save #####################################################
plt.subplots_adjust(wspace=0.4, hspace=0.4)
plt.savefig(mydir + '/results/figures/Fig1-constrained.png', dpi=200, bbox_inches = "tight")
#plt.show()
plt.close()
