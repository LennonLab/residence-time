from __future__ import division
import  matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
from scipy.optimize import curve_fit
from math import pi


mydir = os.path.expanduser('~/GitHub/residence-time2/Emergence')
tools = os.path.expanduser(mydir + "/tools")


def xfrm(X, _max): return -_max-np.array(X)

def laplace_reg(X, loc, scale, const):
    laplace = const*(1/(2*scale)) * np.exp((-np.abs(X - loc))/scale)
    return laplace

def norm_reg(X, mu, std, const):
    norm = const * (1/(np.sqrt(2*(std**2)*pi))) * np.exp(-((X - mu)**2)/(2*(std**2)))
    return norm


def assigncolor(xs):
    cDict = {}
    clrs = []
    for x in xs:
        if x not in cDict:
            if x <= 1: c = 'r'
            elif x <= 2: c = 'Orange'
            elif x <= 3: c = 'Green'
            elif x <= 4: c = 'DodgerBlue'
            elif x <= 5: c = 'Plum'
            else: c = 'Purple'
            cDict[x] = c

        clrs.append(cDict[x])
    return clrs


def curve(fig, x, y):
    b = 100
    x, y = (np.array(t) for t in zip(*sorted(zip(x, y))))
    Xi = xfrm(x, max(x))
    bins = np.linspace(np.min(Xi), np.max(Xi)+1, b)
    ii = np.digitize(Xi, bins)

    pcts = np.array([np.percentile(y[ii==i], ci) for i in range(1, len(bins)) if len(y[ii==i]) > 0])
    xran = np.array([np.mean(x[ii==i]) for i in range(1, len(bins)) if len(y[ii==i]) > 0])

    loc, scale, const = float(v), 1, 1
    popt, pcov = curve_fit(laplace_reg, xran, pcts, [loc, scale, const])
    model = laplace_reg(xran, *popt)
    plt.plot(xran, model, color='r', lw=0.5)

    mu, std, const = float(v), 1, 1
    popt, pcov = curve_fit(norm_reg, xran, pcts, [mu, std, const])
    model = norm_reg(xran, *popt)
    plt.plot(xran, model, color='b', lw=0.5)

    return fig



df = pd.read_csv(mydir + '/ModelTypes/Costs-Growth/results/simulated_data/SimData.csv')

df2 = pd.DataFrame({'N' : df['active.total.abundance'].groupby(df['sim']).mean()})
df2['P'] = df['ind.production'].groupby(df['sim']).mean()

df2['tau'] = df['V'].groupby(df['sim']).mean()/df['Q'].groupby(df['sim']).mean()

df2['size'] = df['active.avg.per.capita.size'].groupby(df['sim']).mean()
df2['G'] = df['active.growth'].groupby(df['sim']).mean()
df2['D'] = df['active.dispersal'].groupby(df['sim']).mean()
df2['M'] = df['active.maint'].groupby(df['sim']).mean()

df2['x1'] = df2['D'] * df2['tau'] * 0.2
df2['x2'] = df2['M'] * df2['tau'] * 0.2
df2['x3'] = df2['G'] * df2['tau'] * 0.2

clrs = assigncolor(np.log10(df2['tau']))

v = 0
fs = 8
fig = plt.figure()
sz = 0.5
w = 0.75
b = 100
ci = 99
fit = 'n'
yscale = 'log'

#### plot figures ##############################################################
xlab =  r"$log_{10}(\mu*\tau)$"

#### N vs. Tau #################################################################
fig.add_subplot(3, 3, 1)
x = np.log10(df2['x1'])
y = df2['N']
if yscale == 'log': y = np.log10(y)

plt.axvline(v, color='k', ls=':', lw = w)
plt.scatter(x, y, s = sz, color=clrs, linewidths=0.0, edgecolor=None)
plt.ylabel(r"$log_{10}(N_{a})$", fontsize=fs)
plt.xlabel(xlab, fontsize=fs)
plt.tick_params(axis='both', which='major', labelsize=fs-2)
if fit == 'y': fig = curve(fig, x, y)

#### production vs. Tau ########################################################
fig.add_subplot(3, 3, 2)
x = np.log10(df2['x1'])
y = df2['P']
if yscale == 'log': y = np.log10(y)

plt.axvline(v, color='k', ls=':', lw = w)
plt.scatter(x, y, s = sz, color=clrs, linewidths=0.0, edgecolor=None)
plt.ylabel(r"$log_{10}(P)$", fontsize=fs)
plt.xlabel(xlab, fontsize=fs)
plt.tick_params(axis='both', which='major', labelsize=fs-2)
if fit == 'y': fig = curve(fig, x, y)



#### plot figures ##############################################################
xlab =  r"$log_{10}(B*\tau)$"

#### N vs. Tau #################################################################
fig.add_subplot(3, 3, 4)
x = np.log10(df2['x2'])
y = df2['N']
if yscale == 'log': y = np.log10(y)

plt.axvline(v, color='k', ls=':', lw = w)
plt.scatter(x, y, s = sz, color=clrs, linewidths=0.0, edgecolor=None)
plt.ylabel(r"$log_{10}(N_{a})$", fontsize=fs)
plt.xlabel(xlab, fontsize=fs)
plt.tick_params(axis='both', which='major', labelsize=fs-2)
if fit == 'y': fig = curve(fig, x, y)

#### production vs. Tau ########################################################
fig.add_subplot(3, 3, 5)
x = np.log10(df2['x2'])
y = df2['P']
if yscale == 'log': y = np.log10(y)

plt.axvline(v, color='k', ls=':', lw = w)
plt.scatter(x, y, s = sz, color=clrs, linewidths=0.0, edgecolor=None)
plt.ylabel(r"$log_{10}(P)$", fontsize=fs)
plt.xlabel(xlab, fontsize=fs)
plt.tick_params(axis='both', which='major', labelsize=fs-2)
if fit == 'y': fig = curve(fig, x, y)


#### plot figures ##############################################################
xlab =  r"$log_{10}(G*\tau)$"

#### N vs. Tau #################################################################
fig.add_subplot(3, 3, 7)
x = np.log10(df2['x3'])
y = df2['N']
if yscale == 'log': y = np.log10(y)

plt.axvline(v, color='k', ls=':', lw = w)
plt.scatter(x, y, s = sz, color=clrs, linewidths=0.0, edgecolor=None)
plt.ylabel(r"$log_{10}(N_{a})$", fontsize=fs)
plt.xlabel(xlab, fontsize=fs)
plt.tick_params(axis='both', which='major', labelsize=fs-2)
if fit == 'y': fig = curve(fig, x, y)

#### production vs. Tau ########################################################
fig.add_subplot(3, 3, 8)
x = np.log10(df2['x3'])
y = df2['P']
if yscale == 'log': y = np.log10(y)

plt.axvline(v, color='k', ls=':', lw = w)
plt.scatter(x, y, s = sz, color=clrs, linewidths=0.0, edgecolor=None)
plt.ylabel(r"$log_{10}(P)$", fontsize=fs)
plt.xlabel(xlab, fontsize=fs)
plt.tick_params(axis='both', which='major', labelsize=fs-2)
if fit == 'y': fig = curve(fig, x, y)


#### Final Format and Save #####################################################
plt.subplots_adjust(wspace=0.5, hspace=0.5)
plt.savefig(mydir + '/ModelTypes/Costs-Growth/results/figures/BMR_vs_Tau-3x3.png', dpi=200, bbox_inches = "tight")
plt.close()
