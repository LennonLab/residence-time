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

#### plot figure ###############################################################
xlab = r"$log_{10}$"+'(' + r"$\tau$" +')'
fs = 6 # fontsize
fig = plt.figure()

w = 1

#### N vs. Tau #################################################################
fig.add_subplot(3, 3, 1)

sims = list(set(dat['sim'].tolist()))

for sim in sims:
    print 'sim:', sim

    df = dat[dat['sim'] == sim]
    df2 = pd.DataFrame({'width' : df['width'].groupby(df['ct']).mean()})
    df2['flow'] = df['flow.rate'].groupby(df['ct']).mean()
    df2['tau'] = np.log10(df2['width']**3/df2['flow'])

    df2['N'] = np.log10(df['total.abundance']).groupby(df['ct']).mean()
    df2['sim'] = df['sim'].groupby(df['ct']).max()

    df2 = df2.replace([np.inf, -np.inf], np.nan).dropna()

    r1 = lambda: randint(0,255)
    r2 = lambda: randint(0,255)
    r3 = lambda: randint(0,255)
    clr = '#%02X%02X%02X' % (r1(),r2(),r3())

    x = df2['tau'].tolist()
    y = df2['N'].tolist()

    if len(x) < 5: continue

    #f = interpolate.interp1d(x, y, kind='quadratic')
    #newx = np.arange(min(x), max(x), 0.01)
    #newy = f(newx)
    #plt.plot(newx, newy, color=clr)

    ius = InterpolatedUnivariateSpline(x, y)
    xi = np.linspace(min(x), max(x), 101)
    yi = ius(xi)
    plt.plot(xi, yi, color=clr)

    #lowess = sm.nonparametric.lowess(y, x, frac=0.1)
    #plt.plot(lowess[:, 0], lowess[:, 1])

plt.ylabel(r"$log_{10}$"+'(' + r"$N$" + ')', fontsize=fs+3)
plt.xlabel(xlab, fontsize=fs+3)
plt.tick_params(axis='both', which='major', labelsize=fs)


#### production vs. Tau ########################################################
fig.add_subplot(3, 3, 2)

for sim in sims:
    print 'sim:', sim

    df = dat[dat['sim'] == sim]
    df2 = pd.DataFrame({'width' : df['width'].groupby(df['ct']).mean()})
    df2['flow'] = df['flow.rate'].groupby(df['ct']).mean()
    df2['tau'] = np.log10(df2['width']**3/df2['flow'])
    df2['Prod'] = df['ind.production'].groupby(df['ct']).mean()

    #df2 = df2.replace([np.inf, -np.inf], np.nan).dropna()

    r1 = lambda: randint(0,255)
    r2 = lambda: randint(0,255)
    r3 = lambda: randint(0,255)
    clr = '#%02X%02X%02X' % (r1(),r2(),r3())

    x = df2['tau']#.tolist()
    y = df2['Prod']#.tolist()
    if len(x) < 5: continue

    #f = interpolate.interp1d(x, y, kind='slinear')
    #newx = np.arange(min(x), max(x), 0.01)
    #newy = f(newx)
    #plt.plot(newx, newy, color=clr)

    ius = InterpolatedUnivariateSpline(x, y)
    xi = np.linspace(min(x), max(x), 101)
    yi = ius(xi)
    plt.plot(xi, yi, color=clr)

    #lowess = sm.nonparametric.lowess(y, x, frac=0.1)
    #plt.plot(lowess[:, 0], lowess[:, 1])

plt.ylabel(r"$log_{10}$"+'(' + r"$Productivity$" + ')', fontsize=fs+3)
plt.xlabel(xlab, fontsize=fs+3)
plt.tick_params(axis='both', which='major', labelsize=fs)


#### S vs. Tau #################################################################
fig.add_subplot(3, 3, 4)

for sim in sims:
    print 'sim:', sim

    df = dat[dat['sim'] == sim]
    df2 = pd.DataFrame({'width' : df['width'].groupby(df['ct']).mean()})
    df2['flow'] = df['flow.rate'].groupby(df['ct']).mean()
    df2['tau'] = np.log10(df2['width']**3/df2['flow'])
    df2['S'] = np.log10(df['species.richness']).groupby(df['ct']).mean()

    #df2 = df2.replace([np.inf, -np.inf], np.nan).dropna()

    r1 = lambda: randint(0,255)
    r2 = lambda: randint(0,255)
    r3 = lambda: randint(0,255)
    clr = '#%02X%02X%02X' % (r1(),r2(),r3())

    x = df2['tau']#.tolist()
    y = df2['S']#.tolist()
    if len(x) < 5: continue

    #f = interpolate.interp1d(x, y, kind='slinear')
    #newx = np.arange(min(x), max(x), 0.01)
    #newy = f(newx)
    #plt.plot(newx, newy, color=clr)

    ius = InterpolatedUnivariateSpline(x, y)
    xi = np.linspace(min(x), max(x), 101)
    yi = ius(xi)
    plt.plot(xi, yi, color=clr)

    #lowess = sm.nonparametric.lowess(y, x, frac=0.1)
    #plt.plot(lowess[:, 0], lowess[:, 1])

plt.ylabel(r"$log_{10}$"+'(' + r"$S$" +')', fontsize=fs+3)
plt.xlabel(xlab, fontsize=fs+3)
plt.tick_params(axis='both', which='major', labelsize=fs)


#### E vs. Tau #################################################################
fig.add_subplot(3, 3, 5)

for sim in sims:
    print 'sim:', sim

    df = dat[dat['sim'] == sim]
    df2 = pd.DataFrame({'width' : df['width'].groupby(df['ct']).mean()})
    df2['flow'] = df['flow.rate'].groupby(df['ct']).mean()
    df2['tau'] = np.log10(df2['width']**3/df2['flow'])
    df2['E'] = df['simpson.e'].groupby(df['ct']).mean()

    #df2 = df2.replace([np.inf, -np.inf], np.nan).dropna()

    r1 = lambda: randint(0,255)
    r2 = lambda: randint(0,255)
    r3 = lambda: randint(0,255)
    clr = '#%02X%02X%02X' % (r1(),r2(),r3())


    x = df2['tau']#.tolist()
    y = df2['E']#.tolist()
    if len(x) < 5: continue

    #f = interpolate.interp1d(x, y, kind='slinear')
    #newx = np.arange(min(x), max(x), 0.01)
    #newy = f(newx)
    #plt.plot(newx, newy, color=clr)

    ius = InterpolatedUnivariateSpline(x, y)
    xi = np.linspace(min(x), max(x), 101)
    yi = ius(xi)
    plt.plot(xi, yi, color=clr)

    #lowess = sm.nonparametric.lowess(y, x, frac=0.1)
    #plt.plot(lowess[:, 0], lowess[:, 1])


plt.ylabel(r"$log_{10}$"+'(' + r"$Evenness$" +')', fontsize=fs+3)
plt.xlabel(xlab, fontsize=fs+3)
plt.tick_params(axis='both', which='major', labelsize=fs)
#plt.text(4.7, 0.3, 'D', color = 'y', fontweight='bold')


#### W vs. Tau #################################################################
ax5 = fig.add_subplot(3, 3, 7)

for sim in sims:
    print 'sim:', sim

    df = dat[dat['sim'] == sim]
    df2 = pd.DataFrame({'width' : df['width'].groupby(df['ct']).mean()})
    df2['flow'] = df['flow.rate'].groupby(df['ct']).mean()
    df2['tau'] = np.log10(df2['width']**3/df2['flow'])
    df2['W'] = df['Whittakers.turnover'].groupby(df['ct']).mean()

    #df2 = df2.replace([np.inf, -np.inf], np.nan).dropna()

    r1 = lambda: randint(0,255)
    r2 = lambda: randint(0,255)
    r3 = lambda: randint(0,255)
    clr = '#%02X%02X%02X' % (r1(),r2(),r3())

    x = df2['tau']#.tolist()
    y = df2['W']#.tolist()
    if len(x) < 5: continue

    #f = interpolate.interp1d(x, y, kind='slinear')
    #newx = np.arange(min(x), max(x), 0.01)
    #newy = f(newx)
    #plt.plot(newx, newy, color=clr)

    ius = InterpolatedUnivariateSpline(x, y)
    xi = np.linspace(min(x), max(x), 101)
    yi = ius(xi)
    plt.plot(xi, yi, color=clr)

    #lowess = sm.nonparametric.lowess(y, x, frac=0.1)
    #plt.plot(lowess[:, 0], lowess[:, 1])


plt.ylabel(r"$log_{10}$"+'(' + r"$\beta$" +')', fontsize=fs+3)
plt.xlabel(xlab, fontsize=fs+3)
plt.tick_params(axis='both', which='major', labelsize=fs)
#plt.text(1.1, -3.0, 'E', color = 'y', fontweight='bold')


#### dormancy vs. Tau ########################################################
fig.add_subplot(3, 3, 8)

for sim in sims:
    print 'sim:', sim

    df = dat[dat['sim'] == sim]
    df2 = pd.DataFrame({'width' : df['width'].groupby(df['ct']).mean()})
    df2['flow'] = df['flow.rate'].groupby(df['ct']).mean()
    df2['tau'] = np.log10(df2['width']**3/df2['flow'])
    df2['Dorm'] = df['Percent.Dormant'].groupby(df['ct']).mean()

    #df2 = df2.replace([np.inf, -np.inf], np.nan).dropna()

    r1 = lambda: randint(0,255)
    r2 = lambda: randint(0,255)
    r3 = lambda: randint(0,255)
    clr = '#%02X%02X%02X' % (r1(),r2(),r3())

    x = df2['tau']#.tolist()
    y = df2['Dorm']#.tolist()
    if len(x) < 5: continue

    #f = interpolate.interp1d(x, y, kind='slinear')
    #newx = np.arange(min(x), max(x), 0.01)
    #newy = f(newx)
    #plt.plot(newx, newy, color=clr)

    ius = InterpolatedUnivariateSpline(x, y)
    xi = np.linspace(min(x), max(x), 101)
    yi = ius(xi)
    plt.plot(xi, yi, color=clr)

    #lowess = sm.nonparametric.lowess(y, x, frac=0.1)
    #plt.plot(lowess[:, 0], lowess[:, 1])


plt.ylabel(r"$log_{10}$"+'(' + r"$Dormant$" +')', fontsize=fs+3)
plt.xlabel(xlab, fontsize=fs+3)
plt.tick_params(axis='both', which='major', labelsize=fs)
#plt.text(4.7, 0.2, 'F', color = 'y', fontweight='bold')

#### Final Format and Save #####################################################
plt.subplots_adjust(wspace=0.4, hspace=0.4)
plt.savefig(mydir + '/results/figures/Fig1.png', dpi=200, bbox_inches = "tight")
plt.close()
