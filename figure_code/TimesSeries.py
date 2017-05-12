from __future__ import division
import  matplotlib.pyplot as plt
from random import randint
import pandas as pd
import numpy as np
import os


mydir = os.path.expanduser('~/GitHub/residence-time/Emergence')
tools = os.path.expanduser(mydir + "/tools")

df1 = pd.read_csv(mydir + '/results/simulated_data/Mason-SimData.csv')
df2 = pd.read_csv(mydir + '/results/simulated_data/Karst-SimData.csv')
df3 = pd.read_csv(mydir + '/results/simulated_data/BigRed2-SimData.csv')

frames = [df1, df2, df3]
df = pd.concat(frames)


def assigncolor(xs):
    cDict = {}
    clrs = []
    for x in xs:
        if x not in cDict:
            r1 = lambda: randint(0,255)
            r2 = lambda: randint(0,255)
            r3 = lambda: randint(0,255)
            c = '#%02X%02X%02X' % (r1(),r2(),r3())
            cDict[x] = c

        clrs.append(cDict[x])
    return clrs


def figplot(df3, xlab, ylab, fig, n):
    fig.add_subplot(2, 2, n)

    sims = list(set(df2['sim']))
    for sim in sims:

        df4 = df3[df3['sim'] == sim]

        clr = df4['clrs'].values.tolist()[0]
        if n == 1: plt.plot(df4['ct'], df4['N'], lw=0.5, color= clr)
        if n == 2: plt.plot(df4['ct'], df4['S'], lw=0.5, color= clr)
        if n == 3: plt.plot(df4['ct'], df4['R'], lw=0.5, color= clr)
        if n == 4: plt.plot(df4['ct'], df4['Dorm'], lw=0.5, color= clr)

    plt.tick_params(axis='both', labelsize=6)
    plt.xlabel(xlab, fontsize=8)
    plt.ylabel(ylab, fontsize=8)

    return fig


p, fr, _lw, w, sz, fs = 2, 0.2, 1.5, 1, 20, 6

df2 = pd.DataFrame({'area' : df['area']})
df2['R'] = np.log10(df['total.res'])
df2['sim'] = df['sim']
df2['ct'] = df['ct']

df2['flow'] = df['flow.rate']
df2['tau'] = np.log10(df2['area']/df2['flow'])

df2['N'] = np.log10(df['total.abundance'])
df2['S'] = np.log10(df['species.richness'])
df2['Dorm'] = df['percent.dormant']

clrs = assigncolor(df2['tau'])
df2['clrs'] = clrs

fig = plt.figure()
xlab = r"$log_{10}$"+'(' + r"$\tau$" +')'

fig = figplot(df2, xlab, r"$N$", fig, 1)
fig = figplot(df2, xlab, r"$S$", fig, 2)
fig = figplot(df2, xlab, 'Resource', fig, 3)
fig = figplot(df2, xlab, '%Dormant', fig, 4)

plt.subplots_adjust(wspace=0.5, hspace=0.45)
plt.savefig(mydir + '/results/figures/TimeSeries.png', dpi=200, bbox_inches = "tight")
plt.close()
