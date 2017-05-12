from __future__ import division
import  matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import sys

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
            if x < 1: c = 'r'
            elif x < 2: c = 'Orange'
            elif x < 3: c = 'Gold'
            elif x < 4: c = 'Green'
            elif x < 5: c = 'Blue'
            else: c = 'DarkViolet'
            cDict[x] = c

        clrs.append(cDict[x])
    return clrs


df2 = pd.DataFrame({'area' : df['area'].groupby(df['sim']).mean()})
df2['sim'] = df['sim'].groupby(df['sim']).mean()
df2['flow'] = df['flow.rate'].groupby(df['sim']).mean()
df2['tau'] = df2['area']/df2['flow']
df2['dil'] = 1/df2['tau']

df2['N'] = np.log10(df['total.abundance'].groupby(df['sim']).max())
df2['Prod'] = np.log10(df['ind.production'].groupby(df['sim']).max())
df2['S'] = np.log10(df['species.richness'].groupby(df['sim']).max())
#df2['N'] = df['total.abundance'].groupby(df['sim']).max()
#df2['Prod'] = df['ind.production'].groupby(df['sim']).max()
#df2['S'] = df['species.richness'].groupby(df['sim']).max()

df2['M'] = df['avg.per.capita.maint'].groupby(df['sim']).median()
df2 = df2.replace([np.inf, -np.inf, 0], np.nan).dropna()

df2['phi'] = df2['M']

df2['x'] = np.log10(df2['phi']) / np.log10(df2['tau'])

clrs = assigncolor(np.log10(df2['tau']))
df2['clrs'] = clrs

#### plot figure ###############################################################

xlab =  r"log($BMR$)/log($\tau$)"
fs = 8 # fontsize
fig = plt.figure()

xl = -2
xh = 0
sz = 10

#### N vs. Tau #################################################################
fig.add_subplot(3, 3, 1)
plt.axvline(-1, color='k', ls='--', lw = 1)
plt.scatter(df2['x'], df2['N'], s = sz, color=df2['clrs'], linewidths=0.2, edgecolor='w')
plt.ylabel(r"$log_{10}(N)$", fontsize=fs+2)
plt.xlabel(xlab, fontsize=fs)
plt.tick_params(axis='both', which='major', labelsize=fs-2)
plt.xlim(xl, xh)
plt.ylim(0.5, 5.2)

#### production vs. Tau ########################################################
#dat = dat.convert_objects(convert_numeric=True).dropna()
fig.add_subplot(3, 3, 2)
plt.axvline(-1, color='k', ls='--', lw = 1)
plt.scatter(df2['x'], df2['Prod'], s = sz, color=df2['clrs'], linewidths=0.2, edgecolor='w')
plt.ylabel(r"$log_{10}(P)$", fontsize=fs+2)
plt.xlabel(xlab, fontsize=fs)
plt.tick_params(axis='both', which='major', labelsize=fs-2)
plt.xlim(xl, xh)
plt.ylim(0, 5)

#### S vs. Tau #################################################################
fig.add_subplot(3, 3, 3)
plt.axvline(-1, color='k', ls='--', lw = 1)
plt.scatter(df2['x'], df2['S'], s = sz, color=df2['clrs'], linewidths=0.2, edgecolor='w')
plt.ylabel(r"$log_{10}(S)$", fontsize=fs+2)
plt.xlabel(xlab, fontsize=fs)
plt.tick_params(axis='both', which='major', labelsize=fs-2)
plt.xlim(xl, xh)
plt.ylim(0.2, 3.5)

#### Final Format and Save #####################################################
plt.subplots_adjust(wspace=0.4, hspace=0.4)
plt.savefig(mydir + '/results/figures/BMR_vs_Tau.png', dpi=200, bbox_inches = "tight")
plt.close()
