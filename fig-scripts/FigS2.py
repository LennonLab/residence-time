from __future__ import division
import  matplotlib.pyplot as plt
import pandas as pd
import linecache
import numpy as np
import os
import sys

import random
import scipy as sc
from scipy import stats

import statsmodels.stats.api as sms
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.outliers_influence import summary_table

mydir = os.path.expanduser('~/GitHub/residence-time')
sys.path.append(mydir+'/tools')
mydir2 = os.path.expanduser("~/")

df = pd.read_csv(mydir + '/results/simulated_data/SimData.csv')

df2 = pd.DataFrame({'width' : df['width']})
df2['flow'] = df['flow.rate']
df2['tau'] = np.log10((df['height'] * df['length'] * df2['width'])/df2['flow'])

df2['N'] = np.log10(df['total.abundance'])
df2['S'] = np.log10(df['species.richness'])
df2['E'] = np.log10(df['simpson.e'])
df2['W'] = df['Whittakers.turnover']
df2['sim'] = df['sim']

data = mydir + '/results/simulated_data/RAD-Data.csv'

def e_simpson(sad): # based on 1/D, not 1 - D
    sad = filter(lambda a: a != 0, sad)

    D = 0.0
    N = sum(sad)
    S = len(sad)

    for x in sad:
        D += (x*x) / (N*N)

    E = round((1.0/D)/S, 4)

    if E < 0.0 or E > 1.0:
        print 'Simpsons Evenness =',E
    return E


def logmodulo_skew(sad):
    skew = stats.skew(sad)
    # log-modulo transformation of skewnness
    lms = np.log10(np.abs(float(skew)) + 1)
    if skew < 0:
        lms = lms * -1
    return lms



RADs = []
with open(data) as f:
    for d in f:
        d = list(eval(d))
        sim = d.pop(0)
        ct = d.pop(0)
        RADs.append(d)


Nmaxs = []
Rs = []

for i, rad in enumerate(RADs):    
    Nmaxs.append(max(rad))
    rare = logmodulo_skew(rad)
    Rs.append(rare)


Nmaxs = np.log10(Nmaxs).tolist()
Rs = np.log10(Rs).tolist()

df2['Nmax'] = Nmaxs
df2['Rs'] = Rs
df2 = df2[df2['Rs'] > -2]
#df2 = df2[df2['flow'] < 0.01]
#df2 = df2[df2['tau'] > 1]

#df2 = df2[df2['W'] != 1]
#df2 = df2[df2['W'] > 0]
#df2 = df2[df2['W'] < 2]

metrics = ['Rarity, '+r'$log_{10}$',
        'Dominance, '+r'$log_{10}$',
        'Evenness, ' +r'$log_{10}$',
        'Richness, ' +r'$log_{10}$',]


fig = plt.figure()
fs = 12 # font size used across figures
for index, metric in enumerate(metrics):
    fig.add_subplot(2, 2, index+1)

    metlist = []
    if index == 0: metlist = list(df2['Rs'])
    elif index == 1: metlist = list(df2['Nmax'])
    elif index == 2: metlist = list(df2['E'])
    elif index == 3: metlist = list(df2['S'])

    print len(df2['N']), len(metlist)
    df2['y'] = list(metlist)

    f = smf.ols('y ~ N', df2).fit()

    r2 = round(f.rsquared,2)
    Int = f.params[0]
    Coef = f.params[1]

    gd = 15
    mct = 0
    plt.hexbin(df2['N'], metlist, mincnt=mct, gridsize = gd, bins='log', cmap=plt.cm.jet)
    
    if index == 0:
        plt.text(1, 0.4, r'$rarity$'+ ' = '+str(round(10**Int,2))+'*'+r'$N$'+'$^{'+str(round(Coef,2))+'}$', fontsize=fs-2, color='k')
        plt.text(1, 0.3,  r'$r^2$' + '=' +str(r2), fontsize=fs-2, color='k')


    elif index == 1:
        plt.text(1, 3.8, r'$Nmax$'+ ' = '+str(round(10**Int,2))+'*'+r'$N$'+'$^{'+str(round(Coef,2))+'}$', fontsize=fs-2, color='k')
        plt.text(1, 3.5,  r'$r^2$' + '=' +str(r2), fontsize=fs-2, color='k')


    elif index == 2:
        plt.text(1, -1.0, r'$Ev$'+ ' = '+str(round(10**Int,2))+'*'+r'$N$'+'$^{'+str(round(Coef,2))+'}$', fontsize=fs-2, color='k')
        plt.text(1, -1.3,  r'$r^2$' + '=' +str(r2), fontsize=fs-2, color='k')

    elif index == 3:
        plt.text(1, 1.8, r'$S$'+ ' = '+str(round(10**Int,2))+'*'+r'$N$'+'$^{'+str(round(Coef,2))+'}$', fontsize=fs-2, color='k')
        plt.text(1, 1.6,  r'$r^2$' + '=' +str(r2), fontsize=fs-2, color='k')


    plt.xlabel('$log$'+r'$_{10}$'+'($N$)', fontsize=fs)
    plt.ylabel(metric, fontsize=fs)
    plt.tick_params(axis='both', which='major', labelsize=fs-3)


#### Final Format and Save #####################################################
plt.subplots_adjust(wspace=0.4, hspace=0.4)
plt.savefig(mydir + '/results/figures/FigS2.png', dpi=600, bbox_inches = "tight")
#plt.show()
plt.close()
