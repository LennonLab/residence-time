from __future__ import division
import  matplotlib.pyplot as plt
import pandas as pd
#import linecache
import numpy as np
import os
import sys

#import random
#import scipy as sc
from scipy import stats

#import statsmodels.stats.api as sms
#import statsmodels.api as sm
import statsmodels.formula.api as smf
#from statsmodels.stats.outliers_influence import summary_table

mydir = os.path.expanduser('~/GitHub/residence-time')
sys.path.append(mydir+'/tools')
mydir2 = os.path.expanduser("~/")

df1 = pd.read_csv(mydir + '/results/simulated_data/SimData.csv')
df2 = pd.DataFrame({'width' : df1['width']})
df2['flow'] = df1['flow.rate']
df2['tau'] = np.log10((df1['height'] * df1['length'] * df2['width'])/df2['flow'])
print min(df2['tau']), max(df2['tau'])
#sys.exit()

data = mydir + '/results/simulated_data/protected/RAD-Data.csv'

def e_simpson(sad): # based on 1/D, not 1 - D
    sad = filter(lambda a: a != 0, sad)
    D = 0.0
    N = sum(sad)
    S = len(sad)

    for x in sad:
        D += (x*x) / (N*N)

    E = round((1.0/D)/S, 4)
    return E


def logmodulo_skew(sad):
    skew = stats.skew(sad)
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

        if len(d) >= 10: RADs.append(d)


Ns = []
Ss = []
Evs = []
Nmaxs = []
Rs = []

for rad in RADs:
    Ns.append(sum(rad))
    Nmaxs.append(max(rad))
    Ss.append(len(rad))
    ev = e_simpson(rad)
    Evs.append(ev)
    rare = logmodulo_skew(rad)
    Rs.append(rare)


metrics = ['Rarity exponent', 'Dominance exponent', 'Evenness exponent', 'Richness exponent',]

xs = np.linspace(100, max(Ns)*0.7, 40)
#xs = np.linspace(1, 6, 40)

fig = plt.figure()
fs = 12 # font size used across figures
for index, metric in enumerate(metrics):
    fig.add_subplot(2, 2, index+1)

    ys = []
    for x in xs:
        df3 = pd.DataFrame({'N' : Ns})
        df3['S'] = Ss
        df3['E'] = Evs
        df3['R'] = Rs
        df3['D'] = Nmaxs
        df3['tau'] = df2['tau']
        df3['width'] = df2['width']
        #df3 = df3[df3['width'] == 1]
        df4 = df3[df3['N'] >= x]
        #df4 = df4[df4['N'] < x+900]
        #df4 = df3[df3['tau'] >= x]
        #df4 = df4[df4['tau'] < x+1]

        metlist = []
        if index == 0: metlist = np.log10(df4['R'])
        elif index == 1: metlist = np.log10(df4['D'])
        elif index == 2: metlist = np.log10(df4['E'])
        elif index == 3: metlist = np.log10(df4['S'])

        if index == 3: print x, len(df4['N']), len(metlist)

        d = pd.DataFrame({'N': np.log10(df4['N'])})
        d['y'] = list(metlist)
        f = smf.ols('y ~ N', d).fit()

        #d = pd.DataFrame({'tau': np.log10(df4['tau'])})
        #d['y'] = list(metlist)
        #f = smf.ols('y ~ tau', d).fit()

        r2 = round(f.rsquared,2)
        Int = f.params[0]
        Coef = f.params[1]
        ys.append(Coef)

    plt.scatter(xs, ys, color = '0.6', alpha= 1 , s = 20, linewidths=0.5, edgecolor='0.3')
    if index == 0: plt.fill_between([0,2000], 0.11, 0.14, color='k', lw=0.0, alpha=0.3)
    elif index == 1: plt.fill_between([0,2000], 0.92, 0.99, color='k', lw=0.0, alpha=0.3)
    elif index == 2: plt.fill_between([0,2000], -0.24, -0.21, color='k', lw=0.0, alpha=0.3)
    elif index == 3: plt.fill_between([0,2000], 0.24, 0.39, color='k', lw=0.0, alpha=0.3)

    #plt.xlim(1000, 2500)
    plt.xlim(0.9*min(xs), 1.1*max(xs))
    plt.xlabel('Minimum '+r'($N$)', fontsize=fs)
    plt.ylabel(metric, fontsize=fs)
    plt.tick_params(axis='both', which='major', labelsize=fs-3)


#### Final Format and Save #####################################################
plt.subplots_adjust(wspace=0.4, hspace=0.4)
plt.savefig(mydir + '/results/figures/FigS2v2.png', dpi=600, bbox_inches = "tight")
#plt.show()
plt.close()
