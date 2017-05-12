from __future__ import division
import  matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
from scipy import stats
import sys
from random import shuffle
from math import isnan, isinf
from scipy.stats.kde import gaussian_kde
from numpy import empty
import statsmodels.formula.api as smf
from statsmodels.stats.outliers_influence import summary_table

mydir = os.path.expanduser('~/GitHub/residence-time/Emergence')
tools = os.path.expanduser(mydir + "/tools")

sys.path.append(tools + "/DiversityTools/macroecotools")
import macroecotools as mct
sys.path.append(tools + "/DiversityTools/macroeco_distributions")
from macroeco_distributions import pln, pln_solver
sys.path.append(tools + "/DiversityTools/mete")
import mete


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

def get_kdens_choose_kernel(_list,kernel):
    """ Finds the kernel density function across a sample of SADs """
    density = gaussian_kde(_list)
    n = len(_list)
    xs = np.linspace(0, 1, n)
    density.covariance_factor = lambda : kernel
    density._compute_covariance()
    D = [xs,density(xs)]
    return D

def get_rad_pln(S, mu, sigma, lower_trunc = True):
    """Obtain the predicted RAD from a Poisson lognormal distribution"""
    abundance = list(empty([S]))
    rank = range(1, int(S) + 1)
    cdf_obs = [(rank[i]-0.5) / S for i in range(0, int(S))]
    j = 0
    cdf_cum = 0
    i = 1
    while j < S:
        cdf_cum += pln.pmf(i, mu, sigma, lower_trunc)
        while cdf_cum >= cdf_obs[j]:
            abundance[j] = i
            j += 1
            if j == S:
                abundance.reverse()
                return abundance
        i += 1

def get_rad_from_obs(ab, dist):
    mu, sigma = pln_solver(ab)
    pred_rad = get_rad_pln(len(ab), mu, sigma)
    return pred_rad


fs = 14
p = 1
fr = 0.2
_lw = 0.5
w = 1
sz = 5

fig = plt.figure()
fig.add_subplot(2, 2, 1)

data1 = mydir + '/results/simulated_data/Karst-RAD-Data.csv'
data2 = mydir + '/results/simulated_data/Mason-RAD-Data.csv'
data3 = mydir + '/results/simulated_data/BigRed2-RAD-Data.csv'

RADs = []
Sets = [data1, data2, data3]
for data in Sets:
    with open(data) as f:
        for d in f:
            d = list(eval(d))

            sim = d.pop(0)
            ct = d.pop(0)

            d = sorted(d, reverse=True)
            RADs.append(d)

print 'Number of RADs:', len(RADs)
mete_r2s = []
pln_r2s = []

ct = 0
shuffle(RADs)
for obs in RADs:
    N = int(sum(obs))
    S = int(len(obs))
    s = obs.count(1)

    if S > 2 and N > S and s/S < 0.5:
        ct += 1
        result = mete.get_mete_rad(S, N)
        pred1 = np.log10(result[0])
        obs1 = np.log10(obs)
        mete_r2 = mct.obs_pred_rsquare(np.array(obs1), np.array(pred1))
        mete_r2s.append(mete_r2)

        pred = get_rad_from_obs(obs, 'pln')
        pred1 = np.log10(pred)
        pln_r2 = mct.obs_pred_rsquare(np.array(obs1), np.array(pred1))
        pln_r2s.append(pln_r2)

        print ct, 'N:', N, ' S:', S, ' n:', len(pln_r2s), ' |  mete:', mete_r2, '  pln:', pln_r2
    if len(pln_r2s) > 100: break

kernel = 0.5
D = get_kdens_choose_kernel(pln_r2s, kernel)
plt.plot(D[0],D[1],color = 'crimson', lw=3, alpha = 0.99, label= 'Poisson lognormal')
D = get_kdens_choose_kernel(mete_r2s, kernel)
plt.plot(D[0],D[1],color = 'steelblue', lw=3, alpha = 0.99,label= 'log-series')

plt.xlim(0.0, 1)
plt.legend(loc=2, fontsize=fs-4, frameon=False)
plt.xlabel('$r$'+r'$^{2}$', fontsize=fs)
plt.ylabel('probability density', fontsize=fs-1)
plt.tick_params(axis='both', labelsize=fs-4)




fig.add_subplot(2, 2, 2)
mydir = os.path.expanduser('~/GitHub/residence-time/Emergence')
tools = os.path.expanduser(mydir + "/tools")
data1 = mydir + '/results/simulated_data/Karst-SAR-Data.csv'
data2 = mydir + '/results/simulated_data/Mason-SAR-Data.csv'
data3 = mydir + '/results/simulated_data/BigRed2-SAR-Data.csv'

z_nest = []
z_rand = []

Sets = [data1, data2, data3]
for data in Sets:
    with open(data) as f:
        for d in f:
            d = list(eval(d))
            sim = d.pop(0)
            ct = d.pop(0)
            z1 = d[0]
            if isnan(z1) == False and isinf(z1) == False:
                z_nest.append(z1)

            if len(z_nest) > 1000: break

print len(z_nest), 'SARs'
kernel = 0.2
D = get_kdens_choose_kernel(z_nest, kernel)
plt.plot(D[0],D[1],color = '0.2', lw=3, alpha = 0.99)
plt.xlabel('Nested SAR '+'$z$'+'-values', fontsize=fs-2)
plt.ylabel('probability density', fontsize=fs-1)
plt.tick_params(axis='both', labelsize=fs-4)


fig.add_subplot(2, 2, 3)
mydir = os.path.expanduser('~/GitHub/residence-time/Emergence')
tools = os.path.expanduser(mydir + "/tools")

df1 = pd.read_csv(mydir + '/results/simulated_data/Mason-SimData.csv')
df2 = pd.read_csv(mydir + '/results/simulated_data/Karst-SimData.csv')
df3 = pd.read_csv(mydir + '/results/simulated_data/BigRed2-SimData.csv')
frames = [df1, df2, df3]
df = pd.concat(frames)

df2 = pd.DataFrame({'area' : df['area'].groupby(df['sim']).mean()})
df2['flow'] = df['flow.rate'].groupby(df['sim']).mean()
df2['tau'] = np.log10(df2['area']/df2['flow'])
df2['size'] = df['avg.per.capita.size'].groupby(df['sim']).max()

df2['G'] = df['avg.per.capita.growth'].groupby(df['sim']).max()
df2['M'] = df['avg.per.capita.maint'].groupby(df['sim']).max()
df2['D'] = df['avg.per.capita.active.dispersal'].groupby(df['sim']).max()
df2['RF'] = df['avg.per.capita.rpf'].groupby(df['sim']).max()
df2['E'] = df['avg.per.capita.efficiency'].groupby(df['sim']).max()
df2['MF'] = df['avg.per.capita.mf'].groupby(df['sim']).max()

df2 = df2.replace([np.inf, -np.inf, 0], np.nan).dropna()

df2['phi'] = df2['M'] * df2['MF']
clrs = assigncolor(df2['tau'])
df2['clrs'] = clrs

xlab = r"$log_{10}$"+'(Body size)'
ylab = r"$log_{10}$"+'('+r'$BMR$' + ')'

df2 = df2.replace([np.inf, -np.inf, 0], np.nan).dropna()

clrs = df2['clrs']
x = df2['size']
y = df2['phi']
print len(x), 'points for MTE plot'

n = 1
x = np.log10(x)
y = np.log10(y)
y2 = list(y)
x2 = list(x)

d = pd.DataFrame({'size': list(x2)})
d['rate'] = list(y2)
f = smf.ols('rate ~ size', d).fit()

coef = f.params[1]
st, data, ss2 = summary_table(f, alpha=0.05)
fitted = data[:,2]
mean_ci_low, mean_ci_upp = data[:,4:6].T

plt.scatter(x2, y2, color = clrs, alpha= 1 , s = sz, linewidths=0.2, edgecolor='w')
plt.plot(x2, fitted,  color='k', ls='-', lw=2.0, alpha=0.9)
plt.xlabel(xlab, fontsize=fs)
plt.ylabel(ylab, fontsize=fs)
plt.text(-2.5, -0.75, '$z$ = '+str(round(coef,2)), backgroundcolor='w')
plt.tick_params(axis='both', labelsize=fs-4)
plt.xlim(0.9*min(x2), 1.1*max(x2))
plt.ylim(-5, 0)
plt.legend(loc=2, fontsize=fs-4, frameon=False)



fig.add_subplot(2, 2, 4)
df1 = pd.read_csv(mydir + '/results/simulated_data/Mason-SimData.csv')
df2 = pd.read_csv(mydir + '/results/simulated_data/Karst-SimData.csv')
df3 = pd.read_csv(mydir + '/results/simulated_data/BigRed2-SimData.csv')

frames = [df1, df2, df3]
df = pd.concat(frames)

df2 = pd.DataFrame({'area' : df['area'].groupby(df['sim']).mean()})
df2['flow'] = df['flow.rate'].groupby(df['sim']).mean()
df2['tau'] = np.log10(df2['area']/df2['flow'])

df2['NS'] = np.log10(df['avg.pop.size'].groupby(df['sim']).mean())
df2['var'] = np.log10(df['pop.var'].groupby(df['sim']).mean())

df2 = df2[df2['var'] > 0.5]
df2 = df2.replace([np.inf, -np.inf], np.nan).dropna()
clrs = assigncolor(df2['tau'])
df2['clrs'] = clrs

Nlist = df2['NS'].tolist()
Vlist = df2['var'].tolist()
print len(Nlist), 'points for Taylors Law'

plt.scatter(Nlist, Vlist, color=df2['clrs'], s = sz, linewidths=0.2, edgecolor='w')
m, b, r, p, std_err = stats.linregress(Nlist, Vlist)
Nlist = np.array(Nlist)
plt.plot(Nlist, m*Nlist + b, '-', color='k', lw=2.0)
xlab = r'$log_{10}(\mu)$'
ylab = r'$log_{10}(\sigma^{2})$'
plt.xlabel(xlab, fontsize=fs)
plt.text(0.1, 4.2, '$z$ = '+str(round(m,2)), backgroundcolor='w')
plt.tick_params(axis='both', labelsize=fs-4)
plt.ylabel(ylab, fontsize=fs)
plt.legend(loc=2, fontsize=fs-4, frameon=False)
plt.xlim(0, 2)
plt.ylim(0.5, 5)

#### Final Format and Save #####################################################
plt.subplots_adjust(wspace=0.4, hspace=0.4)
plt.savefig(mydir + '/results/figures/MacroEco.png', dpi=200, bbox_inches = "tight")
plt.close()
