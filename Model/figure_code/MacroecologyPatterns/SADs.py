from __future__ import division
import  matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
from scipy import stats
import sys
from scipy.stats.kde import gaussian_kde
from numpy import empty
from random import shuffle

mydir = os.path.expanduser('~/GitHub/residence-time2/Emergence')
tools = os.path.expanduser(mydir + "/tools")

sys.path.append(tools + "/DiversityTools/macroecotools")
import macroecotools as mct
sys.path.append(tools + "/DiversityTools/macroeco_distributions")
from macroeco_distributions import pln, pln_solver
sys.path.append(tools + "/DiversityTools/mete")
import mete


def get_kdens_choose_kernel(_list,kernel):
    """ Finds the kernel density function across a sample of SADs """
    density = gaussian_kde(_list)
    n = len(_list)
    #xs = np.linspace(0, 1, n)
    xs = np.linspace(min(_list), max(_list), n)
    density.covariance_factor = lambda : kernel
    density._compute_covariance()
    D = [xs,density(xs)]
    return D


def get_pln(S, mu, sigma, lower_trunc = True):
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

def get_pln_from_obs(ab, dist):
    mu, sigma = pln_solver(ab)
    pred_rad = get_pln(len(ab), mu, sigma)
    return pred_rad


fs = 6
p = 1
_lw = 0.5
w = 1
sz = 1
a = 0.99
minct = 100

fig = plt.figure()

fig.add_subplot(3, 3, 1)
data1 = mydir + '/ModelTypes/Costs-Growth/results/simulated_data/active.RAD-Data.csv'
Sets = [data1]

RADs = []
for data in Sets:
    with open(data) as f:
        for d in f:
            d = list(eval(d))
            sim = d.pop(0)
            tau = d.pop(0)
            ct = d.pop(0)
            rad = d.pop(0)
            rad = sorted(rad, reverse=True)
            if sum(rad) > 100 and len(rad) > 9:
                RADs.append(rad)

print 'Number of RADs:', len(RADs)
mete_r2s = []
pln_r2s = []
zipf_r2s = []

ct = 0
shuffle(RADs)
for obs in RADs:
    N = int(sum(obs))
    S = int(len(obs))
    s = obs.count(1)

    if S > 9 and N > 9:
        ct += 1
        pred = mete.get_mete_rad(S, N)[0]
        mete_r2 = mct.obs_pred_rsquare(obs, np.array(pred))
        mete_r2s.append(mete_r2)

        pred = get_pln_from_obs(obs, 'pln')
        pred = np.log10(pred)
        obs1 = np.log10(obs)
        pln_r2 = mct.obs_pred_rsquare(obs1, pred)
        pln_r2s.append(pln_r2)

        print ct, 'N:', N, ' S:', S, ' n:', len(pln_r2s), ' |  mete:', mete_r2, '  pln:', pln_r2
    if len(pln_r2s) > minct: break

kernel = 0.5
D = get_kdens_choose_kernel(pln_r2s, kernel)
plt.plot(D[0],D[1],color = 'crimson', lw=2, alpha = 0.99, label= 'Poisson lognormal')
D = get_kdens_choose_kernel(mete_r2s, kernel)
plt.plot(D[0],D[1],color = 'steelblue', lw=2, alpha = 0.99,label= 'log-series')

plt.xlim(0.0, 1)
plt.legend(loc=2, fontsize=fs-1, frameon=False)
plt.xlabel('$r$'+r'$^{2}$', fontsize=fs)
plt.ylabel('probability density', fontsize=fs-1)
plt.tick_params(axis='both', labelsize=fs-1)
plt.xlim(0.4, 1.0)
plt.title('Active', fontsize=fs)


fig.add_subplot(3, 3, 2)
data1 = mydir + '/ModelTypes/Costs-Growth/results/simulated_data/dormant.RAD-Data.csv'
Sets = [data1]

RADs = []
for data in Sets:
    with open(data) as f:
        for d in f:
            d = list(eval(d))
            sim = d.pop(0)
            tau = d.pop(0)
            ct = d.pop(0)
            rad = d.pop(0)
            rad = sorted(rad, reverse=True)
            if sum(rad) > 100 and len(rad) > 9:
                RADs.append(rad)

print 'Number of RADs:', len(RADs)
mete_r2s = []
pln_r2s = []
zipf_r2s = []

ct = 0
shuffle(RADs)
for obs in RADs:
    N = int(sum(obs))
    S = int(len(obs))
    s = obs.count(1)

    if S > 9 and N > 9:
        ct += 1
        pred = mete.get_mete_rad(S, N)[0]
        mete_r2 = mct.obs_pred_rsquare(obs, np.array(pred))
        mete_r2s.append(mete_r2)

        pred = get_pln_from_obs(obs, 'pln')
        pred = np.log10(pred)
        obs1 = np.log10(obs)
        pln_r2 = mct.obs_pred_rsquare(obs1, pred)
        pln_r2s.append(pln_r2)

        print ct, 'N:', N, ' S:', S, ' n:', len(pln_r2s), ' |  mete:', mete_r2, '  pln:', pln_r2
    if len(pln_r2s) > minct: break

kernel = 0.5
D = get_kdens_choose_kernel(pln_r2s, kernel)
plt.plot(D[0],D[1],color = 'crimson', lw=2, alpha = 0.99, label= 'Poisson lognormal')
D = get_kdens_choose_kernel(mete_r2s, kernel)
plt.plot(D[0],D[1],color = 'steelblue', lw=2, alpha = 0.99,label= 'log-series')

plt.xlim(0.0, 1)
plt.legend(loc=2, fontsize=fs-1, frameon=False)
plt.xlabel('$r$'+r'$^{2}$', fontsize=fs)
plt.ylabel('probability density', fontsize=fs-1)
plt.tick_params(axis='both', labelsize=fs-1)
plt.xlim(0.4, 1.0)
plt.title('Dormant', fontsize=fs)



fig.add_subplot(3, 3, 3)
data1 = mydir + '/ModelTypes/Costs-Growth/results/simulated_data/RAD-Data.csv'
Sets = [data1]

RADs = []
for data in Sets:
    with open(data) as f:
        for d in f:
            d = list(eval(d))
            sim = d.pop(0)
            tau = d.pop(0)
            ct = d.pop(0)
            rad = d.pop(0)
            rad = sorted(rad, reverse=True)
            if sum(rad) > 100 and len(rad) > 9:
                RADs.append(rad)

print 'Number of RADs:', len(RADs)
mete_r2s = []
pln_r2s = []
zipf_r2s = []

ct = 0
shuffle(RADs)
for obs in RADs:
    N = int(sum(obs))
    S = int(len(obs))
    s = obs.count(1)

    if S > 9 and N > 9:
        ct += 1
        pred = mete.get_mete_rad(S, N)[0]
        mete_r2 = mct.obs_pred_rsquare(obs, np.array(pred))
        mete_r2s.append(mete_r2)

        pred = get_pln_from_obs(obs, 'pln')
        pred = np.log10(pred)
        obs1 = np.log10(obs)
        pln_r2 = mct.obs_pred_rsquare(obs1, pred)
        pln_r2s.append(pln_r2)

        print ct, 'N:', N, ' S:', S, ' n:', len(pln_r2s), ' |  mete:', mete_r2, '  pln:', pln_r2
    if len(pln_r2s) > minct: break

kernel = 0.5
D = get_kdens_choose_kernel(pln_r2s, kernel)
plt.plot(D[0],D[1],color = 'crimson', lw=2, alpha = 0.99, label= 'Poisson lognormal')
D = get_kdens_choose_kernel(mete_r2s, kernel)
plt.plot(D[0],D[1],color = 'steelblue', lw=2, alpha = 0.99,label= 'log-series')

plt.xlim(0.0, 1)
plt.legend(loc=2, fontsize=fs-1, frameon=False)
plt.xlabel('$r$'+r'$^{2}$', fontsize=fs)
plt.ylabel('probability density', fontsize=fs-1)
plt.tick_params(axis='both', labelsize=fs-1)
plt.xlim(0.4, 1.0)
plt.title('All', fontsize=fs)

#### Final Format and Save #####################################################
plt.subplots_adjust(wspace=0.4, hspace=0.4)
plt.savefig(mydir + '/ModelTypes/Costs-Growth/results/figures/Supplement/SupFig1.png', dpi=400, bbox_inches = "tight")
plt.close()
