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

data = mydir + '/results/simulated_data/forfigs/RAD-Data.csv'


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
        
        if len(d) >= 10:
            RADs.append(d)


Ns = []
Ss = []
Evs = []
Nmaxs = []
Rs = []

for rad in RADs:
    
    if len(rad) > 9:
        Ns.append(sum(rad))
        Nmaxs.append(max(rad))
        Ss.append(len(rad))
        
        ev = e_simpson(rad)
        Evs.append(ev)
        rare = logmodulo_skew(rad)
        Rs.append(rare)
    
Ns = np.log10(Ns).tolist()
Ss = np.log10(Ss).tolist()
Evs = np.log10(Evs).tolist()
Nmaxs = np.log10(Nmaxs).tolist()
Rs = np.log10(Rs).tolist()
            
                                      
metrics = ['Rarity, '+r'$log_{10}$',
        'Dominance, '+r'$log_{10}$',
        'Evenness, ' +r'$log_{10}$',
        'Richness, ' +r'$log_{10}$',] #+r'$(S)^{2}$']


fig = plt.figure()
fs = 12 # font size used across figures
for index, metric in enumerate(metrics):
    fig.add_subplot(2, 2, index+1)

    metlist = []
    if index == 0: metlist = list(Rs)
    elif index == 1: metlist = list(Nmaxs)
    elif index == 2: metlist = list(Evs)
    elif index == 3: metlist = list(Ss)

    print len(Ns), len(metlist)
    d = pd.DataFrame({'N': list(Ns)})
    d['y'] = list(metlist)

    f = smf.ols('y ~ N', d).fit()
    
    r2 = round(f.rsquared,2)
    Int = f.params[0]
    Coef = f.params[1]

    st, data, ss2 = summary_table(f, alpha=0.05)
    # ss2: Obs, Dep Var Population, Predicted Value, Std Error Mean Predict,
    # Mean ci 95% low, Mean ci 95% upp, Predict ci 95% low, Predict ci 95% upp,
    # Residual, Std Error Residual, Student Residual, Cook's D

    fitted = data[:,2]
    #predict_mean_se = data[:,3]
    mean_ci_low, mean_ci_upp = data[:,4:6].T
    ci_low, ci_upp = data[:,6:8].T
    ci_Ns = data[:,0]

    gd = 20
    mct = 1
    plt.hexbin(Ns, metlist, mincnt=mct, gridsize = gd, bins='log', cmap=plt.cm.jet)
    #plt.scatter(Ns, metlist, color = 'SkyBlue', alpha= 1 , s = 8, linewidths=0.5, edgecolor='Steelblue')
    #plt.fill_between(ci_Ns, mean_ci_low, mean_ci_upp, color='b', lw=0.0, alpha=0.3)
    #plt.plot(ci_Ns, fitted,  color='b', ls='--', lw=0.5, alpha=0.9)

    plt.xlim(1.0, 2.8)
    
    if index == 0:
        plt.text(1.1, -1.2, r'$rarity$'+ ' = '+str(round(10**Int,2))+'*'+r'$N$'+'$^{'+str(round(Coef,2))+'}$', fontsize=fs-2, color='k')
        plt.text(1.1, -1.4,  r'$r^2$' + '=' +str(r2), fontsize=fs-2, color='k')

        
    elif index == 1:
        
        plt.text(1.1, 2.3, r'$Nmax$'+ ' = '+str(round(10**Int,2))+'*'+r'$N$'+'$^{'+str(round(Coef,2))+'}$', fontsize=fs-2, color='k')
        plt.text(1.1, 2.1,  r'$r^2$' + '=' +str(r2), fontsize=fs-2, color='k')
        

    elif index == 2:
        
        plt.text(1.05, -1.0, r'$Ev$'+ ' = '+str(round(10**Int,2))+'*'+r'$N$'+'$^{'+str(round(Coef,2))+'}$', fontsize=fs-2, color='k')
        plt.text(1.05, -1.1,  r'$r^2$' + '=' +str(r2), fontsize=fs-2, color='k')
        
    elif index == 3:
        
        plt.text(1.1, 1.5, r'$S$'+ ' = '+str(round(10**Int,2))+'*'+r'$N$'+'$^{'+str(round(Coef,2))+'}$', fontsize=fs-2, color='k')
        plt.text(1.1, 1.45,  r'$r^2$' + '=' +str(r2), fontsize=fs-2, color='k')
        
        
    plt.xlabel('$log$'+r'$_{10}$'+'($N$)', fontsize=fs)
    plt.ylabel(metric, fontsize=fs)
    plt.tick_params(axis='both', which='major', labelsize=fs-3)


#### Final Format and Save #####################################################
plt.subplots_adjust(wspace=0.4, hspace=0.4)
plt.savefig(mydir + '/results/figures/FigS2.png', dpi=600, bbox_inches = "tight")
#plt.show()
plt.close()