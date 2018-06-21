from __future__ import division
import  matplotlib.pyplot as plt
from random import shuffle
import os
import sys
import numpy as np
from scipy import stats

mydir = os.path.expanduser('~/GitHub/residence-time2/Emergence')
tools = os.path.expanduser(mydir + "/tools")



def assigncolor(x):
    c = 0
    if x <= 1: c = 'r'
    elif x <= 2: c = 'Orange'
    elif x <= 3: c = 'gold'
    elif x <= 4: c = 'Green'
    elif x <= 5: c = 'DodgerBlue'
    elif x <= 6: c = 'Plum'
    return c



path = mydir + '/ModelTypes/Costs-Growth/results/simulated_data'
labels = ['/dormant.RAD-Data.csv', '/active.RAD-Data.csv', '/RAD-Data.csv']
figlabels = ['dormant', 'active', 'all']

fig = plt.figure()
for il, label in enumerate(labels):

    fig.add_subplot(3, 3, il+1)
    data = path + label

    RADs = []
    simDict = {}
    with open(data) as f:
        ct = 0
        for d in f:
            ct += 1
            d = list(eval(d))

            sim = d[0]
            if sim not in simDict:
                simDict[sim] = {'sim' : [d[0]]}
                simDict[sim]['tau'] = [d[1]]
                simDict[sim]['cts'] = [d[2]]
                simDict[sim]['rads'] = [d[3]]
                simDict[sim]['slists'] = [d[4]]

            else:
                cts = simDict[sim]['cts']
                cts.append(d[2])
                simDict[sim]['cts'] = cts

                rads = simDict[sim]['rads']
                rads.append(d[3])
                simDict[sim]['rads'] = rads

                slists = simDict[sim]['slists']
                slists.append(d[4])
                simDict[sim]['slists'] = slists


    p, fr, _lw, w, sz, fs = 2, 0.2, 1.5, 1, 0.1, 6
    simList = list(simDict)
    shuffle(simList)

    Ns = []
    Vars = []
    clrs = []

    for sim in simList:
        rads = simDict[sim]['rads']
        slists = simDict[sim]['slists']
        tau = np.log10(simDict[sim]['tau'])

        dlist = []
        for ls in slists: dlist.extend(ls)
        if len(dlist) == 0: continue

        sp = list(set(dlist))
        sp.sort()

        for s in sp:
            if dlist.count(s) > 2:

                abl = []
                for i, ls in enumerate(slists):
                    if s in ls:
                        ind = ls.index(s)
                        ab = rads[i][ind]
                        abl.append(ab)
                    else: abl.append(0)

                if max(abl) < 2: continue
                ns = sum(abl)/len(abl)
                var = np.var(abl)
                if ns > 0 and var > 0:
                    Ns.append(ns)
                    Vars.append(var)
                    clr = assigncolor(tau)
                    clrs.append(clr)

    plt.xscale('log')
    plt.yscale('log')

    plt.scatter(Ns, Vars, c=clrs, s = sz, linewidths=0.0, edgecolor='k', alpha=0.5)
    m, b, r, p, std_err = stats.linregress(np.log10(Ns), np.log10(Vars))
    plt.plot(np.arange(min(Ns), max(Ns), 0.1), 10**b * np.arange(min(Ns), max(Ns), 0.1)**m, ls='-', color='k', lw=0.5, label = 'slope = '+str(round(m,2)))

    plt.legend(loc=2, fontsize=fs-1, frameon=False)

    plt.tick_params(axis='both', labelsize=6)
    xlab = r'$\mu$'
    ylab = r'$\sigma^{2}$'
    plt.xlabel(xlab, fontsize=8)
    plt.ylabel(ylab, fontsize=8)
    plt.title(figlabels[il], fontsize=10)

plt.subplots_adjust(wspace=0.4, hspace=0.4)
plt.savefig(mydir + '/ModelTypes/Costs-Growth/results/figures/Supplement/SupFig2.png', dpi=200, bbox_inches = "tight")
plt.close()
