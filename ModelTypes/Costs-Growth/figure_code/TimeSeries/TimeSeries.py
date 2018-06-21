from __future__ import division
import  matplotlib.pyplot as plt
from random import randint, shuffle
from collections import Counter
import os
import sys
import numpy as np


mydir = os.path.expanduser('~/GitHub/residence-time2/Emergence')
tools = os.path.expanduser(mydir + "/tools")


def most_common(lst):
    data = Counter(lst)
    return max(lst, key=data.get)

def assigncolor(x):
    c = 0
    if x <= 1: c = 'firebrick'
    elif x <= 1.3: c = 'r'
    elif x <= 1.6: c = 'crimson'

    elif x <= 1.9: c = 'orangered'
    elif x <= 2.3: c = 'darkorange'
    elif x <= 2.5: c = 'orange'

    elif x <= 2.8: c = 'gold'
    elif x <= 3.2: c = 'yellow'

    elif x <= 3.5: c = 'greenyellow'
    elif x <= 3.8: c = 'springgreen'
    elif x <= 4.1: c = 'Green'

    elif x <= 4.4: c = 'skyblue'
    elif x <= 4.7: c = 'DodgerBlue'
    elif x <= 5.0: c = 'b'

    elif x <= 5.3: c = 'Plum'
    elif x <= 5.6: c = 'violet'
    elif x <= 6: c = 'purple'
    return c


path = mydir + '/ModelTypes/Costs-Growth/results/simulated_data'

labels = ['/active.RAD-Data.csv', '/dormant.RAD-Data.csv', '/RAD-Data.csv']
figlabels = ['active', 'dormant', 'all']

for il, label in enumerate(labels):
    rad_data = path + label
    Sets = [rad_data]

    RADs = []
    for data in Sets:
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


    p, fr, _lw, w, sz, fs = 2, 0.2, 1.5, 1, 20, 6
    simList = list(simDict)
    shuffle(simList)

    Abs1 = []
    Abs2 = []
    Abs3 = []
    Abs4 = []
    clrs1 = []
    clrs2 = []
    clrs3 = []
    clrs4 = []

    for sim in simList:
        rads = simDict[sim]['rads']
        slists = simDict[sim]['slists']
        tau = np.log10(simDict[sim]['tau'])

        dlist = []
        for ls in slists: dlist.extend(ls)

        if len(dlist) == 0: continue

        DomS = most_common(dlist)
        abl = []

        for i, ls in enumerate(slists):
            if DomS in ls:
                ind = ls.index(DomS)
                ab = rads[i][ind]
                abl.append(ab)
            else: abl.append(0)

        if abl.count(0) > 10: continue

        clr = assigncolor(tau)
        if max(abl) < 10:
            Abs1.append(abl)
            clrs1.append(clr)
        elif max(abl) < 100:
            Abs2.append(abl)
            clrs2.append(clr)
        elif max(abl) < 1000:
            Abs3.append(abl)
            clrs3.append(clr)
        else:
            Abs4.append(abl)
            clrs4.append(clr)


    fig = plt.figure()
    fig.add_subplot(2, 2, 1)

    for i, abl in enumerate(Abs1):
        t = 10*np.array(range(len(abl)))
        clr = clrs1[i]
        plt.plot(t, abl, color=clr, lw=0.5)
    xlab = 'Time step after burn-in'
    ylab = 'Population size'
    plt.tick_params(axis='both', labelsize=4)
    plt.xlabel(xlab, fontsize=6)
    plt.ylabel(ylab, fontsize=6)


    fig.add_subplot(2, 2, 2)
    for i, abl in enumerate(Abs2):
        t = 10*np.array(range(len(abl)))
        clr = clrs2[i]
        plt.plot(t, abl, color=clr, lw=0.5)
    xlab = 'Time step after burn-in'
    ylab = 'Population size'
    plt.tick_params(axis='both', labelsize=4)
    plt.xlabel(xlab, fontsize=6)
    plt.ylabel(ylab, fontsize=6)


    fig.add_subplot(2, 2, 3)
    for i, abl in enumerate(Abs3):
        t = 10*np.array(range(len(abl)))
        clr = clrs3[i]
        plt.plot(t, abl, color=clr, lw=0.5)
    xlab = 'Time step after burn-in'
    ylab = 'Population size'
    plt.tick_params(axis='both', labelsize=4)
    plt.xlabel(xlab, fontsize=6)
    plt.ylabel(ylab, fontsize=6)

    fig.add_subplot(2, 2, 4)
    for i, abl in enumerate(Abs4):
        t = 10*np.array(range(len(abl)))
        clr = clrs4[i]
        plt.plot(t, abl, color=clr, lw=0.5)
    xlab = 'Time step after burn-in'
    ylab = 'Population size'
    plt.tick_params(axis='both', labelsize=4)
    plt.xlabel(xlab, fontsize=6)
    plt.ylabel(ylab, fontsize=6)


    plt.subplots_adjust(wspace=0.4, hspace=0.4)
    plt.savefig(mydir + '/ModelTypes/Costs-Growth/results/figures/'+figlabels[il]+'TimeSeries.png', dpi=400, bbox_inches = "tight")
    if figlabels[il] == 'all':
        plt.savefig(mydir + '/ModelTypes/Costs-Growth/results/figures/Supplement/SupFig7.png', dpi=400, bbox_inches = "tight")
    plt.close()
