from __future__ import division
import  matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os


p, fr, _lw, w, fs, sz = 2, 0.75, 0.5, 1, 6, 0.5
minS = 1.2
minN = 0

mydir = os.path.expanduser('~/GitHub/residence-time2/Emergence')
tools = os.path.expanduser(mydir + "/tools")

df = pd.read_csv(mydir + '/ModelTypes/Costs-Growth/results/simulated_data/SimData.csv')
df = df[df['total.abundance'] > 0]

def xfrm(X, _max): return _max-np.array(X)


def assigncolor(xs, kind):
    clrs = []

    if kind == 'Q':
        for x in xs:
            if x >= 10**-0.5: c = 'r'
            elif x >= 10**-1: c = 'Orange'
            elif x >= 10**-1.5: c = 'Lime'
            elif x >= 10**-2: c = 'Green'
            elif x >= 10**-2.5: c = 'DodgerBlue'
            else: c = 'Plum'
            clrs.append(c)

    elif kind == 'V':
        for x in xs:
            if x <= 0.5: c = 'r'
            elif x <= 1: c = 'Orange'
            elif x <= 1.5: c = 'gold'
            elif x <= 2: c = 'Green'
            elif x <= 2.5: c = 'DodgerBlue'
            elif x <= 3: c = 'Plum'
            clrs.append(c)
    return clrs



def figplot(clrs, x, y, xlab, ylab, fig, n, w):
    fig.add_subplot(4, 4, n)

    #y = np.array(y) + 0.01
    #plt.yscale('log')

    plt.scatter(x, y, color=clrs, s=sz, linewidths=0.0)#, edgecolor='w')

    '''
    b = 20
    ci = 99
    if n == 5: ci = 1
    x, y = (np.array(t) for t in zip(*sorted(zip(x, y))))

    Xi = xfrm(x, max(x)*1.05)
    bins = np.linspace(np.min(Xi), np.max(Xi)+1, b)
    ii = np.digitize(Xi, bins)

    pcts = np.array([np.percentile(y[ii==i], ci) for i in range(1, len(bins)) if len(y[ii==i]) > 0])
    xran = np.array([np.mean(x[ii==i]) for i in range(1, len(bins)) if len(y[ii==i]) > 0])

    lowess = sm.nonparametric.lowess(pcts, xran, frac=fr)
    x, y = lowess[:, 0], lowess[:, 1]
    plt.plot(x, y, lw=0.5, color='k')
    '''


    #if w == 3:
    #    x, y = (np.array(t) for t in zip(*sorted(zip(x, y))))
    #    x = np.array(x)
    #    y =  20 #- (1 + (20/10)*(3 #- x)**2)
    #    plt.plot(x, y, lw=0.5, c='0.5')



    plt.tick_params(axis='both', labelsize=5)
    plt.xlabel(xlab, fontsize=5)
    plt.ylabel(ylab, fontsize=5)
    #plt.ylim(min(y), max(y))

    return fig



df2 = pd.DataFrame({'V' : df['V'].groupby(df['sim']).mean()})
df2['Q'] = df['Q'].groupby(df['sim']).mean()
df2['tau'] = np.log10(df2['V']/df2['Q'])

df2['E1'] = df['efficiency1'].groupby(df['sim']).mean()
df2['E2'] = df['efficiency1'].groupby(df['sim']).min()
df2['E3'] = df['efficiency1'].groupby(df['sim']).max()
df2['E4'] = df['active.efficiency1'].groupby(df['sim']).mean()
df2['E5'] = df['active.efficiency1'].groupby(df['sim']).min()
df2['E6'] = df['active.efficiency1'].groupby(df['sim']).max()
df2['E7'] = np.log10(df['avg.per.capita.efficiency1e'].groupby(df['sim']).mean())
df2['E8'] = df['avg.per.capita.efficiency1e'].groupby(df['sim']).min()
df2['E9'] = np.log10(df['avg.per.capita.efficiency1e'].groupby(df['sim']).max())
df2['E10'] = np.log10(df['active.avg.per.capita.efficiency1e'].groupby(df['sim']).mean())
df2['E11'] = df['active.avg.per.capita.efficiency1e'].groupby(df['sim']).min()
df2['E12'] = np.log10(df['active.avg.per.capita.efficiency1e'].groupby(df['sim']).max())

df2['N'] = df['total.abundance'].groupby(df['sim']).mean()
df2['S'] = df['species.richness'].groupby(df['sim']).mean()
df2 = df2[df2['S'] >= minS]
df2 = df2[df2['N'] >= minN]

df2['clrs'] = assigncolor(np.log10(df2['V']), 'V')
#df2['clrs'] = assigncolor(df2['flow'], 'Q')

df2 = df2.replace([np.inf, -np.inf], np.nan).dropna()

xlab = r"$log_{10}$"+'(' + r"$\tau$" +')'
fig = plt.figure()

ylab = 'Avg specialization (All)'
fig = figplot(df2['clrs'], df2['tau'], df2['E1'], xlab, ylab, fig, 1, w=1)

ylab = 'Min specialization (All)'
fig = figplot(df2['clrs'], df2['tau'], df2['E2'], xlab, ylab, fig, 2, w=1)

ylab = 'Max specialization (All)'
fig = figplot(df2['clrs'], df2['tau'], df2['E3'], xlab, ylab, fig, 3, w=1)

ylab = 'Avg specialization (Active)'
fig = figplot(df2['clrs'], df2['tau'], df2['E4'], xlab, ylab, fig, 5, w=1)

ylab = 'Min specialization (Active)'
fig = figplot(df2['clrs'], df2['tau'], df2['E5'], xlab, ylab, fig, 6, w=1)

ylab = 'Max specialization (Active)'
fig = figplot(df2['clrs'], df2['tau'], df2['E6'], xlab, ylab, fig, 7, w=1)

ylab = 'Avg specialization (All)'
fig = figplot(df2['clrs'], df2['tau'], df2['E7'], xlab, ylab, fig, 9, w=1)

ylab = 'Min specialization (All)'
fig = figplot(df2['clrs'], df2['tau'], df2['E8'], xlab, ylab, fig, 10, w=1)

ylab = 'Max specialization (All)'
fig = figplot(df2['clrs'], df2['tau'], df2['E9'], xlab, ylab, fig, 11, w=1)

ylab = 'Avg specialization (Active)'
fig = figplot(df2['clrs'], df2['tau'], df2['E10'], xlab, ylab, fig, 13, w=1)

ylab = 'Min specialization (Active)'
fig = figplot(df2['clrs'], df2['tau'], df2['E11'], xlab, ylab, fig, 14, w=1)

ylab = 'Max specialization (Active)'
fig = figplot(df2['clrs'], df2['tau'], df2['E12'], xlab, ylab, fig, 15, w=1)

plt.subplots_adjust(wspace=0.6, hspace=0.5)
plt.savefig(mydir + '/ModelTypes/Costs-Growth/results/figures/Specialization1_vs_Tau-Unweighted.png', dpi=200, bbox_inches = "tight")
plt.close()


df2 = pd.DataFrame({'V' : df['V'].groupby(df['sim']).mean()})
df2['Q'] = df['Q'].groupby(df['sim']).mean()
df2['tau'] = np.log10(df2['V']/df2['Q'])

df2['E1'] = df['efficiency2'].groupby(df['sim']).mean()
df2['E2'] = df['efficiency2'].groupby(df['sim']).min()
df2['E3'] = df['efficiency2'].groupby(df['sim']).max()
df2['E4'] = df['active.efficiency2'].groupby(df['sim']).mean()
df2['E5'] = df['active.efficiency2'].groupby(df['sim']).min()
df2['E6'] = df['active.efficiency2'].groupby(df['sim']).max()
df2['E7'] = np.log10(df['avg.per.capita.efficiency2e'].groupby(df['sim']).mean())
df2['E8'] = df['avg.per.capita.efficiency2e'].groupby(df['sim']).min()
df2['E9'] = np.log10(df['avg.per.capita.efficiency2e'].groupby(df['sim']).max())
df2['E10'] = np.log10(df['active.avg.per.capita.efficiency2e'].groupby(df['sim']).mean())
df2['E11'] = df['active.avg.per.capita.efficiency2e'].groupby(df['sim']).min()
df2['E12'] = np.log10(df['active.avg.per.capita.efficiency2e'].groupby(df['sim']).max())

df2['N'] = df['total.abundance'].groupby(df['sim']).mean()
df2['S'] = df['species.richness'].groupby(df['sim']).mean()
df2 = df2[df2['S'] >= minS]
df2 = df2[df2['N'] >= minN]

df2['clrs'] = assigncolor(np.log10(df2['V']), 'V')
#df2['clrs'] = assigncolor(df2['flow'], 'Q')

df2 = df2.replace([np.inf, -np.inf], np.nan).dropna()

xlab = r"$log_{10}$"+'(' + r"$\tau$" +')'
fig = plt.figure()

ylab = 'Avg specialization (All)'
fig = figplot(df2['clrs'], df2['tau'], df2['E1'], xlab, ylab, fig, 1, w=2)

ylab = 'Min specialization (All)'
fig = figplot(df2['clrs'], df2['tau'], df2['E2'], xlab, ylab, fig, 2, w=2)

ylab = 'Max specialization (All)'
fig = figplot(df2['clrs'], df2['tau'], df2['E3'], xlab, ylab, fig, 3, w=2)

ylab = 'Avg specialization (Active)'
fig = figplot(df2['clrs'], df2['tau'], df2['E4'], xlab, ylab, fig, 5, w=2)

ylab = 'Min specialization (Active)'
fig = figplot(df2['clrs'], df2['tau'], df2['E5'], xlab, ylab, fig, 6, w=2)

ylab = 'Max specialization (Active)'
fig = figplot(df2['clrs'], df2['tau'], df2['E6'], xlab, ylab, fig, 7, w=2)

ylab = 'Avg specialization (All)'
fig = figplot(df2['clrs'], df2['tau'], df2['E7'], xlab, ylab, fig, 9, w=2)

ylab = 'Min specialization (All)'
fig = figplot(df2['clrs'], df2['tau'], df2['E8'], xlab, ylab, fig, 10, w=2)

ylab = 'Max specialization (All)'
fig = figplot(df2['clrs'], df2['tau'], df2['E9'], xlab, ylab, fig, 11, w=2)

ylab = 'Avg specialization (Active)'
fig = figplot(df2['clrs'], df2['tau'], df2['E10'], xlab, ylab, fig, 13, w=2)

ylab = 'Min specialization (Active)'
fig = figplot(df2['clrs'], df2['tau'], df2['E11'], xlab, ylab, fig, 14, w=2)

ylab = 'Max specialization (Active)'
fig = figplot(df2['clrs'], df2['tau'], df2['E12'], xlab, ylab, fig, 15, w=2)


plt.subplots_adjust(wspace=0.6, hspace=0.5)
plt.savefig(mydir + '/ModelTypes/Costs-Growth/results/figures/Specialization2_vs_Tau-Unweighted.png', dpi=200, bbox_inches = "tight")
plt.close()



df2 = pd.DataFrame({'V' : df['V'].groupby(df['sim']).mean()})
df2['Q'] = df['Q'].groupby(df['sim']).mean()
df2['tau'] = np.log10(df2['V']/df2['Q'])

df2['E1'] = df['efficiency3'].groupby(df['sim']).mean()
df2['E2'] = df['efficiency3'].groupby(df['sim']).min()
df2['E3'] = df['efficiency3'].groupby(df['sim']).max()
df2['E4'] = df['active.efficiency3'].groupby(df['sim']).mean()
df2['E5'] = df['active.efficiency3'].groupby(df['sim']).min()
df2['E6'] = df['active.efficiency3'].groupby(df['sim']).max()
df2['E7'] = np.log10(df['avg.per.capita.efficiency3e'].groupby(df['sim']).mean())
df2['E8'] = np.log10(df['avg.per.capita.efficiency3e'].groupby(df['sim']).min())
df2['E9'] = np.log10(df['avg.per.capita.efficiency3e'].groupby(df['sim']).max())
df2['E10'] = np.log10(df['active.avg.per.capita.efficiency3e'].groupby(df['sim']).mean())
df2['E11'] = np.log10(df['active.avg.per.capita.efficiency3e'].groupby(df['sim']).min())
df2['E12'] = np.log10(df['active.avg.per.capita.efficiency3e'].groupby(df['sim']).max())

df2['N'] = df['total.abundance'].groupby(df['sim']).mean()
df2['S'] = df['species.richness'].groupby(df['sim']).mean()
df2 = df2[df2['S'] >= minS]
df2 = df2[df2['N'] >= minN]

df2['clrs'] = assigncolor(np.log10(df2['V']), 'V')
#df2['clrs'] = assigncolor(df2['flow'], 'Q')

df2 = df2.replace([np.inf, -np.inf], np.nan).dropna()

xlab = r"$log_{10}$"+'(' + r"$\tau$" +')'
fig = plt.figure()

ylab = 'Avg specialization (All)'
fig = figplot(df2['clrs'], df2['tau'], df2['E1'], xlab, ylab, fig, 1, w=3)

ylab = 'Min specialization (All)'
fig = figplot(df2['clrs'], df2['tau'], df2['E2'], xlab, ylab, fig, 2, w=3)

ylab = 'Max specialization (All)'
fig = figplot(df2['clrs'], df2['tau'], df2['E3'], xlab, ylab, fig, 3, w=3)

ylab = 'Avg specialization (Active)'
fig = figplot(df2['clrs'], df2['tau'], df2['E4'], xlab, ylab, fig, 5, w=3)

ylab = 'Min specialization (Active)'
fig = figplot(df2['clrs'], df2['tau'], df2['E5'], xlab, ylab, fig, 6, w=3)

ylab = 'Max specialization (Active)'
fig = figplot(df2['clrs'], df2['tau'], df2['E6'], xlab, ylab, fig, 7, w=3)

ylab = 'Avg specialization (All)'
fig = figplot(df2['clrs'], df2['tau'], df2['E7'], xlab, ylab, fig, 9, w=3)

ylab = 'Min specialization (All)'
fig = figplot(df2['clrs'], df2['tau'], df2['E8'], xlab, ylab, fig, 10, w=3)

ylab = 'Max specialization (All)'
fig = figplot(df2['clrs'], df2['tau'], df2['E9'], xlab, ylab, fig, 11, w=3)

ylab = 'Avg specialization (Active)'
fig = figplot(df2['clrs'], df2['tau'], df2['E10'], xlab, ylab, fig, 13, w=3)

ylab = 'Min specialization (Active)'
fig = figplot(df2['clrs'], df2['tau'], df2['E11'], xlab, ylab, fig, 14, w=3)

ylab = 'Max specialization (Active)'
fig = figplot(df2['clrs'], df2['tau'], df2['E12'], xlab, ylab, fig, 15, w=3)

plt.subplots_adjust(wspace=0.6, hspace=0.5)
plt.savefig(mydir + '/ModelTypes/Costs-Growth/results/figures/Specialization3_vs_Tau-Unweighted.png', dpi=200, bbox_inches = "tight")
plt.close()
