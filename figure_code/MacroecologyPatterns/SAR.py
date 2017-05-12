from __future__ import division
import  matplotlib.pyplot as plt
import numpy as np
import os
from scipy.stats.kde import gaussian_kde
from math import isnan, isinf

mydir = os.path.expanduser('~/GitHub/residence-time/Emergence')
tools = os.path.expanduser(mydir + "/tools")
data1 = mydir + '/results/simulated_data/Karst-SAR-Data.csv'
data2 = mydir + '/results/simulated_data/Mason-SAR-Data.csv'
data3 = mydir + '/results/simulated_data/BigRed2-SAR-Data.csv'


def get_kdens_choose_kernel(_list,kernel):
    """ Finds the kernel density function across a sample of SADs """
    density = gaussian_kde(_list)
    n = len(_list)
    xs = np.linspace(0, 1, n)
    density.covariance_factor = lambda : kernel
    density._compute_covariance()
    D = [xs,density(xs)]
    return D


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

fs = 20
fig = plt.figure()
fig.add_subplot(1, 1, 1)
kernel = 0.2

D = get_kdens_choose_kernel(z_nest, kernel)
plt.plot(D[0],D[1],color = 'k', lw=3, alpha = 0.99, label= 'Nested SAR '+'$z$'+'-values')

plt.legend(loc='best', fontsize=fs, frameon=False)
plt.xlabel('$z$-value', fontsize=fs+3)
plt.ylabel('density', fontsize=fs)
plt.tick_params(axis='both', labelsize=fs-4)

#### Final Format and Save #####################################################
plt.subplots_adjust(wspace=0.4, hspace=0.4)
plt.savefig(mydir + '/results/figures/SAR.png', dpi=200, bbox_inches = "tight")
plt.close()
