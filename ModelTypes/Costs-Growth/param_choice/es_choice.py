from __future__ import division
from random import shuffle, choice, randint
from numpy import mean, log10, array, linspace, log10
from numpy.random import uniform, binomial
import  matplotlib.pyplot as plt
from scipy.stats.kde import gaussian_kde
import numpy as np


def get_kdens_choose_kernel(_list,kernel):
    """ Finds the kernel density function across a sample of SADs """
    density = gaussian_kde(_list)
    n = len(_list)
    #xs = np.linspace(0, 1, n)
    xs = linspace(min(_list), max(_list), n)
    density.covariance_factor = lambda : kernel
    density._compute_covariance()
    D = [xs,density(xs)]
    return D



nr = 20

es1 = []
es2 = []
es3 = []

for i in range(1000):

    ps = np.array(range(1, 21))/sum(range(1, 21))
    en = np.random.choice(range(1, 20+1), size = 1, replace=False, p = ps)
    es = 10**uniform(0, 3, en)
    es = es.tolist() + [0]*(nr-en)
    es = array(es)/sum(es)

    es1.append(np.var(es))

    ls = filter(lambda a: a != 0, es)
    es2.append(np.var(ls))
    es3.append(len(ls))




fig = plt.figure()

fig.add_subplot(2, 2, 1)
kernel = 0.5

D = get_kdens_choose_kernel(es1, kernel)
plt.plot(D[0],D[1],color = 'r', lw=1, alpha = 0.99, label= 'es1')


fig.add_subplot(2, 2, 2)

D = get_kdens_choose_kernel(es2, kernel)
plt.plot(D[0],D[1],color = 'b', lw=1, alpha = 0.99, label= 'es2')


fig.add_subplot(2, 2, 3)

D = get_kdens_choose_kernel(es3, kernel)
plt.plot(D[0],D[1],color = '0.3', lw=1, alpha = 0.99, label= 'es3')

plt.legend(loc=2, fontsize=10, frameon=False)
plt.show()
