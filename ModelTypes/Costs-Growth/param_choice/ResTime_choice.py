from __future__ import division
from random import shuffle, choice, randint
from numpy import mean, log10, array, linspace, log10
from numpy.random import uniform, binomial
import  matplotlib.pyplot as plt
from scipy.stats.kde import gaussian_kde

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



taus1 = []
for i in range(1000):
    tau = 10**uniform(0, 6)
    taus1.append(tau)


taus2 = []
for i in range(1000):
    tau = uniform(1, 1000000)
    taus2.append(tau)


taus3 = []
for i in range(1000):
    h = 10**uniform(0, 3)
    u = 10**uniform(-3, 0)
    tau = h/u
    taus3.append(tau)


taus4 = []
for i in range(1000):
    h = uniform(1, 1000)
    u = uniform(0.001, 1)
    tau = h/u
    taus4.append(tau)


fig = plt.figure()

fig.add_subplot(2, 2, 1)
kernel = 0.5

D = get_kdens_choose_kernel(log10(taus1), kernel)
plt.plot(D[0],D[1],color = 'crimson', lw=1, alpha = 0.99, label= 'log')

D = get_kdens_choose_kernel(log10(taus2), kernel)
plt.plot(D[0],D[1],color = 'steelblue', lw=1, alpha = 0.99,label= 'arith')

D = get_kdens_choose_kernel(log10(taus3), kernel)
plt.plot(D[0],D[1],color = '0.3', lw=1, alpha = 0.99,label= 'h/u, log')

D = get_kdens_choose_kernel(log10(taus4), kernel)
plt.plot(D[0],D[1],color = 'blue', lw=1, alpha = 0.99,label= 'h/u, arith')

plt.legend(loc=2, fontsize=10, frameon=False)




fig.add_subplot(2, 2, 2)
kernel = 0.5

D = get_kdens_choose_kernel(log10(taus1), kernel)
plt.plot(D[0],D[1],color = 'crimson', lw=1, alpha = 0.99, label= 'log')

D = get_kdens_choose_kernel(log10(taus3), kernel)
plt.plot(D[0],D[1],color = '0.3', lw=1, alpha = 0.99,label= 'h/u, log')

taus = taus1 + taus3
D = get_kdens_choose_kernel(log10(taus), kernel)
plt.plot(D[0],D[1],color = 'k', lw=2, alpha = 0.99, label= 'log')



hs1 = []
us1 = []
for i in range(2000):

    #tau = 10**uniform(0, 6)
    #h = 0
    #u = 0
    #while h < 1 or h > 1000 or u < 0.001 or u > 1.0:
        #if binomial(1, 0.5) == 1:
        #if binomial(1, 0.5) == 1:
        #    h = 10**uniform(0, 3)
        #    u = h/tau
        #elif binomial(1, 0.5):
        #    u = uniform(0.001, 1)
        #    h = u*tau
        ##else:
    h = 10**uniform(0, 3)
    u = 10**uniform(-3, 0)


    hs1.append(h)
    us1.append(u)



fig.add_subplot(2, 2, 3)

kernel = 0.5
D = get_kdens_choose_kernel(log10(hs1), kernel)
plt.plot(D[0],D[1],color = 'crimson', lw=2, alpha = 0.99, label= 'log')
plt.legend(loc=2, fontsize=10, frameon=False)


fig.add_subplot(2, 2, 4)
D = get_kdens_choose_kernel(log10(us1), kernel)
plt.plot(D[0],D[1],color = 'crimson', lw=2, alpha = 0.99, label= 'log')
plt.legend(loc=2, fontsize=10, frameon=False)
plt.show()
