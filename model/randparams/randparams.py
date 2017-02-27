from __future__ import division
from random import randint
import numpy as np

def get_rand_params(width=0):
    """ Get random model parameter values. Others are chosen in bide.py """

    seedCom = 1000 # size of starting community

    if width == 0:
        width = 10
    elif width < 20:
        width += 1
    elif width >= 20:
        width = 10

    height = float(width)
    length = float(height)

    low = np.random.uniform(4, 4)
    rates = np.linspace(0, -low, 100)
    rates = 10**rates
    rates = rates.tolist()
    rates.extend(rates)

    nN = randint(3, 3)
    amp = np.random.uniform(10**-1, 10**-1)
    freq = np.random.uniform(10**-1, 10**-1)
    phase = np.random.uniform(10**-1, 10**-1)
    m = np.random.uniform(0.1, 0.1)

    r = randint(20, 20)
    rmax = np.random.uniform(10, 10)

    dormlim = np.random.uniform(0.1, 0.1)
    gmax = np.random.uniform(0.1, 0.1)
    dmax = np.random.uniform(0.3, 0.3)
    pmax = np.random.uniform(0.1, 0.1)
    maintmax = np.random.uniform(0.01, 0.01)

    return [width, height, length, seedCom, m, r, nN, rmax, gmax, maintmax, dmax, amp, freq, phase, rates, pmax, maintmax, dormlim]
