from __future__ import division
from random import choice, randint
import numpy as np
import sys

def get_rand_params(width=0):
    """ Get random model parameter values. Others are chosen in bide.py """

    seedCom = 1000 # size of starting community

    if width == 0:
        width = 1
    elif width < 100:
        width += 5
    elif width >= 100:
        width = 1

    height = float(width)
    length = float(height)

    low = np.random.uniform(6, 6)
    rates = np.linspace(0, -low, 50)
    rates = 10**rates
    rates = rates.tolist()
    rates.extend(rates)

    nN = randint(3, 3)
    amp = np.random.uniform(10**-1, 10**-1)
    freq = np.random.uniform(10**-1, 10**-1)
    phase = np.random.uniform(10**-1, 10**-1)
    m = np.random.uniform(0.01, 0.01)

    r = randint(10, 10)
    rmax = np.random.uniform(10, 10)

    dormlim = 0.1
    gmax = 0.9
    dmax = 0.7
    pmax = 0.6
    maintmax = 0.01

    return [width, height, length, seedCom, m, r, nN, rmax, gmax, maintmax, dmax, amp, freq, phase, rates, pmax, maintmax, dormlim]
