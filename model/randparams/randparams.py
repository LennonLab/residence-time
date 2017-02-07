from __future__ import division
from random import choice, randint
import numpy as np
import sys

def get_rand_params(width=0):
    """ Get random model parameter values. Others are chosen in bide.py """

    seedCom = 1000 # size of starting community

    width = choice([2, 2])

    height = float(width)
    length = float(height)

    low = np.random.uniform(4.5, 5.5)
    rates = np.linspace(0, -low, 10)
    rates = 10**rates

    nN = randint(1, 1)
    amp = np.random.uniform(10**-1, 10**-1)
    freq = np.random.uniform(10**-1, 10**-1)
    phase = np.random.uniform(10**-1, 10**-1)
    m = np.random.uniform(0.001, 0.001)

    r = randint(100, 100)
    rmax = np.random.uniform(100, 100)

    dormlim = 0.1
    gmax = 0.9
    dmax = 0.7
    pmax = 0.1
    maintmax = 0.01

    return [width, height, length, seedCom, m, r, nN, rmax, gmax, maintmax, dmax, amp, freq, phase, rates, pmax, maintmax, dormlim]
