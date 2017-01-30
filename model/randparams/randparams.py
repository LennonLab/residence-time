from __future__ import division
from random import choice, randint
import numpy as np
import sys

def get_rand_params(width=0):
    """ Get random model parameter values. Others are chosen in bide.py """

    seedCom = 1000 # size of starting community

    if width == 0:
        width += 1
    elif width == 10:
        width = 1
    else: width += 1

    height = int(width)
    length = int(height)

    low = np.random.uniform(2.5, 3.5)
    rates = np.linspace(0, -low, 10)
    rates = 10**rates

    nN = randint(3, 3)
    amp = np.random.uniform(10**-1, 10**-1)
    freq = np.random.uniform(10**-1, 10**-1)
    phase = np.random.uniform(10**-1, 10**-1)
    m = np.random.uniform(10**-1, 10**-1)

    r = randint(100, 100)
    rmax = np.random.uniform(100, 100)

    dormlim = np.random.uniform(10**-2, 10**-1)
    gmax = 0.9 #np.random.uniform(9*10**-2, 9*10**-1)
    dmax = 0.9 #np.random.uniform(9*10**-1, 9*10**-1)
    pmax = 0.1 #np.random.uniform(10**-1, 10**-1)
    maintmax = 0.01 #np.random.uniform(10**-2, 10**-2)

    return [width, height, length, seedCom, m, r, nN, rmax, gmax, maintmax, dmax, amp, freq, phase, rates, pmax, maintmax, dormlim]
