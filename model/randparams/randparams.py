from __future__ import division
from random import choice, randint
import numpy as np
import sys

def get_rand_params(width=0):
    """ Get random model parameter values. Others are chosen in bide.py """

    envgrads = []
    seedCom = 1000 # size of starting community

    if width == 0:
        width += 1
    elif width == 10:
        width = 1
    else: width += 1

    width = 4
    height = int(width)
    length = int(height)

    low = np.random.uniform(3.5, 4.5)
    rates = np.linspace(1, -low, 100)
    rates = 10**rates
    #rates = np.array([choice(rates)])

    num_envgrads = randint(2, 2)
    for i in range(num_envgrads):
        x = np.random.uniform(0, width)
        y = np.random.uniform(0, height)
        envgrads.append([x, y])

    nN = randint(3, 3)
    amp = np.random.uniform(0.01, 0.01)
    freq = np.random.uniform(0.01, 0.01)
    phase = np.random.uniform(0.01, 0.01)
    pulse = np.random.uniform(0.01, 0.01)
    m = np.random.uniform(0.00000001, 0.00000001)
    flux = 'yes'

    r = randint(20, 20)
    rmax = np.random.uniform(10, 10)

    gmax = np.random.uniform(0.9, 0.9)
    dmax = np.random.uniform(0.9, 0.9)
    pmax = np.random.uniform(10**-1, 10**-1)
    maintmax = np.random.uniform(10**-2, 10**-2)
    mmax = np.random.uniform(20, 20)

    return [width, height, length, seedCom, m, r, nN, rmax, gmax, maintmax, dmax, amp, freq, flux, pulse, phase, envgrads, rates, pmax, mmax]
