from __future__ import division
from random import choice, randint
import numpy as np
import sys
import os

def get_rand_params():
    """ Get random model parameter values. Others are chosen in bide.py """

    envgrads = []
    seedCom = 100 # size of starting community
    rates = []

    #rates = np.array([1.0, 0.75, 0.25, 0.1, 0.075, 0.025, 0.01, 0.0075, 0.0025, 0.001, 0.00075, 0.0005, 0.00025, 0.0001])
    #rates = np.array([1.0, 0.5, 0.1, 0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001])
    #rates = np.array([1.0, 0.5, 0.1])
    #rates = np.array([1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.09, 0.08, 0.07, 0.06, 0.05, 0.04, 0.03, 0.02, 0.01, 0.009, 0.008, 0.007, 0.006, 0.005, 0.004, 0.003, 0.002, 0.001, 0.0005, 0.0003, 0.0001])
    rates = np.array([1.0, 0.7, 0.5, 0.3, 0.1, 0.07, 0.05, 0.03, 0.01, 0.007, 0.005, 0.003, 0.001, 0.0005, 0.0003, 0.0001])

    motion = 'fluid'

    width = 20
    height = 10

    num_envgrads = 1
    for i in range(num_envgrads):
        x = 0.5 #np.random.uniform(0, width)
        y = 0.5 #np.random.uniform(0, height)
        envgrads.append([x, y])

    nNi = 1
    nP  = 1
    nC  = 1

    amp = 0.001
    freq = 0.001
    phase = 0.0
    pulse = 0.001
    flux = 'yes'

    disturb = 0.0000001
    m = 0.5
    speciation = 0.0001

    reproduction = 'fission'
    alpha = 0.99
    barriers = 0

    rmax = 500
    pmax = 0.5
    mmax = 100

    dorm = 'no'
    imm = 'yes'

    return [width, height, alpha, motion, reproduction, speciation, seedCom, m, nNi, nP, nC, rmax, amp, freq, flux, pulse, phase, disturb, envgrads, barriers, rates, pmax, mmax, dorm, imm]
