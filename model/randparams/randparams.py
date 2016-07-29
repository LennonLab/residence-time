from __future__ import division
from random import choice, randint
import numpy as np
import sys
import os

def get_rand_params(fixed):
    """ Get random model parameter values. Others are chosen in bide.py """

    envgrads = []
    seedCom = 100 # size of starting community
    rates = []

    if fixed is True:

        #rates = np.array([1.0, 0.75, 0.25, 0.1, 0.075, 0.025, 0.01, 0.0075, 0.0025, 0.001])
        rates = np.array([1.0, 0.8, 0.6, 0.4, 0.2, 0.1, 0.08, 0.06, 0.04, 0.02, 0.01, 0.008, 0.006, 0.004, 0.002, 0.001])
        #rates = np.array([0.1, 0.01, 0.001])
        #rates = np.array([1.0, 0.75, 0.5, 0.45, 0.4, 0.35, 0.3, 0.25, 0.1, 0.075, 0.05, 0.025, 0.01, 0.0075, 0.005, 0.0025, 0.001])

        #rates = np.random.uniform(-3.5, 0.0)
        #rates = [round(10**rates, 5)]

        motion = 'fluid'

        #width = width = randint(5, 21)
        width = 10
        height = 10

        num_envgrads = 2
        for i in range(num_envgrads):
            x = np.random.uniform(0, width)
            y = np.random.uniform(0, height)
            envgrads.append([x, y])

        nNi = 1 # max number of Nitrogen types
        nP = 1  # max number of Phosphorus types
        nC = 1  # max number of Carbon types

        amp = 0.001
        freq = 0.001
        phase = 0.0
        pulse = 0.001
        flux = 'yes'

        disturb = 0.0000001
        m = 0.0000001
        speciation = 0.0000001

        reproduction = 'fission'
        alpha = 0.99
        barriers = 0

        r = 120
        rmax = 100

        gmax = 0.9
        dmax = 0.5
        pmax = 0.5

        maintmax = 0.01
        mmax = 60

        # TO EXPLORE A SINGLE SET OF VALUES FOR MODEL PARAMETERS
    return [width, height, alpha, motion, reproduction, speciation, seedCom, m, r, nNi, nP, nC, rmax, gmax, maintmax, dmax, amp, freq, flux, pulse, phase, disturb, envgrads, barriers, rates, pmax, mmax]
