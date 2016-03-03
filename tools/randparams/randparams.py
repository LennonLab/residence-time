from __future__ import division
from random import choice, randint
import numpy as np
import sys
import os

def get_rand_params():
    """ Get random model parameter values. Others are chosen in bide.py """

<<<<<<< HEAD
    #motion = choice(['fluid', 'random_walk']) # 'fluid', 'unidirectional'
    motion = 'fluid'
    width = choice([20, 30, 40, 50, 60, 70, 80, 90, 100])
    #height = choice([20, 30, 40, 50, 60, 70, 80, 90, 100])
    
    width = 10
    height = 20
=======
    motion = choice(['fluid', 'random_walk']) # 'fluid', 'unidirectional'
    width = choice([20, 30, 40, 50, 60, 70, 80, 90, 100])
    height = choice([20, 30, 40, 50, 60, 70, 80, 90, 100])
>>>>>>> 75121159ac30fed40f6ec1f3c4d2509240014862
    barriers = randint(1, 4)

    pulse = np.random.uniform(0.01, 1.0)
    flux = choice(['yes'])

    # Sine wave: y(t) = amplitude * sin(2 * pi * frequency * t + phase)
    # if phi = 0, then there will be 0 amplitude at time 0
    amp = np.random.uniform(0.05, 0.5) # A
    freq = np.random.uniform(0.01, 0.1) # f
    phase = randint(0, 16) # 0 = in phase; 16 = entirely out of phase

<<<<<<< HEAD
    disturb = np.random.uniform(0.00001, 0.0001)

    #rates = np.array([1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.075, 0.05, 0.01, 0.0075, 0.005, 0.001, 0.0005, 0.0001])  # inflow speeds
    #rates = np.array([1.0, 0.5, 0.1, 0.05, 0.01])  # inflow speeds
    #rates = np.array([1.0, 0.1, 0.01])  # inflow speeds
    rates = np.array([1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.09, 0.08, 0.07, 0.06, 0.05, 0.04, 0.03, 0.02, 0.01])
    #                  0.01, 0.008, 0.006, 0.004, 0.002, 0.001, 0.0008, 0.0006, 0.0004, 0.0002])
    
    
=======
    disturb = np.random.uniform(0.0001, 0.001)

    #rates = np.array([1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.075, 0.05, 0.01, 0.0075, 0.005, 0.001, 0.0005, 0.0001])  # inflow speeds
    rates = np.array([1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2,
                      0.1, 0.09, 0.08, 0.07, 0.06, 0.05, 0.04, 0.03, 0.02,
                      0.01, 0.008, 0.006, 0.004, 0.002,
                      0.001, 0.0008, 0.0006, 0.0004, 0.0002])
    #rates = [1.0, 0.9]
    #rates = [0.001, 0.0001]

>>>>>>> 75121159ac30fed40f6ec1f3c4d2509240014862
    alpha = np.random.uniform(0.95, 0.99)
    reproduction = choice(['fission'])
    speciation = np.random.uniform(0.005, 0.05)

    seedCom = 100 # size of starting community
    m = np.random.uniform(0.01, 0.1) # m = probability of immigration

<<<<<<< HEAD
    r = randint(10, 100) #resource particles flowing in per time step
    rmax = randint(100, 1000) # maximum resource particle size
=======
    r = randint(50, 100) #resource particles flowing in per time step
    rmax = randint(100, 200) # maximum resource particle size
>>>>>>> 75121159ac30fed40f6ec1f3c4d2509240014862

    nNi = randint(1, 10) # max number of Nitrogen types
    nP = randint(1, 10) # max number of Phosphorus types
    nC = randint(1, 10) # max number of Carbon types

    envgrads = []
    num_envgrads = randint(1, 10)
<<<<<<< HEAD
    
=======
    #num_envgrads = 4

>>>>>>> 75121159ac30fed40f6ec1f3c4d2509240014862
    for i in range(num_envgrads):
        x = np.random.uniform(0, width)
        y = np.random.uniform(0, height)
        envgrads.append([x, y])

<<<<<<< HEAD
    gmax = np.random.uniform(0.1, 0.9)
    maintmax = np.random.uniform(0.0001*gmax, 0.001*gmax) # maximum metabolic maintanence cost
    dmax = np.random.uniform(0.001, 0.1) # probability of dispersing in a given time step

    # TO EXPLORE A SINGLE SET OF VALUES FOR MODEL PARAMETERS

=======
    gmax = np.random.uniform(0.1, 0.5)
    maintmax = np.random.uniform(0.0001, 0.001) # maximum metabolic maintanence cost
    dmax = np.random.uniform(0.01, 0.1) # probability of dispersing in a given time step


    # TO EXPLORE A SINGLE SET OF VALUES FOR MODEL PARAMETERS

    nNi = 4 # max number of Nitrogen types
    nP = 4 # max number of Phosphorus types
    nC = 4 # max number of Carbon types

    #amp = 0.05
    #freq = 0.01
    #phase = 0
    #pulse = 0.01
    #motion = 'fluid'
    #motion = 'random_walk'

    #disturb = 0.00001
    #m = 0.1
    #speciation = 0.01
    #maintmax = 0.001

    #reproduction = 'fission'
    #alpha = 0.99
    #width = 50
    #height = 10
    barriers = 2
    #r = 100
    #gmax = 0.5
    #rmax = 100
    #dmax = 0.01

>>>>>>> 75121159ac30fed40f6ec1f3c4d2509240014862
    return [width, height, alpha, motion, reproduction, speciation, \
            seedCom, m, r, nNi, nP, nC, rmax, gmax, maintmax, dmax, amp, freq, \
            flux, pulse, phase, disturb, envgrads, barriers, rates]
