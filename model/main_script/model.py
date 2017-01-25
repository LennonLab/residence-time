from __future__ import division
import matplotlib.animation as animation
import matplotlib.pyplot as plt

#import statsmodels.tsa.stattools as sta
#from math import isnan
from random import choice, shuffle #, randint
#from scipy import stats
import numpy as np
from numpy import sin, pi, mean
import sys
import os
import time
#import psutil

mydir = os.path.expanduser("~/")
sys.path.append(mydir + "GitHub/residence-time/model/metrics")
import metrics
sys.path.append(mydir + "GitHub/residence-time/model/LBM")
import LBM
sys.path.append(mydir + "GitHub/residence-time/model/bide")
import bide
sys.path.append(mydir + "GitHub/residence-time/model/randparams")
import randparams as rp
sys.path.append(mydir + "GitHub/residence-time/model/spatial")
#import spatial


""" To generate movies:
1.) uncomment line 'ani.save' on or near line 364
2.) adjust the frames variable on or near line 364, to change movie length
3.) change plot_system = 'no' to 'yes' on or near line 66

Because generating animations requires computing time and memory, doing so can
be computationally demanding. To quicken the process, use plot_system = 'no' on
or near line 66.
"""

# https://www.quantstart.com/articles/Basics-of-Statistical-Mean-Reversion-Testing
# http://statsmodels.sourceforge.net/0.5.0/generated/statsmodels.tsa.stattools.adfuller.html


GenPath = mydir + 'GitHub/residence-time/results/simulated_data/'


OUT = open(GenPath + 'SimData.csv','w+')
print>>OUT, 'sim,ct,motion,dormancy,immigration,res.inflow,N.types,max.res.val,max.growth.rate,max.met.maint,max.active.dispersal,barriers,starting.seed,flow.rate,width,height,viscosity,total.abundance,ind.production,biomass.prod.N,resource.tau,particle.tau,individual.tau,resource.particles,resource.concentration,species.richness,simpson.e,N.max,tracer.particles,speciation,Whittakers.turnover,avg.per.capita.growth,avg.per.capita.maint,avg.per.capita.N.efficiency,avg.per.capita.active.dispersal,amplitude,flux,frequency,phase,disturbance,spec.growth,spec.disp,spec.maint,N.active,N.dormant,Percent.Dormant,avg.per.capita.RPF,avg.per.capita.MF'
OUT.close()

OUT = open(GenPath + 'RAD-Data.csv', 'w+')
OUT.close()

OUT = open(GenPath + 'SpList-Data.csv', 'w+')
OUT.close()



def nextFrame(arg):

    """ Function called for each successive animation frame; arg is the frame number """

    global mmax, pmax, ADList, AVG_DIST, SpecDisp, SpecMaint, SpecGrowth
    global fixed, p, BurnIn, t, num_sims, width, height, Rates, u0, rho, ux, uy
    global n0, nN, nS, nE, nW, nNE, nNW, nSE, nSW, SpColorDict, GrowthDict, N_RD
    global DispDict, MaintDict, one9th, four9ths, one36th, barrier
    global gmax, dmax, maintmax, IndIDs, Qs, IndID, IndTimeIn, IndExitAge, indX
    global indY,  Ind_scatImage, SpeciesIDs, EnvD, TY, tracer_scatImage, TTimeIn
    global TIDs, TExitAge, TX, RTypes, RX, RY, RID, RIDs, RVals, RTimeIn, RExitAge
    global resource_scatImage, bN, bS, bE, bW, bNE, bNW, bSE, bSW, ct1, Mu, Maint
    global motion, reproduction, speciation, seedCom, m, r, nNi, rmax, sim
    global RAD, splist, N, ct, splist2, WT, RDens, RDiv, RRich, S, ES
    global Ev, BP, SD, Nm, sk, T, R, prod_i, prod_q, viscosity, alpha, dorm, imm
    global Ts, Rs, PRODIs, Ns, TTAUs, INDTAUs, RDENs, RDIVs, RRICHs, Ss, ESs, EVs
    global BPs, SDs, NMAXs, SKs, MUs, MAINTs, PRODNs, PRODPs, PRODCs, lefts, bottoms
    global Gs, Ms, NRs, PRs, CRs, Ds, RTAUs, GrowthList, MaintList, N_RList, MFDList, RPFList
    global DispList, amp, freq, flux, pulse, phase, disturb, envgrads
    global barriers, MainFactorDict, RPFDict, SpecRPF, SpecMF,t

    ct += 1
    plot_system = 'no'

    # fluctuate flow according to amplitude, frequency, & phase
    u1 = u0 + u0*(amp * sin(2*pi * ct * freq + phase))
    if u1 > 1.0: u1 = 1.0

    processes = range(1,14)
    shuffle(processes)

    for num in processes:

        if num == 1:
            # 1. Fluid dynamics
            nN, nS, nE, nW, nNE, nNW, nSE, nSW, barrier = LBM.stream([nN, nS, nE, nW, nNE, nNW, nSE, nSW, barrier])
            rho, ux, uy, n0, nN, nS, nE, nW, nNE, nNW, nSE, nSW = LBM.collide(viscosity, rho, ux, uy, n0, nN, nS, nE, nW, nNE, nNW, nSE, nSW, u0)

        elif num == 2: # Inflow of tracers
            TIDs, TTimeIn, TX, TY = bide.NewTracers(motion,TIDs, TX, TY, TTimeIn, width, height, u0, ct)

        elif num == 3: # moving tracer particles
            if len(TIDs) > 0:
                TIDs, TTimeIn, TExitAge, TX, TY  = bide.fluid_movement('tracer', TIDs, TTimeIn, TExitAge, TX, TY, ux, uy, width, height, u0)

        elif num == 4: # 4. Inflow of resources
            RTypes, RVals, RX, RY, RIDs, RID, RTimeIn = bide.ResIn(motion, RTypes, RVals, RX, RY,  RID, RIDs, RTimeIn, r, rmax, nNi, width, height, u1)

        elif num == 5: # 5. Resource flow
            Lists = [RTypes, RIDs, RID, RVals]
            if len(RTypes) > 0:
                RTypes, RTimeIn, RExitAge, RX, RY, RIDs, RID, RVals = bide.fluid_movement('resource', Lists, RTimeIn, RExitAge, RX, RY,  ux, uy, width, height, u0)

        elif num == 6: # 6. Inflow of individuals (immigration)
            mmax, pmax, dmax, gmax, maintmax, motion, seedCom, SpeciesIDs, IndTimeIn, IndExitAge, indX, indY,  width, height, MaintDict, MainFactorDict, RPFDict, EnvD, envgrads, GrowthDict, DispDict, SpColorDict, IndIDs, IndID, Qs, N_RD, nNi, u1, alpha, GrowthList, MaintList, MFDList, RPFList, N_RList, DispList, ADList, ct, m = bide.immigration(mmax, pmax, dmax, gmax, maintmax, motion, seedCom, SpeciesIDs, IndTimeIn, IndExitAge, indX, indY,  width, height, MaintDict, MainFactorDict, RPFDict, EnvD, envgrads, GrowthDict, DispDict, SpColorDict, IndIDs, IndID, Qs, N_RD, nNi, u1, alpha, GrowthList, MaintList, MFDList, RPFList, N_RList, DispList, ADList, ct, m)

        elif num == 7: # 7. dispersal
            Lists = [SpeciesIDs, IndIDs, IndID, Qs, DispDict, GrowthList, MaintList, MFDList, RPFList, N_RList, DispList, ADList]
            if len(SpeciesIDs) > 0:
                SpeciesIDs, IndTimeIn, IndExitAge, indX, indY, IndIDs, IndID, Qs, GrowthList, MaintList, MFDList, RPFList, N_RList, DispList, ADList = bide.fluid_movement('individual', Lists, IndTimeIn, IndExitAge, indX, indY,  ux, uy, width, height, u0)

        elif num == 8: # 8. Forage
            if len(SpeciesIDs) > 0:
                RVals, RX, RY, reproduction, speciation, SpeciesIDs, IndTimeIn, IndExitAge, indX, indY, Qs, IndIDs, IndID, width, height, GrowthDict, DispDict, SpColorDict, N_RD, MaintDict, MainFactorDict, RPFDict, EnvD, envgrads, nNi, GrowthList, MaintList, MFDList, RPFList, N_RList, DispList, ADList = bide.nearest_forage(RVals, RX, RY, reproduction, speciation, SpeciesIDs, IndTimeIn, IndExitAge, indX, indY, Qs, IndIDs, IndID, width, height, GrowthDict, DispDict, SpColorDict, N_RD, MaintDict, MainFactorDict, RPFDict, EnvD, envgrads, nNi, GrowthList, MaintList, MFDList, RPFList, N_RList, DispList, ADList)

        elif num == 9: # Consume
            if len(SpeciesIDs) > 0:
                RPFDict, RTypes, RVals, RIDs, RID, RX, RY,  RTimeIn, RExitAge, SpeciesIDs, Qs, IndIDs, IndID, IndTimeIn, IndExitAge, indX, indY,  width, height, GrowthDict, N_RD, DispDict, GrowthList, MaintList, MFDList, RPFList, MainFactorDict, N_RList, DispList, ADList = bide.consume(RPFDict, RTypes, RVals, RIDs, RID, RX, RY,  RTimeIn, RExitAge, SpeciesIDs, Qs, IndIDs, IndID, IndTimeIn, IndExitAge, indX, indY,  width, height, GrowthDict, N_RD, DispDict, GrowthList, MaintList, MFDList, RPFList, MainFactorDict, N_RList, DispList, ADList)

        elif num == 10: # Transition to or from dormancy
            if len(SpeciesIDs) > 0:
                SpeciesIDs, IndTimeIn, IndExitAge, indX, indY, IndIDs, Qs, DispList, GrowthList, MaintList, MFDList, RPFList, N_RList, MainFactorDict, RPFDict, ADList = bide.transition(SpeciesIDs, IndTimeIn, IndExitAge, indX, indY, IndIDs, Qs, DispList, GrowthList, MaintList, MFDList, RPFList, N_RList, MainFactorDict, RPFDict, ADList)

        elif num == 11: # Maintenance
            if len(SpeciesIDs) > 0:
                SpeciesIDs, IndTimeIn, IndExitAge, indX, indY, IndIDs, Qs, GrowthList, MaintList, MFDList, RPFList, N_RList, DispList, ADList = bide.maintenance(SpeciesIDs, IndTimeIn, IndExitAge, indX, indY, SpColorDict, MaintDict, MainFactorDict, RPFDict, EnvD, IndIDs, Qs, GrowthList, MaintList, MFDList, RPFList, N_RList, DispList, ADList)

        elif num == 12: # Reproduction
            p1, TNQ1 = metrics.getprod(Qs)

            if len(SpeciesIDs) > 0:
                SpeciesIDs, IndTimeIn, IndExitAge, indX, indY, Qs, IndIDs, ID, GrowthDict, DispDict, GrowthList, MaintList, MFDList, RPFList, N_RList, DispList, ADList, MainFactorDict, RPFDict = bide.reproduce(reproduction, speciation, SpeciesIDs, IndTimeIn, IndExitAge, indX, indY, Qs, IndIDs, IndID, width, height, GrowthDict, DispDict, SpColorDict, N_RD, MaintDict, MainFactorDict, RPFDict, EnvD, envgrads, nNi, GrowthList, MaintList, MFDList, RPFList, N_RList, DispList, ADList)

            p2, TNQ2 = metrics.getprod(Qs)

            PRODI = p2 - p1
            PRODN = TNQ2 - TNQ1

        elif num == 13: # Disturbance
            x = np.random.binomial(1, disturb)
            if x == 1 and len(indX) > 0:
                minN = min([1000, 0.5*len(indX)])
                SpeciesIDs, IndTimeIn, IndExitAge, indX, indY, SpColorDict, MaintDict, MainFactorDict, RPFDict, EnvD, IndIDs, Qs, GrowthList, MaintList, MFDList, RPFList, N_RList, DispList, ADList, minN = bide.decimate(SpeciesIDs, IndTimeIn, IndExitAge, indX, indY, SpColorDict, MaintDict, MainFactorDict, RPFDict, EnvD, IndIDs, Qs, GrowthList, MaintList, MFDList, RPFList, N_RList, DispList, ADList, minN)


    ax = fig.add_subplot(111)
    plt.tick_params(axis='both', which='both', bottom='off', top='off', left='off', right='off', labelbottom='off', labelleft='off')

    if len(SpeciesIDs) >= 1:  RAD, splist = bide.GetRAD(SpeciesIDs)
    else: RAD, splist, N, S = [], [], 0, 0

    RAD, splist = bide.GetRAD(SpeciesIDs)
    if len(RAD) > 1:
        RAD, splist = zip(*sorted(zip(RAD, splist), reverse=True))
    RAD = list(RAD)

    N, S, tt, R = sum(RAD), len(RAD), len(TIDs), len(RIDs)
    RDENS = R/(height*width)
    numD = ADList.count('d')
    numA = N - numD

    percD = 0
    if N > 0: percD = 100*(numD/N)

    Title = ['Individuals consume resources, grow, reproduce, and die as they move through the environment. \nAverage speed on the x-axis is '+str(u0)+' units per time step. '+str(len(TExitAge))+' tracers have passed through.\nActive N: '+str(numA)+', S: '+str(S)+', tracers: '+str(len(TExitAge))+', resources: '+str(R)+', ct: '+str(ct)+', %dormant: '+str(round(percD, 2)) + ', ' + str(len(Ns))]

    txt.set_text(' '.join(Title))
    ax.set_ylim(0, height)
    ax.set_xlim(0, width)

    if plot_system == 'yes':

        resource_scatImage.remove()
        tracer_scatImage.remove()
        Ind_scatImage.remove()

        colorlist = []
        sizelist = []
        for i, val in enumerate(SpeciesIDs):

            if ADList[i] == 'a':
                colorlist.append('r')
            else: colorlist.append('0.3')

            #colorlist.append(SpColorDict[val])
            sizelist.append(Qs[i][0] * 1000)

        resource_scatImage = ax.scatter(RX, RY, s = RVals*1, c = 'w', edgecolor = 'SpringGreen', lw = 0.6, alpha=0.3)

        Ind_scatImage = ax.scatter(indX, indY, s = sizelist, c = colorlist, edgecolor = '0.2', lw = 0.2, alpha=0.9)
        tracer_scatImage = ax.scatter(TX, TY, s = 200, c = 'r', marker='*', lw=0.0, alpha=0.6)

    Ns.append(N)

    minct = 0
    tau = np.log10(width/u1)

    if tau <= 2:     minct = 200
    elif tau <= 2.5: minct = 300
    elif tau <= 3:   minct = 400
    elif tau <= 3.5: minct = 500
    elif tau <= 4:   minct = 600
    elif tau <= 4.5: minct = 700
    elif tau <= 5:   minct = 800
    elif tau <= 5.5: minct = 900
    else: minct = 1000

    if BurnIn == 'not done':
        if len(indX) == 0 or ct >= minct:
            BurnIn = 'done'
            Ns = [Ns[-1]] # only keep the most recent N value

    if BurnIn == 'done':

        RTAU, INDTAU, TTAU = 0, 0, 0
        if len(RExitAge) > 0:
            RTAU = mean(RExitAge)
        if len(IndExitAge) > 0:
            INDTAU = mean(IndExitAge)
        if len(TExitAge) > 0:
            TTAU = mean(TExitAge)

        # Number of tracers, resource particles, and individuals
        T, R, N = len(TIDs), len(RIDs), len(SpeciesIDs)

        if N >= 1 and ct%10 == 0:

            spD = DispDict.values()
            spM = MaintDict.values()
            spG = GrowthDict.values()

            if len(spD)   > 0: SpecDisp = mean(spD)
            if len(spM)   > 0: SpecMaint = mean(spM)
            if len(spG)   > 0: SpecGrowth = mean(spG)

            ES = metrics.e_simpson(RAD)

            wt = 0
            if len(Ns) == 1:
                splist2 = list(splist)
            if len(Ns) > 1:
                wt = metrics.WhittakersTurnover(splist, splist2)
                splist2 = list(splist)

            G = mean(GrowthList)
            M = mean(MaintList)
            avgMF = mean(MFDList)
            avgRPF = mean(RPFList)
            Disp = mean(DispList)

            Nmeans = [np.var(x) for x in zip(*N_RList)]
            NR = mean(Nmeans)

            OUT = open(GenPath + 'SimData.csv', 'a')

            outlist = [sim, ct, motion, dorm, m, r, nNi, rmax, gmax, maintmax, dmax, \
            barriers, seedCom, u0, width, height, viscosity, N, PRODI, PRODN, RTAU, \
            TTAU, INDTAU, R, RDENS, S, ES, Nm, T, speciation, wt, G, M, NR, Disp, \
            amp, flux, freq, phase, disturb, SpecGrowth, SpecDisp, SpecMaint, numA, \
            numD, percD, avgRPF, avgMF]

            outlist = str(outlist).strip('[]')
            outlist = outlist.replace(" ", "")
            print>>OUT, outlist
            OUT.close()

            rad = str(RAD).strip('[]')
            rad = rad.replace(" ", "")
            OUT = open(GenPath + 'RAD-Data.csv', 'a')
            print>>OUT, sim, ',', ct,',',  rad
            OUT.close()

            splist = str(splist).strip('[]')
            splist = splist.replace(" ", "")
            OUT = open(GenPath + 'SpList-Data.csv', 'a')
            print>>OUT, sim, ',', ct,',', splist
            OUT.close()


        if len(Ns) > 50:

            t = time.clock() - t
            print 'sim:', '%4s' % sim, 'tau:', '%5s' %  round(tau,2), 'width:', '%4s' %  width,'  N:', '%4s' %  N, 'S:', '%4s' % S, 'R:', '%4s' % R, '%D:', '%4s' % round(percD,1)
            t = 0

            Rates = np.roll(Rates, -1, axis=0)
            u0 = Rates[0]

            n0, nN, nS, nE, nW, nNE, nNW, nSE, nSW, barrier, rho, ux, uy, bN, bS, bE, bW, bNE, bNW, bSE, bSW = LBM.SetLattice(u0, viscosity, width, height, lefts, bottoms, barriers)
            u1 = u0 + u0*(amp * sin(2*pi * ct * freq + phase))

            # Lists
            SpColorList, RColorList, RAD, splist, splist2, TIDs, TTimeIn, TExitAge, TX, TY, RTypes, RTimeIn, RExitAge, RX, RY, RIDs, RVals, SpeciesIDs, IndTimeIn, IndExitAge, indX, indY, IndIDs, Qs, N_RD, GrowthList, MaintList, MFDList, RPFList, N_RList, DispList, ADList = [list([]) for _ in xrange(32)]
            # Scalars
            u1, numA, numD, RDENS, RDiv, RRich, S, ES, Ev, BP, SD, Nm, sk, Mu, Maint, ct, IndID, RID, N, ct, ct1, T, R, PRODI, PRODN = [0]*25

            p = 0
            sim += 1
            BurnIn = 'not done'

            SpColorDict, GrowthDict, MaintDict, MainFactorDict, RPFDict, N_RD, RColorDict, DispDict, EnvD = {}, {}, {}, {}, {}, {}, {}, {}, {}


            if u0 == max(Rates):

                if len(Rates) > 1: print '\n'

                width, height, alpha, motion, reproduction, speciation, seedCom, m, r, nNi, rmax, gmax, maintmax, dmax, amp, freq, flux, pulse, phase, disturb, envgrads, barriers, Rates, pmax, mmax, dorm, imm = rp.get_rand_params()
                lefts, bottoms = [], []

                for i in range(barriers):
                    lefts.append(np.random.uniform(0.3, .7))
                    bottoms.append(np.random.uniform(0.1, 0.8))

                n0, nN, nS, nE, nW, nNE, nNW, nSE, nSW, barrier, rho, ux, uy, bN, bS, bE, bW, bNE, bNW, bSE, bSW = LBM.SetLattice(u0, viscosity, width, height, lefts, bottoms, barriers)
                u1 = u0 + u0*(amp * sin(2*pi * ct * freq + phase))



            ####################### REPLACE ENVIRONMENT ########################
            ax = fig.add_subplot(111)



#######################  COMMUNITY PARAMETERS  #########################

# Lists
SpColorList, RColorList, RAD, splist, splist2, TIDs, TTimeIn, TExitAge, TX, TY, RTypes, RTimeIn, RExitAge, RX, RY, RIDs, RVals, SpeciesIDs, IndTimeIn, IndExitAge, indX, indY, IndIDs, Qs, N_RD, GrowthList, MaintList, MFDList, RPFList, N_RList, DispList, ADList = [list([]) for _ in xrange(32)]
# Scalars
nNi, u1, numA, numD, RDENS, RDiv, RRich, S, ES, Ev, BP, SD, Nm, sk, Mu, Maint, ct, IndID, RID, N, ct, ct1, T, R, PRODI, PRODN = [0]*26

# Dictionaries
SpColorDict, GrowthDict, MaintDict, MainFactorDict, RPFDict, N_RD, RColorDict, DispDict, EnvD = {}, {}, {}, {}, {}, {}, {}, {}, {}


################ MODEL INPUTS ##################################
width, height, alpha, motion, reproduction, speciation, seedCom, m, r, nNi, rmax, gmax, maintmax, dmax, amp, freq, flux, pulse, phase, disturb, envgrads, barriers, Rates, pmax, mmax, dorm, imm = rp.get_rand_params()
lefts, bottoms = [], []

for b in range(barriers):
    lefts.append(np.random.uniform(0.3, .7))
    bottoms.append(np.random.uniform(0.1, 0.8))

###############  SIMULATION VARIABLES, DIMENSIONAL & MODEL CONSTANTS  ##########
num_sims, sim, viscosity = 100000, 1, 10 # viscosity is unitless but required by LBM model
u0 = Rates[0]  # initial in-flow speed

#####################  Lattice Boltzmann PARAMETERS  ###################
n0, nN, nS, nE, nW, nNE, nNW, nSE, nSW, barrier, rho, ux, uy, bN, bS, bE, bW, bNE, bNW, bSE, bSW = LBM.SetLattice(u0, viscosity, width, height, lefts, bottoms, barriers)

############### INITIALIZE GRAPHICS ############################################
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111) # initiate first plot
Ind_scatImage = ax.scatter([0],[0], alpha=0)
tracer_scatImage = ax.scatter([0],[0], alpha=0)
resource_scatImage = ax.scatter([0],[0], alpha=0)
Title = ['','']
txt = fig.suptitle(' '.join(Title), fontsize = 12)

t = time.clock()
Ns = []
BurnIn = 'not done'
p = 0.0

ani = animation.FuncAnimation(fig, nextFrame, frames=110, interval=40, blit=False) # 20000 frames is a long movie
plt.show()
#ani.save(mydir+'/GitHub/residence-time/results/movies/examples/2015_10_05_1751_hydrobide.avi', bitrate=5000)
