from __future__ import division
from random import choice, shuffle #, randint
import numpy as np
from numpy import sin, pi, mean
import sys
import os
import time

mydir = os.path.expanduser("~/")
sys.path.append(mydir + "GitHub/residence-time/model/metrics")
import metrics
sys.path.append(mydir + "GitHub/residence-time/model/bide")
import bide2 as bide
sys.path.append(mydir + "GitHub/residence-time/model/randparams")
import randparams as rp
sys.path.append(mydir + "GitHub/residence-time/model/spatial")


GenPath = mydir + 'GitHub/residence-time/results/simulated_data/'


OUT = open(GenPath + 'SimData.csv','w+')
print>>OUT, 'sim,ct,res.inflow,N.types,max.res.val,max.growth.rate,max.met.maint,max.active.dispersal,starting.seed,flow.rate,height,length,width,total.abundance,ind.production,biomass.prod.N,resource.tau,particle.tau,individual.tau,resource.particles,resource.concentration,species.richness,simpson.e,N.max,tracer.particles,Whittakers.turnover,avg.per.capita.growth,avg.per.capita.maint,avg.per.capita.N.efficiency,avg.per.capita.active.dispersal,amplitude,flux,frequency,phase,N.active,N.dormant,Percent.Dormant,avg.per.capita.RPF,avg.per.capita.MF'
OUT.close()

OUT = open(GenPath + 'RAD-Data.csv', 'w+')
OUT.close()


#######################  COMMUNITY PARAMETERS  #########################

CRList, Ns, SpColorList, RColorList, RAD, splist, splist2, TIDs, TX, TY, RTypes, RX, RY, RZ, RIDs, RVals, SpeciesIDs, indX, indY, indZ, IndIDs, Qs, N_RD, GrowthList, MaintList, MFDList, RPFList, N_RList, DispList, ADList = [list([]) for _ in xrange(30)]
nNi, u1, numA, numD, RDENS, RDiv, RRich, S, ES, Ev, BP, SD, Nm, sk, Mu, Maint, IndID, RID, N, ct, T, R, PRODI, PRODN = [0]*24
SpColorDict, GrowthDict, MaintDict, MainFactorDict, RPFDict, N_RD, RColorDict, DispDict, EnvD = {}, {}, {}, {}, {}, {}, {}, {}, {}

################ MODEL INPUTS ##################################
width, height, length, seedCom, m, r, nNi, rmax, gmax, maintmax, dmax, amp, freq, flux, pulse, phase, envgrads, Rates, pmax, mmax = rp.get_rand_params()

###############  SIMULATION VARIABLES, DIMENSIONAL & MODEL CONSTANTS  ##########
u0 = Rates[0]  # initial in-flow speed

processes = range(1, 10)
t = time.clock()
BurnIn = 'not done'
p, sim = 0.0, 1

while sim < 100000:

    ct += 1
    numc = 0
    plot_system = 'no'

    # fluctuate flow according to amplitude, frequency, & phase
    u1 = u0 + u0*(amp * sin(2*pi * ct * freq + phase))
    if u1 > 1.0: u1 = 1.0

    shuffle(processes)
    for num in processes:

        if num == 1: # Inflow of resources
            RTypes, RVals, RX, RY, RZ, RIDs, RID = bide.ResIn(RTypes, RVals, RX, RY, RZ, RID, RIDs, r, rmax, nNi, height, length, width, u1)

        if num == 2: # Resource flow

            RTypes, RIDs, RID, RVals, RX, RY, RZ, height, length, width, u0 = bide.res_flow(RTypes, RIDs, RID, RVals, RX, RY, RZ, height, length, width, u0)

        if num == 3: # Inflow of individuals (immigration)
            CRList, mmax, pmax, dmax, gmax, maintmax, seedCom, SpeciesIDs, indX, indY, indZ, height, length, width, MaintDict, MainFactorDict, RPFDict, EnvD, envgrads, GrowthDict, DispDict, SpColorDict, IndIDs, IndID, Qs, N_RD, nNi, u1, GrowthList, MaintList, MFDList, RPFList, N_RList, DispList, ADList, ct, m = bide.immigration(CRList, mmax, pmax, dmax, gmax, maintmax, seedCom, SpeciesIDs, indX, indY, indZ, height, length, width, MaintDict, MainFactorDict, RPFDict, EnvD, envgrads, GrowthDict, DispDict, SpColorDict, IndIDs, IndID, Qs, N_RD, nNi, u1, GrowthList, MaintList, MFDList, RPFList, N_RList, DispList, ADList, ct, m)

        elif num == 4: # Dispersal
            Lists = [CRList, SpeciesIDs, IndIDs, IndID, Qs, DispDict, GrowthList, MaintList, MFDList, RPFList, N_RList, DispList, ADList]
            CRList, SpeciesIDs, indX, indY, indZ, IndIDs, IndID, Qs, GrowthList, MaintList, MFDList, RPFList, N_RList, DispList, ADList = bide.ind_flow('individual', Lists, indX, indY, indZ, height, length, width, u0)

        elif num == 5: # Consume
            numc, CRList, RPFDict, RTypes, RVals, RIDs, RID, RX, RY, RZ, SpeciesIDs, Qs, IndIDs, IndID, indX, indY, indZ, height, length, width, GrowthDict, N_RD, DispDict, GrowthList, MaintList, MFDList, RPFList, MainFactorDict, N_RList, DispList, ADList = bide.consume(numc, CRList, RPFDict, RTypes, RVals, RIDs, RID, RX, RY, RZ, SpeciesIDs, Qs, IndIDs, IndID, indX, indY, indZ, height, length, width, GrowthDict, N_RD, DispDict, GrowthList, MaintList, MFDList, RPFList, MainFactorDict, N_RList, DispList, ADList)

        elif num == 6: # Transition to dormancy
            CRList, SpeciesIDs, indX, indY, indZ, IndIDs, Qs, DispList, GrowthList, MaintList, MFDList, RPFList, N_RList, MainFactorDict, RPFDict, ADList = bide.to_dormant(CRList, SpeciesIDs, indX, indY, indZ, IndIDs, Qs, DispList, GrowthList, MaintList, MFDList, RPFList, N_RList, MainFactorDict, RPFDict, ADList)

        elif num == 7: # Transition to activity
            CRList, SpeciesIDs, indX, indY, indZ, IndIDs, Qs, DispList, GrowthList, MaintList, MFDList, RPFList, N_RList, MainFactorDict, RPFDict, ADList = bide.to_active(CRList, SpeciesIDs, indX, indY, indZ, IndIDs, Qs, DispList, GrowthList, MaintList, MFDList, RPFList, N_RList, MainFactorDict, RPFDict, ADList)

        elif num == 8: # Maintenance
            CRList, SpeciesIDs, indX, indY, indZ, IndIDs, Qs, GrowthList, MaintList, MFDList, RPFList, N_RList, DispList, ADList = bide.maintenance(CRList, SpeciesIDs, indX, indY, indZ, SpColorDict, MaintDict, MainFactorDict, RPFDict, EnvD, IndIDs, Qs, GrowthList, MaintList, MFDList, RPFList, N_RList, DispList, ADList)

        elif num == 9: # Reproduction
            p1, TNQ1 = metrics.getprod(Qs)

            CRList, SpeciesIDs, indX, indY, indZ, Qs, IndIDs, IndID, height, length, width, GrowthDict, DispDict, SpcolorDict, N_RD, MD, MFD, RPD, EnvD, envGs, nNi, GList, MList, MFDList, RPDList, NList, DList, ADList = bide.reproduce(CRList, SpeciesIDs, indX, indY, indZ, Qs, IndIDs, IndID, height, length, width, GrowthDict, DispDict, SpColorDict, N_RD, MaintDict, MainFactorDict, RPFDict, EnvD, envgrads, nNi, GrowthList, MaintList, MFDList, RPFList, N_RList, DispList, ADList)

            p2, TNQ2 = metrics.getprod(Qs)
            PRODI = p2 - p1
            PRODN = TNQ2 - TNQ1


    R, N, S = len(RIDs), len(SpeciesIDs), len(list(set(SpeciesIDs)))
    Ns.append(N)

    if N > 4000:
        # Lists
        CRList, Ns, SpColorList, RColorList, RAD, splist, splist2, TIDs, TX, TY, RTypes, RX, RY, RZ, RIDs, RVals, SpeciesIDs, indX, indY, indZ, IndIDs, Qs, GrowthList, MaintList, MFDList, RPFList, N_RList, DispList, ADList = [list([]) for _ in xrange(29)]
        # Scalars
        u1, numA, numD, RDENS, RDiv, RRich, S, ES, Ev, BP, SD, Nm, sk, Mu, Maint, IndID, RID, N, ct, T, R, PRODI, PRODN = [0]*23
        # Dictionaries
        SpColorDict, GrowthDict, MaintDict, MainFactorDict, RPFDict, N_RD, RColorDict, DispDict, EnvD = {}, {}, {}, {}, {}, {}, {}, {}, {}

        p = 0
        sim += 1
        print '\n'
        BurnIn = 'not done'
        continue

    RDENS = R/(width**3)
    numD = ADList.count(0)
    numA = N - numD

    percD = 0
    if N > 0: percD = 100*(numD/N)

    tau = np.log10((width**3)/u1)

    print 'sim:', '%4s' % sim, '  ct:', '%4s' % ct, '  tau:', '%5s' %  round(tau,3), '  width:', '%4s' %  round(width,1), 'flow:', '%5s' %  round(u0,4), '   N:', '%4s' %  N, '   S:', '%3s' %  S, '  R:', '%3s' % R, '  C:', '%4s' % numc, '  D:', '%4s' % round(percD,2)

    minct = 400 + (5**tau)

    if BurnIn == 'not done':
        if len(indX) == 0 or ct >= minct:
            BurnIn = 'done'
            Ns = [Ns[-1]] # only keep the most recent N value

    if BurnIn == 'done':

        RAD, splist = [], []
        if len(SpeciesIDs) >= 1:
            RAD, splist = bide.GetRAD(SpeciesIDs)

        RTAU, INDTAU, TTAU = 0, 0, 0

        # Number of tracers, resource particles, and individuals
        T, R, N = len(TIDs), len(RIDs), len(SpeciesIDs)

        if N >= 1 and ct%10 == 0:
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

            outlist = [sim, ct, m, r, nNi, rmax, gmax, maintmax, dmax, seedCom, u0, height, length, width, N, PRODI, PRODN, RTAU, TTAU, \
            INDTAU, R, RDENS, S, ES, Nm, T, wt, G, M, NR, Disp, amp, flux, freq, phase, numA, numD, percD, avgRPF, avgMF]

            outlist = str(outlist).strip('[]')
            outlist = outlist.replace(" ", "")
            print>>OUT, outlist
            OUT.close()

            rad = str(RAD).strip('[]')
            rad = rad.replace(" ", "")
            OUT = open(GenPath + 'RAD-Data.csv', 'a')
            print>>OUT, sim, ',', ct,',',  rad
            OUT.close()


        if len(Ns) > 100:
            print 'sim:', '%4s' % sim, 'tau:', '%5s' %  round(tau,2), 'volume:', '%4s' %  width**3,'  flow:', '%6s' %  round(u0,4), '  N:', '%4s' %  N, 'S:', '%4s' % S, 'R:', '%4s' % R, '%D:', '%4s' % round(percD,2)

            Rates = np.roll(Rates, -1, axis=0)
            u0 = Rates[0]

            CRList, Ns, SpColorList, RColorList, RAD, splist, splist2, TIDs, TX, TY, RTypes, RX, RY, RZ, RIDs, RVals, SpeciesIDs, indX, indY, indZ, IndIDs, Qs, GrowthList, MaintList, MFDList, RPFList, N_RList, DispList, ADList = [list([]) for _ in xrange(29)]
            u1, numA, numD, RDENS, RDiv, RRich, S, ES, Ev, BP, SD, Nm, sk, Mu, Maint, IndID, RID, N, ct, T, R, PRODI, PRODN = [0]*23
            SpColorDict, GrowthDict, MaintDict, MainFactorDict, RPFDict, N_RD, RColorDict, DispDict, EnvD = {}, {}, {}, {}, {}, {}, {}, {}, {}

            p, t, BurnIn = 0, 0, 'not done'
            sim += 1

            if u0 == max(Rates):
                width, height, length, seedCom, m, r, nNi, rmax, gmax, maintmax, dmax, amp, freq, flux, pulse, phase, envgrads, Rates, pmax, mmax = rp.get_rand_params(width)
