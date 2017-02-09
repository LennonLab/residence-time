from __future__ import division
from random import choice, shuffle #, randint
import numpy as np
from numpy import sin, pi, mean
import sys
import os
import time
from scipy import stats

mydir = os.path.expanduser("~/")
sys.path.append(mydir + "GitHub/residence-time/model/metrics")
import metrics
sys.path.append(mydir + "GitHub/residence-time/model/bide")
import bide
sys.path.append(mydir + "GitHub/residence-time/model/randparams")
import randparams as rp
sys.path.append(mydir + "GitHub/residence-time/model/spatial")


GenPath = mydir + 'GitHub/residence-time/results/simulated_data/'

'''
OUT = open(GenPath + 'SimData.csv','w+')
print>>OUT, 'sim,ct,immigration.rate,res.inflow,N.types,max.res.val,max.growth.rate,max.met.maint,max.active.dispersal,starting.seed,\
flow.rate,height,length,width,total.abundance,ind.production,biomass.prod.N,resource.particles,resource.concentration,\
species.richness,simpson.e,N.max,logmod.skew,Whittakers.turnover,amplitude,frequency,phase,\
active.biomass,dormant.biomass,active.avg.per.capita.growth,dormant.avg.per.capita.growth,\
active.avg.per.capita.maint,dormant.avg.per.capita.maint,active.avg.per.capita.efficiency,dormant.avg.per.capita.efficiency,\
active.avg.per.capita.active.dispersal,dormant.avg.per.capita.active.dispersal,active.avg.per.capita.rpf,dormant.avg.per.capita.rpf,\
active.avg.per.capita.mf,dormant.avg.per.capita.mf,N.active,S.active,N.dormant,S.Dormant,Percent.Dormant,dorm.limit'
OUT.close()

OUT = open(GenPath + 'RAD-Data.csv', 'w+')
OUT.close()
'''

#######################  COMMUNITY PARAMETERS  #########################

CRList, Ns, SpColorList, RColorList, RAD, splist, splist2, TIDs, TX, TY, RTypes, RX, RY, RZ, RIDs, RVals, SpeciesIDs, indX, indY, indZ, IndIDs, Qs, N_RD, GrowthList, MaintList, MFDList, RPFList, N_RList, DispList, ADList = [list([]) for _ in xrange(30)]
nN, u1, numA, numD, RDENS, RDiv, RRich, S, ES, Ev, BP, SD, Nm, sk, Mu, Maint, IndID, RID, N, ct, T, R, PRODI, PRODN = [0]*24
SpColorDict, GrowthDict, MaintDict, MainFactorDict, RPFDict, N_RD, RColorDict, DispDict = {}, {}, {}, {}, {}, {}, {}, {}

################ MODEL INPUTS ##################################
width, height, length, seedCom, m, r, nN, rmax, gmax, maintmax, dmax, amp, freq, phase, rates, pmax, mmax, dormlim = rp.get_rand_params()

###############  SIMULATION VARIABLES, DIMENSIONAL & MODEL CONSTANTS  ##########
u0 = rates[0]  # initial in-flow speed

processes = range(1, 10)
t = time.clock()
BurnIn = 'not done'
p, sim, ctr2 = 0.0, 18, 162


while sim < 100000:
    ct += 1
    numc = 0
    plot_system = 'no'

    # fluctuate flow according to amplitude, frequency, & phase
    u1 = float(u0) #+ u0*(amp * sin(2*pi * ct * freq + phase))
    if u1 > 1.0:
        u1 = u0

    shuffle(processes)
    for num in processes:

        if num == 1: # Inflow of resources
            RTypes, RVals, RX, RY, RZ, RIDs, RID = bide.ResIn(RTypes, RVals, RX, RY, RZ, RID, RIDs, r, rmax, nN, height, length, width, u1)

        if num == 2: # Resource flow

            RTypes, RIDs, RID, RVals, RX, RY, RZ, height, length, width, u0 = bide.res_flow(RTypes, RIDs, RID, RVals, RX, RY, RZ, height, length, width, u0)

        if num == 3: # Inflow of individuals (immigration)
            CRList, mmax, pmax, dmax, gmax, maintmax, seedCom, SpeciesIDs, indX, indY, indZ, height, length, width, MaintDict, MainFactorDict, RPFDict, GrowthDict, DispDict, SpColorDict, IndIDs, IndID, Qs, N_RD, nN, u1, GrowthList, MaintList, MFDList, RPFList, N_RList, DispList, ADList, ct, m = bide.immigration(CRList, mmax, pmax, dmax, gmax, maintmax, seedCom, SpeciesIDs, indX, indY, indZ, height, length, width, MaintDict, MainFactorDict, RPFDict, GrowthDict, DispDict, SpColorDict, IndIDs, IndID, Qs, N_RD, nN, u1, GrowthList, MaintList, MFDList, RPFList, N_RList, DispList, ADList, ct, m)

        elif num == 4: # Dispersal
            Lists = [CRList, SpeciesIDs, IndIDs, IndID, Qs, DispDict, GrowthList, MaintList, MFDList, RPFList, N_RList, DispList, ADList]
            CRList, SpeciesIDs, indX, indY, indZ, IndIDs, IndID, Qs, GrowthList, MaintList, MFDList, RPFList, N_RList, DispList, ADList = bide.ind_flow('individual', Lists, indX, indY, indZ, height, length, width, u0)

        elif num == 5: # Consume
            numc, CRList, RPFDict, RTypes, RVals, RIDs, RID, RX, RY, RZ, SpeciesIDs, Qs, IndIDs, IndID, indX, indY, indZ, height, length, width, GrowthDict, N_RD, DispDict, GrowthList, MaintList, MFDList, RPFList, MainFactorDict, N_RList, DispList, ADList = bide.consume(numc, CRList, RPFDict, RTypes, RVals, RIDs, RID, RX, RY, RZ, SpeciesIDs, Qs, IndIDs, IndID, indX, indY, indZ, height, length, width, GrowthDict, N_RD, DispDict, GrowthList, MaintList, MFDList, RPFList, MainFactorDict, N_RList, DispList, ADList)

        elif num == 6: # Transition to dormancy
            CRList, SpeciesIDs, indX, indY, indZ, IndIDs, Qs, DispList, GrowthList, MaintList, MFDList, RPFList, N_RList, MainFactorDict, RPFDict, ADList = bide.to_dormant(CRList, SpeciesIDs, indX, indY, indZ, IndIDs, Qs, DispList, GrowthList, MaintList, MFDList, RPFList, N_RList, MainFactorDict, RPFDict, ADList, dormlim)

        elif num == 7: # Transition to activity
            CRList, SpeciesIDs, indX, indY, indZ, IndIDs, Qs, DispList, GrowthList, MaintList, MFDList, RPFList, N_RList, MainFactorDict, RPFDict, ADList = bide.to_active(CRList, SpeciesIDs, indX, indY, indZ, IndIDs, Qs, DispList, GrowthList, MaintList, MFDList, RPFList, N_RList, MainFactorDict, RPFDict, ADList)

        elif num == 8: # Maintenance
            CRList, SpeciesIDs, indX, indY, indZ, IndIDs, Qs, GrowthList, MaintList, MFDList, RPFList, N_RList, DispList, ADList = bide.maintenance(CRList, SpeciesIDs, indX, indY, indZ, SpColorDict, MaintDict, MainFactorDict, RPFDict, IndIDs, Qs, GrowthList, MaintList, MFDList, RPFList, N_RList, DispList, ADList)

        elif num == 9: # Reproduction
            p1, TNQ1 = metrics.getprod(Qs)

            CRList, SpeciesIDs, indX, indY, indZ, Qs, IndIDs, IndID, height, length, width, GrowthDict, DispDict, SpcolorDict, N_RD, MD, MFD, RPD, nN, GList, MList, MFDList, RPDList, NList, DList, ADList = bide.reproduce(u0, CRList, SpeciesIDs, indX, indY, indZ, Qs, IndIDs, IndID, height, length, width, GrowthDict, DispDict, SpColorDict, N_RD, MaintDict, MainFactorDict, RPFDict, nN, GrowthList, MaintList, MFDList, RPFList, N_RList, DispList, ADList)

            p2, TNQ2 = metrics.getprod(Qs)
            PRODI = p2 - p1
            PRODN = TNQ2 - TNQ1


    R, N, S = len(RIDs), len(SpeciesIDs), len(list(set(SpeciesIDs)))

    if N > 10000:
        Lists = [CRList, SpeciesIDs, IndIDs, IndID, Qs, DispDict, GrowthList, MaintList, MFDList, RPFList, N_RList, DispList, ADList]
        CRList, SpeciesIDs, indX, indY, indZ, IndIDs, Qs, GrowthList, MaintList, MFDList, RPFList, N_RList, DispList, ADList = bide.decimate(Lists, indX, indY, indZ, height, length, width, u0)


    Ns.append(N)

    RDENS = R/(width**3)
    numD = ADList.count(0)
    numA = N - numD

    percD = 0
    if N > 0: percD = 100*(numD/N)

    tau = np.log10((width**3)/u0)
    minct = 600 + (2**tau)

    print 'sim:', '%4s' % sim, 'ct:', '%3s' % ctr2, '  t:', '%6s' % str(round(minct - ct)), '  tau:', '%5s' %  round(tau,3), '  width:', '%4s' %  round(width,1), 'flow:', '%5s' %  round(u0,4), '   N:', '%4s' %  N, '   S:', '%3s' %  S, '  R:', '%3s' % R, '  C:', '%4s' % numc, '  D:', '%4s' % round(percD,2)


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

            Lists = [SpeciesIDs, IndIDs, IndID, Qs, DispDict, GrowthList, MaintList, MFDList, RPFList, N_RList, DispList, ADList]
            aLists, dLists = metrics.separateCom(Lists)
            a_SpeciesIDs, a_IndIDs, a_Qs, a_GrowthList, a_MaintList, a_MFDList, a_RPFList, a_N_RList, a_DispList = aLists
            d_SpeciesIDs, d_IndIDs, d_Qs, d_GrowthList, d_MaintList, d_MFDList, d_RPFList, d_N_RList, d_DispList = dLists

            aRAD, asplist = bide.GetRAD(a_SpeciesIDs)
            dRAD, dsplist = bide.GetRAD(d_SpeciesIDs)
            aCOM = []
            dCOM = []

            for sp in splist:
                if sp in asplist:
                    i = asplist.index(sp)
                    aCOM.append(aRAD[i])
                else: aCOM.append(0)

                if sp in dsplist:
                    i = dsplist.index(sp)
                    dCOM.append(dRAD[i])
                else: dCOM.append(0)


            ES = metrics.e_simpson(RAD)
            Nm = max(RAD)

            skew = stats.skew(RAD)
            # log-modulo transformation of skewnness
            lms = np.log10(np.abs(float(skew)) + 1)
            if skew < 0:
                lms = lms * -1

            wt = 0
            if len(Ns) == 1:
                splist2 = list(splist)
            if len(Ns) > 1:
                wt = metrics.WhittakersTurnover(splist, splist2)
                splist2 = list(splist)

            SA = len(aRAD)
            SD = len(dRAD)

            a_G = mean(a_GrowthList)
            a_M = mean(a_MaintList)
            a_avgMF = mean(a_MFDList)
            a_avgRPF = mean(a_RPFList)
            a_Disp = mean(a_DispList)

            d_G = mean(d_GrowthList)
            d_M = mean(d_MaintList)
            d_avgMF = mean(d_MFDList)
            d_avgRPF = mean(d_RPFList)
            d_Disp = mean(d_DispList)

            a_NR = mean(a_N_RList)
            if sum(a_N_RList) == 0:
                a_NR = 0

            d_NR = mean(d_N_RList)
            if sum(d_N_RList) == 0:
                d_NR = 0

            a_Q = np.mean(a_Qs)
            d_Q = np.mean(d_Qs)


            Nmeans = [np.var(x) for x in zip(*N_RList)]
            NR = mean(Nmeans)
            OUT = open(GenPath + 'SimData.csv', 'a')
            outlist = [sim, ctr2, m, r, nN, rmax, gmax, maintmax, dmax, seedCom,\
            u0, height, length, width, N, PRODI, PRODN, R, RDENS,\
            S, ES, Nm, lms, wt, amp, freq, phase,\
            a_Q, d_Q, a_G, d_G, a_M, d_M, a_NR, d_NR,\
            a_Disp, d_Disp, a_avgRPF, d_avgRPF,\
            a_avgMF, d_avgMF,numA, SA, numD, SD, percD, dormlim]

            outlist = str(outlist).strip('[]')
            outlist = outlist.replace(" ", "")
            print>>OUT, outlist
            OUT.close()

            rad = str(RAD).strip('[]')
            rad = rad.replace(" ", "")
            OUT = open(GenPath + 'RAD-Data.csv', 'a')
            print>>OUT, sim, ',', ct,',',  rad
            OUT.close()


        if len(Ns) > 1000:

            ctr2 += 1
            print 'sim:', '%4s' % sim, 'tau:', '%5s' %  round(tau,2), 'volume:', '%4s' %  width**3,'  flow:', '%6s' %  round(u0,4), '  N:', '%4s' %  N, 'S:', '%4s' % S, 'R:', '%4s' % R, '%D:', '%4s' % round(percD,2)

            rates = np.roll(rates, -1, axis=0)
            u0 = rates[0]

            CRList, Ns, SpColorList, RColorList, RAD, splist, splist2, TIDs, TX, TY, RTypes, RX, RY, RZ, RIDs, RVals, SpeciesIDs, indX, indY, indZ, IndIDs, Qs, GrowthList, MaintList, MFDList, RPFList, N_RList, DispList, ADList = [list([]) for _ in xrange(29)]
            u1, numA, numD, RDENS, RDiv, RRich, S, ES, Ev, BP, SD, Nm, sk, Mu, Maint, IndID, RID, N, ct, T, R, PRODI, PRODN = [0]*23
            #SpColorDict, GrowthDict, MaintDict, MainFactorDict, RPFDict, N_RD, RColorDict, DispDict = {}, {}, {}, {}, {}, {}, {}, {}

            p, t, BurnIn = 0, 0, 'not done'

            if u0 == max(rates):
                #ctr2 = 0
                width, height, length, seedCom, m, r, nN, rmax, gmax, maintmax, dmax, amp, freq, phase, rates, pmax, mmax, dormlim = rp.get_rand_params(width)
                SpColorDict, GrowthDict, MaintDict, MainFactorDict, RPFDict, N_RD, RColorDict, DispDict = {}, {}, {}, {}, {}, {}, {}, {}
                sim += 1
