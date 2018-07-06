from __future__ import division
from lazyme.string import color_print
import numpy as np


def merge_dicts(*dict_args):
    """
    Given any number of dicts, shallow copy and merge into a new dict,
    precedence goes to key value pairs in latter dicts.
    """
    result = {}
    for dictionary in dict_args:
        result.update(dictionary)
    return result


def shannon(sad):
    if len(sad) == 0: return float('NaN')
    sad = filter(lambda a: a != 0, sad)
    sad = np.array(sad)
    H = sum(-sad*np.log(sad))
    return H


def e_simpson(sad):
    if len(sad) == 0: return float('NaN')
    sad = filter(lambda a: a != 0, sad)
    D, N, S = [0, sum(sad), len(sad)]
    for x in sad: D += (x*x) / (N*N)
    E = round((1.0/D)/S, 4)
    return E

def WhittakersTurnover(site1, site2):
    if len(site1) == 0 or len(site2) == 0: return float('NaN')

    set1, set2 = set(site1), set(site2)
    gamma = set1.intersection(set2) # Gamma species pool
    s = len(gamma)  # Gamma richness
    bw = ((len(set1) - s) + (len(set2) - s))
    bw = bw/np.mean([len(set1), len(set2)])
    return bw

def GetRAD(vector):
    RAD, unique = [], list(set(vector))
    for val in unique: RAD.append(vector.count(val))
    return RAD, unique



def output(iD, sD, rD, ps, sim, t, ct, prod, splist2, asplist2, dsplist2, D, minlim, pD, clr, Ri):

    h, h1, hvar, r, u, u1, uvar, nr, im = ps
    nN, rmax, gmax, maintmax, dmax, pmax = 3, 1, 1, 1, 1, 1
    N, S, R = 0, 0, 0

    G, M, MF, RP, Di, Sz, S_NR, S_GL, S_ML, S_MF, S_RP, S_Di = [0]*12
    aG, aM, aMF, aRP, aDi, aSz, aS_NR, aS_GL, aS_ML, aS_MF, aS_RP, aS_Di = [0]*12
    dG, dM, dMF, dRP, dDi, dSz, dS_NR, dS_GL, dS_ML, dS_MF, dS_RP, dS_Di = [0]*12

    SpIDs, IndIDs, Qs, GList, MList, MFList, AList = [list([]) for _ in xrange(7)]
    aAList, dAList, aQs, dQs, RepList, aRepList, dRepList = [list([]) for _ in xrange(7)]
    RPList, DiList, DoList, ADList, SzList = [list([]) for _ in xrange(5)]
    RIDs, Rvals, Rtypes, indX, a_indX, d_indX = [list([]) for _ in xrange(6)]
    aIndIDs, aSpIDs, aGList, aMList, aMFList, aRPList, aDiList, aSzList, aQs = [list([]) for _ in xrange(9)]
    dIndIDs, dSpIDs, dGList, dMList, dMFList, dRPList, dDiList, dSzList, dQs = [list([]) for _ in xrange(9)]

    S_GList, S_MList, S_MFList, S_RPList, S_DiList = [list([]) for _ in xrange(5)]
    aS_GList, aS_MList, aS_MFList, aS_RPList, aS_DiList = [list([]) for _ in xrange(5)]
    dS_GList, dS_MList, dS_MFList, dS_RPList, dS_DiList = [list([]) for _ in xrange(5)]

    aNRList1, aNRList2, aNRList3 = [list([]) for _ in xrange(3)]
    dNRList1, dNRList2, dNRList3 = [list([]) for _ in xrange(3)]
    NRList1, NRList2, NRList3 = [list([]) for _ in xrange(3)]

    aNRList1e, aNRList2e, aNRList3e = [list([]) for _ in xrange(3)]
    dNRList1e, dNRList2e, dNRList3e = [list([]) for _ in xrange(3)]
    NRList1e, NRList2e, NRList3e = [list([]) for _ in xrange(3)]

    S_NRList1, S_NRList2, S_NRList3 = [list([]) for _ in xrange(3)]
    aS_NRList1, aS_NRList2, aS_NRList3 = [list([]) for _ in xrange(3)]
    dS_NRList1, dS_NRList2, dS_NRList3 = [list([]) for _ in xrange(3)]



    for k, v in rD.items():
        RIDs.append(k)
        Rvals.append(v['v'])
        Rtypes.append(v['t'])

    R, tR, Rrich = len(RIDs), sum(Rvals), len(list(set(Rtypes)))
    Rdens = R/h

    for k, v in iD.items():

        IndIDs.append(k)
        AList.append(v['age'])
        SpIDs.append(v['sp'])
        GList.append(v['gr'])

        if v['st'] == 'a': MList.append(v['mt'])
        elif v['st'] == 'd':
            MList.append(v['mt'] * v['mf'])

        MFList.append(v['mf'])
        RPList.append(v['rp'])

        NRList1.append(np.var(v['ef']))
        ls = v['ef']
        ls = filter(lambda a: a != 0, ls)
        NRList2.append(np.var(ls))
        NRList3.append(1/len(ls))

        NRList1e.append(np.var(v['e_ef']))
        ls = v['e_ef']
        ls = filter(lambda a: a != 0, ls)
        NRList2e.append(np.var(ls))
        NRList3e.append(len(ls))

        DiList.append(v['di'])
        SzList.append(v['sz'])
        Qs.append(v['q'])
        indX.append(v['x'])

        ri = v['q']/v['mt']
        rp = ri/(1+ri) * v['sz']/(1+v['sz'])
        RepList.append(rp)

        if v['st'] == 'a':
            aIndIDs.append(k)
            aSpIDs.append(v['sp'])
            aGList.append(v['gr'])
            aMList.append(v['mt'])
            aMFList.append(v['mf'])
            aRPList.append(v['rp'])
            aAList.append(v['age'])
            aQs.append(v['q'])

            ri = v['q']/v['mt']
            rp = ri/(1+ri) * v['sz']/(1+v['sz'])
            aRepList.append(rp)

            aNRList1.append(np.var(v['ef']))
            ls = v['ef']
            ls = filter(lambda a: a != 0, ls)
            aNRList2.append(np.var(ls))
            aNRList3.append(1/len(ls))

            aNRList1e.append(np.var(v['e_ef']))
            ls = v['e_ef']
            ls = filter(lambda a: a != 0, ls)
            aNRList2e.append(np.var(ls))
            aNRList3e.append(len(ls))

            aDiList.append(v['di'])
            aSzList.append(v['sz'])
            aQs.append(v['q'])
            a_indX.append(v['x'])

        elif v['st'] == 'd':
            dIndIDs.append(k)
            dSpIDs.append(v['sp'])
            dGList.append(v['gr'])
            dMList.append(v['mt'] * v['mf'])
            dMFList.append(v['mf'])
            dRPList.append(v['rp'])
            dAList.append(v['age'])
            dQs.append(v['q'])

            ri = v['q']/v['mt']
            rp = ri/(1+ri) * v['sz']/(1+v['sz'])
            dRepList.append(rp)

            dNRList1.append(np.var(v['ef']))
            ls = v['ef']
            ls = filter(lambda a: a != 0, ls)
            dNRList2.append(np.var(ls))
            dNRList3.append(1/len(ls))

            dNRList1e.append(np.var(v['e_ef']))
            ls = v['e_ef']
            ls = filter(lambda a: a != 0, ls)
            dNRList2e.append(np.var(ls))
            dNRList3e.append(len(ls))

            dDiList.append(v['di'])
            dSzList.append(v['sz'])
            dQs.append(v['q'])
            d_indX.append(v['x'])

    NR1 = np.mean(NRList1)
    NR1e = np.mean(NRList1e)
    S_NR1 = np.mean(list(set(NRList1)))

    NR2 = np.mean(NRList2)
    NR2e = np.mean(NRList2e)
    S_NR2 = np.mean(list(set(NRList2)))

    NR3 = np.mean(NRList3)
    NR3e = np.mean(NRList3e)
    S_NR3 = np.mean(list(set(NRList3)))

    aNR1 = np.mean(aNRList1)
    aNR1e = np.mean(aNRList1e)
    aS_NR1 = np.mean(list(set(aNRList1)))

    aNR2 = np.mean(aNRList2)
    aNR2e = np.mean(aNRList2e)
    aS_NR2 = np.mean(list(set(aNRList2)))

    aNR3 = np.mean(aNRList3)
    aNR3e = np.mean(aNRList3e)
    aS_NR3 = np.mean(list(set(aNRList3)))

    dNR1 = np.mean(dNRList1)
    dNR1e = np.mean(dNRList1e)
    dS_NR1 = np.mean(list(set(dNRList1)))

    dNR2 = np.mean(dNRList2)
    dNR2e = np.mean(dNRList2e)
    dS_NR2 = np.mean(list(set(dNRList2)))

    dNR3 = np.mean(dNRList3)
    dNR3e = np.mean(dNRList3e)
    dS_NR3 = np.mean(list(set(dNRList3)))

    N = len(IndIDs)
    aN = len(aIndIDs)
    dN = len(dIndIDs)
    S = len(list(set(SpIDs)))
    aS = len(list(set(aSpIDs)))
    dS = len(list(set(dSpIDs)))

    RAD, splist = GetRAD(SpIDs)
    ES = e_simpson(RAD)
    if len(RAD) == 0: Nm = float('NaN')
    else: Nm = max(RAD)

    aRAD, asplist = GetRAD(aSpIDs)
    aES = e_simpson(aRAD)
    if len(aRAD) == 0: aNm = float('NaN')
    else: aNm = max(aRAD)

    dRAD, dsplist = GetRAD(dSpIDs)
    dES = e_simpson(dRAD)
    if len(dRAD) == 0: dNm = float('NaN')
    else: dNm = max(dRAD)

    if S > 0:
        x = np.array(RAD)
        skw = sum((x - np.mean(x))**3)/((S-1)*np.std(x)**3)
        lms = np.log10(abs(float(skw)) + 1)
        if skw < 0: lms = lms * -1
    else: lms = float('NaN')

    if aS > 0:
        ax = np.array(aRAD)
        askw = sum((ax - np.mean(ax))**3)/((aS-1)*np.std(ax)**3)
        alms = np.log10(abs(float(askw)) + 1)
        if askw < 0: alms = alms * -1
    else: alms = float('NaN')

    if dS > 0:
        dx = np.array(dRAD)
        dskw = sum((dx - np.mean(dx))**3)/((dS-1)*np.std(dx)**3)
        dlms = np.log10(abs(float(dskw)) + 1)
        if dskw < 0: dlms = dlms * -1
    else: dlms = float('NaN')

    wt = WhittakersTurnover(splist, splist2)
    splist2 = list(splist)
    awt = WhittakersTurnover(asplist, asplist2)
    asplist2 = list(asplist)
    dwt = WhittakersTurnover(dsplist, dsplist2)
    dsplist2 = list(dsplist)

    avgA = np.mean(AList)
    varA = np.var(AList)
    G = np.mean(GList)
    M = np.mean(MList)
    MF = np.mean(MFList)
    RP = np.mean(RPList)
    Di = np.mean(DiList)
    Sz = np.mean(SzList)
    S_G = np.mean(list(set(GList)))
    S_M = np.mean(list(set(MList)))
    S_MF = np.mean(list(set(MFList)))
    S_RP = np.mean(list(set(RPList)))
    S_Di = np.mean(list(set(DiList)))

    avg_active_A = np.mean(aAList)
    aRep = np.mean(aRepList)
    Rep = np.mean(RepList)
    dRep = np.mean(dRepList)
    aG = np.mean(aGList)
    aM = np.mean(aMList)
    aMF = np.mean(aMFList)
    aRP = np.mean(aRPList)
    aDi = np.mean(aDiList)
    aSz = np.mean(aSzList)
    aS_G = np.mean(list(set(aGList)))
    aS_M = np.mean(list(set(aMList)))
    aS_MF = np.mean(list(set(aMFList)))
    aS_RP = np.mean(list(set(aRPList)))
    aS_Di = np.mean(list(set(aDiList)))

    avg_dormant_A = np.mean(dAList)
    dG = np.mean(dGList)
    dM = np.mean(dMList)
    dMF = np.mean(dMFList)
    dRP = np.mean(dRPList)
    dDi = np.mean(dDiList)
    dSz = np.mean(dSzList)
    dS_G = np.mean(list(set(dGList)))
    dS_M = np.mean(list(set(dMList)))
    dS_MF = np.mean(list(set(dMFList)))
    dS_RP = np.mean(list(set(dRPList)))
    dS_Di = np.mean(list(set(dDiList)))

    Q = np.mean(Qs)
    aQ = np.mean(aQs)
    dQ = np.mean(dQs)

    Nvar = np.var(RAD)
    aNvar = np.var(aRAD)
    dNvar = np.var(dRAD)

    avgN = np.mean(RAD)
    aavgN = np.mean(aRAD)
    davgN = np.mean(dRAD)

    OUT = open('results/data/SimData.csv', 'a')
    outlist = [sim, clr, t, ct, im, r, nN, rmax, gmax, maintmax, dmax, 1000, u, h,\
    N, aN, dN, prod, R, Rdens, Rrich, S, ES, avgN, Nvar, Nm, lms, wt, Q, G, M, \
    NR1, NR2, NR3, NR1e, NR2e, NR3e,\
    Di, RP, MF, Sz, aS, aES, aavgN, aNvar, aNm, alms, awt, aQ,\
    aG, aM, aNR1, aNR2, aNR3, aNR1e, aNR2e, aNR3e, aDi, aRP, aMF, aSz, dS, dES, \
    davgN, dNvar, dNm, dlms, dwt, dQ, dG, dM, dNR1, dNR2, dNR3, dNR1e, dNR2e, dNR3e, \
    dDi, dRP, dMF, dSz, pD, nr, tR, S_NR1, S_NR2, S_NR3, S_G, S_M, S_MF, S_RP, S_Di, \
    aS_NR1, aS_NR2, aS_NR3, aS_G, aS_M, aS_MF, aS_RP, aS_Di, dS_NR1, dS_NR2, \
    dS_NR3, dS_G, dS_M, dS_MF, dS_RP, dS_Di, avgA, avg_active_A,
    avg_dormant_A, varA, aRep, Rep, dRep, h1, hvar, u1, uvar]

    outlist = str(outlist).strip('[]')
    outlist = outlist.replace(" ", "")
    print>>OUT, outlist
    OUT.close()

    OUT = open('results/data/RAD-Data.csv', 'a')
    print>>OUT, sim, ',', h/u, ',', ct,',',  RAD, ',', splist
    OUT.close()

    OUT = open('results/data/active.RAD-Data.csv', 'a')
    print>>OUT, sim, ',', h/u, ',', ct,',',  aRAD, ',', asplist
    OUT.close()

    OUT = open('results/data/dormant.RAD-Data.csv', 'a')
    print>>OUT, sim, ',', h/u, ',', ct, ',',  dRAD, ',', dsplist
    OUT.close()

    Rp = R #(R/h)/((R/h)+1)
    string = 'sim:'+'%4s' % str(sim)+' ct:'+'%5s' % str(int(round(minlim-ct, 0)))
    string += '  tau:''%6s' % str(round(np.log10(h/u), 2))
    string +=  ' N:'+'%5s' % str(N)+' S:'+'%5s' % str(S)
    string += '  R:'+'%6s' % str(round(Rp, 4))+' P:'+'%4s' % str(round(prod, 2))
    string += '  %D:'+'%5s' % str(round(100*pD,1))
    string += '  Q:'+'%6s' % str(round(Q,2))
    string += '  Sz:'+'%6s' % str(round(Sz,2))
    string += '  Ri:'+'%6s' % str(round(Ri,2))
    color_print(string, color='green')

    return splist2, asplist2, dsplist2



def headings():

    headings = 'sim,color,time,ct,immigration.rate,inflowing.res.dens,N.types,max.res.val,max.growth.rate,'
    headings += 'max.met.maint,max.dispersal,starting.seed,Q,V,'
    headings += 'total.abundance,active.total.abundance,dormant.total.abundance,ind.production,'
    headings += 'resource.particles,resource.concentration,resource.richness,'

    headings += 'species.richness,simpson.e,avg.pop.size,'
    headings += 'pop.var,N.max,logmod.skew,whittakers.turnover,'
    headings += 'total.biomass,avg.per.capita.growth,avg.per.capita.maint,'
    headings += 'avg.per.capita.efficiency1,avg.per.capita.efficiency2,avg.per.capita.efficiency3,'
    headings += 'avg.per.capita.efficiency1e,avg.per.capita.efficiency2e,avg.per.capita.efficiency3e,'
    headings += 'avg.per.capita.dispersal,'
    headings += 'avg.per.capita.rpf,avg.per.capita.mf,avg.per.capita.size,'

    headings += 'active.species.richness,active.simpson.e,active.avg.pop.size,'
    headings += 'active.pop.var,active.N.max,active.logmod.skew,active.whittakers.turnover,'
    headings += 'active.total.biomass,active.avg.per.capita.growth,active.avg.per.capita.maint,'
    headings += 'active.avg.per.capita.efficiency1,active.avg.per.capita.efficiency2,active.avg.per.capita.efficiency3,'
    headings += 'active.avg.per.capita.efficiency1e,active.avg.per.capita.efficiency2e,active.avg.per.capita.efficiency3e,'
    headings += 'active.avg.per.capita.dispersal,'
    headings += 'active.avg.per.capita.rpf,active.avg.per.capita.mf,active.avg.per.capita.size,'

    headings += 'dormant.species.richness,dormant.simpson.e,dormant.avg.pop.size,'
    headings += 'dormant.pop.var,dormant.N.max,dormant.logmod.skew,dormant.whittakers.turnover,'
    headings += 'dormant.total.biomass,dormant.avg.per.capita.growth,dormant.avg.per.capita.maint,'
    headings += 'dormant.avg.per.capita.efficiency1,dormant.avg.per.capita.efficiency2,dormant.avg.per.capita.efficiency3,'
    headings += 'dormant.avg.per.capita.efficiency1e,dormant.avg.per.capita.efficiency2e,dormant.avg.per.capita.efficiency3e,'
    headings += 'dormant.avg.per.capita.dispersal,'
    headings += 'dormant.avg.per.capita.rpf,dormant.avg.per.capita.mf,dormant.avg.per.capita.size,'

    headings += 'percent.dormant,inflowing.res.rich,total.res,'

    headings += 'efficiency1,efficiency2,efficiency3,'
    headings += 'growth,maint,mf,rpf,dispersal,'
    headings += 'active.efficiency1,active.efficiency2,active.efficiency3,'
    headings += 'active.growth,active.maint,active.mf,active.rpf,active.dispersal,'
    headings += 'dormant.efficiency1,dormant.efficiency2,dormant.efficiency3,'
    headings += 'dormant.growth,dormant.maint,dormant.mf,dormant.rpf,dormant.dispersal,'
    headings += 'avg.age,avg.active.age,avg.dormant.age,var.age,active.repro.p,repro.p,dormant.repro.p,h1,h.var,u1,u.var'
    return headings


def clear():
    OUT = open('results/data/RAD-Data.csv', 'w+').close()
    OUT = open('results/data/active.RAD-Data.csv', 'w+').close()
    OUT = open('results/data/dormant.RAD-Data.csv', 'w+').close()
    OUT = open('results/data/SimData.csv','w+')

    h = headings()
    print>>OUT, h
    OUT.close()
    return
