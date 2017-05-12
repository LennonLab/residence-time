from __future__ import division
from random import shuffle, randint, choice
import numpy as np
from math import isnan
import time
import copy
import sys

def e_simpson(sad): # based on 1/D, not 1 - D
    sad = filter(lambda a: a != 0, sad)
    D, N, S = [0, sum(sad), len(sad)]
    for x in sad: D += (x*x) / (N*N)
    E = round((1.0/D)/S, 4)
    return E

def WhittakersTurnover(site1, site2):
    if len(site1) == 0 or len(site2) == 0: return float('NaN')

    set1, set2 = set(site1), set(site2)
    gamma = set1.intersection(set2) # Gamma species pool
    s = len(gamma)          # Gamma richness
    bw = ((len(set1) - s) + (len(set2) - s))/np.mean([len(set1), len(set2)])
    return bw

def GetRAD(vector):
    RAD, unique = [], list(set(vector))
    for val in unique: RAD.append(vector.count(val))
    return RAD, unique


def immigration(sD, iD, ps, sd=1):
    h, r, u, nr = ps

    for j in range(sd):
        if sd < 100 and np.random.binomial(1, u) == 0: continue

        p = np.random.randint(1, 1000)
        if p not in sD:
            sD[p] = {'gr' : 10**np.random.uniform(-3, 0)} # growth rate
            sD[p]['di'] = 10**np.random.uniform(-3, 0) # active dispersal rate
            sD[p]['rp'] = 10**np.random.uniform(-3, 0) # RPF factor
            sD[p]['mt'] = 10**np.random.uniform(-3, 0) # maintenance
            sD[p]['mf'] = 10**np.random.uniform(-3, 0)
            es = 10**np.random.uniform(0, 2, nr)
            es = es/sum(es) # growth efficiencies
            sD[p]['ef'] = es.tolist()

        ID = time.time()
        iD[ID] = copy.copy(sD[p])
        iD[ID]['sp'] = p
        iD[ID]['x'] = np.random.uniform(0, h)
        iD[ID]['sz'] = np.random.uniform(0.1, 1)
        iD[ID]['q'] = np.random.uniform(0.1, 1)
        iD[ID]['st'] = 'a'

    return [sD, iD]


def ind_disp(iD, ps):
    h, r, u, nr = ps
    for k, v in iD.items():
        if v['st'] == 'a':
            iD[k]['q'] -= v['di'] * v['q']
            iD[k]['x'] -= v['di']
            if iD[k]['x'] > h or iD[k]['x'] < 0: del iD[k]

    return iD


def consume(iD, rD, ps):
    h, r, u, nr = ps
    keys = list(iD)
    shuffle(keys)
    for k in keys:

        if iD[k]['st'] == 'd': continue
        if len(list(rD)) == 0: return [iD, rD]

        c = choice(list(rD))
        e = iD[k]['ef'][rD[c]['t']] * iD[k]['q']
        iD[k]['q'] += min([rD[c]['v'], e])
        rD[c]['v'] -= min([rD[c]['v'], e])
        
	if rD[c]['x'] > h or rD[c]['v'] <= 0: del rD[c]

    return [iD, rD]


def grow(iD):
    for k, v in iD.items():
        if v['st'] == 'd': continue
        iD[k]['sz'] += v['sz']*v['gr']
        iD[k]['q'] -= v['sz']*v['gr']

	if iD[k]['sz'] <= 0 or iD[k]['q'] <= 0:            
            del iD[k]
        elif np.isnan(iD[k]['sz']) or np.isnan(iD[k]['q']):
            del iD[k]

    return iD


def maintenance(iD):
    for k, v in iD.items():
        m = v['mt']
        if v['st'] == 'd': m = v['mt'] * v['mf']

        iD[k]['q'] -= m * v['q']
        if iD[k]['q'] < 0:
            iD[k]['q'] += m * v['q']
            iD[k]['sz'] -= m * v['sz']
        
	if iD[k]['sz'] < m or iD[k]['q'] < m: 
	    del iD[k]
	elif np.isnan(iD[k]['sz']) or np.isnan(iD[k]['q']):
	    del iD[k]
    return iD


def ind_flow(iD, ps):
    h, r, u, nr = ps
    for k, val in iD.items():
        iD[k]['x'] += u
        if iD[k]['x'] > h: del iD[k]
    return iD


def reproduce(sD, iD, ps, n = 0):
    for k, v in iD.items():
        if v['st'] == 'd' or v['q'] <= 0 or np.isnan(v['sz']): continue

        if np.random.binomial(1, v['gr']) == 1:
            n += 1
            iD[k]['q'] = v['q']/2
            iD[k]['sz'] = v['sz']/2

            i = float(time.time())
            iD[i] = copy.copy(iD[k])
            if np.random.binomial(1, 10**-3) == 1:
                sD[i] = copy.copy(sD[v['sp']])
                iD[i] = copy.copy(sD[i])
		iD[i]['sp'] = i

            iD[i]['q'] = float(iD[k]['q'])
            iD[i]['sz'] = float(iD[k]['sz'])
            iD[i]['x'] = float(iD[k]['x'])
            iD[i]['st'] = 'a'

    return [sD, iD, n]



def res_flow(rD, ps):
    h, r, u, nr = ps
    for k, v in rD.items():
        rD[k]['x'] = v['x'] + u
        if rD[k]['x'] > h or rD[k]['v'] <= 0: del rD[k]
    return rD


def ResIn(rD, ps):
    h, r, u, nr = ps
    for i in range(r):
        if np.random.binomial(1, u) == 1:
            ID = time.time()
            rD[ID] = {'t' : randint(0, nr-1)}
            rD[ID]['v'] = 10**np.random.uniform(0, 2)
            rD[ID]['x'] = np.random.uniform(0, h)
    return rD


def transition(iD):
    for k, v in iD.items():
        if v['st'] == 'a' and v['q'] <= v['mt']: 
	    iD[k]['st'] = 'd'

        elif v['st'] == 'd' and np.random.binomial(1, v['rp']) == 1:
            iD[k]['q'] -= v['rp'] * v['q']
            iD[k]['st'] = 'a'
	
	if iD[k]['sz'] <= 0 or iD[k]['q'] <= 0:            
            del iD[k]
        elif np.isnan(iD[k]['sz']) or np.isnan(iD[k]['q']):
            del iD[k] 
    
    return iD


def SARt1(X1s, indC, SpID1s, h): # nested

    newX, newS = [], []
    Xs, SpIDs = list(X1s), list(SpID1s)
    xh, xl = float(h), 0

    shuffle(SpIDs)
    species = []
    areas = []
    while (xh - xl) > 1:
        for i, x in enumerate(Xs):
            if x >= xl and x <= xh:
                newX.append(x)
                newS.append(SpIDs[i])

        s = len(list(set(newS)))
        if s > 0:
            species.append(s)
            a = (xh - xl)
            areas.append(a)

        xl += 1
        xh -= 1

        Xs = list(newX)
        SpIDs = list(newS)
        newX, newS = [], []

    if len(areas) == 0 or len(species) == 0: return float('NaN')

    areas.reverse()
    areas = np.log10(areas)
    species.reverse()
    species = np.log10(species)

    A = np.vstack([areas, np.ones(len(areas))]).T
    m, c = np.linalg.lstsq(A, species)[0]
    return m


def output(iD, sD, rD, ps, sim, N, R, ct, prod, splist2):

    h, r, u, nr = ps
    m, nN, rmax, gmax, maintmax, dmax, pmax, dormlim = 1, 3, 1, 1, 1, 1, 1, 1
    N, S, R = 0, 0, 0

    indC, SpIDs, IndIDs, Qs, GList, MList, MFDList = [list([]) for _ in xrange(7)]
    RPList, NRList, DiList, DoList, ADList, SzList = [list([]) for _ in xrange(6)]
    RIDs, Rvals, Rtypes, indX, indY = [list([]) for _ in xrange(5)]

    for k, v in rD.items():
            RIDs.append(k)
            Rvals.append(v['v'])
            Rtypes.append(v['t'])

    for k, v in iD.items():
            IndIDs.append(k)
            SpIDs.append(v['sp'])
            GList.append(v['gr'])
            MList.append(v['mt'])
            MFDList.append(v['mf'])
            RPList.append(v['rp'])
            NRList.append(v['ef'])
            DiList.append(v['di'])
            ADList.append(v['st'])
            SzList.append(v['sz'])
            Qs.append(v['q'])
            indX.append(v['x'])

    N = len(IndIDs)
    S = len(list(set(SpIDs)))
    if N > 0 and S > 0:

        numD = ADList.count('d')
        pD = numD/N

        RAD, splist = GetRAD(SpIDs)
        R, N, tR = len(RIDs), len(SpIDs), sum(Rvals)
        ES = e_simpson(RAD)
        Nm = max(RAD)

	x = np.array(RAD)
        skw = sum((x - np.mean(x))**3)/((S-1)*np.std(x)**3)
        lms = np.log10(abs(float(skw)) + 1)
        if skw < 0: lms = lms * -1

        wt = WhittakersTurnover(splist, splist2)
        splist2 = list(splist)
        G = np.mean(GList)
        M = np.mean(MList)
        avgMF = np.mean(MFDList)
        avgRPF = np.mean(RPList)
        Disp = np.mean(DiList)
        Size = np.mean(SzList)

        List = []
        for n in NRList: List.append(np.var(n))
        NR = np.mean(List)

        Q = np.mean(Qs)
        avgN = N/S
        Nvar = np.var(RAD)
        Rdens = R/(h*h)

        OUT = open('SimData.csv', 'a')
        outlist = [sim, ct, m, r, nN, rmax, gmax, maintmax, dmax, 1000, u, h,\
        N, prod, prod, R, Rdens, S, ES, avgN, Nvar, Nm, lms, wt, Q, G, M, NR, \
        Disp, avgRPF, avgMF, Size, pD, dormlim, nr, tR]

        outlist = str(outlist).strip('[]')
        outlist = outlist.replace(" ", "")
        print>>OUT, outlist
        OUT.close()

        rad = str(RAD).strip('[]')
        rad = rad.replace(" ", "")
        OUT = open('RAD-Data.csv', 'a')
        print>>OUT, sim, ',', ct,',',  rad
        OUT.close()

        z1 = SARt1(indX, indC, SpIDs, h)
        if isnan(z1) == False:
            zs = str([z1]).strip('[]')
            zs = zs.replace(" ", "")
            OUT = open('SAR-Data.csv', 'a')
            print>>OUT, sim, ',', ct,',',  zs
            OUT.close()

        print 'sim:', '%3s' % sim, 'ct:', '%3s' % ct,'  N:', '%4s' %  N,
        print '  S:', '%4s' %  S, '  R:', '%4s' % R, '  Rtot:', '%4s' % round(tR, 2),
	print ' area:', h, ' u0:', '%4s' % round(u, 4)

    return splist2


def headings():
    headings = 'sim,ct,immigration.rate,'
    headings += 'res.inflow,N.types,max.res.val,'
    headings += 'max.growth.rate,max.met.maint,max.active.dispersal,'
    headings += 'starting.seed,flow.rate,area,'
    headings += 'total.abundance,ind.production,biomass.prod.N,'
    headings += 'resource.particles,resource.concentration,species.richness,'
    headings += 'simpson.e,avg.pop.size,pop.var,'
    headings += 'N.max,logmod.skew,whittakers.turnover,'
    headings += 'total.biomass,avg.per.capita.growth,avg.per.capita.maint,'
    headings += 'avg.per.capita.efficiency,avg.per.capita.active.dispersal,'
    headings += 'avg.per.capita.rpf,avg.per.capita.mf,avg.per.capita.size,'
    headings += 'percent.dormant,dorm.limit,inflowing.res.rich,total.res'
    return headings


def clear():
    OUT = open('RAD-Data.csv', 'w+').close()
    OUT = open('SAR-Data.csv', 'w+').close()
    OUT = open('SimData.csv','w+')
    h = headings()
    print>>OUT, h
    OUT.close()
    return


def iter_procs(iD, sD, rD, ps, ct):
    pr = 0
    procs = [0,1,2,3,4,5,6,7,8,9]
    shuffle(procs)
    for p in procs:
        if p is 0: rD = ResIn(rD, ps)
        elif p is 1: rD = res_flow(rD, ps)
        elif p is 2: sD, iD = immigration(sD, iD, ps)
        elif p is 3: iD = ind_flow(iD, ps)
        elif p is 4: iD = ind_disp(iD, ps)
        elif p is 5: iD, rD = consume(iD, rD, ps)
        elif p is 6: iD = grow(iD)
        elif p is 7: iD = transition(iD)
        elif p is 8: iD = maintenance(iD)
        elif p is 9: sD, iD, pr = reproduce(sD, iD, ps)

    return [iD, sD, rD, len(list(iD)), len(list(rD)), ct+1, pr]


def run_model(sim):
    rD, sD, iD, ct, splist2 = {}, {}, {}, 0, []
    h = np.random.randint(1, 1000) #int(round(10**np.random.uniform(0, 3)))
    r = np.random.randint(1, 10)
    nr = np.random.randint(1, 3)
    u = 10**np.random.uniform(-4, 0)
    ps = [h, r, u, nr]
    sD, iD = immigration(sD, iD, ps, 100)

    print '\n'
    while ct < 5000:
        iD, sD, rD, N, R, ct, prod = iter_procs(iD, sD, rD, ps, ct)
        if ct > 500 and ct%50 == 0:
            splist2 = output(iD, sD, rD, ps, sim, N, R, ct, prod, splist2)


######################## RUN MODELS ############################################
clear()
for sim in range(1, 10**4): run_model(sim)
