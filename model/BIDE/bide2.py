# -*- coding: utf-8 -*-
from __future__ import division
from random import randint, choice
import numpy as np
import sys
import math

cost = 0.06
tpoint = 0.01

def checkVal(val, line):
    if val < 0:
        print 'line',line,': error: val < 0:', val
        sys.exit()
    return


def decomposition(i, n):
    ct = 1
    gn = []

    while ct < n:
        ct += 1
        val = np.random.uniform(0.0, i)
        gn.append(val)
        i -= val

    gn.append(1.0 - sum(gn))
    return gn



def GetRAD(vector):
    RAD = []
    unique = list(set(vector))

    for val in unique:
        RAD.append(vector.count(val)) # the abundance of each Sp_

    return RAD, unique # the rad and the specieslist



def get_color(ID, colorD): # FUNCTION TO ASSIGN COLORS TO Sp_

    r1 = lambda: randint(0,255)
    r2 = lambda: randint(0,255)
    r3 = lambda: randint(0,255)

    color = '#%02X%02X%02X' % (r1(),r2(),r3())
    colorD[ID] = color

    return colorD



def ResIn(Type, Vals, Xs, Ys, Zs, ID, IDs, numr, rmax, nN, h, l, w, u0):

    for r in range(numr):
        x = np.random.binomial(1, 0.99*u0)

        if x == 1:

            rval = np.random.uniform(rmax/1, rmax)
            rtype = randint(0, nN-1)
            Type.append(rtype)
            Vals.append(rval)
            IDs.append(ID)
            ID += 1

            Xs.append(float(np.random.uniform(0, h)))
            Ys.append(float(np.random.uniform(0, l)))
            Zs.append(float(np.random.uniform(0, w)))


    return [Type, Vals, Xs, Ys, Zs, IDs, ID]



def immigration(CRList, mfmax, p_max, d_max, g_max, m_max, seed, Sp, Xs, Ys, Zs, h, l, w, MD, MFD, RPD,
        EnvD, envGs, GD, DispD, colorD, IDs, ID, Qs, N_RD, nN, u0, GList, MList, MFDList, RPDList, NList, DList, ADList, ct, m):

    if u0 > 1.0:
        u0 = 1.0


    sd = int(seed)
    if ct > 1:
        sd = 1

    for i in range(sd):
        x = 0

        if sd > 1:
            x = 1
        else:
            x = np.random.binomial(1, u0*m)


        if x == 1:
            prop = np.random.randint(1, 10000)
            Sp.append(prop)

            Xs.append(float(np.random.uniform(0, h)))
            Ys.append(float(np.random.uniform(0, l)))
            Zs.append(float(np.random.uniform(0, w)))

            IDs.append(ID)
            ID += 1
            Qn = float(np.random.uniform(0.25, 0.75))

            Qs.append(Qn)

            if prop not in colorD:

                # species growth rate
                GD[prop] = np.random.uniform(g_max/10, g_max)

                # species RPF factor
                RPD[prop] = np.random.uniform(p_max/10, p_max)

                # species active dispersal rate
                DispD[prop] = np.random.uniform(d_max/10, d_max)

                # species maintenance
                MD[prop] =  np.random.uniform(m_max/10, m_max)

                # species maintenance factor
                mfd = float(np.random.logseries(0.95, 1))+1

                if mfd > 40: mfd = 40

                MFD[prop] = mfd

                # species environmental gradient optima
                glist = []
                for j in envGs:
                    x = np.random.uniform(0.001, 0.09*w)
                    y = np.random.uniform(0.001, 0.09*h)
                    z = np.random.uniform(0.001, 0.09*h)
                    glist.append([x,y,z])
                EnvD[prop] = glist

                # A set of specific growth rates for three major types of resources
                N_RD[prop] = list(decomposition(1, nN))

            state = choice([1, 1])
            ADList.append(state)

            i = GD[prop]
            GList.append(i)

            i = MFD[prop]
            MFDList.append(i)

            i = MD[prop]

            if state == 1:
                MList.append(i)
            elif state == 0:
                MList.append(i/MFD[prop])

            i = N_RD[prop]
            NList.append(i)

            i = DispD[prop]
            DList.append(i)



            i = RPD[prop]
            RPDList.append(i)

            CRList.append('none')

    return [CRList, mfmax, p_max, d_max, g_max, m_max, seed, Sp, Xs, Ys, Zs, h, l, w, MD, MFD, RPD,
        EnvD, envGs, GD, DispD, colorD, IDs, ID, Qs, N_RD, nN, u0, GList, MList, MFDList, RPDList, NList, DList, ADList, ct, m]



def ind_flow(TypeOf, List, Xs, Ys, Zs, h, l, w, u0):

    Type, IDs, ID, Vals = [], [], int(), []

    CRList, Sp_IDs, IDs, ID, Qs, DispD, GList, MList, MFDList, RPDList, N_RList, DList, ADList = List

    if Xs == []: return [CRList, Sp_IDs, Xs, Ys, Zs, IDs, ID, Qs, GList, MList, MFDList, RPDList, N_RList, DList, ADList]


    Qs = np.array(Qs)
    DList = np.array(DList)
    Qs = Qs - (Qs * DList * (2*cost) * u0)

    trials = 1 - np.random.binomial(1, DList) # 0 = stay; 1 = go

    Xs = np.array(Xs) + (trials * u0)
    Ys = np.array(Ys) + (trials * u0)
    Zs = np.array(Zs) + (trials * u0)

    i1 = np.where(Xs > h)[0].tolist()
    i2 = np.where(Ys > l)[0].tolist()
    i3 = np.where(Zs > w)[0].tolist()
    index = np.array(list(set(i1 + i2 + i3)))

    Qs = np.delete(Qs, index).tolist()
    Sp_IDs = np.delete(Sp_IDs, index).tolist()
    IDs = np.delete(IDs, index).tolist()
    Xs = np.delete(Xs, index).tolist()
    Ys = np.delete(Ys, index).tolist()
    Zs = np.delete(Zs, index).tolist()
    GList = np.delete(GList, index).tolist()
    MList = np.delete(MList, index).tolist()
    MFDList = np.delete(MFDList, index).tolist()
    RPDList = np.delete(RPDList, index).tolist()
    N_RList = np.delete(N_RList, index, axis=0).tolist()
    DList = np.delete(DList, index).tolist()
    ADList = np.delete(ADList, index).tolist()
    CRList = np.delete(CRList, index).tolist()

    return [CRList, Sp_IDs, Xs, Ys, Zs, IDs, ID, Qs, GList, MList, MFDList, RPDList, N_RList, DList, ADList]



def res_flow(Type, IDs, ID, Vals, Xs, Ys, Zs, h, l, w, u0):

    if len(Xs) == 0:
        return [Type, IDs, ID, Vals, Xs, Ys, Zs, h, l, w, u0]

    Xs = np.array(Xs) + u0
    Ys = np.array(Ys) + u0
    Zs = np.array(Zs) + u0

    i1 = np.where(Xs > h)[0].tolist()
    i2 = np.where(Ys > l)[0].tolist()
    i3 = np.where(Zs > w)[0].tolist()

    index = np.array(list(set(i1 + i2 + i3)))

    Type = np.delete(Type, index).tolist()
    Xs = np.delete(Xs, index).tolist()
    Ys = np.delete(Ys, index).tolist()
    Zs = np.delete(Zs, index).tolist()
    Vals = np.delete(Vals, index).tolist()
    IDs = np.delete(IDs, index).tolist()


    return [Type, IDs, ID, Vals, Xs, Ys, Zs, h, l, w, u0]



def maintenance(CRList, Sp_IDs, Xs, Ys, Zs, colorD, MD, MFD, RPD, EnvD, IDs, Qs, GrowthList, MList, MFDList, RPDList, N_RList, DispList, ADList):

    if Sp_IDs == []: return [CRList, Sp_IDs, Xs, Ys, Zs, IDs, Qs, GrowthList, MList, MFDList, RPDList, N_RList, DispList, ADList]

    Qs = np.array(Qs)
    MList = np.array(MList)
    Qs = Qs - MList
    index = np.where(Qs < 0.0)[0]

    Qs = np.delete(Qs, index).tolist()
    Sp_IDs = np.delete(Sp_IDs, index).tolist()
    IDs = np.delete(IDs, index).tolist()
    Xs = np.delete(Xs, index).tolist()
    Ys = np.delete(Ys, index).tolist()
    Zs = np.delete(Zs, index).tolist()
    GrowthList = np.delete(GrowthList, index).tolist()
    MList = np.delete(MList, index).tolist()
    MFDList = np.delete(MFDList, index).tolist()
    RPDList = np.delete(RPDList, index).tolist()
    N_RList = np.delete(N_RList, index, axis=0).tolist()
    DispList = np.delete(DispList, index).tolist()
    ADList = np.delete(ADList, index).tolist()
    CRList = np.delete(CRList, index).tolist()

    return [CRList, Sp_IDs, Xs, Ys, Zs, IDs, Qs, GrowthList, MList, MFDList, RPDList, N_RList, DispList, ADList]




def to_active(CRList, Sp_IDs, Xs, Ys, Zs, IDs, Qs, DList, GList, MList, MFDList, RPDList, N_RList, MFD, RPD, ADList):

    if Sp_IDs == []:
        return [CRList, Sp_IDs, Xs, Ys, Zs, IDs, Qs, DList, GList, MList, MFDList, RPDList, N_RList, MFD, RPD, ADList]

    MList = np.array(MList)
    MFDList = np.array(MFDList)
    Qs = np.array(Qs)
    ADList = np.array(ADList)
    RPDList = np.array(RPDList)

    dormantQs = np.array(Qs) * (1-ADList)
    trials = np.random.binomial(1, RPDList) # 0 = no change; 1 = resucitate
    tdIs = (1-ADList) * trials
    rQs = dormantQs * tdIs

    rQs = rQs - (rQs * RPDList * cost * tdIs)
    nQs = Qs * (1-tdIs)
    Qs = rQs + nQs

    ADList = ADList + tdIs
    cMList = (MList*MFDList) * tdIs
    nMList = MList * (1-tdIs)
    MList = cMList + nMList

    ADList = ADList.tolist()
    MList = MList.tolist()
    MFDList = MFDList.tolist()
    RPDList = RPDList.tolist()
    Qs = Qs.tolist()

    return [CRList, Sp_IDs, Xs, Ys, Zs, IDs, Qs, DList, GList, MList, MFDList, RPDList, N_RList, MFD, RPD, ADList]


def to_dormant(CRList, Sp_IDs, Xs, Ys, Zs, IDs, Qs, DList, GList, MList, MFDList, RPDList, N_RList, MFD, RPD, ADList):

    if Sp_IDs == []:
        return [CRList, Sp_IDs, Xs, Ys, Zs, IDs, Qs, DList, GList, MList, MFDList, RPDList, N_RList, MFD, RPD, ADList]

    MList = np.array(MList)
    MFDList = np.array(MFDList)
    ADList = np.array(ADList)

    activeQs = np.array(Qs) * ADList

    todormantQs = np.array(activeQs)
    todormantQs[todormantQs > 0.3] = 0
    tdIs = np.array(todormantQs)
    tdIs[tdIs > 0] = 1

    ADList = ADList - tdIs

    cMList = (MList/MFDList) * tdIs
    nMList = MList * (1-tdIs)
    MList = cMList + nMList

    ADList = ADList.tolist()
    MList = MList.tolist()
    MFDList = MFDList.tolist()

    return [CRList, Sp_IDs, Xs, Ys, Zs, IDs, Qs, DList, GList, MList, MFDList, RPDList, N_RList, MFD, RPD, ADList]





def consume(numc, CRList, RPFDict, Rtypes, Rvals, RIDs, RID, RX, RY, RZ, SpIDs, Qs, IIDs, IID, IX, IY, IZ, h, l, w, GrowthDict, N_RD, DispDict, GList, MList, MFDList, RPDList, MFD, N_RList, DList, ADList):
    numc = 0

    if len(Rtypes) == 0 or len(SpIDs) == 0:
        return [numc, CRList, RPFDict, Rtypes, Rvals, RIDs, RID, RX, RY, RZ, SpIDs, Qs, IIDs, IID, IX, IY, IZ, h, l, w, GrowthDict, N_RD, DispDict, GList, MList, MFDList, RPDList, MFD, N_RList, DList, ADList]


    for ii in range(len(IIDs)):

        if len(Rvals) == 0 or len(SpIDs) == 0:
            return [numc, CRList, RPFDict, Rtypes, Rvals, RIDs, RID, RX, RY, RZ, SpIDs, Qs, IIDs, IID, IX, IY, IZ, h, l, w, GrowthDict, N_RD, DispDict, GList, MList, MFDList, RPDList, MFD, N_RList, DList, ADList]


        i = randint(0, len(IIDs)-1)
        Q = Qs[i]
        state = ADList[i]

        j = randint(0, len(Rvals)-1)
        Rval = Rvals[j]
        rtype = Rtypes[j]

        mu1 = GList[i]
        if Q == Q: #- Q * mu1 * cost > MList[i]: #dist <= i_radius + r_radius and
            numc += 1
            # An energetic cost to growth, i.e., pay a bigger cost for growing faster. No such thing as a free lunch.
            Q -= Q * mu1 * cost

            if state == 0:
                ADList[i] = 1
                mfd = MFDList[i]
                MList[i] = MList[i]*mfd

            efficiency = N_RList[i][rtype]
            mu = mu1 * efficiency

            if Rval > mu * Q: # Increase cell quota
                Rval = Rval - (mu * Q)
                Q += (mu * Q)

            else:
                Q += Rval
                Rval = 0.0

            if Q > 1.0:
                Rval = Q - 1.0
                Q = 1.0

            if Rval <= 0.0:

                Rvals.pop(j)
                Rtypes.pop(j)
                RIDs.pop(j)
                RX.pop(j)
                RY.pop(j)
                RZ.pop(j)

            else:
                Rvals[j] = Rval

            if Q < 0.0:
                Q = 0.0
            Qs[i] = Q


    return [numc, CRList, RPFDict, Rtypes, Rvals, RIDs, RID, RX, RY, RZ, SpIDs, Qs, IIDs, IID, IX, IY, IZ, h, l, w, GrowthDict, N_RD, DispDict, GList, MList, MFDList, RPDList, MFD, N_RList, DList, ADList]




def reproduce(CRList, Sp_IDs, Xs, Ys, Zs, Qs, IDs, ID, h, l, w, GD, DispD, colorD, N_RD, MD, MFD, RPD, EnvD, envGs, nN, GList, MList, MFDList, RPDList, NList, DList, ADList):

    if Sp_IDs == []:
        return [CRList, Sp_IDs, Xs, Ys, Zs, Qs, IDs, ID, h, l, w, GD, DispD, colorD, N_RD, MD, MFD, RPD, EnvD, envGs, nN, GList, MList, MFDList, RPDList, NList, DList, ADList]

    for j in range(len(IDs)):

        i = randint(0, len(IDs)-1)
        state = ADList[i]
        Q = Qs[i]

        if Q > 1.0:
            Q == 1.0

        if state == 0 or Q < MList[i]: continue

        p = np.random.binomial(1, Q)
        if p == 1 and Q >= 0.5: # individual is large enough to reproduce
            spID = Sp_IDs[i]
            X = Xs[i]
            Y = Ys[i]
            Z = Zs[i]

            Qs[i] = Q/2.0
            Qs.append(Q/2.0)
            ID += 1

            p = np.random.binomial(1, 0.001)
            prop = float(spID)
            if p == 1:
                # speciate
                max_spid = max(list(set(Sp_IDs)))
                prop = max_spid+1

                g_max = GD[spID]
                m_max = MD[spID]
                nrd = N_RD[spID]
                d_max = DispD[spID]
                mfd = MFD[spID]
                rpd = RPD[spID]

                # speciescolor
                colorD = get_color(prop, colorD)

                # species growth rate
                GD[prop] = g_max

                # species maintenance
                MD[prop] = m_max

                # species maintenance factor
                MFD[prop] = mfd

                # species RPF factor
                RPD[prop] = rpd

                # species active dispersal rate
                DispD[prop] = d_max

                # species environmental gradient optima
                glist = []
                for j in envGs:
                    x = np.random.uniform(0.001, 0.09*w)
                    y = np.random.uniform(0.001, 0.09*h)
                    glist.append([x,y])
                EnvD[prop] = glist

                # A set of specific growth rates for three major types of resources
                N_RD[prop] = nrd

            IDs.append(ID)
            GList.append(GD[prop])
            MList.append(MD[prop])
            NList.append(N_RD[prop])
            DList.append(DispD[prop])
            MFDList.append(MFD[prop])
            RPDList.append(RPD[prop])
            CRList.append(CRList[i])
            Sp_IDs.append(prop)
            ADList.append(1)

            Xs.append(float(X))
            Ys.append(float(Y))
            Zs.append(float(Z))


    return [CRList, Sp_IDs,  Xs, Ys, Zs, Qs, IDs, ID, h, l, w, GD, DispD,
        colorD, N_RD, MD, MFD, RPD, EnvD, envGs, nN, GList, MList, MFDList, RPDList,
        NList, DList, ADList]
