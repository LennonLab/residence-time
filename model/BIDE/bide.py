# -*- coding: utf-8 -*-
from __future__ import division
from random import randint, choice
import numpy as np
import sys
import math

cost = 0.1

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
        x = np.random.binomial(1, u0)

        if x == 1:

            rval = np.random.uniform(rmax/1, rmax)
            rtype = randint(0, nN-1)
            Type.append(rtype)
            Vals.append(rval)
            IDs.append(ID)
            ID += 1

            Xs.append(float(np.random.uniform(0.1, 0.9*h)))
            Ys.append(float(np.random.uniform(0.1, 0.9*l)))
            Zs.append(float(np.random.uniform(0.1, 0.9*w)))


    return [Type, Vals, Xs, Ys, Zs, IDs, ID]



def immigration(CRList, mfmax, p_max, d_max, g_max, m_max, seed, Sp, Xs, Ys, Zs, h, l, w, MD, MFD, RPD,
        GD, DispD, colorD, IDs, ID, Qs, N_RD, nN, u0, GList, MList, MFDList, RPDList, NList, DList, ADList, ct, m):

    sd = int(seed)
    if ct > 1:
        sd = 0

    for i in range(sd):
        x = 0

        if sd > 1:
            x = 1
        else:
            x = np.random.binomial(1, u0*m)

        if x == 1:
            prop = np.random.randint(1, 10000)
            Sp.append(prop)

            Xs.append(float(np.random.uniform(0, 0.1*h)))
            Ys.append(float(np.random.uniform(0, 0.1*l)))
            Zs.append(float(np.random.uniform(0, 0.1*w)))

            IDs.append(ID)
            ID += 1
            Qn = float(np.random.uniform(0.2, 0.3))

            Qs.append(Qn)

            if prop not in colorD:

                # species growth rate
                #x = np.random.uniform(0.0, 1.0)
                #y = 0.99*10**(x-1)
                #GD[prop] = y #np.random.beta(2, 1, size=1)[0]
                GD[prop] = np.random.uniform(g_max/100, g_max)


                # species active dispersal rate
                #x = np.random.uniform(0.0, 1.0)
                #y = 0.6*2**(x-1)
                #DispD[prop] = y
                DispD[prop] = np.random.uniform(d_max/100, d_max)


                # species RPF factor
                #x = np.random.uniform(0.0, 1.0)
                #y = 10**(x-1)
                RPD[prop] = np.random.uniform(p_max/100, p_max)


                # species maintenance
                MD[prop] =  np.random.uniform(m_max/100, m_max)


                # species maintenance factor
                mfd = float(np.random.logseries(0.95, 1))+1
                if mfd > 40: mfd = 40
                MFD[prop] = mfd


                # A set of specific growth rates for three major types of resources
                N_RD[prop] = list(decomposition(1, nN))


            state = choice([1, 1]) # 1 is active
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
        GD, DispD, colorD, IDs, ID, Qs, N_RD, nN, u0, GList, MList, MFDList, RPDList, NList, DList, ADList, ct, m]



def ind_flow(TypeOf, List, Xs, Ys, Zs, h, l, w, u0):

    Type, IDs, ID, Vals = [], [], int(), []

    CRList, Sp_IDs, IDs, ID, Qs, DispD, GList, MList, MFDList, RPDList, N_RList, DList, ADList = List

    if Xs == []: return [CRList, Sp_IDs, Xs, Ys, Zs, IDs, ID, Qs, GList, MList, MFDList, RPDList, N_RList, DList, ADList]


    Qs = np.array(Qs)
    DList = np.array(DList)
    Qs = Qs - (Qs * DList * cost * u0)

    trials1 = 1 - np.random.binomial(1, DList) # 0 = stay put; 1 = flow

    Xs = np.array(Xs) + (trials1 * u0)
    Ys = np.array(Ys) + (trials1 * u0)
    Zs = np.array(Zs) + (trials1 * u0)

    #trials2 = np.random.binomial(1, DList) # 1 = disperse; 0 = flow
    #Xs = np.array(Xs) - (trials2 * u0 * 0.1)
    #Ys = np.array(Ys) - (trials2 * u0 * 0.1)
    #Zs = np.array(Zs) - (trials2 * u0 * 0.1)

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



def maintenance(CRList, Sp_IDs, Xs, Ys, Zs, colorD, MD, MFD, RPD, IDs, Qs, GrowthList, MList, MFDList, RPDList, N_RList, DispList, ADList):

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


def to_dormant(CRList, Sp_IDs, Xs, Ys, Zs, IDs, Qs, DList, GList, MList, MFDList, RPDList, N_RList, MFD, RPD, ADList, dormlim):

    if Sp_IDs == []:
        return [CRList, Sp_IDs, Xs, Ys, Zs, IDs, Qs, DList, GList, MList, MFDList, RPDList, N_RList, MFD, RPD, ADList]

    MList = np.array(MList)
    MFDList = np.array(MFDList)
    ADList = np.array(ADList)

    activeQs = np.array(Qs) * ADList

    todormantQs = np.array(activeQs)
    todormantQs[todormantQs > dormlim] = 0
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

    iids = list(IIDs)

    while len(iids) > 0:

        if len(Rtypes) == 0 or len(SpIDs) == 0:
            return [numc, CRList, RPFDict, Rtypes, Rvals, RIDs, RID, RX, RY, RZ, SpIDs, Qs, IIDs, IID, IX, IY, IZ, h, l, w, GrowthDict, N_RD, DispDict, GList, MList, MFDList, RPDList, MFD, N_RList, DList, ADList]

        ind = choice(iids)
        i = iids.index(ind)
        iids.pop(i)

        state = ADList[i]
        if state == 0:
            rp = RPDList[i]
            x = np.random.binomial(1, rp)

            if x == 1:
                Q = Qs[i]
                Q -= Q * rp * cost
                ADList[i] = 1
                MFDList[i] = MList[i] * MFDList[i]

            else: continue

        Q = Qs[i]
        j = randint(0, len(Rvals)-1)
        Rval = Rvals[j]
        rtype = Rtypes[j]

        mu1 = GList[i]
        numc += 1

        # An energetic cost to growth, i.e., pay a bigger cost for growing faster. No such thing as a free lunch.
        Q -= Q * mu1 * cost

        efficiency = N_RList[i][rtype]
        mu = mu1 * efficiency

        if Rval > mu * Q: # Increase cell quota
            Rval -= (mu * Q)
            Q += (mu * Q)

        else:
            Q += Rval
            Rval = 0.0

        if Q > 1:
            Rval += (Q - 1)
            Q = 1

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




def reproduce(u0, CRList, Sp_IDs, Xs, Ys, Zs, Qs, IDs, ID, h, l, w, GD, DispD, colorD, N_RD, MD, MFD, RPD, nN, GList, MList, MFDList, RPDList, NList, DList, ADList):

    if Sp_IDs == []:
        return [CRList, Sp_IDs, Xs, Ys, Zs, Qs, IDs, ID, h, l, w, GD, DispD, colorD, N_RD, MD, MFD, RPD, nN, GList, MList, MFDList, RPDList, NList, DList, ADList]

    iids = list(IDs)

    while len(iids) > 0:

        ind = choice(iids)
        i = iids.index(ind)
        iids.pop(i)

        state = ADList[i]
        Q = Qs[i]

        if Q > 1.0:
            Q == 1.0

        if state == 0 or Q < 0.5: continue

        p = np.random.binomial(1, Q)
        if p == 1: # individual is large enough to reproduce
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
                o = np.random.uniform(1, 1)
                o_gmax = g_max * o
                if o_gmax > 1 or o_gmax < 0:
                    GD[prop] = g_max
                else:
                    GD[prop] = o_gmax


                # species maintenance
                o = np.random.uniform(1, 1)
                o_mmax = m_max * o
                if o_mmax > 1 or o_mmax < 0:
                    MD[prop] = m_max
                else:
                    MD[prop] = o_mmax

                # species maintenance factor
                MFD[prop] = mfd

                # species RPF factor
                o = np.random.uniform(1, 1)
                o_rpd = rpd * o
                if o_rpd > 1 or o_rpd < 0:
                    RPD[prop] = rpd
                else:
                    RPD[prop] = o_rpd

                RPD[prop] = rpd

                # species active dispersal rate
                o = np.random.uniform(1, 1)
                o_dmax = d_max * o
                if o_dmax > 1 or o_dmax < 0:
                    DispD[prop] = d_max
                else:
                    DispD[prop] = o_dmax

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

            nX = float(X) - u0
            if nX < 0: nX = 0

            nY = float(Y) - u0
            if nY < 0: nY = 0

            nZ = float(Z) - u0
            if nZ < 0: nZ = 0

            Xs.append(float(X) - u0)
            Ys.append(float(Y) - u0)
            Zs.append(float(Z) - u0)


    return [CRList, Sp_IDs,  Xs, Ys, Zs, Qs, IDs, ID, h, l, w, GD, DispD,
        colorD, N_RD, MD, MFD, RPD, nN, GList, MList, MFDList, RPDList,
        NList, DList, ADList]


def decimate(Lists, Xs, Ys, Zs, height, length, width, u0):

    Type, IDs, ID, Vals = [], [], int(), []
    CRList, SpIDs, IDs, ID, Qs, DispD, GList, MList, MFDList, RPDList, NList, DList, ADList = Lists

    lim = len(Xs)/2
    while len(IDs) > lim:

        i = randint(0, len(IDs)-1)

        IDs.pop(i)
        GList.pop(i)
        MList.pop(i)
        NList.pop(i)
        DList.pop(i)
        MFDList.pop(i)
        RPDList.pop(i)
        CRList.pop(i)
        SpIDs.pop(i)
        ADList.pop(i)
        Xs.pop(i)
        Ys.pop(i)
        Zs.pop(i)
        Qs.pop(i)

    return [CRList, SpIDs, Xs, Ys, Zs, IDs, Qs, GList, MList, MFDList, RPDList, NList, DList, ADList]
