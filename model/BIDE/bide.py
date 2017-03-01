# -*- coding: utf-8 -*-
from __future__ import division
from random import randint, choice
import numpy as np
import sys
from math import sqrt


cost = 0.1


def get_closest(RIDs, RX, RY, RZ, Rtypes, coords):
    closest = 0

    if len(RIDs) == 0:
        return closest

    x1, y1, z1 = coords
    Try = min([20, len(RIDs)])
    minDist, ct = 10**10, 0

    while ct < Try:
        ct += 1
        j = randint(0, len(RIDs)-1)
        x = RX[j]
        y = RY[j]
        z = RZ[j]

        dist = sqrt((x1 - x)**2 + (y1 - y)**2 + (z1 - z)**2)

        if dist < minDist:
            minDist = dist
            closest = RIDs[j]

        return closest



def checkVal(val, line):
    if val < 0:
        print 'line',line,': error: val < 0:', val
        sys.exit()
    return


def decomposition(i, n):
    gn = choice([[1/3, 1/3, 1/3], [0.5, 0.3, 0.2], [0.6, 0.2, 0.2],
                [0.8, 0.1, 0.1], [0.99, 0.005, 0.005], [0.9, 0.05, 0.05]])
    return gn


def decomposition1(i, n):
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

    ID += 1
    for r in range(numr):
        x = np.random.binomial(1, u0*0.5)

        if x == 1:

            rval = np.random.uniform(rmax/1, rmax)
            rtype = randint(0, nN-1)
            Type.append(rtype)
            Vals.append(rval)
            IDs.append(ID)
            ID += 1

            Xs.append(float(np.random.uniform(0, 0.1*h)))
            Ys.append(float(np.random.uniform(0, 0.1*l)))
            Zs.append(float(np.random.uniform(0, 0.1*w)))


    return [Type, Vals, Xs, Ys, Zs, IDs, ID]



def immigration(SizeList, mfmax, p_max, d_max, g_max, m_max, seed, Sp, Xs, Ys, Zs, h, l, w, MD, MFD, RPD,
        GD, DispD, colorD, IDs, ID, Qs, N_RD, nN, u0, GList, MList, MFDList, RPDList, NList, DList, ADList, ct, m):

    sd = int(seed)
    if ct > 1:
        sd = 1

    for i in range(sd):
        x = 0

        if m == 0 and ct > 1:
            break

        if sd > 1:
            x = 1
        else:
            x = np.random.binomial(1, u0*m)

        if x == 1:
            prop = np.random.randint(1, 1000)
            #prop = np.random.logseries(0.99, 1)[0]
            Sp.append(prop)

            Xs.append(float(np.random.uniform(0, 0.1*h)))
            Ys.append(float(np.random.uniform(0, 0.1*l)))
            Zs.append(float(np.random.uniform(0, 0.1*w)))

            IDs.append(ID)
            ID += 1
            Qn = float(np.random.uniform(0.2, 0.2))
            size = float(np.random.uniform(0.01, 0.01))

            Qs.append(Qn)

            if prop not in colorD:

                # species growth rate
                GD[prop] = np.random.uniform(g_max/100, g_max)

                # species active dispersal rate
                DispD[prop] = np.random.uniform(d_max/100, d_max)

                # species RPF factor
                RPD[prop] = np.random.uniform(p_max/100, p_max)

                # species maintenance
                MD[prop] = np.random.uniform(m_max/10, m_max)

                # species maintenance factor
                mfd = np.random.logseries(0.95, 1)[0]
                mfd += 1
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

            SizeList.append(size)

    return [SizeList, mfmax, p_max, d_max, g_max, m_max, seed, Sp, Xs, Ys, Zs, h, l, w, MD, MFD, RPD,
        GD, DispD, colorD, IDs, ID, Qs, N_RD, nN, u0, GList, MList, MFDList, RPDList, NList, DList, ADList, ct, m]



def ind_flow(TypeOf, List, Xs, Ys, Zs, h, l, w, u0):

    Type, IDs, ID, Vals = [], [], int(), []

    SizeList, Sp_IDs, IDs, ID, Qs, DispD, GList, MList, MFDList, RPDList, N_RList, DList, ADList = List

    if Xs == []: return [SizeList, Sp_IDs, Xs, Ys, Zs, IDs, ID, Qs, GList, MList, MFDList, RPDList, N_RList, DList, ADList]

    ADList = np.array(ADList)
    Qs = np.array(Qs)
    DList = np.array(DList)

    trials = ADList * np.random.binomial(1, DList) # 1 = stay put; 0 = flow

    Qs = Qs - (DList * SizeList * ADList * cost * 2)
    trials = 1 - trials  # 0 = stay put; 1 = flow

    Xs = np.array(Xs) + (trials * u0)
    Ys = np.array(Ys) + (trials * u0)
    Zs = np.array(Zs) + (trials * u0)

    i1 = np.where(Xs > h)[0].tolist()
    i2 = np.where(Ys > l)[0].tolist()
    i3 = np.where(Zs > w)[0].tolist()
    i4 = np.where(Qs <= 0.0)[0].tolist()
    index = np.array(list(set(i1 + i2 + i3 + i4)))

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
    SizeList = np.delete(SizeList, index).tolist()


    return [SizeList, Sp_IDs, Xs, Ys, Zs, IDs, ID, Qs, GList, MList, MFDList, RPDList, N_RList, DList, ADList]



def ind_disp(SizeList, Sp_IDs, Xs, Ys, Zs, IDs, ID, Qs, GList, MList, MFDList, RPDList, N_RList, DList, ADList, h, w, l):

    if Xs == []: return [SizeList, Sp_IDs, Xs, Ys, Zs, IDs, ID, Qs, GList, MList, MFDList, RPDList, N_RList, DList, ADList]

    ADList = np.array(ADList)
    Qs = np.array(Qs)
    DList = np.array(DList)

    Qs = Qs - (DList * SizeList * ADList * cost * 2)
    Xs = np.array(Xs) - (DList * ADList)
    Ys = np.array(Ys) - (DList * ADList)
    Zs = np.array(Zs) - (DList * ADList)

    i1 = np.where(Xs > h)[0].tolist()
    i2 = np.where(Ys > l)[0].tolist()
    i3 = np.where(Zs > w)[0].tolist()
    i4 = np.where(Qs <= 0.0)[0].tolist()
    index = np.array(list(set(i1 + i2 + i3 + i4)))

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
    SizeList = np.delete(SizeList, index).tolist()

    return [SizeList, Sp_IDs, Xs, Ys, Zs, IDs, ID, Qs, GList, MList, MFDList, RPDList, N_RList, DList, ADList]



def search(SizeList, Sp_IDs, Xs, Ys, Zs, IDs, ID, Qs, GList, MList, MFDList, RPDList, N_RList, DList, ADList, h, l, w, u0, RTypes, RVals, RXs, RYs, RZs, RIDs):

    if Xs == []:
        return [SizeList, Sp_IDs, Xs, Ys, Zs, IDs, ID, Qs, GList, MList, MFDList, RPDList, N_RList, DList, ADList, h, l, w, u0, RTypes, RVals, RXs, RYs, RZs, RIDs]

    for i, ind in enumerate(Xs):

        if ADList[i] == 0: continue

        x1, y1, z1 = Xs[i], Ys[i], Zs[i]
        coords = [x1, y1, z1]
        closest = get_closest(RIDs, RXs, RYs, RZs, RTypes, coords)

        if closest in RIDs:
            ri = RIDs.index(closest)
            x2 = RXs[ri]
            y2 = RYs[ri]
            z2 = RZs[ri]

        else:
            x2 = np.random.uniform(0, h)
            y2 = np.random.uniform(0, l)
            z2 = np.random.uniform(0, w)

        disp = DList[i]
        x = np.abs(x1 - x2)
        if x1 > x2:
            x1 -= np.random.uniform(0, disp*x)
        elif x1 < x2:
            x1 += np.random.uniform(0, disp*x)

        y = np.abs(y1 - y2)
        if y1 > y2:
            y1 -= np.random.uniform(0, disp*y)
        elif y1 < y2:
            y1 += np.random.uniform(0, disp*y)

        z = np.abs(z1 - z2)
        if z1 > z2:
            z1 -= np.random.uniform(0, disp*z)
        elif z1 < z2:
            z1 += np.random.uniform(0, disp*z)

        Q = Qs[i]
        Q -= disp * SizeList[i] * cost * 2
        Qs[i] = Q

    return [SizeList, Sp_IDs, Xs, Ys, Zs, IDs, ID, Qs, GList, MList, MFDList, RPDList, N_RList, DList, ADList, h, l, w, u0, RTypes, RVals, RXs, RYs, RZs, RIDs]




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



def maintenance(SizeList, Sp_IDs, Xs, Ys, Zs, colorD, MD, MFD, RPD, IDs, Qs, GrowthList, MList, MFDList, RPDList, N_RList, DispList, ADList):

    if Sp_IDs == []: return [SizeList, Sp_IDs, Xs, Ys, Zs, IDs, Qs, GrowthList, MList, MFDList, RPDList, N_RList, DispList, ADList]

    Qs = np.array(Qs)
    MList = np.array(MList)
    Qs = Qs - MList
    index = np.where(Qs <= 0.0)[0]

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
    SizeList = np.delete(SizeList, index).tolist()

    return [SizeList, Sp_IDs, Xs, Ys, Zs, IDs, Qs, GrowthList, MList, MFDList, RPDList, N_RList, DispList, ADList]



def grow(Qs, GList, ADList, SizeList):

    if Qs == []: return [Qs, GList, ADList, SizeList]

    ADList = np.array(ADList)
    Qs = np.array(Qs)
    GList = np.array(GList)
    SizeList = np.array(SizeList)

    SizeList = SizeList + (SizeList * GList * ADList)
    Qs = Qs - (GList * ADList * cost)

    return [Qs.tolist(), GList.tolist(), ADList.tolist(), SizeList.tolist()]




def to_active(SizeList, Sp_IDs, Xs, Ys, Zs, IDs, Qs, DList, GList, MList, MFDList, RPDList, N_RList, MFD, RPD, ADList):

    if Sp_IDs == []:
        return [SizeList, Sp_IDs, Xs, Ys, Zs, IDs, Qs, DList, GList, MList, MFDList, RPDList, N_RList, MFD, RPD, ADList]

    MList = np.array(MList)
    MFDList = np.array(MFDList)
    Qs = np.array(Qs)
    ADList = np.array(ADList)
    RPDList = np.array(RPDList)

    dormantQs = np.array(Qs) * (1-ADList)
    trials = np.random.binomial(1, RPDList) # 0 = no change; 1 = resucitate
    tdIs = (1-ADList) * trials
    rQs = dormantQs * tdIs

    rQs = rQs - (RPDList * tdIs * cost)
    nQs = Qs * (1 - tdIs)
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

    return [SizeList, Sp_IDs, Xs, Ys, Zs, IDs, Qs, DList, GList, MList, MFDList, RPDList, N_RList, MFD, RPD, ADList]


def to_dormant(SizeList, Sp_IDs, Xs, Ys, Zs, IDs, Qs, DList, GList, MList, MFDList, RPDList, N_RList, MFD, RPD, ADList, dormlim):

    if Sp_IDs == []:
        return [SizeList, Sp_IDs, Xs, Ys, Zs, IDs, Qs, DList, GList, MList, MFDList, RPDList, N_RList, MFD, RPD, ADList]

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

    return [SizeList, Sp_IDs, Xs, Ys, Zs, IDs, Qs, DList, GList, MList, MFDList, RPDList, N_RList, MFD, RPD, ADList]





def consume(numc, SizeList, RPFDict, Rtypes, Rvals, RIDs, RID, RX, RY, RZ, SpIDs, Qs, IIDs, IID, IX, IY, IZ, h, l, w, GrowthDict, N_RD, DispDict, GList, MList, MFDList, RPDList, MFD, N_RList, DList, ADList):
    numc = 0

    if len(Rtypes) == 0 or len(SpIDs) == 0:
        return [numc, SizeList, RPFDict, Rtypes, Rvals, RIDs, RID, RX, RY, RZ, SpIDs, Qs, IIDs, IID, IX, IY, IZ, h, l, w, GrowthDict, N_RD, DispDict, GList, MList, MFDList, RPDList, MFD, N_RList, DList, ADList]

    iids = list(IIDs)

    while len(iids) > 0:

        if len(Rtypes) == 0 or len(SpIDs) == 0:
            return [numc, SizeList, RPFDict, Rtypes, Rvals, RIDs, RID, RX, RY, RZ, SpIDs, Qs, IIDs, IID, IX, IY, IZ, h, l, w, GrowthDict, N_RD, DispDict, GList, MList, MFDList, RPDList, MFD, N_RList, DList, ADList]

        ind = choice(iids)
        i = iids.index(ind)
        iids.pop(i)

        i = IIDs.index(ind)
        state = ADList[i]

        if state == 0:
            continue

        Q = Qs[i]
        j = randint(0, len(Rvals)-1)
        Rval = Rvals[j]
        rtype = Rtypes[j]
        numc += 1

        eff = N_RList[i][rtype]

        if Rval > eff * Q: # Increase cell quota
            Rval -= (eff * Q)
            Q += (eff * Q)

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

    return [numc, SizeList, RPFDict, Rtypes, Rvals, RIDs, RID, RX, RY, RZ, SpIDs, Qs, IIDs, IID, IX, IY, IZ, h, l, w, GrowthDict, N_RD, DispDict, GList, MList, MFDList, RPDList, MFD, N_RList, DList, ADList]




def reproduce(u0, SizeList, Sp_IDs, Xs, Ys, Zs, Qs, IDs, ID, h, l, w, GD, DispD, colorD, N_RD, MD, MFD, RPD, nN, GList, MList, MFDList, RPDList, NList, DList, ADList):

    if Sp_IDs == []:
        return [SizeList, Sp_IDs, Xs, Ys, Zs, Qs, IDs, ID, h, l, w, GD, DispD, colorD, N_RD, MD, MFD, RPD, nN, GList, MList, MFDList, RPDList, NList, DList, ADList]

    iids = list(IDs)

    while len(iids) > 0:

        ind = choice(iids)
        i = iids.index(ind)
        iids.pop(i)

        state = ADList[i]
        Q = Qs[i]
        size = SizeList[i]

        if Q > 1.0:
            Q == 1.0

        if state == 0 or size < 0.5 or Q < 0.5: continue

        p = size
        if p > 1.0: p = 1.0

        if np.random.binomial(1, p) == 1: # individual is large enough to reproduce
            spID = Sp_IDs[i]
            X = Xs[i]
            Y = Ys[i]
            Z = Zs[i]

            Qs[i] = Q/2.0
            Qs.append(Q/2.0)
            SizeList[i] = size/2
            SizeList.append(size/2)
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
            Sp_IDs.append(prop)
            ADList.append(1)

            nX = float(X) - u0
            if nX < 0: nX = 0

            nY = float(Y) - u0
            if nY < 0: nY = 0

            nZ = float(Z) - u0
            if nZ < 0: nZ = 0

            Xs.append(float(X) - u0*0.01)
            Ys.append(float(Y) - u0*0.01)
            Zs.append(float(Z) - u0*0.01)


    return [SizeList, Sp_IDs,  Xs, Ys, Zs, Qs, IDs, ID, h, l, w, GD, DispD,
        colorD, N_RD, MD, MFD, RPD, nN, GList, MList, MFDList, RPDList,
        NList, DList, ADList]
