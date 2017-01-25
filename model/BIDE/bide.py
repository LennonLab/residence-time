# -*- coding: utf-8 -*-
from __future__ import division
from random import randint, choice
import numpy as np
import sys
import math

limit = 0.1


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


def get_closest(RIDs, RX, RY, RZ, Rtypes, restype, coords):

    closest = 'none'
    res_indices = [xi for xi, x in enumerate(Rtypes) if x == restype or x == 'dead']

    if len(res_indices) == 0:
        return closest

    x1, y1, z1 = coords
    Try = min([40, len(res_indices)])
    minDist, ct = 10**10, 0

    while ct < Try:
        ct += 1
        j = randint(0, len(res_indices)-1)
        x = RX[j]
        y = RY[j]
        z = RZ[j]

        dist = math.sqrt((x1 - x)**2 + (y1 - y)**2 + (z1 - z)**2)

        if dist < minDist:
            minDist = dist
            closest = RIDs[j]

        return closest


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



def NewTracers(motion, IDs, Xs, Ys, t_In, w, h, u0, ct):

    if ct == 1:
        for i in range(1):
            IDs.append(0)
            t_In.append(0)
            Ys.append(float(np.random.uniform(0.05*h, 0.95*h)))
            Xs.append(float(np.random.uniform(0.05*w, 0.9*w)))
    else:
        x = np.random.binomial(1, u0/10)
        if x == 1:
            IDs.append(0)
            t_In.append(0)
            Ys.append(float(np.random.uniform(0.05*h, 0.95*h)))
            Xs.append(float(np.random.uniform(0.05*w, 0.9*w)))

    return [IDs, t_In, Xs, Ys]



def ResIn(motion, Type, Vals, Xs, Ys, ID, IDs, t_In, numr, rmax, nN, w, h, u0):

    for r in range(numr):
        x = np.random.binomial(1, 0.99*u0)

        if x == 1:

            rval = randint(rmax/10, rmax)
            rtype = randint(0, nN-1)
            rtype = 'N'+str(rtype)

            Vals.append(rval)
            IDs.append(ID)
            Type.append(rtype)
            t_In.append(0)
            ID += 1

            Ys.append(float(np.random.uniform(0.05*h, 0.95*h)))
            Xs.append(float(np.random.uniform(0.05*w, 0.9*w)))


    return [Type, Vals, Xs, Ys, IDs, ID, t_In]



def immigration(mfmax, p_max, d_max, g_max, m_max, motion, seed, Sp, t_In, I_xAge, Xs, Ys, w, h, MD, MFD, RPD,
        EnvD, envGs, GD, DispD, colorD, IDs, ID, Qs, N_RD, nN, u0, alpha, GList, MList, MFDList, RPDList, NList, DList, ADList, ct, m):

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

            Ys.append(float(np.random.uniform(0.05*h, 0.9*h)))
            Xs.append(float(np.random.uniform(0.05*w, 0.9*w)))


            IDs.append(ID)
            t_In.append(0)
            ID += 1
            Qn = float(np.random.uniform(0.5, 0.5))

            Qs.append([Qn])

            if prop not in colorD:
                # speciescolor
                colorD = get_color(prop, colorD)

                # species growth rate
                #gs = np.random.logseries(0.95, 10)
                #gs = gs/max(gs)
                #g = choice(gs)
                g = np.random.uniform(g_max/100, g_max)
                GD[prop] = g

                # species maintenance
                md =  np.random.uniform(m_max/100, m_max)
                MD[prop] = md

                # species maintenance factor
                MFD[prop] = float(np.random.logseries(0.95, 1))

                # species RPF factor
                RPD[prop] = np.random.uniform(p_max/100, p_max)

                # species active dispersal rate
                DispD[prop] =  np.random.uniform(d_max/100, d_max)

                # species environmental gradient optima
                glist = []
                for j in envGs:
                    x = np.random.uniform(0.1, 0.9*w)
                    y = np.random.uniform(0.1, 0.9*h)
                    glist.append([x,y])
                EnvD[prop] = glist

                # A set of specific growth rates for three major types of resources
                N_RD[prop] = list(decomposition(1, nN))

            state = 'a'
            ADList.append(state)

            i = GD[prop]
            GList.append(i)

            i = MD[prop]
            MList.append(i)

            i = N_RD[prop]
            NList.append(i)

            i = DispD[prop]
            DList.append(i)

            i = MFD[prop]
            MFDList.append(i)

            i = RPD[prop]
            RPDList.append(i)

    return [mfmax, p_max, d_max, g_max, m_max, motion, seed, Sp, t_In, I_xAge, Xs, Ys, w, h, MD, MFD, RPD,
        EnvD, envGs, GD, DispD, colorD, IDs, ID, Qs, N_RD, nN, u0, alpha, GList, MList, MFDList, RPDList, NList, DList, ADList, ct, m]



def fluid_movement(TypeOf, List, t_In, xAge, Xs, Ys, ux, uy, w, h, u0):

    Type, IDs, ID, Vals = [], [], int(), []

    if TypeOf == 'resource':
        Type, IDs, ID, Vals = List
    elif TypeOf == 'individual':
        Type, IDs, ID, Vals, DispD, GrowthList, MList, MFDList, RPDList, N_RList, DispList, ADList = List
    else:
        IDs = List

    if Xs == []:
        if TypeOf == 'tracer':
            return [IDs, t_In, xAge, Xs, Ys]
        elif TypeOf == 'individual':
            return [Type, t_In, xAge,Xs, Ys, IDs, ID, Vals, GrowthList, MList, MFDList, RPDList,
                N_RList, DispList, ADList]
        elif TypeOf == 'resource':
            return [Type, t_In, xAge, Xs, Ys, IDs, ID, Vals]

    ux = np.reshape(ux, (w*h)) # ux is the macroscopic x velocity
    uy = np.reshape(uy, (w*h)) # uy is the macroscopic y velocity

    # dispersal inside the system
    n = len(Xs)
    for j in range(n):

        i = randint(0, len(Xs)-1)
        X = int(round(Xs[i]))
        Y = int(round(Ys[i]))

        index =  int(round(X + Y * w))

        if index > len(ux) - 1:
            index = len(ux) - 1
        if index > len(uy) - 1:
            index = len(uy) - 1

        k = 0
        if TypeOf == 'individual' and ADList[i] == 'a':

            # A cost for active dispersal, larger dispersal means a bigger cost
            r1 = float(Vals[i][0])
            r1 -= DispList[i] * r1 * 0.01 * u0 # Larger individuals pay a bigger cost
            Vals[i] = [r1]
            k = np.random.binomial(1, DispList[i])

        if k == 0:
            Xs[i] += ux[index]
            Ys[i] += uy[index]

        y = Ys[i]

        if 0.0 > y:
            Ys[i] = 0.0
        elif y >= h:
            Ys[i] = h - 0.0

        t_In[i] += 1
        if Xs[i] <= 0:
            Xs[i] = 0.0

        if Xs[i] >= w - limit:

            xAge.append(t_In[i])
            Xs.pop(i)
            Ys.pop(i)
            t_In.pop(i)
            IDs.pop(i)

            if TypeOf == 'resource' or TypeOf == 'individual':
                Type.pop(i)
                Vals.pop(i)

            if TypeOf == 'individual':
                GrowthList.pop(i)
                MList.pop(i)
                MFDList.pop(i)
                RPDList.pop(i)
                N_RList.pop(i)
                DispList.pop(i)
                ADList.pop(i)

    ux = np.reshape(ux, (h, w))
    uy = np.reshape(uy, (h, w))

    if TypeOf == 'tracer':
        return [IDs, t_In, xAge, Xs, Ys]
    elif TypeOf == 'individual':
        return [Type, t_In, xAge, Xs, Ys, IDs, ID, Vals, GrowthList, MList, MFDList, RPDList,
            N_RList, DispList, ADList]
    elif TypeOf == 'resource':
        return [Type, t_In, xAge, Xs, Ys, IDs, ID, Vals]




def maintenance(Sp_IDs, t_In, xAge, Xs, Ys, colorD, MD, MFD, RPD, EnvD, IDs, Qs, GrowthList,
        MList, MFDList, RPDList, N_RList, DispList, ADList):

    if Sp_IDs == []:
        return [Sp_IDs, t_In, xAge, Xs, Ys, IDs, Qs, GrowthList, MList, MFDList, RPDList, N_RList,
                DispList, ADList]


    for j in range(len(IDs)):

        i = randint(0, len(IDs)-1)

        r1 = Qs[i][0]
        r1 -= MList[i]

        if r1 <= 0.0:   # starved

            Qs.pop(i)
            xAge.append(t_In[i])
            t_In.pop(i)
            Sp_IDs.pop(i)
            IDs.pop(i)
            Xs.pop(i)
            Ys.pop(i)
            GrowthList.pop(i)
            MList.pop(i)
            MFDList.pop(i)
            RPDList.pop(i)
            N_RList.pop(i)
            DispList.pop(i)
            ADList.pop(i)

        else: Qs[i] = [r1]

    return [Sp_IDs, t_In, xAge, Xs, Ys, IDs, Qs, GrowthList, MList, MFDList, RPDList, N_RList,
            DispList, ADList]





def transition(Sp_IDs, t_In, xAge, Xs, Ys, IDs, Qs, DList, GList, MList, MFDList, RPDList, N_RList, MFD, RPD, ADList):

    if Sp_IDs == []:
        return [Sp_IDs, t_In, xAge, Xs, Ys, IDs, Qs, DList, GList, MList, MFDList, RPDList, N_RList, MFD, RPD, ADList]

    for j in range(len(IDs)):

        i = randint(0, len(IDs)-1)
        state = ADList[i]
        mfd = MFDList[i]
        rpf = RPDList[i]

        # The individual's cell quota
        Q = Qs[i]
        QN = Q[0]

        if QN <= 0.0:
            Qs.pop(i)
            xAge.append(t_In[i])
            t_In.pop(i)
            Sp_IDs.pop(i)
            IDs.pop(i)
            Xs.pop(i)
            Ys.pop(i)
            GList.pop(i)
            MList.pop(i)
            MFDList.pop(i)
            RPDList.pop(i)
            N_RList.pop(i)
            DList.pop(i)
            ADList.pop(i)
            continue

        if state == 'd':

            x = np.random.binomial(1, rpf) # make this probability a randomly chosen variable
            if x == 1:

                # An energetic cost to resuscitate
                QN -= 0.02 * QN # becoming active costs energy, larger individuals pay a greater cost

                if QN <= 0.0:
                    Qs.pop(i)
                    xAge.append(t_In[i])
                    t_In.pop(i)
                    Sp_IDs.pop(i)
                    IDs.pop(i)
                    Xs.pop(i)
                    Ys.pop(i)
                    GList.pop(i)
                    MList.pop(i)
                    MFDList.pop(i)
                    RPDList.pop(i)
                    N_RList.pop(i)
                    DList.pop(i)
                    ADList.pop(i)
                    continue


                else:
                    Qs[i] = [QN]
                    ADList[i] = 'a'
                    MList[i] = MList[i]*mfd

        elif state == 'a':
            if QN <= 0.1:  # go dormant

                Qs[i] = [QN]
                ADList[i] = 'd'
                MList[i] = MList[i]/mfd


    return [Sp_IDs, t_In, xAge, Xs, Ys, IDs, Qs, DList, GList, MList, MFDList, RPDList, N_RList, MFD, RPD, ADList]




def decimate(Sp_IDs, t_In, I_xAge, Xs, Ys, xAge, colorD, MD, MFD, RPD, EnvD, IDs, Qs, GrowthList,
            MList, MFDList, RPDList, N_RList, DispList, ADList, minN = 10000):

    if Sp_IDs == []:
        return [Sp_IDs, t_In, I_xAge, Xs, Ys, xAge, colorD, MD, MFD, RPD, EnvD, IDs, Qs, GrowthList,
            MList, MFDList, RPDList, N_RList, DispList, ADList, minN]

    while len(IDs) > 0.9*minN:

        i = randint(0, len(IDs)-1)

        Qs.pop(i)
        xAge.append(t_In[i])
        t_In.pop(i)
        Sp_IDs.pop(i)
        IDs.pop(i)
        Xs.pop(i)
        Ys.pop(i)
        GrowthList.pop(i)
        MList.pop(i)
        MFDList.pop(i)
        RPDList.pop(i)
        N_RList.pop(i)
        DispList.pop(i)
        ADList.pop(i)

    return [Sp_IDs, t_In, I_xAge, Xs, Ys, xAge, colorD, MD, MFD, RPD, EnvD, IDs, Qs, GrowthList,
            MList, MFDList, RPDList, N_RList, DispList, ADList, minN]




def consume(RPFDict, R_Types, R_Vals, R_IDs, R_ID, R_Xs, R_Ys, R_t_In, R_xAge, Sp_IDs, Qs, I_IDs, I_ID, I_t_In, I_xAge, I_Xs, I_Ys,  w, h, GrowthDict, N_RD, DispDict, GList, MList, MFDList, RPDList, MFD, N_RList, DList, ADList):

    if not len(R_Types) or not len(Sp_IDs):
        return [RPFDict, R_Types, R_Vals, R_IDs, R_ID, R_Xs, R_Ys, R_t_In, R_xAge, Sp_IDs, Qs, I_IDs, I_ID, I_t_In, I_xAge, I_Xs, I_Ys,  w, h, GrowthDict, N_RD, DispDict, GList, MList, MFDList, RPDList, MFD, N_RList, DList, ADList]

    I_Boxes = [list([]) for _ in xrange(w*h)]
    R_Boxes = [list([]) for _ in xrange(w*h)]

    index = 0
    for i, val in enumerate(I_IDs):
        rX = int(round(I_Xs[i]))
        rY = int(round(I_Ys[i]))

        index = int(round(rX + (rY * w)))

        if index > len(I_Boxes) - 1:
            index = len(I_Boxes) - 1
        elif index < 0:
            index = 0

        I_Boxes[index].append(val)

    index = 0
    for i, val in enumerate(R_IDs):

        rX = int(round(R_Xs[i]))
        rY = int(round(R_Ys[i]))
        index = int(round(rX + (rY * w)))

        if index > len(R_Boxes) - 1:
            index = len(R_Boxes) - 1
        elif index < 0:
            index = 0

        R_Boxes[index].append(val)


    for i, box in enumerate(I_Boxes):
        if not len(box): continue

        R_Box = R_Boxes[i]

        for ind in box: # The individuals
            if not len(R_Box): break

            R_ID = choice(R_Box)
            boxI_ex = R_Box.index(R_ID)

            # The food
            j = R_IDs.index(R_ID)
            R_val = R_Vals[j]
            R_type = R_Types[j]

            rtype = list(R_type)
            R = rtype.pop(0)
            rnum = int(''.join(rtype))

            # The Individual
            ID = I_IDs.index(ind)

            # the species
            mu1 = float(GList[ID])

            state = ADList[ID]

            # The individual's cell quota
            Q = Qs[ID][0]

            if Q <= 0.0:
                Qs.pop(ID)
                I_xAge.append(I_t_In[ID])
                I_t_In.pop(ID)
                Sp_IDs.pop(ID)
                I_IDs.pop(ID)
                I_Xs.pop(ID)
                I_Ys.pop(ID)
                GList.pop(ID)
                MList.pop(ID)
                MFDList.pop(ID)
                RPDList.pop(ID)
                N_RList.pop(ID)
                DList.pop(ID)
                ADList.pop(ID)

                continue

            if state == 'd':
                ADList[ID] = 'a'

                mfd = MFDList[ID]
                MList[ID] = MList[ID]*mfd

            # An energetic cost to growth, i.e., pay a bigger cost for growing faster. No such thing as a free lunch.
            Q -= Q * mu1 * 0.02 # the faster you grow, the more energy it takes, and hence the more efficient you need to be

            efficiency = N_RList[ID][rnum]
            mu = mu1 * efficiency

            if R_val > (mu * Q): # Increase cell quota
                R_val = R_val - (mu * Q)
                Q += (mu * Q)

            else:
                Q += R_val
                R_val = 0.0

            if Q > 1.0:
                R_val = Q - 1.0
                Q = 1.0
                R_Vals[j] = R_val

            if R_val <= 0.0:
                R_Box.pop(boxI_ex)
                R_Vals.pop(j)
                R_xAge.append(R_t_In[j])
                R_t_In.pop(j)
                R_Types.pop(j)
                R_IDs.pop(j)
                R_Xs.pop(j)
                R_Ys.pop(j)

            if Q < 0.0: Q = 0.0

            if R == 'N':
                Qs[ID] = [Q]
            if R == 'P':
                Qs[ID] = [Q]
            if R == 'C':
                Qs[ID] = [Q]


    return [RPFDict, R_Types, R_Vals, R_IDs, R_ID, R_Xs, R_Ys, R_t_In, R_xAge, Sp_IDs, Qs, I_IDs, I_ID, I_t_In, I_xAge, I_Xs, I_Ys,  w, h, GrowthDict, N_RD, DispDict, GList, MList, MFDList, RPDList, MFD, N_RList, DList, ADList]



def reproduce(repro, spec, Sp_IDs, t_In, I_xAge, Xs, Ys, Qs, IDs, ID, w, h, GD, DispD,
        colorD, N_RD, MD, MFD, RPD, EnvD, envGs, nN, GList, MList, MFDList, RPDList,
        NList, DList, ADList):

    if Sp_IDs == []:
        return [Sp_IDs, t_In, I_xAge, Xs, Ys, Qs, IDs, ID, GD, DispD, GList, MList, MFDList, RPDList,
                NList, DList, ADList, MFD, RPD]

    if repro == 'fission':

        for j in range(len(IDs)):

            i = randint(0, len(IDs)-1)

            state = ADList[i]

            if state == 'd':
                continue

            Q = Qs[i][0]

            if Q <= 0.0:
                Qs.pop(i)
                I_xAge.append(t_In[i])
                t_In.pop(i)
                Sp_IDs.pop(i)
                IDs.pop(i)
                Xs.pop(i)
                Ys.pop(i)
                GList.pop(i)
                MList.pop(i)
                MFDList.pop(i)
                RPDList.pop(i)
                NList.pop(i)
                DList.pop(i)
                ADList.pop(i)

                continue

            pq = float(Q)
            if pq < 0.0: pq = 0.0
            p = np.random.binomial(1, pq)

            if p == 1 and pq >= 0.5: # individual is large enough to reproduce

                spID = Sp_IDs[i]
                X = Xs[i]
            	Y = Ys[i]

                pg = []
                sp_opts = EnvD[spID]

                for g, opt in enumerate(sp_opts):

                    x, y = envGs[g]
                    pg.append(1 - (abs(X - x)/max([X,x])))
                    pg.append(1 - (abs(Y - y)/max([Y,y])))


                if np.mean(pg) > 1 or np.mean(pg) < 0:
                    print pg
                    sys.exit()

                p = np.mean(pg)
                p = np.random.binomial(1, p)
                if p == 1: # the environment is suitable for reproduction

                    Qs[i] = [Q/2.0]
                    Qs.append([Q/2.0])

                    ID += 1
                    IDs.append(ID)
                    t_In.append(0)

                    p = np.random.binomial(1, spec)
                    p = 0



                    i = GD[spID]
                    GList.append(i)

                    i = MD[spID]
                    MList.append(i)

                    i = N_RD[spID]
                    NList.append(i)

                    i = DispD[spID]
                    DList.append(i)

                    i = MFD[spID]
                    MFDList.append(i)

                    i = RPD[spID]
                    RPDList.append(i)

                    Sp_IDs.append(spID)
                    ADList.append('a')

                    Xs.append(float(X))

                    newY = float(np.random.uniform(Y-0.1, Y+0.1, 1))
                    if limit > newY: newY = 0
                    elif newY > h: newY = h - limit
                    Ys.append(newY)


    return [Sp_IDs, t_In, I_xAge, Xs, Ys, Qs, IDs, ID, GD, DispD, GList, MList, MFDList, RPDList,
                NList, DList, ADList, MFD, RPD]




def nearest_forage(RVals, RX, RY, repro, spec, Sp_IDs, t_In, I_xAge, Xs, Ys, Qs, IDs, ID, w, h, GD, DispD,
        colorD, N_RD, MD, MFD, RPD, EnvD, envGs, nN, GList, MList, MFDList, RPDList,
        NList, DList, ADList):

    if Sp_IDs == [] or RVals == []:
        return [RVals, RX, RY, repro, spec, Sp_IDs, t_In, I_xAge, Xs, Ys, Qs, IDs, ID, w, h, GD, DispD,
        colorD, N_RD, MD, MFD, RPD, EnvD, envGs, nN, GList, MList, MFDList, RPDList,
        NList, DList, ADList]

    n = len(IDs)
    n = min([50, n])
    r = len(RVals)

    for j in range(n):
        i = randint(0, len(IDs)-1)

        state = ADList[i]
        if state == 'd':
            continue

        x1 = Xs[i]
        y1 = Ys[i]

        MinDist = 10000
        dist = 0

        rx = 0
        ry = 0

        for j in range(r):

            x2 = RX[j]
            y2 = RY[j]

            dist = math.sqrt((x1 - x2)**2 + (y1 - y2)**2)
            if dist < MinDist:
                MinDist = dist
                rx = x2
                ry = y2

        dist = DList[i]

        # A cost for active dispersal
        r1 = Qs[i][0]
        r1 -= r1 * dist * 0.02 # greater cost for larger individuals, cost is multiplied across distance

        if r1 <= 0.0:
            #print 'r1:',r1
            Qs.pop(i)
            I_xAge.append(t_In[i])
            t_In.pop(i)
            Sp_IDs.pop(i)
            IDs.pop(i)
            Xs.pop(i)
            Ys.pop(i)
            GList.pop(i)
            MList.pop(i)
            MFDList.pop(i)
            RPDList.pop(i)
            NList.pop(i)
            DList.pop(i)
            ADList.pop(i)

            continue

        Qs[i] = [r1]

        if x1 > rx:
            x1 -= dist

        elif x1 < rx:
            x1 += dist

        if y1 > ry:
            y1 -= dist

        elif y1 < ry:
            y1 += dist

        if x1 > w: x1 = w
        elif x1 < 0: x1 = 0
        if y1 > h: y1 = h
        elif y1 < 0: y1 = 0

        Xs[i] = x1
        Ys[i] = y1

    return [RVals, RX, RY, repro, spec, Sp_IDs, t_In, I_xAge, Xs, Ys, Qs, IDs, ID, w, h, GD, DispD,
        colorD, N_RD, MD, MFD, RPD, EnvD, envGs, nN, GList, MList, MFDList, RPDList,
        NList, DList, ADList]
