# -*- coding: utf-8 -*-
from __future__ import division
from random import randint, choice
import numpy as np
import sys
import math
#from math import modf
#import decimal
import time

limit = 0.1

def coord(d):
    return float(np.random.uniform(0.1*d, 0.9*d))


def GetIndParam(means):
    vals = []

    if isinstance(means, float) or isinstance(means, int):
        std = means/100.0
        vals = np.random.normal(means, std)
        if vals < 0.0:
            vals = -1*vals

    else:
        for val in means:
            std = val/100.0
            i = np.random.normal(val, std)
            if i < 0.0:
                i = -i
            vals.append(i)

    return vals



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
        for i in range(10):
            IDs.append(0)
            t_In.append(0)
            Ys.append(float(np.random.uniform(0.1*h, 0.9*h)))
            Xs.append(float(np.random.uniform(0.1*w, 0.101*w)))
    else:
        x = np.random.binomial(1, u0)
        if x == 1:
            IDs.append(0)
            t_In.append(0)
            Ys.append(float(np.random.uniform(0.1*h, 0.9*h)))
            Xs.append(float(np.random.uniform(0.1*w, 0.101*w)))

    return [IDs, t_In, Xs, Ys]



def ResIn(motion, Type, Vals, Xs, Ys, ID, IDs, t_In, numr, rmax, nN, nP, nC, w, h, u0):


    for r in range(numr):
        x = np.random.binomial(1, 0.99*u0)

        if x == 1:
            rval = int(np.random.random_integers(1, rmax, 1))
            nr = choice(['N', 'P', 'C'])

            if nr == 'N':
                rtype = int(np.random.random_integers(0, nN-1, 1))
                rtype = 'N'+str(rtype)

            if nr == 'P':
                rtype = int(np.random.random_integers(0, nP-1, 1))
                rtype = 'P'+str(rtype)

            if nr == 'C':
                rtype = int(np.random.random_integers(0, nC-1, 1))
                rtype = 'C'+str(rtype)

            Vals.append(rval)
            IDs.append(ID)
            Type.append(rtype)
            t_In.append(0)
            ID += 1


            Ys.append(float(np.random.uniform(0.01*h, 0.99*h)))
            Xs.append(float(np.random.uniform(0.01*w, 0.99*w)))


    return [Type, Vals, Xs, Ys, IDs, ID, t_In]



def immigration(mfmax, p_max, d_max, g_max, m_max, motion, seed, ip, Sp, Xs, Ys, w, h, MD, MFD, RPD,
        EnvD, envGs, GD, DispD, colorD, IDs, ID, t_In, Qs, N_RD, P_RD, C_RD,
        nN, nP, nC, u0, alpha, GList, MList, NList, PList, CList, DList, ADList):

    if u0 > 1.0:
        u0 = 1.0

    for m in range(seed):
        x = 0

        if seed > 1:
            x = 1
        else:
            x = np.random.binomial(1, u0*ip)

        if x == 1:

            if seed > 1:
                prop = np.random.randint(1, 1000)
                #prop = float(np.random.logseries(alpha, 1))
            else:
                prop = np.random.randint(1, 1000)
                #prop = float(np.random.logseries(alpha, 1))

            Sp.append(prop)

            Ys.append(float(np.random.uniform(0.01*h, 0.99*h)))
            Xs.append(float(np.random.uniform(0.01*w, 0.99*w)))


            IDs.append(ID)
            t_In.append(0)
            ID += 1
            Qn = float(np.random.uniform(0.1, 0.1))
            Qp = float(np.random.uniform(0.1, 0.1))
            Qc = float(np.random.uniform(0.1, 0.1))

            Qs.append([Qn, Qp, Qc])

            if prop not in colorD:
                # speciescolor
                colorD = get_color(prop, colorD)

                # species growth rate
                g = np.random.uniform(g_max/1, g_max)
                GD[prop] = g

                # species maintenance
                md = np.random.uniform(m_max/1, m_max)
                MD[prop] = md

                # species maintenance factor
                MFD[prop] = np.random.uniform(1, mfmax)

                # species RPF factor
                RPD[prop] = np.random.uniform(p_max/100, p_max)

                # species active dispersal rate
                DispD[prop] = np.random.uniform(d_max/1, d_max)

                # species environmental gradient optima
                glist = []
                for j in envGs:
                    x = np.random.uniform(0.0, w)
                    y = np.random.uniform(0.0, h)
                    glist.append([x,y])
                EnvD[prop] = glist

                # species growth efficiency
                GE = np.random.uniform(0.99, 1.0, nN)

                # species Nitrogen use efficiency
                N_RD[prop] = GE #np.random.uniform(0.01, 1.0, nN)

                # species Phosphorus use efficiency
                P_RD[prop] = GE #np.random.uniform(0.01, 1.0, nP)

                # species Carbon use efficiency
                C_RD[prop] = GE #np.random.uniform(0.01, 1.0, nC)

            state = 'a'
            ADList.append(state)

            means = GD[prop]
            i = GetIndParam(means)
            GList.append(i)

            means = MD[prop]
            i = GetIndParam(means)
            MList.append(i)

            means = N_RD[prop]
            n = GetIndParam(means)
            means = P_RD[prop]
            p = GetIndParam(means)
            means = C_RD[prop]
            c = GetIndParam(means)

            NList.append(n)
            PList.append(p)
            CList.append(c)

            means = DispD[prop]
            i = GetIndParam(means)
            DList.append(i)

    return [Sp, Xs, Ys, MD, MFD, RPD, EnvD, GD, DispD, colorD, IDs, ID, t_In, Qs, N_RD,
            P_RD, C_RD, GList, MList, NList, PList, CList, DList, ADList]



def fluid_movement(TypeOf, List, t_In, xAge, Xs, Ys, ux, uy, w, h, u0):

    Type, IDs, ID, Vals = [], [], int(), []

    if TypeOf == 'resource':
        Type, IDs, ID, Vals = List
    elif TypeOf == 'individual':
        Type, IDs, ID, Vals, DispD, GrowthList, MList, N_RList, P_RList, C_RList, DispList, ADList = List
    else:
        IDs = List

    if Xs == []:
        if TypeOf == 'tracer':
            return [IDs, Xs, Ys, xAge, t_In]
        elif TypeOf == 'individual':
            return [Type, Xs, Ys, xAge, IDs, ID, t_In, Vals, GrowthList,
                    MList, N_RList, P_RList, C_RList, DispList, ADList]
        elif TypeOf == 'resource':
            return [Type, Xs, Ys, xAge, IDs, ID, t_In, Vals]

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
        if TypeOf == 'individual':

            # A cost for active dispersal, larger dispersal means a bigger cost
            r1,r2,r3 = Vals[i]
            if ADList[i] == 'a':
                r1 -= DispList[i]*0.001
                r2 -= DispList[i]*0.001
                r3 -= DispList[i]*0.001
            Vals[i] = [r1, r2, r3]

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
                N_RList.pop(i)
                P_RList.pop(i)
                C_RList.pop(i)
                DispList.pop(i)
                ADList.pop(i)

    ux = np.reshape(ux, (h, w))
    uy = np.reshape(uy, (h, w))

    if TypeOf == 'tracer':
        return [IDs, Xs, Ys, xAge, t_In]
    elif TypeOf == 'individual':
        return [Type, Xs, Ys, xAge, IDs, ID, t_In, Vals, GrowthList, MList,
            N_RList, P_RList, C_RList, DispList, ADList]
    elif TypeOf == 'resource':
        return [Type, Xs, Ys, xAge, IDs, ID, t_In, Vals]




def maintenance(Sp_IDs, Xs, Ys, xAge, colorD, MD, MFD, RPD, EnvD, IDs, t_In, Qs, GrowthList,
        MList, N_RList, P_RList, C_RList, DispList, ADList):

    numD = 0
    if Sp_IDs == []:
        return [Sp_IDs, Xs, Ys, xAge, IDs, t_In, Qs, GrowthList, MList, N_RList,
                P_RList, C_RList, DispList, ADList]


    for j in range(len(IDs)):

        i = randint(0, len(IDs)-1)

        minVal = MList[i]
        #state = ADList[i]

        Q = Qs[i]
        r1,r2,r3 = Q
        r1 -= MList[i]
        r2 -= MList[i]
        r3 -= MList[i]
        Q = [r1, r2, r3]

        if min(Q) < minVal:   # starved

            numD += 1

            Qs.pop(i)
            xAge.append(t_In[i])
            t_In.pop(i)
            Sp_IDs.pop(i)
            IDs.pop(i)
            Xs.pop(i)
            Ys.pop(i)
            GrowthList.pop(i)
            MList.pop(i)
            N_RList.pop(i)
            P_RList.pop(i)
            C_RList.pop(i)
            DispList.pop(i)
            ADList.pop(i)

        else: Qs[i] = Q

    #if numD > 0: print numD

    return [Sp_IDs, Xs, Ys, xAge, IDs, t_In, Qs, GrowthList, MList, N_RList,
            P_RList, C_RList, DispList, ADList]





def transition(Sp_IDs, IDs, Qs, GrowthList, MList, MFD, RPD, ADList):

    if Sp_IDs == []:
        return [Sp_IDs, IDs, Qs, GrowthList, MList, ADList]

    for j in range(len(IDs)):

        i = randint(0, len(IDs)-1)
        spid = Sp_IDs[i]
        state = ADList[i]

        mfd = float(MFD[spid])
        val = Qs[i]

        # The individual's cell quota
        Q = Qs[i]
        QN = Q[0]
        QP = Q[1]
        QC = Q[2]

        rpf = RPD[spid]

        if state == 'd':

            x = np.random.binomial(1, rpf) # make this probability a randomly chosen variable

            if x == 1:

                # An energetic cost to resuscitate
                QN -= rpf*0.01 # becoming active costs energy
                QP -= rpf*0.01
                QC -= rpf*0.01
                Qs[i] = [QN, QP, QC]

                ADList[i] = 'a'
                MList[i] = float(MList[i])*float(mfd)

        elif state == 'a':
            if min(val) <= MList[i]*10:  # go dormant

                # An energetic cost to resuscitate
                QN -= rpf*0.01 # becoming active costs energy
                QP -= rpf*0.01
                QC -= rpf*0.01
                Qs[i] = [QN, QP, QC]

                ADList[i] = 'd'
                MList[i] = float(MList[i])/float(mfd)


    return [Sp_IDs, IDs, Qs, GrowthList, MList, ADList]




def decimate(Sp_IDs, Xs, Ys, xAge, colorD, MD, MFD, RPD, EnvD, IDs, t_In, Qs, GrowthList,
            MList, N_RList, P_RList, C_RList, DispList, ADList, minN = 10000):

    if Sp_IDs == []:
        return [Sp_IDs, Xs, Ys, xAge, IDs, t_In, Qs, GrowthList, MList,
        N_RList, P_RList, C_RList, DispList, ADList]

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
        N_RList.pop(i)
        P_RList.pop(i)
        C_RList.pop(i)
        DispList.pop(i)
        ADList.pop(i)

    return [Sp_IDs, Xs, Ys, xAge, IDs, t_In, Qs, GrowthList, MList, N_RList,
            P_RList, C_RList, DispList, ADList]




def consume(RPFDict, R_Types, R_Vals, R_IDs, R_ID, R_Xs, R_Ys, R_t_In, R_xAge, Sp_IDs,
        Qs, I_IDs, I_ID, I_t_In, I_Xs, I_Ys, w, h, GD, N_RD, P_RD, C_RD, DispD,
        GrowthList, MList, MFD, N_RList, P_RList, C_RList, DispList, ADList):

    if not len(R_Types) or not len(Sp_IDs):
        List = [R_Types, R_Vals, R_IDs, R_ID, R_t_In, R_xAge, R_Xs]
        List += [R_Ys, Sp_IDs, Qs, I_IDs, I_ID, I_t_In]
        List += [I_Xs, I_Ys, GrowthList, MList, N_RList,
                P_RList, C_RList, DispList, ADList]
        return List

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
            sp = Sp_IDs[ID]
            mu1 = float(GrowthList[ID])

            state = ADList[ID]

            # The individual's cell quota
            Q = Qs[ID]
            QN = Q[0]
            QP = Q[1]
            QC = Q[2]

            if state == 'd':
                rpf = RPFDict[sp]
                x = np.random.binomial(1, rpf) # make this probability a randomly chosen variable

                if x == 1:
                    ADList[ID] = 'a'

                    # An energetic cost to resuscitate
                    QN -= rpf*0.05 # becoming active costs energy
                    QP -= rpf*0.05
                    QC -= rpf*0.05

                    mfd = float(MFD[sp])
                    MList[ID] = MList[ID]*mfd

            # An energetic cost to growth, i.e., pay a bigger cost for growing faster. No such thing as a free lunch.
            QN -= mu1*0.001 # the faster you grow, the more energy it takes, and hence the more efficient you need to be
            QP -= mu1*0.001
            QC -= mu1*0.001

            Q = 0.0
            efficiency = 0.0

            if R == 'N':
                efficiency = N_RList[ID][rnum]
                Q = QN

            if R == 'P':
                efficiency = P_RList[ID][rnum]
                Q = QP

            if R == 'C':
                efficiency = C_RList[ID][rnum]
                Q = QC

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
                Qs[ID] = [Q, QP, QC]
            if R == 'P':
                Qs[ID] = [QN, Q, QC]
            if R == 'C':
                Qs[ID] = [QN, QP, Q]


    return [R_Types, R_Vals, R_IDs, R_ID, R_t_In, R_xAge, R_Xs, R_Ys, Sp_IDs,
            Qs, I_IDs, I_ID, I_t_In, I_Xs, I_Ys, GrowthList, MList, N_RList,
            P_RList, C_RList, DispList, ADList]



def reproduce(repro, spec, Sp_IDs, Qs, IDs, ID, t_In, Xs, Ys, w, h, GD, DispD,
        colorD, N_RD, P_RD, C_RD, MD, MFD, RPD, EnvD, envGs, nN, nP, nC, GList, MList,
        NList, PList, CList, DList, ADList):

    if Sp_IDs == []:
        return [Sp_IDs, Qs, IDs, ID, t_In, Xs, Ys, GD, DispD, GList, MList,
                NList, PList, CList, DList, ADList, MFD, RPD]

    if repro == 'fission':

        for j in range(len(IDs)):

            i = randint(0, len(IDs)-1)

            state = ADList[i]

            if state == 'd':
                continue

            Q = Qs[i]
            pq = float(np.mean(Q))
            if pq < 0.0: pq = 0.0
            p = np.random.binomial(1, pq)

            if p == 1: # individual is large enough to reproduce

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

                    QN = Q[0]
                    QP = Q[1]
                    QC = Q[2]

                    Qs[i] = [QN/2.0, QP/2.0, QC/2.0]
                    Qs.append([QN/2.0, QP/2.0, QC/2.0])

                    ID += 1
                    IDs.append(ID)
                    t_In.append(t_In[i])

                    p = np.random.binomial(1, spec)
                    p = 0
                    if p == 1:

                        # speciate
                        t = time.clock()
                        spID_new = (spID+t)*t

                        # new species color
                        colorD = get_color(spID_new, colorD)

                        # new species growth rate
                        p = 1 #np.random.binomial(1, 0.1)
                        mu = 0
                        if p == 1:
                            mu = GetIndParam(GD[spID])
                            GD[spID_new] = mu
                        else:
                            mu = GD[spID]
                            GD[spID_new] = GD[spID]

                        # new species maintenance
                        p = 1 #np.random.binomial(1, 0.1)
                        if p == 1:
                            MD[spID_new] = GetIndParam(MD[spID])
                        else:
                            MD[spID_new] = MD[spID]

                        # new species maintenance factor
                        p = 1 #np.random.binomial(1, 0.1)
                        if p == 1:
                            MFD[spID_new] = GetIndParam(MFD[spID])
                        else:
                            MFD[spID_new] = MFD[spID]

                        # new species resuscitation probability
                        p = 1 #np.random.binomial(1, 0.1)
                        if p == 1:
                            RPD[spID_new] = GetIndParam(RPD[spID])
                        else:
                            RPD[spID_new] = RPD[spID]

                        # species environmental gradient optima
                        glist = []
                        for j, g in enumerate(envGs):
                            p = np.random.binomial(1, 0.1)
                            if p == 1:
                                x = np.random.uniform(0.0, w)
                                y = np.random.uniform(0.0, h)
                            else:
                                x = EnvD[spID][j][0]
                                y = EnvD[spID][j][1]

                            glist.append([x, y])
                            EnvD[spID_new] = glist

                        # new species active dispersal rate
                        p = 1 #np.random.binomial(1, 0.1)
                        if p == 1:
                            DispD[spID_new] = GetIndParam(DispD[spID])
                        else:
                            DispD[spID_new] = DispD[spID]

                        # new species resource use efficiencies
                        p = 1 #np.random.binomial(1, 0.1)
                        if p == 1:
                            GE = np.random.uniform(0.01, 1.0, nN)
                            P_RD[spID_new] = GE
                            C_RD[spID_new] = GE
                            N_RD[spID_new] = GE # high growth rate leads to low efficiency
                            #P_RD[spID_new] = np.random.uniform(0.01, 1.0, nP) * (1-g) # high growth rate leads to low efficiency
                        else:
                            N_RD[spID_new] = N_RD[spID]
                            C_RD[spID_new] = C_RD[spID]
                            P_RD[spID_new] = P_RD[spID]

                        spID = spID_new

                    means = GD[spID]
                    i = GetIndParam(means)
                    GList.append(i)

                    means = MD[spID]
                    i = GetIndParam(means)
                    MList.append(i)

                    means = N_RD[spID]
                    i = GetIndParam(means)
                    NList.append(i)

                    means = P_RD[spID]
                    i = GetIndParam(means)
                    PList.append(i)

                    means = C_RD[spID]
                    i = GetIndParam(means)
                    CList.append(i)

                    means = DispD[spID]
                    i = GetIndParam(means)
                    DList.append(i)

                    Sp_IDs.append(spID)

                    ADList.append('a')

                    newX = float(np.random.uniform(X-0.1, X, 1))
                    if limit > newX: newX = 0
                    if newX > w - limit: newX = w - limit
                    Xs.append(newX)

                    newY = float(np.random.uniform(Y-0.1, Y+0.1, 1))
                    if limit > newY: newY = 0
                    elif newY > h: newY = h - limit
                    Ys.append(newY)


    listlen = [len(Sp_IDs), len(Qs), len(IDs), len(t_In), len(Xs), len(Ys), len(GList), len(MList), len(DList), len(ADList)]
    if min(listlen) != max(listlen):
        print 'In reproduce'
        print 'min(listlen) != max(listlen)'
        print listlen
        sys.exit()

    return [Sp_IDs, Qs, IDs, ID, t_In, Xs, Ys, GD, DispD, GList, MList,
                NList, PList, CList, DList, ADList, MFD, RPD]




def nearest_forage(RVals, RX, RY, repro, spec, Sp_IDs, Qs, IDs, ID, t_In, Xs, Ys,  w, h, GD, DispD,
        colorD, N_RD, P_RD, C_RD, MD, MFD, RPD, EnvD, envGs, nN, nP, nC, GList, MList,
        NList, PList, CList, DList, ADList):

    if Sp_IDs == []:
        return [Sp_IDs, Qs, IDs, ID, t_In, Xs, Ys, GD, DispD, GList,
                    MList, NList, PList, CList, DList, ADList]

    n = len(IDs)
    n = min([100, n])
    r = len(RVals)

    for j in range(n):
        i = randint(0, len(IDs)-1)

        state = ADList[i]
        if state == 'd':
            #pass
            continue

        x1 = Xs[i]
        y1 = Ys[i]

        MinDist = 10000

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

        spID = Sp_IDs[i]
        dist = DispD[spID]

        # A cost for active dispersal
        r1,r2,r3 = Qs[i]
        r1 -= dist*0.001
        r2 -= dist*0.001
        r3 -= dist*0.001
        Qs[i] = [r1, r2, r3]


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

    return [Sp_IDs, Qs, IDs, ID, t_In, Xs, Ys, GD, DispD, GList, MList,
                NList, PList, CList, DList, ADList]
