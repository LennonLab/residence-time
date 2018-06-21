from __future__ import division
from random import choice, randint, shuffle
from numpy.random import binomial, uniform
import time
import copy



def immigration(sD, iD, ps):
    h, r, u, nr, im, Si = ps
    if binomial(1, im*u) == 0: return sD, iD

    sz = 10**uniform(0, 2)
    q = sz * 10**uniform(-2, 0)
    p = randint(0, Si-1)
    ID = time.time()
    iD[ID] = copy.copy(sD[p])
    iD[ID].update({'age' : 0, 'sp' : p, 'time.in.state' : 0,
        'x' : 0, 'st':'a', 'sz': sz, 'q': q, 'e_ef': [0]*nr})
    return sD, iD



def ResIn(rD, ps):
    h, r, u, nr, im, Si = ps
    for i in range(r):
        if binomial(1, u) == 1:
            rD[time.time()] = {'t' : choice(range(nr)),
            'x' : 0, 'age' : 0, 'v': 10**uniform(0, 1)}
    return rD



def res_flow(rD, ps):
    h, r, u, nr, im, Si = ps
    for k, v in rD.items():
        rD[k]['age'] += 1
        rD[k]['x'] += u
        if rD[k]['x'] > h or v['v'] <= 0: del rD[k]
    return rD



def ind_flow(iD, ps):
    h, r, u, nr, im, Si = ps
    for k, v in iD.items():
        iD[k]['x'] += u
        if iD[k]['x'] >= h: del iD[k]
    return iD



def ind_disp(iD, ps):
    h, r, u, nr, im, Si = ps
    for k, v in iD.items():

        if v['st'] == 'a':
            d = min([v['di'] * v['sz'], v['q']])
            iD[k]['x'] -= d
            iD[k]['q'] -= d

        if iD[k]['q'] < 0: del iD[k]
    return iD



def grow(iD, ps):
    h, r, u, nr, im, Si = ps
    for k, v in iD.items():

        if v['st'] == 'a':
            g = min([v['gr'] * v['sz'], v['q']])
            iD[k]['sz'] += g
            iD[k]['q'] -= g

        if iD[k]['q'] < 0: del iD[k]
    return iD



def maintenance(iD, ps):
    h, r, u, nr, im, Si = ps
    for k, v in iD.items():

        iD[k]['age'] += 1
        if v['st'] == 'd': iD[k]['q'] -= v['mt'] * v['mf']
        else: iD[k]['q'] -= v['mt']
        if iD[k]['q'] < 0: del iD[k]

    return iD



def consume(iD, rD, ps):
    h, r, u, nr, im, Si = ps
    D = 0

    ids = iD.keys()
    if len(ids) == 0: return iD, rD, D

    shuffle(ids)
    for i in ids:
        rlen = len(rD.keys())
        D = rlen/h
        p = D/(1+D)
        if rlen == 0: return iD, rD, D
        if binomial(1, p) == 0: continue

        if iD[i]['st'] == 'a':
            i2 = randint(0, rlen-1)
            c = rD.keys()[i2]
            t = rD[c]['t']
            e = iD[i]['ef'][t]
            if e > 0: iD[i]['e_ef'][t] += 1
            ci = rD[c]['v'] * e
            ci = min([ci * iD[i]['sz'], ci])
            iD[i]['q'] += ci
            rD[c]['v'] -= ci
            if rD[c]['v'] == 0: del rD[c]

    return iD, rD, D



def reproduce(sD, iD, ps):
    n=0
    h, r, u, nr, im, Si = ps
    for k, v in iD.items():
        if v['q'] > 0 and v['st'] == 'a':
            ri = v['q']/v['mt']
            p = ri/(1+ri) * v['sz']/(1+v['sz'])

            if binomial(1, p) == 1:
                n += 1
                iD[k]['q'] = v['q']/2
                iD[k]['sz'] = v['sz']/2
                iD[k]['age'] = 0
                i = float(time.time())
                iD[i] = copy.copy(iD[k])
                iD[i]['e_ef'] = [0]*nr

    return sD, iD, n



def transition(iD, ps):
    h, r, u, nr, im, Si = ps
    if len(list(iD)) == 0: return iD

    for k, v in iD.items():
        ri =  v['q']/v['mt']
        ap =  1/(1+ri) * v['age']/(1+v['age'])
        dp = v['rp'] * ri/(1+ri)

        if v['st'] == 'd' and binomial(1, dp) == 1:
            iD[k]['st'] = 'a'
        elif v['st'] == 'a' and binomial(1, ap) == 1:
            iD[k]['st'] = 'd'

    return iD
