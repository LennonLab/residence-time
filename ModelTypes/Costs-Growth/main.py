from __future__ import division
from random import shuffle, choice, randint
from lazyme.string import color_print
from numpy import mean, log10, array
from numpy.random import uniform, binomial
import time
import sys
import os

mydir = os.path.expanduser('~/GitHub/residence-time2/Emergence')
sys.path.append(mydir)
import main_fxns as fx
import bide as bd



def iter_procs(iD, sD, rD, ps, ct, minlim):
    pr, D = 0, 0
    pD = float('NaN')
    h, r, u, nr, im, Si = ps
    procs = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    shuffle(procs)
    for p in procs:
        if p is 0: rD = bd.ResIn(rD, ps)
        elif p is 1: rD = bd.res_flow(rD, ps)
        elif p is 2: sD, iD = bd.immigration(sD, iD, ps)
        elif p is 3: iD = bd.ind_flow(iD, ps)
        elif p is 4: iD = bd.ind_disp(iD, ps)
        elif p is 5: iD, rD, D = bd.consume(iD, rD, ps)
        elif p is 6: iD = bd.grow(iD, ps)
        elif p is 7: iD = bd.transition(iD, ps)
        elif p is 8: iD = bd.maintenance(iD, ps)
        elif p is 9: sD, iD, pr = bd.reproduce(sD, iD, ps)

    avgQ = []
    Sz = []
    Ris = []
    SpIDs = []
    states = []
    if len(list(iD)) > 0:
        for k, v in iD.items():
            Ris.append(v['q']/(v['mt']*v['sz']))
            avgQ.append(v['q'])
            Sz.append(v['sz'])
            SpIDs.append(v['sp'])
            states.append(v['st'])
        avgQ = mean(avgQ)
        Sz = mean(Sz)
        N = len(states)
        pD = states.count('d')/N
        S = len(list(set(SpIDs)))
        Ri = mean(Ris)

        return [iD, sD, rD, N, S, len(list(rD)), ct+1, pr, pD, avgQ, Sz, D, Ri]

    else: return [iD, sD, rD, 0, 0, len(list(rD)), ct+1, 0, float('NaN'),
            float('NaN'), float('NaN'), float('NaN'), float('NaN')]



def run_model(sim, h, u, sD, nr, r, im, Si, clr):
    rD, iD, ct, splist2, asplist2, dsplist2 = {}, {}, 0, [], [], []
    ps = [h, r, u, nr, im, Si]
    hvar, uvar = 0, 0
    t = time.time()
    minlim = 1000+(h/u)**0.8

    print '\ntau:', log10(h/u), ' h:', h, ' u:', u

    ct2 = 0
    while ct != -1:
        ps = [h, r, u, nr, im, Si]
        iD, sD, rD, N, S, R, ct, prod, pD, avgQ, Sz, D, Ri = iter_procs(iD, sD, rD, ps, ct, minlim)
        Rp = R
        if ct <= minlim and ct%100 == 0:
            string  = 'sim:'+'%4s' % str(sim)+' ct:'+'%5s' % str(int(round(minlim-ct, 0)))
            string += '  tau:''%6s' % str(round(log10(h/u), 2)) + ' N:'+'%5s' % str(N)+' S:'+'%5s' % str(S)
            string += '  R:'+'%6s' % str(round(Rp, 4))+' P:'+'%4s' % str(round(prod, 2))
            string += '  %D:'+'%5s' % str(round(100*pD,1)) + '  Q:'+'%6s' % str(round(avgQ,2))
            string += '  Sz:'+'%6s' % str(round(Sz,2)) + '  Ri:'+'%6s' % str(round(Ri,2))
            color_print(string)

        elif ct > minlim and ct%10 == 0:
            ct2 += 1
            ps2 = [h, h, hvar, r, u, u, uvar, nr, im]
            splist2, asplist2, dsplist2 = fx.output(iD, sD, rD, ps2, sim, t,
                ct, prod, splist2, asplist2, dsplist2, D, minlim, pD, clr, Ri)
            if ct2 == 100: return


######################## RUN MODELS ############################################
fx.clear()

sim = 0
while sim < 10000:

    tau = 10**uniform(0, 6)
    h, u = 0, 0

    if binomial(1, 1.0) == 1:
        while h < 1 or h > 1000 or u < 0.001 or u > 1.0:
            if binomial(1, 0.5) == 1:
                h = 10**uniform(0, 3)
                u = h/tau
            else:
                u = 10**uniform(-3, 0)
                h = u*tau
    else:
        h = 10**uniform(0, 3)
        u = 10**uniform(-3, 0)


    r = lambda: randint(0,255)
    clr = '#%02X%02X%02X' % (r(),r(),r())

    sD = {}
    nr = 10
    r = 1
    im = 1
    Si = 1000

    for i in range(Si):
        en = choice(range(1, nr+1))
        es = 10**uniform(-3, 0, en)
        es = es.tolist() + [0]*(nr-en)
        es = array(es)/sum(es)
        shuffle(es)

        sD[i] = {'gr' : 10**uniform(-3, 0), 'di' : 10**uniform(-3, 0),
            'mt': 10**uniform(-3, 0), 'ef': es,
            'rp': 10**uniform(-3, 0), 'mf': 10**uniform(-3, 0)}

    run_model(sim, h, u, sD, nr, r, im, Si, clr)
    sim += 1
