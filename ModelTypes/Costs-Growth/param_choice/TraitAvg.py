from __future__ import division
from random import shuffle, choice, randint
from lazyme.string import color_print
from numpy import mean, log10, array
from numpy.random import uniform, binomial
import numpy as np
import time
import sys
import os




es1 = []
es2 = []
es3 = []
es4 = []

for i in range(10000):
    en = choice(range(1, 10+1))
    es = 10**uniform(-3, 0, en)
    es = es.tolist() + [0]*(10-en)
    es = array(es)/sum(es)
    shuffle(es)

    es1.append(np.var(es))
    es = filter(lambda a: a != 0, es)
    es2.append(np.var(es))
    es3.append(1/len(es))

    t = 10**uniform(-3, 0)
    es4.append(t)


print np.mean(es1)
print np.mean(es2)
print np.mean(es3)
print np.mean(es4)
