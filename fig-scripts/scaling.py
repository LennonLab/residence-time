from __future__ import division
import  matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import random
import os
import sys
import linecache

import scipy as sc
from scipy import stats

import statsmodels.stats.api as sms
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.sandbox.regression.predstd import wls_prediction_std
from statsmodels.stats.outliers_influence import summary_table



mydir = os.path.expanduser('~/GitHub/residence-time')
sys.path.append(mydir+'/tools')
mydir2 = os.path.expanduser("~/")

dat = pd.read_csv(mydir + '/results/simulated_data/SimData2.csv')
dat = dat[dat['dormancy'] == ' \'yes\'']
dat['tau'] = np.log10(dat['width']/dat['flow.rate'])
dat['N'] = np.log10(dat['total.abundance'])
dat['S'] = np.log10(dat['species.richness'])
dat['E'] = np.log10(dat['simpson.e'])


f = smf.ols('S ~ N', dat).fit()
print f.summary()


f = smf.ols('E ~ N', dat).fit()
print f.summary()
