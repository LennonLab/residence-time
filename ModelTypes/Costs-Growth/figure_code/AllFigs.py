#!/usr/bin/python
from os.path import expanduser
import sys

mydir = expanduser('~/GitHub/residence-time2/Emergence/ModelTypes/Costs-Growth/figure_code')

sys.path.append(mydir + "/MacroecologyPatterns")
sys.path.append(mydir + "/TaxaDiversityPatterns")
sys.path.append(mydir + "/TimeSeries")
sys.path.append(mydir + "/TraitDiversityPatterns")
sys.path.append(mydir + "/ResourcePatterns")

import Taxa_vs_Tau
import Traits_vs_Tau_3Figs
import BMR_vs_Tau_Multi
import DiversityAbundanceScaling
import MTE
import Taylors_Law


import Specialization_vs_Tau
import Res_vs_Tau
import TimeSeries
import trait_relations


import SupFig5
import SupFig6
import SupFig8
import SupFig9
import SADs
