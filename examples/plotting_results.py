#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 11 09:52:22 2025

@author: eyup
"""

import sys 
import matplotlib.pyplot as plt 


import os 

BEM_2D_path = os.path.join(os.path.expanduser("~"), 'BEM_2D')

plot_path = os.path.join(BEM_2D_path, 'post_processing')

sys.path.append(plot_path)

from plot import Ptool  

sim_path = '/Users/eyup/workspace/runs/BEM25D/TEST_FREESURFACE'

p = Ptool(sim_path)

# p.plot_timeseries()
p.plot_slip_profile()