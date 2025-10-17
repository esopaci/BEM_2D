#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  3 13:52:42 2024

@author: sopaci
"""

import os 
import matplotlib.pyplot as plt
import sys 

BEM_path = os.path.join(os.path.expanduser("~"), 'BEM_2D')
sys.path.append(os.path.join(BEM_path, 'src'))
import numpy as np
sys.path.append(os.path.join(BEM_path, 'post_processing'))
import plot as pl

# define simulation path

path = sys.argv[1]

p =pl.Ptool(path)

# Plot time series and slip profile
p.plot_timeseries()
p. plot_slip_profile1(vmin=-9)

# Read output
ox = p.read_ox_1()

# Find values at a certain depth (zi) 
zi = -7500

ti =700
zz = ox.z.unique()

zii = zz[np.abs(zz - zi).argmin()]

ox1 = ox[ox.z==zii].copy()

ox1 = ox1[ox1.t<700*p.t_yr]


# plotting time series
fig, ax = plt.subplots(1,1,clear = True)
ax.set_xlabel('time [yr]')
ax.set_ylabel('stress [MPa]')
ax.plot(ox1.t/p.t_yr, ox1.tau*1e-6)
fig.savefig('BP1_time_vs_stress.jpg', dpi =100, bbox_inches='tight')


fig, ax = plt.subplots(1,1,clear = True)
ax.set_xlabel('time [yr]')
ax.set_ylabel('log(v) [m/s]')
ax.semilogy(ox1.t/p.t_yr, ox1.v)
fig.savefig('BP1_time_vs_srate.jpg', dpi =100, bbox_inches='tight')


## plot surface reflection
zi = -12.5
ti =700
zz = ox.z.unique()

zii = zz[np.abs(zz - zi).argmin()]
ox2 = ox[ox.z==zii].copy()

## coseismic instances
co = p.get_coseismic_instances()[-2]

t1 = co.iloc[0].t
t2 = co.iloc[-1].t

fig, ax = plt.subplots(1,1,clear = True)

ox_temp = ox1[(ox1.t>t1) & (ox1.t<t2)]
ax.set_xlabel('time [s]')
ax.set_ylabel('v [m/s]')
ax.plot(ox_temp.t-t1, ox_temp.v)
ax.set_xlim([10,60])
fig.suptitle('z=-7.5km')
fig.savefig('BP1_surface_time_vs_srate.jpg', dpi =100, bbox_inches='tight')





