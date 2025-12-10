#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  3 09:01:00 2024

@author: sopaci
"""

import pickle 

import os 
import numpy as np 

import sys 
from scipy import interpolate


home_path = os.path.expanduser('~')

src_path = os.path.join(os.path.join(home_path, 'BEM_2D'), 'src')

sys.path.append(src_path)

post_processing_path = os.path.join(src_path, '../post_processing')
sys.path.append(post_processing_path)


import plot as pp

# The simulation folder
# path = sys.argv[1]
# tbx = float(sys.argv[2])

path = f'{os.path.expanduser("~")}/workspace/runs/conts/leteil/N2678_L12_LinfL8_VPL-13'
tbx = 60
# if len(sys.argv) <=3:
      
#     b0 = 7.5e-3 * kk
#     b1 = 0 
# else:

b0 = 9.59028549e-04
b1 = 2.08575e-22
    
    
# Read the simulation parameters
# file = open(os.path.join(mpath, 'params.pickle'),'rb')
# p0 = pickle.load(file)

# Initiate the post processing tool 
p0 = pp.Ptool(path)
# and read the cosesimic moments into seperate lists
d0 = p0.get_coseismic_instances()
# We will initiate the simulation before the last failure.
# We apply triggering and compare the failure time differences with and without.

ftimes = []
    
for i in range(len(d0)):
    
    max_indices = d0[i].ind_max.unique()
    
    ftimes.append(d0[i].t.iloc[0])
    
# get the recurrence times
dftimes = np.concatenate(([0], np.diff(ftimes)))
ind = dftimes>1e10

ftimes = np.array(ftimes)[ind]
dftimes = dftimes[ind]

# # last rupture information
# df0 = d0[-1]
# df_last = df0.iloc[0]

# get the parameters dictionary
pars = p0.pars

kk=1

y0 = -1000

tb = tbx * dftimes[-1]/100


# Update the triggering dictionary
pars['mass_removal']['mass_removal'] = 1
pars['mass_removal']['t_start'] = ftimes[-1] - tb
pars['mass_removal']['t_onset'] = ftimes[-1] - tb
pars['mass_removal']['t_stop'] = ftimes[-1]
pars['mass_removal']['y0'] = y0
pars['mass_removal']['beta0'] = b0
pars['mass_removal']['beta1'] = b1

zz = pars['mesh']['zz'][:]

pars['print'] = False

# find the closest time instance to strat triggering simulation
df00 = p0.read_vmaxfile()

# The new restart values 
index_restart = (pars['mass_removal']['t_onset'] - df00.t).abs().argmin()

istep = int(df00.iloc[index_restart-1].istep)

ox = p0.read_ox_1()

step0 = (ox.step - istep).abs().argmin()
ox = ox[ox.step == ox.step.iloc[step0]]


f = interpolate.interp1d( ox['z'].to_numpy(), ox['v'].to_numpy(), fill_value='extrapolate')
v_ini = f(zz)
pars['mesh']['v_ini'] = v_ini
f = interpolate.interp1d( ox['z'].to_numpy(), ox['theta'].to_numpy(), fill_value='extrapolate')
theta_ini = f(zz)
pars['mesh']['theta_ini'] = theta_ini
f = interpolate.interp1d( ox['z'].to_numpy(), ox['slip'].to_numpy(), fill_value='extrapolate')
slip_ini = f(zz)
pars['mesh']['slip_ini'] = slip_ini
pars['t_ini'] = ox.t.iloc[0]    #initial time

fname = f'sil_tbx{tbx}_xc{0}_yc{y0}_b0{b0:.3E}_b1{b1:.3E}'
target_folder = os.path.join(path, fname)

if not os.path.exists(target_folder):
    os.makedirs(target_folder)
    
pars['target_path']= target_folder # 
pars['t_f'] = ftimes[-1] 
pars['t_ini'] = ftimes[-1] - tb
pars['ot_interval'] = 100     # 
pars['ox_interval'] = 4      # space snapshot interval 
pars['max_interval'] = 100    # time snapshots

with open(os.path.join(target_folder, 'params.pickle'), 'wb') as handle:
    pickle.dump(pars, handle, protocol=pickle.HIGHEST_PROTOCOL)

from process import process                 
            
process(pars, v_c = 1e-4)