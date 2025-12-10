#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 22 18:11:41 2024
This is an example run for single degree of freedom spring slider 
@author: sopaci
"""
import sys 
import os
import numpy as np 
import matplotlib.pyplot as plt

#BEM_2D path: change depending on your path
BEM_2D_path = os.path.join( os.path.expanduser("~"), 'BEM_2D')
sys.path.append( os.path.join(BEM_2D_path,'src'))

from process import process

import pickle
from parameter_dictionary import p


def v_preseismic(t, a, b, dc, v_dyn, theta_dyn, v_pl, dtau, sigma, K):
    
    B = sigma * b
    
    tp = B / (K * v_pl + dtau)
    
    V = v_dyn * ( np.exp(t/tp) / (t/theta_dyn))**(b/a)
        
    return V


a = 0.01
b = 0.015

N = 1
p['N'] = N 
p['N_lam'] = N
p['computation_kernel'] = 0 
p['t_f'] = 5e8
p['v_pl'] = 0 
p['sigma'] = 1e8 
p['G'] = 30e9 
p['a'] = p['b'] + 0.001 
p['dc'] = 1e-4
p['L'] = 20000 

K = p['G'] / p['L'] 
# rate and state parameters
aa = np.array( [p['a']] )   
bb = np.array( [p['b']] )   
dcc =np.array( [p['dc']] )   


sigma_ini = np.ones( N ) * p['sigma']
slip_ini = np.zeros_like(sigma_ini)
dip_w = np.ones( N) * p['dip']  

p['mesh'] = {}
# define meshing dictionary
p['mesh']['xx'] = np.array([p['L']])
p['mesh']['yy'] = np.array([p['L']])
p['mesh']['zz'] = np.array([p['L']])
p['mesh']['ww'] = np.array([p['L']])
p['mesh']['dip_w'] = dip_w
p['mesh']['aa'] = aa
p['mesh']['bb'] = bb
p['mesh']['dc'] = dcc

p['dt_max'] = 1e5


v_0 = 1e-9
kk = 1e-4
theta_0 =  p['dc'] / v_0 * kk

tau_rate = p['external_load']['tau_rate']
tp = bb* sigma_ini / (K * p['v_pl'] + tau_rate) 


v_0 = np.ones( N) * 1e-9
v_pl = np.ones( N) * 0
mu_0 = np.ones( N) * 0.6  


theta_ini = np.array([theta_0])
v_ini = v_0

p['mesh']['v_0'] = v_0
p['mesh']['v_pl'] = v_pl
p['mesh']['mu_0'] = mu_0
p['mesh']['sigma_ini'] = sigma_ini
p['mesh']['v_ini'] =  v_ini
p['mesh']['theta_ini'] = theta_ini
p['mesh']['slip_ini'] = slip_ini


# target_path
target_folder= os.path.join( os.getcwd(), 'sdf_spring_slider')
if not os.path.exists(target_folder):
    os.makedirs(target_folder)
p['target_path']= target_folder # 
with open(os.path.join(target_folder, 'params.pickle'), 'wb') as handle:
    pickle.dump(p, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
process(p)


