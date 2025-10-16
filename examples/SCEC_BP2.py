#!/usr/bin/env python3
# -*- coding= utf-8 -*-
"""
Created on Thu Feb 22 18=11=41 2024
This example uses okadas solution for equal grids over VS and VW patches 
@author= sopaci
"""

import numpy as np 
import matplotlib.pyplot as plt 
import sys
import os 

sys.path.append( os.path.join(os.path.expanduser("~"),'simcode/src'))

from process import process
import pandas as pd 
import pickle
from parameter_dictionary import p




""" Simulation parameters """

t_yr = 3600*24*365.0        # Seconds in 1 year [s]
t_max = 1200*t_yr           # Simulation time [s]

rho = 2670.0                # Rock density [kg/m3]
Vs = 3464.0                 # Shear wave speed [m/s]
mu = rho * Vs**2            # Shear modulus [Pa]
eta = mu / (2 * Vs)
sigma = 50e6                # Effective normal stress [Pa]
V_pl = 1e-9                 # Plate rate [m/s]
V_ini = V_pl                # Initial velocity [m/s]
H = 15e3                    # Depth extent of VW region [m]
h = 3e3                     # Width of VW-VS transition region [m]
L = 40e3                    # With of frictional fault [m]

# Target element sizes: {25, 50, 100, 200, 400, 800} [m]
dz = 25.0                   # Target cell size
N_approx = L / dz           # Approximate number of fault elements

# Compute the nearest power of 2
logN = np.log2(N_approx)
N = int(np.power(2, np.round(logN)))
dz2 = L / N
print("Target dz: %.1f \t Current dz: %.1f" % (dz, dz2))

dz_slip = 100.0
dN_slip = 1
if dz2 < dz_slip:
    logdN_slip = np.log2(dz_slip / dz2)
    dN_slip = int(np.power(2, np.round(logdN_slip)))

""" Rate-and-state friction parameters """

a0 = 0.010          # Min a
a_max = 0.025       # Max a
b = 0.015           # Constant b
Dc = 0.004          # Critical slip distance
f_ref = 0.6         # Reference friction value
V_ref = 1e-6        # Reference velocity

z = np.arange(dz2 / 2, L, dz2)      # Depth vector
a = a0 + (a_max - a0) * (z - H)/h   # Depth-dependent values of a
a[z < H] = a0                       # Shallow cutoff
a[z >= H + h] = a_max               # Deep cutoff
a= a[::-1]


# Construct initial stress state
tau_exp = np.exp( (f_ref + b*np.log(V_ref / V_ini)) / a_max )
tau_sinh = np.arcsinh(V_ini * tau_exp / (2 * V_ref))
tau_ini = sigma * a_max * tau_sinh + eta * V_ini

# Construct initial "state" state
theta_sinh = np.sinh((tau_ini - eta * V_ini) / (a * sigma))
theta_log = np.log(theta_sinh * 2 * V_ref / V_ini)
theta_ini = (Dc / V_ref) * np.exp((a / b) * theta_log - f_ref / b)

# Requested timeseries output depths
# z_IOT = np.array([0.0, 2.4, 4.8, 7.2, 9.6, 12.0, 14.4, 16.8, 19.2, 24.0, 28.8, 36.0]) * 1e3



p['state'] = 0  # state 0=aging, 1=slip law
p['computation_kernel'] = 2   
p['solver'] = 0     # time solver
p['prefix']= '' 
p['N_fault']= 1 
p['friction_law'] = 1     # friction law 0:original, 1:regularized

# # % SEAS PARAMETERS
p['G'] = mu
p['sigma_n'] = sigma;


p['a'] = a_max
p['b'] = b = b;
p['dc'] = dc = Dc;
p['v_pl'] = V_pl; #% this will need a modification in Qdyn's source. (V_PL != V_SS)
p['v_0'] = V_ini;
p['v_ss'] = V_ref;
p['mu_0'] = f_ref;

p['L'] = L;
p['W'] = -1
p['t_f'] = t_max;
p['nu'] = 0.
p['dip'] = 90    # dipping angle



# # % REQUIRED COMPUTATIONS
p['N'] = N 
p['N_lam'] = N
p['N_ker'] = N * 4

zz = np.linspace(dz2, L+dz2/2, N)

zz_v = -L-3*dz/2 + zz


# # % CHARACTERISTIC LENGTHS
Lc = 2/np.pi*dc*p['G'] *b/(sigma*(b- a0)**2); #% nucleation length [Rubin&Ampuero05]
Lb = p['G']*dc/(sigma*b); # % cohesive length

print('Characteristic lengths');
print('Number of elements with-in the quasi-static cohesive zone: {:1.1f}'.format(Lb/dz));
print('Lc/Lb: %1.1f', Lc/Lb);
print('L/Lc: %1.1f', L/Lc);


# ## Solver parameters
p['t_ini'] = 0    #initial time
p['dt_ini'] = 1e-6  # initial time step
p['dt_max'] = 1e8  # maximum time step allowed by the solver
p['dt_min'] = 1e-6   # minimum time step allowed by the solver 
p['err_tol'] = 1e-8   # error tolerance for the solver
p['max_iter'] = 50    # maximum iteration 
p['ot_interval'] = 100     # 
p['ox_interval'] = 8       # space snapshot interval 
p['max_interval'] = 100    # time snapshots
p['i_step'] = -1          # starting step 
p['print'] = True         # printing on the screen option
p['z_corner'] = -L + 3*dz/2


    
p['mesh'] = {}
# # define meshing dictionary
p['mesh']['xx'] = np.zeros(N)
p['mesh']['yy'] = np.zeros(N)
p['mesh']['zz'] = zz_v
p['mesh']['ww'] = zz_v
p['mesh']['dip_w'] = np.ones(N) * 90
p['mesh']['aa'] = a
p['mesh']['bb'] = np.ones(N) * b
p['mesh']['dc'] = np.ones(N) * dc
p['mesh']['v_0'] = np.ones(N) * V_ref
p['mesh']['v_pl'] = np.ones(N) * V_pl
p['mesh']['mu_0'] = np.ones(N) * f_ref
p['mesh']['sigma_ini'] = np.ones(N) * sigma
p['mesh']['v_ini'] = np.ones(N) * V_ini
p['mesh']['theta_ini'] = theta_ini
p['mesh']['slip_ini'] = np.zeros(N) 


root_target = '/Users/eyup/workspace/runs/SCEC'
    
fname = 'BP2'
target_folder = os.path.join(root_target, fname)
p['target_path']= target_folder # 

if not os.path.exists(target_folder):
    os.makedirs(target_folder)

with open(os.path.join(target_folder, 'params.pickle'), 'wb') as handle:
    pickle.dump(p, handle, protocol=pickle.HIGHEST_PROTOCOL)
                        
process(p)


    