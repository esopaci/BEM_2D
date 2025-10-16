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

year = 365*24*3600;


p['state'] = 0  # state 0=aging, 1=slip law
p['computation_kernel'] = 2 #2 for Spectral BEM    
p['solver'] = 2      # time solver
p['prefix']= '' 
p['N_fault']= 1 
p['friction_law'] = 1     # friction law 0:original, 1:regularized

# % SEAS PARAMETERS
p['rho'] = 2670;
p['c_s'] = 3464;
p['G'] = p['rho']*p['c_s']**2;
p['sigma_n'] = sigma_n = 50e6;
a0 = 0.01;
amax = 0.025;

p['a'] = amax
p['b'] = b = 0.015;
p['dc'] = dc = 8e-3;
p['v_pl'] = v_pl = 1e-9; #% this will need a modification in Qdyn's source. (V_PL != V_SS)
p['v_0'] = v_0 = 1e-9;
p['v_ss'] = v_ss = 1e-6;
p['mu_0'] = mu_0 = 0.6;
p['H'] = H = 15000;
p['h'] = h =3000;
p['L'] = L =40000;
p['W'] = -1
dz = 25; 
p['t_f'] = 700*year;
p['nu'] = 0.0

p['dip'] = 90    # dipping angle



impedance =  0.5*p['G']/p['c_s']

# % REQUIRED COMPUTATIONS
p['N'] = N = int(np.power(2, np.ceil(np.log2(p['L']/dz))))
p['N_lam'] = N
p['N_ker'] = N * 4

zz = np.linspace(dz, L+dz/2, N)


n1 = int(H/L*N);
n2 = int(np.round( (H + h)/L*N ));

aa = np.ones(N) * amax
aa[0:n1] = a0
aa[n1:n2] = a0 + (amax - a0)*(zz[n1:n2] - H - dz/2)/ (h+dz/2) ;
aa[n2:N] = amax;


aa_v = aa[::-1]
# aa_v = aa
zz_v = -L-3*dz/2 + zz


tau0 = sigma_n*amax*np.asinh(v_0/2/v_ss*np.exp( (mu_0 + b*np.log(v_ss/v_0))/amax )) + impedance*v_0;

theta_0 = dc/v_ss*np.exp(aa_v/b*np.log( 2*v_ss/v_0*np.sinh( (tau0 - impedance*v_0)/(aa_v*sigma_n) ) ) - mu_0/b);


# % CHARACTERISTIC LENGTHS
Lc = 2/np.pi*dc*p['G'] *b/(sigma_n*(b- a0)**2); #% nucleation length [Rubin&Ampuero05]
Lb = p['G']*dc/(sigma_n*b); # % cohesive length

print('Characteristic lengths');
print('Number of elements with-in the quasi-static cohesive zone: %1.1f', Lb/dz);
print('Lc/Lb: %1.1f', Lc/Lb);
print('L/Lc: %1.1f', L/Lc);


## Solver parameters
p['t_ini'] = 0    #initial time
p['dt_ini'] = 1e-6  # initial time step
p['dt_max'] = 1e8  # maximum time step allowed by the solver
p['dt_min'] = 1e-6   # minimum time step allowed by the solver 
p['err_tol'] = 1e-8   # error tolerance for the solver
p['max_iter'] = 100    # maximum iteration 
p['ot_interval'] = 100     # 
p['ox_interval'] = 8       # space snapshot interval 
p['max_interval'] = 100    # time snapshots
p['i_step'] = -1          # starting step 
p['print'] = True         # printing on the screen option
p['z_corner'] = -L + 3*dz/2


    
    
p['mesh'] = {}
# define meshing dictionary
p['mesh']['xx'] = np.zeros(N)
p['mesh']['yy'] = np.zeros(N)
p['mesh']['zz'] = zz_v
p['mesh']['ww'] = zz_v
p['mesh']['dip_w'] = np.ones(N) * 90
p['mesh']['aa'] = aa_v
p['mesh']['bb'] = np.ones(N) * b
p['mesh']['dc'] = np.ones(N) * dc
p['mesh']['v_0'] = np.ones(N) * v_ss
p['mesh']['v_pl'] = np.ones(N) * v_pl
p['mesh']['mu_0'] = np.ones(N) * mu_0
p['mesh']['sigma_ini'] = np.ones(N) * sigma_n
p['mesh']['v_ini'] = np.ones(N) * v_0
p['mesh']['theta_ini'] = theta_0
p['mesh']['slip_ini'] = np.zeros(N) 


root_target = '/Users/eyup/workspace/runs/SCEC/SBEM'
    
fname = 'BP1'
target_folder = os.path.join(root_target, fname)
p['target_path']= target_folder # 

if not os.path.exists(target_folder):
    os.makedirs(target_folder)

with open(os.path.join(target_folder, 'params.pickle'), 'wb') as handle:
    pickle.dump(p, handle, protocol=pickle.HIGHEST_PROTOCOL)
                        
        
process(p)


    