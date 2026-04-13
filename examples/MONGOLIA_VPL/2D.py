#!/usr/bin/env python3
# -*- coding= utf-8 -*-
"""
Created on November 2025
This example performs simulations of 2.5D Fault

@author= sopaci
"""

import numpy as np 
import matplotlib.pyplot as plt 
import sys
import os 
from scipy.interpolate import interp1d

## Change to your BEM_2 path
sys.path.append( os.path.join(os.path.expanduser("~"),'BEM_2D/src'))

from process import process
import pandas as pd 
import pickle

from parameter_dictionary import p

year = 365*24*3600; # seconds to year convert

## --------- SIMULATION PARAMETERS --------------##
'''
  lambda_pf=P/sigma_v  : 0.94
  rho (density)        : 2900 kg/m3
  Geothermal gradient  : 35 °C/km
  BDT depth            : 14.6 km
  Seismogenic Width    : 9.6 km
  Seismogenic Length   : 106.1 km
  Mean sigma_n_eff     : 95.6 MPa
  Mean Slip            : 0.3 m
  Moment Magnitude     : 6.6


  lambda_pf=P/sigma_v  : 0.94
  rho (density)        : 2900 kg/m3
  Geothermal gradient  : 15 °C/km
  BDT depth            : 30.4 km
  Seismogenic Width    : 25.4 km
  Seismogenic Length   : 279.4 km
  Mean sigma_n_eff     : 94.8 MPa
  Mean Slip            : 1.7 m
  Moment Magnitude     : 7.6
  
  
  
  b       : 0.01500 
  a-b     : 0.01000 
  a_vs    : 0.02500 
  a_vw    : 0.00500 
'''

# # Slip_rate as input
# VP = float(sys.argv[1])

# # BDT as input 
# BDT = float(sys.argv[2])

VP = 30  # mm/yer
BDT = 14.96 # km



lambda_pf = 0.94
p['v_pl'] = VP * 1e-3 / year
p['L'] = W = (BDT * 2)*1E3   # Fault width twice the BDT
W_tran = 5e3   # TRnsition zone from VS-VW
p['rho'] = rho = 2900

W_bdt = BDT * 1e3
W_seis = W_bdt - W_tran

p['c_s'] = 3464;
p['G'] = MU = 30.0e9


## --------- Frictional Parameters --------- ##
p['b']  = b       = 0.01500 
a_min_b = 0.01000 
a_vs    = 0.02500 
p['a'] = a_vw    = 0.00500 
kx = 8 # Seismogenic zone will be 8 times the nucleation length (RA,2005)
p['sigma_n'] = sigma_clip = 94.8e6 ## max effective normal stress 

# Compute crtictical slip distance to ensure 8 earthquakes 
# can hypothetically nucleate along the seismogenoc depth.
p['dc'] = dc = W_seis/2 * sigma_clip * b * np.pi / MU / kx * (b/a_min_b)**-2

# Nucleation length for slip rate loacalization Dieterich,1992
Lb = MU * dc / sigma_clip / b

# Compute minimum elment size (At least 7 time smaller than the Lb)
dx =  Lb/7
p['N'] = N = int(np.power(2, np.ceil(np.log2(p['L']/dx))))
p['N_lam'] = 1*N
p['N_ker'] = 4*N


p['state'] = 0                  # state 0=aging, 1=slip law
p['computation_kernel'] = 2     #2 for Spectral BEM    
p['solver'] = 0                 # 0:RK Fehlberg, 1: RK Cash-Karp, 2: Adam's method 
p['prefix']= '' 
p['N_fault']= 1 
p['friction_law'] = 1     # friction law 0:original, 1:regularized
p['free_surface'] = 1     # free surface is not applied (periodic fault)

# % SEAS PARAMETERS


p['v_0'] = v_0 = 1e-9;
p['v_ss'] = v_ss = 1e-6;
p['mu_0'] = mu_0 = 0.6;
p['W'] = -1             # For 2D fault : -1, for 2.5D : Fault width


## size of the cell


p['t_f'] = 1E4*year * 3 / VP;
p['nu'] = 0 ## Poisson ratio
p['dip'] = 90    # dipping angle


impedance = 0.5*p['G']/p['c_s']

# % REQUIRED COMPUTATIONS
p['N'] = N = int(np.power(2, np.ceil(np.log2(p['L']/dx))))
p['N_lam'] = 1*N
p['N_ker'] = 4*N


zz = np.linspace(dx, W+dx/2, N)

avary_depth=np.array([[0,a_vs],
                      [-W_tran,a_vw],
                      [-W_seis,a_vw], 
                      [-W_bdt,b],
                      [-W, b + a_min_b* (W-W_bdt)/ 5e3]])

fa_depth = interp1d(avary_depth[:,0], 
                    avary_depth[:,1], kind='linear', 
                    fill_value='extrapolate')


# Compute effective normal stress
sigma_v = 9.81 * rho * zz    # vertical stress
P_f = sigma_v * lambda_pf
# --- pore pressure ---
phi = np.arctan(mu_0)
f_s = 1 / np.tan(2.0*phi)

# --- Geodynamics 3rd edition ----------------------------------------#
# --- Turcotte&Schubert (2002) ---------------------------------------#
# --- Stress calculations --------------------------------------------#

# Differential stress Eq (8.35) 
delta_sigma_xx = 2 * (sigma_v-P_f) /\
        (( 1 + f_s**2 )**(0.5) + f_s ) 
theta = np.pi/2 - p['dip']*np.pi/180
sigma_n = sigma_v + 0.5*delta_sigma_xx*(1+np.cos(2*theta))
sigma_n_eff = sigma_n - P_f
sigma_n_eff[sigma_n_eff<1e5] = 1e5


# plt.plot(sigma_n_eff*1e-6, -zz)

# plt.plot(fa_depth(-zz) - b, -zz)
# plt.axvline(0, ls = ':')



## Mesh parameters

aa = fa_depth(-zz)
bb = np.ones_like(aa) * b
dcc = np.ones_like(aa) * dc

tau0 = sigma_n * aa * np.asinh(v_0/2/v_ss* np.exp( (mu_0 + b*np.log(v_ss/v_0))/aa )) + impedance*v_0

theta_0 = dc/v_ss*np.exp(aa/b*np.log( 2*v_ss/v_0*np.sinh( (tau0 - impedance*v_0)/(aa*sigma_n) ) ) - mu_0/b);
plt.plot(theta_0, -zz*1e-3)

# # % CHARACTERISTIC LENGTHS
Lc = 2/np.pi*dc*p['G'] *b/(sigma_clip*(b-  a_vw)**2); #% nucleation length [Rubin&Ampuero05]

# print('Characteristic lengths');
# print('Number of elements with-in the quasi-static cohesive zone: %1.1f', Lb/dz);
# print('Lc/Lb: %1.1f', Lc/Lb);



# ## Solver parameters
p['t_ini'] = 0    #initial time
p['dt_ini'] = 1e-4  # initial time step
p['dt_max'] = 1e8  # maximum time step allowed by the solver
p['dt_min'] = 1e-4   # minimum time step allowed by the solver 
p['err_tol'] = 1e-7   # error tolerance for the solver
p['max_iter'] = 50    # maximum iteration 
p['ot_interval'] = 100     # 
p['ox_interval'] = 10       # space snapshot interval 
p['max_interval'] = 100    # time snapshots
p['i_step'] = -1          # starting step 
p['print'] = True         # printing on the screen option


    
    
p['mesh'] = {}
# # define meshing dictionary
p['mesh']['xx'] = np.zeros(N)
p['mesh']['yy'] = np.zeros(N)
p['mesh']['zz'] = zz
p['mesh']['ww'] = zz
p['mesh']['dip_w'] = np.ones(N) * 90
p['mesh']['aa'] = aa
p['mesh']['bb'] = bb
p['mesh']['dc'] = dcc
p['mesh']['v_0'] = np.ones(N) * v_ss
p['mesh']['v_pl'] = np.ones(N) * p['v_pl']
p['mesh']['mu_0'] = np.ones(N) * mu_0
p['mesh']['sigma_ini'] = sigma_n_eff
p['mesh']['v_ini'] = np.ones(N) * v_0
p['mesh']['theta_ini'] = theta_0
p['mesh']['slip_ini'] = np.zeros(N) 

print('Lc:', Lc)
print('Lb:', Lb)
print('L/Lc:', W_seis/Lc)
print('dz:',W/N)
print('Lb/dz:', Lb/(W/N))








# Define your target directory
root_target = '/Users/eyup/workspace/Mongolia_latest'
    
fname = f'R2D_VP{VP}_BDT{BDT}'
target_folder = os.path.join(root_target, fname)
p['target_path']= target_folder # 

if not os.path.exists(target_folder):
    os.makedirs(target_folder)
    
fig,ax = plt.subplots(1,2, sharey=True)

ax[0].set_ylabel('Depth [km]')
ax[0].set_xlabel('$\\sigma_n - P$ [MPa]')
ax[1].set_xlabel('a-b [-]' )

ax[0].plot(sigma_n_eff*1e-6, -zz*1e-3)
ax[1].plot( aa-bb, -zz*1e-3)
ax[1].axvline( 0, ls = ':', color = 'k')
ax[1].axhline( -BDT, ls = ':', color = 'k')
ax[1].text(0.01, -BDT, 'BDT')
fig.savefig(os.path.join(target_folder,'setup.jpg'),
            bbox_inches='tight', )

with open(os.path.join(target_folder, 'params.pickle'), 'wb') as handle:
    pickle.dump(p, handle, protocol=pickle.HIGHEST_PROTOCOL)
                        
        
process(p)


    