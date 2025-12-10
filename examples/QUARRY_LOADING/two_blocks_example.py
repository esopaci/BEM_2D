#!/usr/bin/env python3
# -*- coding= utf-8 -*-
"""
Created on Thu Feb 22 18=11=41 2024
This example is for two fault segments.
A velocity strengthening (vs) fault at the top. 
Another VW patched is depper below.   
@author= sopaci
"""

import numpy as np 
import matplotlib.pyplot as plt 
import sys
import os 

sys.path.append( os.path.join(os.getcwd(),'../../src'))

from process import process
import pandas as pd 
import pickle
from parameter_dictionary import p

def main(mode, state, a1, a2, b, dc, D_vw, D_vs, sigma_n, V_PL, root_target):
    
    t_yr = 365* 3600* 24
    
    fname = f'N2_k{mode}_state{state}_avw{a1:.4f}_avs{a2:.4f}_b{b:.4f}_D_vw{D_vw:>05.0f}_D_vs{D_vs:>05.0f}'
    target_folder = os.path.join(root_target, fname)
    
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)
    
    p['state'] = state  # state 0=aging, 1=slip law
    p['friction']= 0    # friction law 0=original, 1=regularized
    p['computation_kernel'] = mode    # stres kernel 0= no-fft, 1=fft
    p['solver'] = 0      # time solver
    p['target_path']= target_folder # 
    p['prefix']= '' 
    p['N_fault']= 1 
    
    
    p['G'] = 30e9  # shear modulus
    p['nu'] = 0.25  # poisson ratio
    p['c_s'] = 3000  # shear wave velocity 
    p['dip'] = 60    # dipping angle
    p['W'] = -1      # width of the fault for 2.5D approximation
    
     
    ## Solver parameters
    p['t_ini'] = 0              # initial time
    p['dt_ini'] = 1e-5          # initial time step
    p['t_f'] = 1E6 * t_yr     # final time
    p['dt_max'] = 1e8           # maximum time step allowed by the solver
    p['dt_min'] = 1e-8          # minimum time step allowed by the solver 
    p['err_tol'] = 1e-7         # error tolerance for the solver
    p['max_iter'] = 50          # maximum iteration 
    p['ot_interval'] = 10       # 
    p['ox_interval'] = 1        # space snapshot interval 
    p['max_interval'] = 10      # time snapshots
    p['i_step'] = -1            # starting step 
    p['print'] = False          # printing on the screen option
    
    '''
    # COMPUTE GRIDS
    # A sigle VW segment with length D_vw is down below and the shallower 
    # VS segment is well gridded. The number of grids on the VS patch is N.
    Total number of grids will be N+1.
    The frist element is the center of the VW fault that is
    ww[0] = z_corner + D_vw / 2.
    The rest VS patch is evenly spaced.
    dw = D_vs/N
    
    ww[1=] = ww[0] + D_vw/2 + np.arange(dw, D_vs + dw/2, dw)
    
    '''
    
    # Number of elements
    # resolution = 12
    # # N = int(np.power(2, np.ceil(np.log2(resolution * (D_vs) / Lb))))
    L = D_vw + D_vs# size of the domain
    # N = int(resolution * D_vs / Lb) + 1
    N = 2
    p['N'] = N
    p['N_lam'] = N 
    p['N_ker'] = N * 2

    #SIZE OF THE DOMAIN
    p['L'] = L  # add it to the dictionary
    
    # maximum depth
    z_corner = -L
    p['z_corner'] = z_corner
    
    # Initialize the computational space    
    # First element is the center of the VW cell.
    ww = np.zeros(N)

    ww[0] = z_corner
    ww[1] = -D_vs

    # # The center of the grids
    dw = np.diff(np.concatenate([ww, [0]]) ) 
    ww2 = ww + dw/2

    xx = np.zeros(N)
    yy = ww2 * np.cos( p['dip'] * np.pi /180 )
    zz = ww2 * np.sin( p['dip'] * np.pi /180 )
    
    # rate and state parameters
    aa = np.zeros_like(xx)   

    bb = np.zeros_like(xx)   
    dcc = np.zeros_like(xx)   
    
    # modify rate and state parameters
    # aa[1:(N-1)//2-1] = b + 0.003
    aa[1:] = a2
    # aa[1:(N-1)//4] = b + 0.001
    aa[0] = a1
    bb[:] = b
    dcc[:] = dc

    v_0 = np.ones( N) * 1e-6 
    v_pl = np.ones( N) * V_PL 
    mu_0 = np.ones( N) * 0.6  
    sigma_ini = np.ones( N) * sigma_n
    sigma_ini[1] /=2
    slip_ini = np.zeros_like(sigma_ini)
    dip_w = np.ones( N) * p['dip']   
    
    p['mesh'] = {}
    # define meshing dictionary
    p['mesh']['xx'] = xx
    p['mesh']['yy'] = yy
    p['mesh']['zz'] = zz
    p['mesh']['ww'] = ww
    p['mesh']['dip_w'] = dip_w
    p['mesh']['aa'] = aa
    p['mesh']['bb'] = bb
    p['mesh']['dc'] = dcc
    p['mesh']['v_0'] = v_0
    p['mesh']['v_pl'] = v_pl
    p['mesh']['mu_0'] = mu_0
    p['mesh']['sigma_ini'] = sigma_ini
    p['mesh']['v_ini'] = v_0 
    p['mesh']['theta_ini'] = dcc/V_PL
    p['mesh']['slip_ini'] = slip_ini
    
    
    with open(os.path.join(target_folder, 'params.pickle'), 'wb') as handle:
        pickle.dump(p, handle, protocol=pickle.HIGHEST_PROTOCOL)
                            
            
    process(p)
     
if __name__ == '__main__':
    
    a1_min_b = -0.001 
    for a2_min_b in np.arange(0.0005):

        G = 30E9
        state = 0
        mode = 1
        V_PL = 1E-13
        
        sigma_n = 20e6
        
        root_target = f'{os.path.expanduser("~")}/workspace/runs/two_blocks/new/VPL{np.log10(V_PL):.0f}/Linf2'
        
        b = 0.012
        
        a1 = a1_min_b + b
        a2 = a2_min_b + b
        dc = 0.001 
        
        Lb = G * dc / sigma_n / b 
        Linf = 1 / np.pi *(b/a1_min_b)**2*Lb
        
        k = 2
        D_vw = k * Linf
        kx = 16
                
        D_vs = kx * Lb 
                        
        main(mode, state, a1, a2, b, dc, D_vw, D_vs, sigma_n, V_PL, root_target)

    