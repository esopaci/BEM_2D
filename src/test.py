#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  9 13:51:35 2024
Testing
@author: sopaci
"""

import settings
from parameter_dictionary import p
import numpy as np
import os 


def test_mesh(p):
    '''
    This function tests meshing

    Parameters
    ----------
    p : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    '''
    try:
        mesh = settings.initialize_mesh(p)
        for key in p.keys():
            print('{}:{}'.format(key, p[f'{key}']))
        print('The mesh has been created with the parameters above!')
        p['mesh'] = mesh
    except:
        print('check input parameters!')
        
    return p

def test_kernel(p, N, kernel):
    
    if kernel == 0:
        p['N'] = 1 
        p['N_lam'] = 1
        p['N_ker'] = 4
        
    elif kernel == 1:
        p['N'] = N 
        p['N_lam'] = N
        p['N_ker'] = N*4
        
    elif kernel == 2:
        p['N'] = N 
        p['N_lam'] = N
        p['N_ker'] = N*4
        
    
    N_lam = p['N_lam']
    N_ker = p['N_ker']
    p['computation_kernel'] = kernel
    
    p = test_mesh(p)
    target_folder = f'/home/sopaci/TESTS/TEST{kernel}'
    FAULT_ID = 1
    KK, y_ini = settings.intialize_input_files( p, target_folder, FAULT_ID)

    KK = p['KK']
    # Initial values
    # y_ini = np.concatenate(
    #     (p['mesh']['v_ini']*1e2,p['mesh']['theta_ini'],
    #       p['mesh']['sigma_ini'],
    #       p['mesh']['slip_ini'])
    #     )

    #
    friction_law, state_law, fun, init_kernel, solver = settings.initialize_kernels( p['state_law'], 
                                                                                    p['friction_law'], 
                                                                                    p['computation_kernel'], 
                                                                                    p['solver'] )
    
    p['mesh']['aa'][N*1//4:N*3//4] = p['mesh']['bb'][N*1//4:N*3//4] - 0.005
    
    
    # #TEST
    try:
        
        # test friction 
        mu = friction_law(y_ini[:N_lam], y_ini[N_lam:2*N_lam], 
                          p['mesh']['aa'], p['mesh']['bb'], p['mesh']['dc'], 
                          p['mesh']['v_0'], p['mesh']['mu_0'])
        
        theta_dt = state_law(y_ini[:N_lam], y_ini[N_lam:2*N_lam],p['mesh']['dc'])
        
        print(friction_law)
        print(state_law)
        print('friction test passed!')
    
        
        
        # test balance equation
        dy_dt = fun(y_ini, p['t_ini'], 
            p['mesh']['aa'], p['mesh']['bb'], p['mesh']['dc'],p['mesh']['v_pl'],
            N_lam, N_lam*4, 0, KK, 
            p['nu'], p['G'], p['c_s'], state_law, 1)
        
        print(fun)
        print('balance equation test passed!')
        
        #test_solver 
        y = y_ini 
        dt = p['dt_ini']
        t = p['t_ini']
        tf = t + 100
        print(f'{"dt":>12}{"time":>12}{"v":>12}{"theta":>12}{"sigma":>12}')
        while t<tf:
        
            t, dt, y, r = solver(t, y, 
                        dt, p['dt_max'], p['dt_min'], 
                        p['mesh']['aa'], p['mesh']['bb'], p['mesh']['dc'],p['mesh']['v_pl'], 
                        N_lam, N_ker, 0, KK, 
                        p['nu'], p['G'], p['c_s'], 
                        fun, state_law, 1, p['max_iter'], tf, 1e-6)    
            
            
            
            print('{:>12.4E}{:>12.4E}{:>12.4E}{:>12.4E}{:>12.4E}'.format(
                        dt, t, y[N_lam//2], y[N_lam +N_lam//2],y[2*N_lam + N_lam//2]) )      
            
        print('solver test passed!')
    except Exception as e:
        print(e)
        
    return y


## testing kernel
# print( '\n\n\n!!!TESTING SPRING SLIDER!!!!\n\n\n' )
# test_kernel(p, 1, 0)
print( '\n\n\n!!!TESTING SPACE KERNEL SOLUTION!!!!\n\n\n' )
y = test_kernel(p, 8, 1)
# print( '\n\n\n!!!TESTING FFT KERNEL SOLUTION!!!!\n\n\n' )
# test_kernel(p, 8, 2)

