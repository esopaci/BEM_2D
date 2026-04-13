#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 21 10:59:10 2024

@author: sopaci
"""


import os
import pickle
import numpy as np
from kernel import space_kernel, fft_kernel, spring_slider_kernel
from friction import original_rsf, regularized_rsf, aging_law, slip_law
from balance_equations import balance_eq_space,  balance_eq_fft_scipy, balance_eq_springslider,balance_eq_fft_periodic, balance_eq_fft_freesurface 
from solver import rkck, rk_routine, scipy_ode

# from scipy.integrate import ode

def initialize_mesh(p):
    '''
    This function generate mesh and initialize the variables:
        v_ini 
        theta_ini
        tau_ini
        sigma_ini.

    Parameters
    ----------
    p : TYPE
        DESCRIPTION.

    Returns
    -------
    mesh.

    '''
    N = p['N'] # NUMBER OF ELEMENTS    
    #nitialize the computational space
    dw = p['L'] / p['N']        # grid size along the domain
    ww = np.arange(dw/2,p['L']+dw/2, dw) # girds
    ww += p['z_corner']
    
    # Fault in 3D grid
    xx = np.zeros(N) 
    yy = ww * np.cos(p['dip'] * np.pi /180)
    zz = ww * np.sin(p['dip'] * np.pi /180)
    
    
    # # rate and state parameters
    aa = np.ones(N) * p['a']  
    bb = np.ones(N) * p['b']   
    dcc = np.ones(N) * p['dc']   
    
    v_0 = np.ones( N) *  p['v_0']
    v_pl = np.ones( N)  *  p['v_pl']
    mu_0 = np.ones( N) *  p['mu_0']
    dip_w = np.ones( N)  *  p['dip']
    
    # Initial values
    sigma_ini = np.ones( N)  *  p['sigma_n']
    v_ini = 0.01 * v_0
    theta_ini = dcc/v_0
    tau_ini = original_rsf(v_ini, theta_ini, aa, bb, dcc, v_0, mu_0) * sigma_ini
    slip_ini = np.zeros_like(tau_ini)
    # define meshing dictionary
    mesh = {
            'xx' : xx,
            'yy' : yy,
            'zz' : zz,
            'ww' : ww,
            'dip_w' : dip_w,
            'aa' : aa,
            'bb' : bb,
            'dc' : dcc,
            'v_0' : v_0,
            'v_pl' : v_pl,
            'mu_0' : mu_0,
            'sigma_ini' : sigma_ini,
            'tau_ini' : tau_ini, 
            'v_ini' : v_ini,
            'theta_ini' : theta_ini,
            'slip_ini' : slip_ini,
            }
    
    return mesh
     
    


def initialize_kernels( state_type, friction_type, mode, solver_type, free_surface ):
    '''
    This function intializes the kernels and methods to be used.

    Parameters
    ----------
    kernel_type : TYPE
        DESCRIPTION.
    state_type : TYPE
        DESCRIPTION.
    friction_type : TYPE
        DESCRIPTION.
    model : TYPE
        DESCRIPTION.
    solver_type : TYPE
        DESCRIPTION.
    N_fault : TYPE
        Number of the faults. Will be modified later. ES!
    Returns
    -------
    kernel : TYPE
        DESCRIPTION.
    friction_law : TYPE
        DESCRIPTION.
    state_law : TYPE
        DESCRIPTION.
    fun : TYPE
        DESCRIPTION.
    solver : TYPE
        DESCRIPTION.

    '''
        


    ## state law for rate and state friction
    if state_type == 0:
        
        state_law = aging_law
        
    elif state_type == 1:
        
        state_law = slip_law
        
    else:
        
        state_law = None
        
    ## Friction law
    if friction_type == 0:
        
        friction_law = original_rsf
        
    elif friction_type == 1:
        
        friction_law = regularized_rsf  
    
    # Balance equation and kernel
    if mode == 0:
        fun = balance_eq_springslider
        init_kernel = spring_slider_kernel

    elif mode ==1:
        if solver_type!=2:

            fun = balance_eq_space
        # else:
        #     fun =balance_eq_space_nonumba
            
        init_kernel = space_kernel
        
    elif mode == 2:
        if solver_type!=2:
            if free_surface == 1:
                fun = balance_eq_fft_freesurface
            else:
                fun = balance_eq_fft_periodic
        # else:
        #     fun =balance_eq_fft_nonumba
        init_kernel = fft_kernel
    
    # Time solver
    if solver_type == 0 :
        solver = rk_routine
    elif solver_type == 1:
        solver = rkck
    elif solver_type == 2:
        # Balance equation and kernel
        solver = scipy_ode

        

        # if mode == 0:
        #     fun = balance_eq_space_scipy
        # elif mode == 1:
        #     fun = balance_eq_fft_scipy
        #     init_kernel = fft_kernel  
       
    
    return ( friction_law, state_law, fun, init_kernel, solver)


def intialize_input_files(p, target_folder, FAULT_ID):
    
    # prepare the folder to save inputs and outputs
    # target_folder = os.path.join(p['root'], p['fname'])

    if not os.path.exists(target_folder):
        os.makedirs(target_folder)
    
        
    (state_law, friction_law, fun, init_kernel, solver) = initialize_kernels( p['state_law'], 
                        p['friction_law'], 
                        p['computation_kernel'], 
                        p['solver'],
                        p['free_surface']
                    )
    
    ww = p['mesh']['ww'] #computation domain.
    N = p['N']
    N_lam = p['N_lam'] # Number of elements to be outputted.
    N_ker = p['N_ker'] # Number of elements in the computation (kernel) domain.
    L = p['L'] # Length of the fault.
    W = p['W'] # Width of the fault.
    G = p['G'] # Shear Modulus.
    nu = p['nu'] # poisson ratio.
    
    # COMPUTE KERNEL
    KK = init_kernel( ww, N_lam, N_ker, L, G, nu, W )
        
    p['KK'] = KK # KERNEL
    
    #Initial values
    v_ini = p['mesh']['v_ini']
    theta_ini = p['mesh']['theta_ini']
    sigma_ini = p['mesh']['sigma_ini']
    slip_ini = p['mesh']['slip_ini']
    y_ini = np.concatenate((v_ini, theta_ini, sigma_ini, slip_ini))
    
    p['y_ini'] = y_ini
    
    with open(os.path.join(target_folder, f'p{FAULT_ID:03}.pickle'), 'wb') as handle:
        pickle.dump(p, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
    return (KK, y_ini)
        
            


    