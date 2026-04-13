#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  9 13:53:11 2024

@author: sopaci
"""

p = {
     ## GENERAL PARAMETERS
     'free_surface': 1,     # 1. Free surface is applied, 0: Peridodic fault no frees surface is applied. Applied only to spectral BEM.  
     'state_law' : 0,       # state 0:aging, 1:slip law
     'friction_law': 0,     # friction law 0:original, 1:regularized
     'computation_kernel' : 1,        # 0: spring-slider, 1: okada kernel, 2:fft kernel
     'solver' : 0,          # time solver 0:Runge-kutta Fehlberg
     'prefix': '',          #

     ### FAULT PARAMETERS
     'v_pl' : 1e-9,     # long term deformation rate
     'G' : 30e9,        # shear modulus
     'nu' : 0.25,       # poisson ratio
     'c_s' : 3000,      # shear wave velocity 
     'dip' : 89,        # dipping angle
      'W' : 50e3,          # width of the fault for 2.5D approximation
      'L' : 50E3,       # length of the fault for 2.5D approximation
      'N' : 8,          # Number of elements         
      'N_lam' : 8,          # Number of elements
      'N_ker' : 8 * 4,          # Number of kernel elements        
      'z_corner' : -5e4,          # maximum depth of the fault         
      'sigma_n' : 100e6,          # normal stress         


     
    ## Solver parameters
    't_ini' : 0,    #initial time
    'dt_ini' : 1e-2,  # initial time step
    't_f' : 100,  # final time
    'dt_max' : 1e7,  # maximum time step allowed by the solver
    'dt_min' : 1e-6,   # minimum time step allowed by the solver 
    'err_tol' : 1e-10,   # error tolerance for the solver
    'max_iter' : 50,    # maximum iteration 
    'ot_interval' : 1,     # snapshots for time step
    'ox_interval' : 1,      # space snapshot interval 
    'max_interval' : 1,    # snapshots form maximum slip rate
    'i_step' : -1,          # starting step 
    'print' : True,         # printing on the screen option
    
    
    
    # Rate and state parameters
    'a' : 0.015,     # direct effect
    'b' : 0.01,    # state effect 
    'dc': 1e-2,     # critical slip distance
    'v_0' : 1e-6,   # reference velocity 
    'mu_0' : 0.6,   # reference friction
     }


## Triggering parameters 
mass_removal = { 
    'mass_removal' : 0, 
    'beta0'     : 7.5e-3, 
    'beta1'     : 0.0,
    't_onset'   : 0,     
    't_stop'   : 0,        
    'x0' : 0, 
    'y0' : 0
    }

## Triggering parameters 
external_load = { 
    'external_load'     : 0,    # 0  no external load, 1: external load applied 
    'stochastic_loading': 0,    # Analoy for stochasti loading condtion. 
    't_start'           : 0,    # Start time of external load    
    't_stop'            : 0,    # Stop of the external load
    'tau_rate'          : 0,    # Magnitude of the external load Pa/s
    }


# p['mesh'] = mesh
p['mass_removal'] = mass_removal
p['external_load'] = external_load

