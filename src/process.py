#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 21 14:02:31 2024

@author: sopaci
"""

import numpy as np 
import kernel
import settings 
import pandas as pd 
import os
from boussinesqs import boussinesq_solution

from scipy.integrate import ode


def process(p, v_c  = 1e-3):

    
    mode = p['computation_kernel'] 
    state_type = p['state_law'] 
    friction_type = p['friction_law'] 
    solver_type = p['solver']
    target_path = p['target_path']
    prefix = p['prefix']
    free_surface = p['free_surface']
    
    # N_fault = p['N_fault']     # Number of faults.  

    # triggering = p['triggering']
    friction_law, state_law, fun, init_kernel, solver = settings.initialize_kernels(state_type, friction_type, mode, solver_type, free_surface )
    G = p['G'] 
    nu = p['nu']
    c_s = p['c_s']
    W = p['W']
    L = p['L']
    sigma_n = p['sigma_n']
    K =G/L
        
    ww = p['mesh']['ww']
    N_lam = p['N_lam']
    N_ker = p['N_ker']
    
    # size of the fault is twice to apply free surface
    if free_surface == 1:
        N_lam2 = N_lam * 2
    elif free_surface == 0:
        N_lam2 = N_lam
    N_kernel = N_lam2
    
    aa = p['mesh']['aa']
    bb = p['mesh']['bb']
    dc = p['mesh']['dc']
    v_0 = p['mesh']['v_0']
    v_pl = p['mesh']['v_pl']
    mu_0 = p['mesh']['mu_0']

    ## 
    XX = p['mesh']['xx']
    YY = p['mesh']['yy']
    ZZ = p['mesh']['zz']
    WW = p['mesh']['ww']
    DIP_W = p['mesh']['dip_w']
       
    ## Triggering parameters 
    mass_removal = p['mass_removal']['mass_removal']
    beta0 = p['mass_removal']['beta0']
    beta1 = p['mass_removal']['beta1']
    t_onset = p['mass_removal']['t_onset']
    t_stop = p['mass_removal']['t_stop']
    x0 = p['mass_removal']['x0']
    y0 = p['mass_removal']['y0']
    
    
    ## external Triggering parameters 
    external_load = p['external_load']['external_load'] 
    stochastic_loading = p['external_load']['stochastic_loading']   # This is an analogy for random loading 
    external_tstart = p['external_load']['t_start']  # onset time of external load
    external_tstop = p['external_load']['t_stop']  # onset time of external load    
    external_load_rate = p['external_load']['tau_rate']      # magnitude of the external load Pa/s
    
    # Create a new Generator instance (recommended method for new code)
    # Used for stochastic loading source
    rng = np.random.default_rng()

    ## Solver parameters
    t = p['t_ini']
    dt = p['dt_ini']
    tf = p['t_f']
    dt_max = p['dt_max'] 
    dt_min = p['dt_min'] 
    tol = p['err_tol'] 
    max_iter = p['max_iter']
    
    ot_interval = p['ot_interval']  
    ox_interval = p['ox_interval']  
    vmax_interval = p['max_interval']  
    
    istep = p['i_step']
        
    CFF = np.zeros( N_lam )
        
    tyr = 365*3600*24
    
    output_vmax = os.path.join(target_path, f'{prefix}_Vmax.txt')
    snaphot_path = os.path.join(target_path, 'snapshots')
    
    if not os.path.isdir(snaphot_path): 
        os.mkdir(snaphot_path) 
    
    # kernel and intial values
    (K, y_ini) = settings.intialize_input_files(p, target_path, 1)  
            
    filevmax = open(output_vmax, "w") 
    
    line_max = '#istep,t,ind_max,v,theta,tau,slip,sigma_n\n'
    filevmax.write(line_max)
    
    #INITIAL VALUES
    y = y_ini
    
    # if solver == 2:
        
    # def fun1(y, t, aa, bb, dc, v_pl, N_lam, N_kernel,CFF, K, nu, G, c_s, state_law, Nfault):
        
    #     return fun(y, t, 
    #                        aa, bb, dc, v_pl, N_lam, N_kernel,
    #                        CFF, K, nu, G, c_s, state_law, 1)
    
    r = ode( fun ).set_integrator('vode', method="adams", 
                                  rtol=tol, 
                                  min_step=dt_min, 
                                  max_step=dt_max,
                                  first_step=dt_min)   
    # r = ode( fun ).set_integrator('dop853',
    #                               atol=tol, 
    #                               max_step=dt_max,
    #                               first_step=dt_min)   
    
    r.set_initial_value( y, t )
    r.set_f_params( aa, bb, dc, v_pl, N_lam, N_kernel,
                   0, K, nu, G, c_s, state_law, 1)
    
    
    # V_dyn = 2 * sigma_n * (aa-bb).max() * c_s / G
    # print(V_dyn)
    while t < tf :
        
        # if y[0]<=0:
        #     break
        
        if mass_removal == 1:
            
            #if ((y[0] >=V_dyn) and (y[N_lam-1] >=V_dyn)):
            #break

            if t>= t_onset and t < t_stop:
                
                dtau, dsigma = boussinesq_solution(t, t_onset, beta0, beta1, XX, YY, ZZ, DIP_W, x0, y0, N_lam)
                
                # mu = friction_law(y[0*N_lam:1*N_lam], y[1*N_lam:2*N_lam], aa, bb, dc, v_0, mu_0)
                # CFF = dtau - mu * dsigma

                CFF = dtau - mu_0 * dsigma
                r.set_f_params( aa, bb, dc, v_pl, N_lam, N_kernel,
                               CFF, K, nu, G, c_s, state_law, 1)
                
            else:
                
                CFF = np.zeros( N_lam )
                r.set_f_params( aa, bb, dc, v_pl, N_lam, N_kernel,
                               CFF, K, nu, G, c_s, state_law, 1)
                
                
        if external_load == 1:
                
            if stochastic_loading == 1 :
                # Generate an array of 5 floats in the range [0.0, 1.0)
                random_loading_coeff = rng.uniform(size=N_lam)
                CFF = external_load_rate * random_loading_coeff
                
                # print(random_loading_coeff, CFF)
            else:
                CFF = np.array([external_load_rate])           
        
        if solver_type==2:
            r.integrate(tf, step=True)
            dt = r.t - t
            t = r.t
            y = r.y
            
            
        else:
        
            t, dt, y, err = solver(t, y, 
                           dt, dt_max, dt_min, 
                           aa, bb, dc, v_pl, 
                           N_lam, N_ker, CFF, 
                           K, nu, G, c_s, 
                           fun, state_law, 1,
                           max_iter, tf, tol)
            

        
        istep += 1


        # Write the maximum values to the file
        if istep % vmax_interval == 0 :
            
            ind_max = np.argmax(y[:N_lam])

            mu_f = friction_law(y[ind_max], y[N_lam+ind_max], aa[ind_max], bb[ind_max], dc[ind_max], v_0[ind_max], mu_0[ind_max])
            tau_f = mu_f * y[2*N_lam+ind_max]
            
            # 1=step, 2=t, 3=ivmax, 4=v, 5=theta, 6=tau, 7=dtau_dt, 8=slip, 9=sigma
            
            line_max = '{:>6.0f}{:>24.16E}{:>8.0f}{:>16.8E}{:>16.8E}{:>16.8E}{:>16.8E}{:16.8E}\n'.format(
                    istep, t, ind_max, y[ind_max], y[N_lam+ind_max], tau_f, y[3*N_lam+ind_max], y[2*N_lam+ind_max])
            
            filevmax.write(line_max)
            
            
            if p['print']:
                print('{:>12.4E}{:>12.4E}{:>12.4E}{:>12.4E}{:>12.4E}'.format(
                    dt, t/tyr, y[ind_max], y[N_lam+ind_max], np.abs(CFF).max()))            
        
        if istep % ot_interval == 0 :
            
            # line_ox = '#step t x y z v theta tau CFF slip sigma\n'
            line_ox = '{:>16}{:>24}{:>16}{:>16}{:>16}{:>16}{:>24}{:>16}{:>16}{:>16}{:>16}\n'.format(
                "Step","Time","X","Y","Z","V","Theta","Tau_f","Tau_e","Slip","Sigma"
                )
            for ii in np.arange(0,N_lam,ox_interval):
                
                mu_f = friction_law(y[ii], y[N_lam+ii], aa[ii], bb[ii], dc[ii], v_0[ii], mu_0[ii])
                tau_f = mu_f * y[2*N_lam+ii]
                
                line_ox += '{:>16.0f}{:>24.16E}{:>16.8E}{:>16.8E}{:>16.8E}{:16.8E}{:>24.16E}{:>16.8E}{:16.8E}{:>16.8E}{:>16.8E}\n'.format(
                        istep, t, XX[ii], YY[ii], ZZ[ii], y[ii], y[N_lam+ii], tau_f, CFF[ii], y[3*N_lam+ii], y[2*N_lam+ii])
            
            snapshot_file = os.path.join(snaphot_path, f'step_{istep:.0f}')
            with open(snapshot_file, "w") as file:
                file.write(line_ox)           

            
    filevmax.close()
