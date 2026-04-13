#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 21 11:12:53 2024

@author: sopaci
"""

import numpy as np 
import numba
from kernel import f_FFT_freesurface, f_FD, f_FFT_periodic
# import okada4py as ok92


# from boussinesqs import boussinesq_solution
# import pyfftw


@numba.njit(nopython=True, parallel = True)
def balance_eq_fft_periodic(y, t, 
                   aa, bb, dcc, vpl, N_lam, N_kernel,
                   dtau_trig, KK, nu, G, c_s, state_fun, Nfault):
    '''
    Periodic fault (No free surface)

    Parameters
    ----------
    y : TYPE
        DESCRIPTION.
    t : TYPE
        DESCRIPTION.
    aa : TYPE
        DESCRIPTION.
    bb : TYPE
        DESCRIPTION.
    dcc : TYPE
        DESCRIPTION.
    vpl : TYPE
        DESCRIPTION.
    N_lam : TYPE
        DESCRIPTION.
    N_kernel : TYPE
        DESCRIPTION.
    dtau_trig : TYPE
        DESCRIPTION.
    KK : TYPE
        DESCRIPTION.
    nu : TYPE
        DESCRIPTION.
    G : TYPE
        DESCRIPTION.
    c_s : TYPE
        DESCRIPTION.
    state_fun : TYPE
        DESCRIPTION.
    Nfault : TYPE
        DESCRIPTION.

    Returns
    -------
    dy_dt : TYPE
        DESCRIPTION.

    '''    
    # initialize the derivative array.
    # It is N_faults * 3
    # Nvar = N_faults * 3
    # dydt = np.zeros( 3*N_var * N_lam ) # total number of variables
    # ff = np.zeros(N_faults * N_lam)    # array for stress transfer
    
    
    # for fid in N_faults:  
        # ind0 = int(fid * Nvar)
        # srate = y[ ind0 : int(ind0 + N_lam)] ##  slip rate 
        # theta = y[ int(ind0 + N_lam) : int(ind0+2*N_lam)] ##  state
        # sigma_n =  y[ int(ind0 + 2*N_lam) : int(ind0 + 3*N_lam)] ## effective normal stress

    # ind0 = 0
    
    # Initiate the derivatives 
    dy_dt = np.zeros(4*N_lam)
    
    v =     y[ 0*N_lam : 1*N_lam] ##  slip rate 
    theta = y[ 1*N_lam : 2*N_lam] ##  state
    sigma = y[ 2*N_lam : 3*N_lam] ## effective normal stress
    # slip =  y[ 3*N_lam : 4*N_lam] ## effective normal stress    
    
    drate = v - vpl #slip deficit
        
        #### Numpy computation######################################################
    f = f_FFT_periodic(KK, drate, N_lam, N_kernel)
        # if N_fault > 1: #In case more than one fault interaction is considered! 
            #pass it for now. Apply stress interaction between different faults   
            # pass 
                    
    
    dtheta_dt = state_fun(v, theta, dcc) # state rate
        
        # slip rate
    dv_dt = (f + dtau_trig - bb * sigma * dtheta_dt / theta) \
                / (0.5 * G / c_s + (aa * sigma) / v)
    
    dy_dt[:N_lam] = dv_dt[:]
    dy_dt[1*N_lam:2*N_lam] = dtheta_dt[:]
    
    # For not the effective normal stress change is ignored!
    dy_dt[2*N_lam:3*N_lam] = 0.0
    
    dy_dt[3*N_lam:4*N_lam] = v 
          
    return dy_dt




@numba.njit(nopython=True, parallel = True)
def balance_eq_fft_freesurface(y, t, 
                   aa, bb, dcc, vpl, N_lam, N_kernel,
                   dtau_trig, KK, nu, G, c_s, state_fun, Nfault):
    
    # initialize the derivative array.
    # It is N_faults * 3
    # Nvar = N_faults * 3
    # dydt = np.zeros( 3*N_var * N_lam ) # total number of variables
    # ff = np.zeros(N_faults * N_lam)    # array for stress transfer
    
    
    # for fid in N_faults:  
        # ind0 = int(fid * Nvar)
        # srate = y[ ind0 : int(ind0 + N_lam)] ##  slip rate 
        # theta = y[ int(ind0 + N_lam) : int(ind0+2*N_lam)] ##  state
        # sigma_n =  y[ int(ind0 + 2*N_lam) : int(ind0 + 3*N_lam)] ## effective normal stress

    # ind0 = 0
    
    # Initiate the derivatives 
    dy_dt = np.zeros(4*N_lam)
    
    v =     y[ 0*N_lam : 1*N_lam] ##  slip rate 
    theta = y[ 1*N_lam : 2*N_lam] ##  state
    sigma = y[ 2*N_lam : 3*N_lam] ## effective normal stress
    # slip =  y[ 3*N_lam : 4*N_lam] ## effective normal stress    
    
    drate = v - vpl #slip deficit
        
        #### Numpy computation######################################################
    f = f_FFT_freesurface(KK, drate, N_lam, N_kernel)
        # if N_fault > 1: #In case more than one fault interaction is considered! 
            #pass it for now. Apply stress interaction between different faults   
            # pass 
                    
    
    dtheta_dt = state_fun(v, theta, dcc) # state rate
        
        # slip rate
    dv_dt = (f + dtau_trig - bb * sigma * dtheta_dt / theta) \
                / (0.5 * G / c_s + (aa * sigma) / v)
    
    dy_dt[:N_lam] = dv_dt[:]
    dy_dt[1*N_lam:2*N_lam] = dtheta_dt[:]
    
    # For not the effective normal stress change is ignored!
    dy_dt[2*N_lam:3*N_lam] = 0.0
    
    dy_dt[3*N_lam:4*N_lam] = v 
          
    return dy_dt

# @numba.njit
def balance_eq_fft_nonumba(t, y, 
                   aa, bb, dcc, vpl, N_lam, N_kernel,
                   dtau_trig, KK, nu, G, c_s, state_fun, Nfault):
    
    # initialize the derivative array.
    # It is N_faults * 3
    # Nvar = N_faults * 3
    # dydt = np.zeros( 3*N_var * N_lam ) # total number of variables
    # ff = np.zeros(N_faults * N_lam)    # array for stress transfer
    
    
    # for fid in N_faults:  
        # ind0 = int(fid * Nvar)
        # srate = y[ ind0 : int(ind0 + N_lam)] ##  slip rate 
        # theta = y[ int(ind0 + N_lam) : int(ind0+2*N_lam)] ##  state
        # sigma_n =  y[ int(ind0 + 2*N_lam) : int(ind0 + 3*N_lam)] ## effective normal stress

    # ind0 = 0
    
    # Initiate the derivatives 
    dy_dt = np.zeros(4*N_lam)
    
    v =     y[ 0*N_lam : 1*N_lam] ##  slip rate 
    theta = y[ 1*N_lam : 2*N_lam] ##  state
    sigma = y[ 2*N_lam : 3*N_lam] ## effective normal stress
    # slip =  y[ 3*N_lam : 4*N_lam] ## effective normal stress    
    
    drate = v - vpl #slip deficit
        
        #### Numpy computation######################################################
    f = f_FFT_freesurface(KK, drate, N_lam, N_kernel)
        # if N_fault > 1: #In case more than one fault interaction is considered! 
            #pass it for now. Apply stress interaction between different faults   
            # pass 
                    
    
    dtheta_dt = state_fun(v, theta, dcc) # state rate
        
        # slip rate
    dv_dt = (f + dtau_trig - bb * sigma * dtheta_dt / theta) \
                / (0.5 * G / c_s + (aa * sigma) / v)
    
    dy_dt[:N_lam] = dv_dt[:]
    dy_dt[1*N_lam:2*N_lam] = dtheta_dt[:]
    
    # For not the effective normal stress change is ignored!
    dy_dt[2*N_lam:3*N_lam] = 0.0
    
    dy_dt[3*N_lam:4*N_lam] = v 
          
    return dy_dt


@numba.jit(nopython=True, parallel = True)
def balance_eq_space(y, t, aa, bb, dcc, vpl, N_lam, N_kernel, dtau_trig, KK, nu, G, c_s, state_fun, N_fault):
    
    
    # Due to the boundary conditions first and the last elements are set to 0!
    
    # ind0 = 0
    
    # Initiate the derivatives 
    dy_dt = np.zeros(4*N_lam)
    
    v =     y[ 0*N_lam : 1*N_lam] ##  slip rate 
    theta = y[ 1*N_lam : 2*N_lam] ##  state
    sigma = y[ 2*N_lam : 3*N_lam] ## effective normal stress
    # slip =  y[ 3*N_lam : 4*N_lam] ## effective normal stress     
    drate = v - vpl #slip deficit
    
    #### Numpy computation######################################################
    f = f_FD(KK, drate, N_lam, N_kernel)

    dtheta_dt = state_fun(v, theta, dcc) # state rate
        
        # slip rate
    dv_dt = (f + dtau_trig - bb * sigma * dtheta_dt / theta) \
                / (0.5 * G / c_s + (aa * sigma) / v)
    
    dy_dt[:N_lam] = dv_dt[:]
    dy_dt[1*N_lam:2*N_lam] = dtheta_dt[:]
    
    # effective normal stress change is ignored!
    dy_dt[2*N_lam:3*N_lam] = 0.0
    dy_dt[3*N_lam:] = v
          
    ############################################################################
    # numba parallelization
    # dy_dt = np.zeros( 2 * N_lam)
    

    # # # slip rate deficit
    # for i in numba.prange(N_lam):
    #     # stress transfer
    #     fi = 0
    #     dtheta_dt = state_fun(drate[i], theta[i], dcc[i])
    #     dy_dt[N_lam + i] = dtheta_dt
    #     for j in numba.prange(N_lam):
    #           fi += (KK[i,j] * (v[j] - vpl[i]) )  
        
        
    #     dtheta_dt = state_fun(v[i], theta[i], dcc[i])
        
    #     dy_dt[i] = (fi + dtau_trig - bb[i]*sigma[i] * dtheta_dt / theta[i])\
    #             / (0.5 * G / c_s + (aa[i] * sigma[i]) / v[i])
    ###########################################################################
    

    return dy_dt

# @numba.njit
def balance_eq_space_nonumba(t, y, aa, bb, dcc, vpl, N_lam, N_kernel, dtau_trig, KK, nu, G, c_s, state_fun, N_fault):
    
    
    # Due to the boundary conditions first and the last elements are set to 0!
    
    # ind0 = 0
    
    # Initiate the derivatives 
    dy_dt = np.zeros(4*N_lam)
    
    v =     y[ 0*N_lam : 1*N_lam] ##  slip rate 
    theta = y[ 1*N_lam : 2*N_lam] ##  state
    sigma = y[ 2*N_lam : 3*N_lam] ## effective normal stress
    # slip =  y[ 3*N_lam : 4*N_lam] ## effective normal stress     
    drate = v - vpl #slip deficit
    
    #### Numpy computation######################################################
    # f = -np.dot(KK, srate - vpl)   # stress transfer functiom
    f = f_FD(KK, drate, N_lam, N_kernel)

    dtheta_dt = state_fun(v, theta, dcc) # state rate
        
        # slip rate
    dv_dt = (f + dtau_trig - bb * sigma * dtheta_dt / theta) \
                / (0.5 * G / c_s + (aa * sigma) / v)
    
    dy_dt[:N_lam] = dv_dt[:]
    dy_dt[1*N_lam:2*N_lam] = dtheta_dt[:]
    
    # effective normal stress change is ignored!
    dy_dt[2*N_lam:3*N_lam] = 0.0
    dy_dt[3*N_lam:] = v
          
          
    
    ############################################################################
    # numba parallelization
    # dy_dt = np.zeros( 2 * N_lam)
    

    # # # slip rate deficit
    # for i in numba.prange(N_lam):
    #     # stress transfer
    #     fi = 0
    #     dtheta_dt = state_fun(srate[i], theta[i], dcc[i])
    #     dy_dt[N_lam + i] = dtheta_dt
    #     for j in numba.prange(N_lam):
    #           fi += (KK[i,j] * (srate[j] - vpl[i]) )  
        
        
    #     dtheta_dt = state_fun(srate[i], theta[i], dcc[i])
        
    #     dy_dt[i] = (fi + dtau - bb[i]*sigma_nn[i] * dtheta_dt / theta[i])\
    #             / (0.5 * G / c_s + (aa[i] * sigma_nn[i]) / srate[i])
    ###########################################################################
    

    return dy_dt


@numba.jit(nopython=True, parallel = True)
def balance_eq_springslider(y, t, aa, bb, dcc, vpl, N_lam, N_kernel, dtau_trig, KK, nu, G, c_s, state_fun, N_fault):
    '''
    spring slider formulation 

    '''
    
    dy_dt = np.zeros(4*N_lam)
    
    v =     y[ 0*N_lam : 1*N_lam] ##  slip rate 
    theta = y[ 1*N_lam : 2*N_lam] ##  state
    sigma = y[ 2*N_lam : 3*N_lam] ## effective normal stress
    # slip =  y[ 3*N_lam : 4*N_lam] ## effective normal stress 
    drate = vpl - v #slip deficit

    #### Numpy computation######################################################
    f = np.dot(KK, drate)   # stress transfer functiom
    

    dtheta_dt = state_fun(v, theta, dcc) # state rate
        
    # slip rate
    dv_dt = (f + dtau_trig - bb * sigma * dtheta_dt / theta) \
                / (0.5 * G / c_s + (aa * sigma) / v)
    
    dy_dt[:N_lam] = dv_dt[:]
    dy_dt[1*N_lam:2*N_lam] = dtheta_dt[:]
    
    # For not the effective normal stress change is ignored!
    dy_dt[2*N_lam:3*N_lam] = 0.0
    dy_dt[3*N_lam:] = v

    return dy_dt

@numba.njit
def balance_eq_fft_scipy(t, y, 
                   aa, bb, dcc, vpl, N_lam, N_kernel,
                   dtau_trig, KK, nu, G, c_s, state_fun, Nfault):
    
    return balance_eq_fft(y, t, 
                       aa, bb, dcc, vpl, N_lam, N_kernel,
                       dtau_trig, KK, nu, G, c_s, state_fun, Nfault)
@numba.njit
def balance_eq_space_scipy(t, y, 
                   aa, bb, dcc, vpl, N_lam, N_kernel,
                   dtau_trig, KK, nu, G, c_s, state_fun, Nfault):
    
    return balance_eq_space(y, t, 
                       aa, bb, dcc, vpl, N_lam, N_kernel,
                       dtau_trig, KK, nu, G, c_s, state_fun, Nfault)

