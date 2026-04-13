#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 21 10:47:18 2024

@author: sopaci
"""
import numba
import numpy as np
from scipy.integrate import ode


def scipy_ode(t, y, 
               dt, dt_max, dt_min, 
               aa, bb, dcc, vpl, 
               N_lam, N_kernel, dtau_trig, 
               KK, nu, G, c_s, 
               fun, state_fun, N_fault, maxiter, tf, tol):
    

    r = ode( fun ).set_integrator('vode', method="adams")       
    r.set_initial_value( y, t )
    r.set_f_params( dt, dt_max, dt_min, 
                    aa, bb, dcc, vpl, 
                    N_lam, N_kernel, dtau_trig, 
                    KK, nu, G, c_s, 
                    fun, state_fun, N_fault, maxiter, tf, tol)

    # RUNNING SIMULATION
    r.integrate(tf, step = True)
    dt = r.t - t
    t = r.t
    y = r.y
        
    return t, dt, y, 0

@numba.njit
def rk4_step(t, y, dt, 
             aa, bb, dcc, 
             vpl, N_lam, N_kernel, dtau_trig, 
             KK, nu, G, c_s, fun, state_fun, N_fault, tol):
    k1 = fun(y, t, 
         aa, bb, dcc, vpl, N_lam, N_kernel, dtau_trig, KK, nu, G, c_s, state_fun, N_fault)
    k2 = fun(t + dt/2, y + dt*k1/2, 
           aa, bb, dcc, 
           vpl, N_lam, N_kernel, dtau_trig, 
           KK, nu, G, c_s, fun, state_fun, N_fault, tol)
    k3 = fun(t + dt/2, y + dt*k2/2,
           aa, bb, dcc, 
           vpl, N_lam, N_kernel, dtau_trig, 
           KK, nu, G, c_s, fun, state_fun, N_fault, tol)
    k4 = fun(t + dt, y + dt*k3, 
           aa, bb, dcc, 
           vpl, N_lam, N_kernel, dtau_trig, 
           KK, nu, G, c_s, fun, state_fun, N_fault, tol)
    
    return y + (dt/6)*(k1 + 2*k2 + 2*k3 + k4)

@numba.njit
def abm(t, y, dt, 
                                     aa, bb, dcc, 
                                     vpl, N_lam, N_kernel, dtau_trig, 
                                     KK, nu, G, c_s, fun, state_fun, N_fault, tol):
    """
    Adaptive 4th-order Adams–Bashforth–Moulton predictor–corrector ODE solver.

    """

    t_0 = t 
    y_0 = y
    # Storage
    t_values = [t_0]
    y_values = [y_0]

    # RK4 for first 3 points
    # Bootstrap 3 points
    for _ in range(3):
        y = rk4_step(t, y, dt, aa, bb, dcc, 
         vpl, N_lam, N_kernel, dtau_trig, 
         KK, nu, G, c_s, fun, state_fun, N_fault, tol)
        t += dt
        t_values.append(t)
        y_values.append(y)

    # Adams–Bashforth predictor
    f_n3 = fun(t_values[-4], y_values[-4], dt, 
               aa, bb, dcc, 
               vpl, N_lam, N_kernel, dtau_trig, 
               KK, nu, G, c_s, fun, state_fun, N_fault, tol)
    f_n2 = fun(t_values[-3], y_values[-3], dt,
               aa, bb, dcc, 
               vpl, N_lam, N_kernel, dtau_trig, 
               KK, nu, G, c_s, fun, state_fun, N_fault, tol)
    f_n1 = fun(t_values[-2], y_values[-2], dt,
               aa, bb, dcc, 
               vpl, N_lam, N_kernel, dtau_trig, 
               KK, nu, G, c_s, fun, state_fun, N_fault, tol)
    f_n  = fun(t_values[-1], y_values[-1], dt,
               aa, bb, dcc, 
               vpl, N_lam, N_kernel, dtau_trig, 
               KK, nu, G, c_s, fun, state_fun, N_fault, tol)

    y_pred = y_values[-1] + (dt/24)*(55*f_n - 59*f_n1 + 37*f_n2 - 9*f_n3)

    # Adams–Moulton corrector
    f_pred = fun(t + dt, y_pred, dt,
                 aa, bb, dcc, 
                 vpl, N_lam, N_kernel, dtau_trig, 
                 KK, nu, G, c_s, fun, state_fun, N_fault, tol)
    y_corr = y_values[-1] + (dt/24)*(9*f_pred + 19*f_n - 5*f_n1 + f_n2)
    

    # Error estimate
    err = np.abs(y_corr - y_pred)
    
    return dt, t, y_corr, err
 


@numba.njit
def rk_step( t, y, dt, aa, bb, dcc, vpl, N_lam, N_kernel, dtau_trig, KK, nu, G, c_s, fun, state_fun, N_fault, tol ):
    
    # Runge-Kutta Fehlberg coefficients
    a2 = 2.500000000000000e-01  # 1/4
    a3 = 3.750000000000000e-01  # 3/8
    a4 = 9.230769230769231e-01  # 12/13
    a5 = 1.000000000000000e+00  # 1
    a6 = 5.000000000000000e-01  # 1/2

    b21 = 2.500000000000000e-01  # 1/4
    b31 = 9.375000000000000e-02  # 3/32
    b32 = 2.812500000000000e-01  # 9/32
    b41 = 8.793809740555303e-01  # 1932/2197
    b42 = -3.277196176604461e+00  # -7200/2197
    b43 = 3.320892125625853e+00  # 7296/2197
    b51 = 2.032407407407407e+00  # 439/216
    b52 = -8.000000000000000e+00  # -8
    b53 = 7.173489278752436e+00  # 3680/513
    b54 = -2.058966861598441e-01  # -845/4104
    b61 = -2.962962962962963e-01  # -8/27
    b62 = 2.000000000000000e+00  # 2
    b63 = -1.381676413255361e+00  # -3544/2565
    b64 = 4.529727095516569e-01  # 1859/4104
    b65 = -2.750000000000000e-01  # -11/40

    r1 = 2.777777777777778e-03  # 1/360
    r3 = -2.994152046783626e-02  # -128/4275
    r4 = -2.919989367357789e-02  # -2197/75240
    r5 = 2.000000000000000e-02  # 1/50
    r6 = 3.636363636363636e-02  # 2/55

    c1 = 1.157407407407407e-01  # 25/216
    c3 = 5.489278752436647e-01  # 1408/2565
    c4 = 5.353313840155945e-01  # 2197/4104
    c5 = -2.000000000000000e-01  # -1/5
    
    
    k1 = dt * fun(y, t, 
                       aa, bb, dcc, vpl, N_lam, N_kernel, dtau_trig, KK, nu, G, c_s, state_fun, N_fault)
    k2 = dt * fun(y + b21 * k1, t + a2 * dt, 
                       aa, bb, dcc, vpl, N_lam, N_kernel, dtau_trig, KK, nu, G, c_s, state_fun, N_fault)
    k3 = dt * fun(y + b31 * k1 + b32 * k2, t + a3 * dt, 
                       aa, bb, dcc, vpl, N_lam, N_kernel, dtau_trig, KK, nu, G, c_s, state_fun, N_fault)
    k4 = dt * fun(y + b41 * k1 + b42 * k2 + b43 * k3, t + a4 * dt, 
                       aa, bb, dcc, vpl, N_lam, N_kernel, dtau_trig, KK, nu, G, c_s, state_fun, N_fault)
    k5 = dt * fun(y + b51 * k1 + b52 * k2 + b53 * k3 + b54 * k4, t + a5 * dt, 
                       aa, bb, dcc, vpl, N_lam, N_kernel, dtau_trig, KK, nu, G, c_s, state_fun, N_fault)
    k6 = dt * fun(y + b61 * k1 + b62 * k2 + b63 * k3 + b64 * k4 + b65 * k5, t + a6 * dt,
                       aa, bb, dcc, vpl, N_lam, N_kernel, dtau_trig, KK, nu, G, c_s, state_fun, N_fault)
    
    r = max(
        np.abs(r1 * k1 + r3 * k3 + r4 * k4 + r5 * k5 + r6 * k6) / dt)
    
    t += dt
    
    y += c1 * k1 + c3 * k3 + c4 * k4 + c5 * k5
    
    return dt, t, y,r 

@numba.njit
def rkck(t, y, 
               dt, dt_max, dt_min, 
               aa, bb, dcc, vpl, 
               N_lam, N_kernel, dtau_trig, 
               KK, nu, G, c_s, 
               fun, state_fun, N_fault, maxiter, tf, tol):
    '''
    Runge Kutta cash-karp

    Parameters
    ----------
    t : TYPE
        DESCRIPTION.
    y : TYPE
        DESCRIPTION.
    dt : TYPE
        DESCRIPTION.
    dt_max : TYPE
        DESCRIPTION.
    dt_min : TYPE
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
    fun : TYPE
        DESCRIPTION.
    state_fun : TYPE
        DESCRIPTION.
    N_fault : TYPE
        DESCRIPTION.
    maxiter : TYPE
        DESCRIPTION.
    tf : TYPE
        DESCRIPTION.
    tol : TYPE
        DESCRIPTION.

    Returns
    -------
    tn : TYPE
        DESCRIPTION.
    dt : TYPE
        DESCRIPTION.
    yn : TYPE
        DESCRIPTION.
    err : TYPE
        DESCRIPTION.

    '''


    err = 2 * tol
    istop=5
    ii = 0
    while (err > tol):
        
        k1 = dt*fun(y, t,
                   aa, bb, dcc, vpl,
                   N_lam, N_kernel, dtau_trig, 
                   KK, nu, G, c_s, state_fun, N_fault)
        
        k2 = dt*fun( y+((1/5)*k1), t+(1/5)*dt, 
                   aa, bb, dcc, vpl,
                   N_lam, N_kernel, dtau_trig, 
                   KK, nu, G, c_s, state_fun, N_fault)
        
        k3 = dt*fun( y+((3/40)*k1)+((9/40)*k2), t+(3/10)*dt,
                   aa, bb, dcc, vpl,
                   N_lam, N_kernel, dtau_trig, 
                   KK, nu, G, c_s, state_fun, N_fault)        
        
        k4 = dt*fun(y+((3/10)*k1)-((9/10)*k2)+((6/5)*k3), t+(3/5)*dt,
                   aa, bb, dcc, vpl,
                   N_lam, N_kernel, dtau_trig, 
                   KK, nu, G, c_s, state_fun, N_fault)        
        
        k5 = dt*fun( y-((11/54)*k1)+((5/2)*k2)-((70/27)*k3)+((35/27)*k4), t+(1/1)*dt,
                   aa, bb, dcc, vpl,
                   N_lam, N_kernel, dtau_trig, 
                   KK, nu, G, c_s, state_fun, N_fault)        
        
        k6 = dt*fun( y+((1631/55296)*k1)+((175/512)*k2)+((575/13824)*k3)+((44275/110592)*k4)+((253/4096)*k5), t+(7/8)*dt,
                   aa, bb, dcc, vpl,
                   N_lam, N_kernel, dtau_trig, 
                   KK, nu, G, c_s, state_fun, N_fault)
        

        dy4 = ((37/378)*k1)+((250/621)*k3)+((125/594)*k4)+((512/1771)*k6)
        dy5 = ((2825/27648)*k1)+((18575/48384)*k3)+((13525/55296)*k4)+((277/14336)*k5)+((1/4)*k6)
        err = 1e-2*tol+max(np.abs(dy4-dy5))
        dt = max(min(0.95 * dt * (tol/err)**(1/5), dt_max), dt_min)
        if ii>=istop:
            break
        
        if t + dt > tf:
            # if time exceeds set t+dt =tf
            
            dt = tf - t
            
            k1 = dt*fun(y, t,
                       aa, bb, dcc, vpl,
                       N_lam, N_kernel, dtau_trig, 
                       KK, nu, G, c_s, state_fun, N_fault)
            
            k2 = dt*fun( y+((1/5)*k1), t+(1/5)*dt, 
                       aa, bb, dcc, vpl,
                       N_lam, N_kernel, dtau_trig, 
                       KK, nu, G, c_s, state_fun, N_fault)
            
            k3 = dt*fun( y+((3/40)*k1)+((9/40)*k2), t+(3/10)*dt,
                       aa, bb, dcc, vpl,
                       N_lam, N_kernel, dtau_trig, 
                       KK, nu, G, c_s, state_fun, N_fault)        
            
            k4 = dt*fun(y+((3/10)*k1)-((9/10)*k2)+((6/5)*k3), t+(3/5)*dt,
                       aa, bb, dcc, vpl,
                       N_lam, N_kernel, dtau_trig, 
                       KK, nu, G, c_s, state_fun, N_fault)        
            
            k5 = dt*fun( y-((11/54)*k1)+((5/2)*k2)-((70/27)*k3)+((35/27)*k4), t+(1/1)*dt,
                       aa, bb, dcc, vpl,
                       N_lam, N_kernel, dtau_trig, 
                       KK, nu, G, c_s, state_fun, N_fault)        
            
            k6 = dt*fun( y+((1631/55296)*k1)+((175/512)*k2)+((575/13824)*k3)+((44275/110592)*k4)+((253/4096)*k5), t+(7/8)*dt,
                       aa, bb, dcc, vpl,
                       N_lam, N_kernel, dtau_trig, 
                       KK, nu, G, c_s, state_fun, N_fault)
            
            tn = tf
            yn = y + dy4
            return tn, dt, yn, err
            break
        
        ii+=1
                
        
    tn = t + dt
    yn = y + dy4
    
    
    
    return tn, dt, yn, err



@numba.njit
def rk_routine(t, y, 
               dt, dt_max, dt_min, 
               aa, bb, dcc, vpl, 
               N_lam, N_kernel, dtau_trig, 
               KK, nu, G, c_s, 
               fun, state_fun, N_fault, maxiter, tf, tol):
    '''
    Runge-Kutta Fehlberg method 

    Parameters
    ----------
    t : TYPE
        DESCRIPTION.
    y : TYPE
        DESCRIPTION.
    dt : TYPE
        DESCRIPTION.
    dt_max : TYPE
        DESCRIPTION.
    dt_min : TYPE
        DESCRIPTION.
    aa : TYPE
        DESCRIPTION.
    bb : TYPE
        DESCRIPTION.
    dcc : TYPE
        DESCRIPTION.
    sigma_nn : TYPE
        DESCRIPTION.
    vpl : TYPE
        DESCRIPTION.
    N_lam : TYPE
        DESCRIPTION.
    N_kernel : TYPE
        DESCRIPTION.
    KK : TYPE
        DESCRIPTION.
    mu : TYPE
        DESCRIPTION.
    c : TYPE
        DESCRIPTION.
    state_type : TYPE
        DESCRIPTION.
    tol : TYPE
        DESCRIPTION.
    maxiter : TYPE
        DESCRIPTION.
    fun : TYPE
        DESCRIPTION.

    Returns
    -------
    t : TYPE
        DESCRIPTION.
    dt : TYPE
        DESCRIPTION.
    y : TYPE
        DESCRIPTION.
    r : TYPE
        DESCRIPTION.

    '''
    
    dt_try = dt

    for ii in range(maxiter):

        dt_try, t_try, y_try, r = rk_step(t, y, dt_try, 
                                          aa, bb, dcc, vpl, 
                                          N_lam, N_kernel, dtau_trig, 
                                          KK, nu, G, c_s, fun, state_fun, 
                                          N_fault, tol )
        

        dt_try = dt_try * min(max(0.9 * np.power(r / tol, -0.20), 0.1), 4.0)
        
        
        # If necessary dt to sustain the error tolerancfe should be even smaller 
        # than the given dt_min continue integration with dt_min 
        if dt_try < dt_min : 
            
            dt_try = dt_min
            
            # ii = maxiter-1
            
            dt_try, t_try, y_try, r = rk_step(t, y, dt_try, 
                                          aa, bb, dcc, vpl, 
                                          N_lam, N_kernel, dtau_trig, 
                                          KK, nu, G, c_s, fun, state_fun, 
                                          N_fault, tol )            
            break
            
            # raise Warning("dt_try should be lower than the dt_min! Solver continues with dt_min")
        
        # If necessary dt is larger than the given dt_max continue integration 
        # with dt_max  
        if dt_try > dt_max: 
            
            if t_try + dt_try > tf :
                dt_try = tf - t
            else:
                dt_try = dt_max
            
            dt_try, t_try, y_try, r = rk_step(t, y, dt_try, 
                                          aa, bb, dcc, vpl, 
                                          N_lam, N_kernel, dtau_trig, 
                                          KK, nu, G, c_s, fun, state_fun, 
                                          N_fault, tol )

            break
        
        ## time has reached tf integration is ending with dt = tf- ti
        if t_try + dt_try > tf :
            
            dt_try = tf - t
            
            dt_try, t_try, y_try, r = rk_step(t, y, dt_try, 
                                              aa, bb, dcc, vpl, 
                                              N_lam, N_kernel, dtau_trig, 
                                              KK, nu, G, c_s, fun, state_fun, 
                                              N_fault, tol )      
            
            break
            
        
        if r <= tol:
            break
        

    t = t_try 
    dt = dt_try 
    y = y_try 

    return t, dt, y, r



    # @numba.njit
    # def adams_routine(t, y, 
    #                dt, dt_max, dt_min, 
    #                aa, bb, dcc, vpl, 
    #                N_lam, N_kernel, dtau_trig, 
    #                KK, nu, G, c_s, 
    #                fun, state_fun, N_fault, maxiter, tf, tol):
        
    #     '''
    #     not working yet (17.10.2025)
    #     '''
        
    #     dt_try = dt
    #     order = 4
    
    #     for ii in range(maxiter):
    
    #         dt_try, t_try, y_try, r = abm(t, y, dt_try, 
    #                                           aa, bb, dcc, vpl, 
    #                                           N_lam, N_kernel, dtau_trig, 
    #                                           KK, nu, G, c_s, fun, state_fun, 
    #                                           N_fault, tol )
        
        
    #         tol_scaled = tol * (1 + np.abs(y_try))
            
    #         # Step size control
    #         if np.all(err <= tol_scaled):
    #             # Accept step
    #             t += dt_try
    #             y_try = y_corr
        
    #             # Predict next step size
    #             safety = 0.9
    #             dt_new = dt_try * min(2.0, max(0.5, safety * (tol_scaled / (r + 1e-15))**(1/(order + 1))))
    #             dt = min(max(h_new, h_min), h_max)
    #         else:
    #             # Reject step → decrease h and retry
    #             dt = max(dt_try * 0.5, h_min)

    # # return np.array(t_values), np.array(y_values)

    #     # dt_try = dt_try * min(max(0.9 * np.power(r / tol, -0.20), 0.1), 4.0)
        
        
    #     # If necessary dt to sustain the error tolerancfe should be even smaller 
    #     # than the given dt_min continue integration with dt_min 
    #     if dt_try < dt_min : 
            
            # dt_try = dt_min
            
            # # ii = maxiter-1
            
            # dt_try, t_try, y_try, r = rk_step(t, y, dt_try, 
            #                               aa, bb, dcc, vpl, 
            #                               N_lam, N_kernel, dtau_trig, 
            #                               KK, nu, G, c_s, fun, state_fun, 
            #                               N_fault, tol )            
            # break
            
            # raise Warning("dt_try should be lower than the dt_min! Solver continues with dt_min")
        
        # If necessary dt is larger than the given dt_max continue integration 
        # with dt_max  
    #     if dt_try > dt_max: 
            
    #         if t_try + dt_try > tf :
    #             dt_try = tf - t
    #         else:
    #             dt_try = dt_max
            
    #         dt_try, t_try, y_try, r = rk_step(t, y, dt_try, 
    #                                       aa, bb, dcc, vpl, 
    #                                       N_lam, N_kernel, dtau_trig, 
    #                                       KK, nu, G, c_s, fun, state_fun, 
    #                                       N_fault, tol )

    #         break
        
    #     ## time has reached tf integration is ending with dt = tf- ti
    #     if t_try + dt_try > tf :
            
    #         dt_try = tf - t
            
    #         dt_try, t_try, y_try, r = rk_step(t, y, dt_try, 
    #                                           aa, bb, dcc, vpl, 
    #                                           N_lam, N_kernel, dtau_trig, 
    #                                           KK, nu, G, c_s, fun, state_fun, 
    #                                           N_fault, tol )      
            
    #         break
            
        
    #     if r <= tol:
    #         break
        

    # t = t_try 
    # dt = dt_try 
    # y = y_try 

    # return t, dt, y, r    


# %timeit rk_routine(t, y, dt, dt_max, dt_min, aa, bb, dcc, sigma_nn, vpl, N_lam, KK, mu, c, tol, maxiter)