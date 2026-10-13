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

    k1 = fun(
        y, t, aa, bb, dcc, vpl, N_lam, N_kernel,
        dtau_trig, KK, nu, G, c_s, state_fun, N_fault
    )

    k2 = fun(
        y + 0.5*dt*k1, t + 0.5*dt,
        aa, bb, dcc, vpl, N_lam, N_kernel,
        dtau_trig, KK, nu, G, c_s, state_fun, N_fault
    )

    k3 = fun(
        y + 0.5*dt*k2, t + 0.5*dt,
        aa, bb, dcc, vpl, N_lam, N_kernel,
        dtau_trig, KK, nu, G, c_s, state_fun, N_fault
    )

    k4 = fun(
        y + dt*k3, t + dt,
        aa, bb, dcc, vpl, N_lam, N_kernel,
        dtau_trig, KK, nu, G, c_s, state_fun, N_fault
    )

    return y + (dt/6.0)*(k1 + 2.0*k2 + 2.0*k3 + k4)






@numba.njit
def abm_adaptive(
    t, y, dt, dt_max, dt_min,
    aa, bb, dcc, vpl,
    N_lam, N_kernel, dtau_trig,
    KK, nu, G, c_s,
    fun, state_fun, N_fault,
    maxiter, tf, tol
):
    """
    Adaptive ABM4 integrator.

    Returns exactly:
        t, dt_next, y, err

    Assumptions:
        - y is a 1D NumPy array.
        - fun(y, t, ...) returns dy/dt.
        - rk4_step(t, y, h, ...) returns the updated state.
        - tol is an absolute error tolerance.
        - ABM4 history is reconstructed whenever dt changes.
    """

    if tol <= 0.0 or dt_min <= 0.0 or dt_max < dt_min:
        return t, dt, y, np.inf

    if t >= tf:
        return t, 0.0, y, 0.0

    if maxiter <= 0:
        return t, dt, y, np.inf

    h_try = min(max(dt, dt_min), dt_max, tf - t)
    last_err = 0.0
    attempts = 0

    while t < tf and attempts < maxiter:

        h_try = min(h_try, dt_max, tf - t)

        # Rebuild four equally spaced states from the
        # last globally accepted (t, y).
        Y = np.empty((4, y.size), dtype=np.float64)
        F = np.empty((4, y.size), dtype=np.float64)

        Y[0, :] = y
        F[0, :] = fun(
            y, t, aa, bb, dcc, vpl,
            N_lam, N_kernel, dtau_trig,
            KK, nu, G, c_s, state_fun, N_fault
        )

        tn = t
        yn = y.copy()
        restart = False

        # ------------------------------------------------
        # 1. RK4 startup: construct three history steps.
        # ------------------------------------------------
        for j in range(1, 4):

            if tn >= tf:
                return tn, h_try, yn, last_err

            h = min(h_try, tf - tn)

            # Full RK4 step
            y_full = rk4_step(
                tn, yn.copy(), h,
                aa, bb, dcc, vpl,
                N_lam, N_kernel, dtau_trig,
                KK, nu, G, c_s,
                fun, state_fun, N_fault, tol
            )

            # Two half RK4 steps for error estimation
            y_half1 = rk4_step(
                tn, yn.copy(), 0.5*h,
                aa, bb, dcc, vpl,
                N_lam, N_kernel, dtau_trig,
                KK, nu, G, c_s,
                fun, state_fun, N_fault, tol
            )

            y_half2 = rk4_step(
                tn + 0.5*h, y_half1.copy(), 0.5*h,
                aa, bb, dcc, vpl,
                N_lam, N_kernel, dtau_trig,
                KK, nu, G, c_s,
                fun, state_fun, N_fault, tol
            )

            err = np.max(np.abs(y_half2 - y_full)) / 15.0
            last_err = err
            attempts += 1

            # If the startup step fails, discard the local
            # history and rebuild it with a smaller step.
            if err > tol:
                if h <= dt_min:
                    return t, h_try, y, err

                factor = max(
                    0.1,
                    min(0.9 * (tol / err)**0.2, 2.0)
                )
                h_try = max(dt_min, h * factor)
                restart = True
                break

            # Richardson-corrected RK4 result
            yn = y_half2 + (y_half2 - y_full) / 15.0
            tn += h

            # A startup step can reach the final time.
            if tn >= tf:
                return tf, h_try, yn, err

            Y[j, :] = yn
            F[j, :] = fun(
                yn, tn, aa, bb, dcc, vpl,
                N_lam, N_kernel, dtau_trig,
                KK, nu, G, c_s, state_fun, N_fault
            )

            if attempts >= maxiter:
                return t, h_try, y, last_err

        if restart:
            continue

        # ------------------------------------------------
        # 2. AB4 predictor.
        # Y[0:4] and F[0:4] now have uniform spacing h.
        # ------------------------------------------------
        h = h_try

        y_n = Y[3, :].copy()

        f_n3 = F[0, :]
        f_n2 = F[1, :]
        f_n1 = F[2, :]
        f_n  = F[3, :]

        y_pred = y_n + (h / 24.0) * (
            55.0 * f_n
            - 59.0 * f_n1
            + 37.0 * f_n2
            - 9.0 * f_n3
        )

        f_pred = fun(
            y_pred, tn + h, aa, bb, dcc, vpl,
            N_lam, N_kernel, dtau_trig,
            KK, nu, G, c_s, state_fun, N_fault
        )

        # ------------------------------------------------
        # 3. AM4 corrector.
        # ------------------------------------------------
        y_corr = y_n + (h / 24.0) * (
            9.0 * f_pred
            + 19.0 * f_n
            - 5.0 * f_n1
            + f_n2
        )

        # Practical predictor-corrector error indicator
        err = np.max(np.abs(y_corr - y_pred))
        last_err = err
        attempts += 1

        # ------------------------------------------------
        # 4. Accept or reject the ABM trial.
        # ------------------------------------------------
        if err <= tol:

            # Accept corrected state
            t_new = tn + h
            y_new = y_corr

            if t_new >= tf:
                return tf, min(h, dt_max), y_new, err

            if err == 0.0:
                factor = 2.0
            else:
                factor = max(
                    0.1,
                    min(0.9 * (tol / err)**0.2, 2.0)
                )

            dt_next = min(h * factor, dt_max)

            # Commit the accepted state. The next iteration
            # reconstructs history if the step size changes.
            t = t_new
            y = y_new
            h_try = max(dt_min, dt_next)

        else:
            # Reject the entire trial block: retain the
            # previous globally accepted t and y.
            if h <= dt_min:
                return t, h_try, y, err

            factor = max(
                0.1,
                min(0.9 * (tol / err)**0.2, 0.5)
            )

            h_try = max(dt_min, h * factor)

    return t, h_try, y, last_err

 


# @numba.njit
# def rk_step( t, y, dt, aa, bb, dcc, vpl, N_lam, N_kernel, dtau_trig, KK, nu, G, c_s, fun, state_fun, N_fault, tol ):
    
#     # Runge-Kutta Fehlberg coefficients
#     a2 = 2.500000000000000e-01  # 1/4
#     a3 = 3.750000000000000e-01  # 3/8
#     a4 = 9.230769230769231e-01  # 12/13
#     a5 = 1.000000000000000e+00  # 1
#     a6 = 5.000000000000000e-01  # 1/2

#     b21 = 2.500000000000000e-01  # 1/4
#     b31 = 9.375000000000000e-02  # 3/32
#     b32 = 2.812500000000000e-01  # 9/32
#     b41 = 8.793809740555303e-01  # 1932/2197
#     b42 = -3.277196176604461e+00  # -7200/2197
#     b43 = 3.320892125625853e+00  # 7296/2197
#     b51 = 2.032407407407407e+00  # 439/216
#     b52 = -8.000000000000000e+00  # -8
#     b53 = 7.173489278752436e+00  # 3680/513
#     b54 = -2.058966861598441e-01  # -845/4104
#     b61 = -2.962962962962963e-01  # -8/27
#     b62 = 2.000000000000000e+00  # 2
#     b63 = -1.381676413255361e+00  # -3544/2565
#     b64 = 4.529727095516569e-01  # 1859/4104
#     b65 = -2.750000000000000e-01  # -11/40

#     r1 = 2.777777777777778e-03  # 1/360
#     r3 = -2.994152046783626e-02  # -128/4275
#     r4 = -2.919989367357789e-02  # -2197/75240
#     r5 = 2.000000000000000e-02  # 1/50
#     r6 = 3.636363636363636e-02  # 2/55

#     c1 = 1.157407407407407e-01  # 25/216
#     c3 = 5.489278752436647e-01  # 1408/2565
#     c4 = 5.353313840155945e-01  # 2197/4104
#     c5 = -2.000000000000000e-01  # -1/5
    
    
#     k1 = dt * fun(y, t, 
#                        aa, bb, dcc, vpl, N_lam, N_kernel, dtau_trig, KK, nu, G, c_s, state_fun, N_fault)
#     k2 = dt * fun(y + b21 * k1, t + a2 * dt, 
#                        aa, bb, dcc, vpl, N_lam, N_kernel, dtau_trig, KK, nu, G, c_s, state_fun, N_fault)
#     k3 = dt * fun(y + b31 * k1 + b32 * k2, t + a3 * dt, 
#                        aa, bb, dcc, vpl, N_lam, N_kernel, dtau_trig, KK, nu, G, c_s, state_fun, N_fault)
#     k4 = dt * fun(y + b41 * k1 + b42 * k2 + b43 * k3, t + a4 * dt, 
#                        aa, bb, dcc, vpl, N_lam, N_kernel, dtau_trig, KK, nu, G, c_s, state_fun, N_fault)
#     k5 = dt * fun(y + b51 * k1 + b52 * k2 + b53 * k3 + b54 * k4, t + a5 * dt, 
#                        aa, bb, dcc, vpl, N_lam, N_kernel, dtau_trig, KK, nu, G, c_s, state_fun, N_fault)
#     k6 = dt * fun(y + b61 * k1 + b62 * k2 + b63 * k3 + b64 * k4 + b65 * k5, t + a6 * dt,
#                        aa, bb, dcc, vpl, N_lam, N_kernel, dtau_trig, KK, nu, G, c_s, state_fun, N_fault)
    
#     err = np.max(np.abs(
#         r1*k1 + r3*k3 + r4*k4 + r5*k5 + r6*k6
#     ))
        
#     t += dt
    
#     y += c1 * k1 + c3 * k3 + c4 * k4 + c5 * k5
    
#     return dt, t, y, err 


@numba.njit
def rkck(t, y, dt, dt_max, dt_min,
         aa, bb, dcc, vpl,
         N_lam, N_kernel, dtau_trig,
         KK, nu, G, c_s,
         fun, state_fun, N_fault, maxiter, tf, tol):

    # Handle an already completed integration
    if t >= tf:
        return t, 0.0, y, 0.0

    if tol <= 0.0:
        raise ValueError("tol must be positive")

    dt_try = min(dt, dt_max, tf - t)

    for ii in range(maxiter):

        # Start from the same accepted state (t, y)
        dt_used = min(dt_try, tf - t)

        # ------------------------------------------------
        # Calculate k1 ... k6 
        # ------------------------------------------------

        k1 = dt_used * fun(
            y, t, aa, bb, dcc, vpl,
            N_lam, N_kernel, dtau_trig,
            KK, nu, G, c_s, state_fun, N_fault
        )

        k2 = dt_used * fun(
            y + (1/5)*k1, t + (1/5)*dt_used,
            aa, bb, dcc, vpl, N_lam, N_kernel,
            dtau_trig, KK, nu, G, c_s,
            state_fun, N_fault
        )

        k3 = dt_used * fun(
            y + (3/40)*k1 + (9/40)*k2,
            t + (3/10)*dt_used,
            aa, bb, dcc, vpl, N_lam, N_kernel,
            dtau_trig, KK, nu, G, c_s,
            state_fun, N_fault
        )

        k4 = dt_used * fun(
            y + (3/10)*k1 - (9/10)*k2 + (6/5)*k3,
            t + (3/5)*dt_used,
            aa, bb, dcc, vpl, N_lam, N_kernel,
            dtau_trig, KK, nu, G, c_s,
            state_fun, N_fault
        )

        k5 = dt_used * fun(
            y - (11/54)*k1 + (5/2)*k2
            - (70/27)*k3 + (35/27)*k4,
            t + dt_used,
            aa, bb, dcc, vpl, N_lam, N_kernel,
            dtau_trig, KK, nu, G, c_s,
            state_fun, N_fault
        )

        k6 = dt_used * fun(
            y + (1631/55296)*k1 + (175/512)*k2
            + (575/13824)*k3 + (44275/110592)*k4
            + (253/4096)*k5,
            t + (7/8)*dt_used,
            aa, bb, dcc, vpl, N_lam, N_kernel,
            dtau_trig, KK, nu, G, c_s,
            state_fun, N_fault
        )

        # Fifth-order and fourth-order increments
        dy5th = (
            (37/378)*k1 + (250/621)*k3
            + (125/594)*k4 + (512/1771)*k6
        )

        dy4th = (
            (2825/27648)*k1 + (18575/48384)*k3
            + (13525/55296)*k4 + (277/14336)*k5
            + (1/4)*k6
        )

        err = np.max(np.abs(dy5th - dy4th))

        # Calculate proposed NEXT step size
        if err == 0.0:
            factor = 4.0
        else:
            factor = min(
                max(0.95 * (tol/err)**0.2, 0.1),
                4.0
            )

        dt_next = min(dt_used * factor, dt_max)

        # Accept the current trial
        if err <= tol:
            tn = t + dt_used
            yn = y + dy5th
            return tn, dt_next, yn, err

        # Cannot meet tolerance at the minimum step
        if dt_used <= dt_min:
            return t, dt_used, y, err

        # Reject trial and retry from original t and y
        dt_try = max(dt_next, dt_min)

    # Retry limit reached: no accepted step
    return t, dt_try, y, err





@numba.njit
def rkf45(t, y, dt, dt_max, dt_min,
          aa, bb, dcc, vpl,
          N_lam, N_kernel, dtau_trig,
          KK, nu, G, c_s,
          fun, state_fun, N_fault, maxiter, tf, tol):

    if tol <= 0.0:
        raise ValueError("tol must be positive")

    if t >= tf:
        return t, 0.0, y, 0.0

    dt_try = min(dt, dt_max, tf - t)
    err = 0.0

    for ii in range(maxiter):

        # Use the actual trial step for all stages
        dt_used = min(dt_try, tf - t)

        k1 = dt_used * fun(
            y, t, aa, bb, dcc, vpl, N_lam, N_kernel,
            dtau_trig, KK, nu, G, c_s, state_fun, N_fault
        )

        k2 = dt_used * fun(
            y + (1/4)*k1, t + (1/4)*dt_used,
            aa, bb, dcc, vpl, N_lam, N_kernel,
            dtau_trig, KK, nu, G, c_s, state_fun, N_fault
        )

        k3 = dt_used * fun(
            y + (3/32)*k1 + (9/32)*k2,
            t + (3/8)*dt_used,
            aa, bb, dcc, vpl, N_lam, N_kernel,
            dtau_trig, KK, nu, G, c_s, state_fun, N_fault
        )

        k4 = dt_used * fun(
            y + (1932/2197)*k1 - (7200/2197)*k2
              + (7296/2197)*k3,
            t + (12/13)*dt_used,
            aa, bb, dcc, vpl, N_lam, N_kernel,
            dtau_trig, KK, nu, G, c_s, state_fun, N_fault
        )

        k5 = dt_used * fun(
            y + (439/216)*k1 - 8*k2
              + (3680/513)*k3 - (845/4104)*k4,
            t + dt_used,
            aa, bb, dcc, vpl, N_lam, N_kernel,
            dtau_trig, KK, nu, G, c_s, state_fun, N_fault
        )

        k6 = dt_used * fun(
            y - (8/27)*k1 + 2*k2
              - (3544/2565)*k3 + (1859/4104)*k4
              - (11/40)*k5,
            t + (1/2)*dt_used,
            aa, bb, dcc, vpl, N_lam, N_kernel,
            dtau_trig, KK, nu, G, c_s, state_fun, N_fault
        )

        # Fourth-order solution (Fehlberg)
        dy4 = (
            (25/216)*k1
            + (1408/2565)*k3
            + (2197/4104)*k4
            - (1/5)*k5
        )

        # Fifth-order embedded solution
        dy5 = (
            (16/135)*k1
            + (6656/12825)*k3
            + (28561/56430)*k4
            - (9/50)*k5
            + (2/55)*k6
        )

        # Absolute error estimate
        err = np.max(np.abs(dy5 - dy4))

        # Calculate the next proposed step size
        if err == 0.0:
            factor = 4.0
        else:
            factor = min(
                max(0.9 * (tol/err)**0.2, 0.1),
                4.0
            )

        dt_next = min(dt_used * factor, dt_max)

        # Accept only if the error is within tolerance
        if err <= tol:
            tn = t + dt_used
            yn = y + dy4

            return tn, dt_next, yn, err

        # Failed tolerance at the minimum step
        if dt_used <= dt_min:
            return t, dt_used, y, err

        # Reject this trial; retain original t and y
        dt_try = max(dt_next, dt_min)

    # Retry limit reached without an accepted step
    return t, dt_try, y, err





# @numba.njit
# def rk_routine(t, y,
#                dt, dt_max, dt_min,
#                aa, bb, dcc, vpl,
#                N_lam, N_kernel, dtau_trig,
#                KK, nu, G, c_s,
#                fun, state_fun, N_fault,
#                maxiter, tf, tol):

#     dt_try = min(dt, dt_max, tf - t)

#     if dt_try <= 0.0:
#         return t, 0.0, y, 0.0

#     for ii in range(maxiter):

#         # Limit the trial step to the remaining integration interval
#         dt_try = min(dt_try, tf - t)

#         # Attempt a step from the CURRENT accepted state
#         _, t_try, y_try, err = rk_step(
#             t, y, dt_try,
#             aa, bb, dcc, vpl,
#             N_lam, N_kernel, dtau_trig,
#             KK, nu, G, c_s,
#             fun, state_fun, N_fault, tol
#         )

#         # Calculate the next proposed step size
#         if err == 0.0:
#             factor = 4.0
#         else:
#             factor = min(
#                 max(0.9 * (err / tol)**(-0.2), 0.1),
#                 4.0
#             )

#         dt_next = min(dt_try * factor, dt_max)

#         # Accept the trial if the error is acceptable
#         if err <= tol:

#             return t_try, dt_next, y_try, err

#         # Error too large: reject the trial.
#         # Do not update t or y.
#         if dt_try <= dt_min:
#             # Tolerance cannot be met at the minimum step.
#             # Return the current accepted state and report err.
#             return t, dt_try, y, err

#         # Reduce the step size and retry from the same t and y
#         dt_try = max(dt_next, dt_min)

#     # Maximum number of attempts reached without acceptance
#     return t, dt_try, y, err




# %timeit rk_routine(t, y, dt, dt_max, dt_min, aa, bb, dcc, sigma_nn, vpl, N_lam, KK, mu, c, tol, maxiter)