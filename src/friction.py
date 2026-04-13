#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 11 11:24:04 2022
Friction
@author: gauss
"""
import numpy as np
import numba

@numba.jit(nopython=True, cache=True)
def original_rsf(vel, state, a, b, dc, v0, f_0):
    """
    

    Parameters
    ----------
    vel : array
        slip rates.
    state : array
        state of the frictional surface (contact history).
    a : array
        direct velocity effect.
    b : array
        state evolution effect.
    dc : array
        critical slip distance.
    sigma_n : array
        effective normal stress.
    v0 : scalar
        reference velocity.
    f_0 : scalar
        friction constant.

    Returns
    -------
    strength : array
        frictional resistance against the driving plate.

    """

    strength = (
            f_0 + a * np.log(vel / v0) + b * np.log(v0 * state / dc)
            )
    return strength


@numba.jit(nopython=True, cache=True)
def regularized_rsf(vel, state, a, b, dc, v0, f_0):
    """
    

    Parameters
    ----------
    vel : array
        slip rates.
    state : array
        state of the frictional surface (contact history).
    a : array
        direct velocity effect.
    b : array
        state evolution effect.
    dc : array
        critical slip distance.
    sigma_n : array
        effective normal stress.
    v0 : scalar
        reference velocity.
    f_0 : scalar
        friction constant.

    Returns
    -------
    strength : array
        frictional resistance against the driving plate.

    """


    strength= (a *
            np.arcsinh(0.5 * vel / v0 *
                       np.exp((f_0 + b * np.log(v0 * state / dc)) / a))
            )
    
    return strength

@numba.njit
def aging_law(vel, state, dc):
    """
    

    Parameters
    ----------
    vel : array
        slip rates
    state : array
        state of the frictional surface (contact history).
        DESCRIPTION.
    dc : array
        direct velocity effect.
    state_type : integer
        0 (aging), 1  (ruina) state evolution formula.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    # vel = np.abs(vel)
    omega = np.multiply(vel, state) / dc
    return 1.0 - omega  # aging law

@numba.njit
def slip_law(vel, state, dc):
    """
    

    Parameters
    ----------
    vel : array
        slip rates.
    state : array
        state of the frictional surface (contact history).
        DESCRIPTION.
    dc : array
        direct velocity effect.
    state_type : integer
        0 (aging), 1  (ruina) state evolution formula.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    # vel = np.abs(vel)
    omega = np.multiply(vel, state) / dc
    return -omega * np.log(omega)  # slip law