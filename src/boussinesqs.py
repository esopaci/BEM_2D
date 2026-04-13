#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 11 09:21:37 2024

@author: sopaci
"""

import numpy as np 
import numba


@numba.njit
def boussinesqs1(xx, yy, zz, dip_w, x0, y0, q, mode):
    
    rho = 2650
    gg = 9.81
    nu = 0.25
    PI = np.pi

    opi = 0.5 / PI

    
    P0 = q * rho * gg
    
    n_dir = np.zeros(3)
    n_f = np.zeros(3)
    tau_n = np.zeros(3)
    
    
    STRESS = np.zeros((3,3))

    if mode == 1:
        #case (1) ! strike-slip (right-lateral)
        Ustrike = 1
        Udip = 0
        n_dir[0] = -1
        n_dir[1] = 0
        n_dir[2] = 0
        
    elif mode == -1:

        # case (-1) ! strike-slip (left-lateral)
        Ustrike = -1
        Udip = 0
        n_dir[0] = 1
        n_dir[1] = 0
        n_dir[2] = 0
        
    elif mode == 2:
        # case (2) ! dip-slip (thrust)
        Ustrike = 0
        Udip = 1
        n_dir[0] = 0
        n_dir[1] = -np.cos(dip_w/180*PI)
        n_dir[2] = -np.sin(dip_w/180*PI)
        
    elif mode == -2:
        # case (-2) ! dip-slip (normal)
        Ustrike = 0
        Udip = -1
        n_dir[0] = 0
        n_dir[1] = np.cos(dip_w/180*np.pi)
        n_dir[2] = np.sin(dip_w/180*np.pi)



    n_f[1] = 0
    n_f[1] = -np.sin(dip_w/180*np.pi)
    n_f[2] =  np.cos(dip_w/180*np.pi)

    dx1 = x0 - xx
    dx1s = dx1*dx1
    dx2 = y0 - yy
    dx2s = dx2 * dx2
    dx3 = np.abs(zz)
    dx3s = dx3 * dx3

    R2 = dx1s + dx2s + dx3s
    R = np.sqrt(R2)
    R3 = R2 * R
    R5 = R3 * R2
    RDX3 = (R + dx3)
    
    
    dummy0 = 1 - 2 * nu
    dummy1 = (2 * R + dx3) / (R * RDX3**2)
    
    STRESS[0,0] = -opi*P0 / R2 * (3 * dx1s * dx3 / R3 - dummy0 * \
        (dx3 / R - R / RDX3 + (dx1s * dummy1)))

    STRESS[1,1] = -opi*P0 / R2 * (3 * dx2s * dx3 / R3 - dummy0 * \
        (dx3 / R - R / RDX3 + (dx2s * dummy1)))

    STRESS[0,1] = -opi*P0 / R2 * dx1 * dx2 * (3 * dx3 / R3 - dummy0 * dummy1)

    STRESS[1,0] = STRESS[0,1]

    STRESS[0,2] = -1.5*P0 / PI / R5 * dx1 * dx3s

    STRESS[2,0] = STRESS[0,2]

    STRESS[1,2] = -1.5*P0 / PI / R5 * dx2 * dx3s

    STRESS[2,1] = STRESS[1,2]

    STRESS[2,2] = -1.5*P0 / PI / R5 * dx3s

    tau_n[0] = STRESS[0,0]*n_f[0] + STRESS[0,1]*n_f[1] + STRESS[0,2]*n_f[2]
    tau_n[1] = STRESS[1,0]*n_f[0] + STRESS[1,1]*n_f[1] + STRESS[0,2]*n_f[2]
    tau_n[2] = STRESS[2,0]*n_f[0] + STRESS[2,1]*n_f[1] + STRESS[0,2]*n_f[2]

    dtau = tau_n[0]*n_dir[0] + tau_n[1]*n_dir[1] + tau_n[2]*n_dir[2]
    dsigma = tau_n[0]*n_f[0] + tau_n[1]*n_f[1] + tau_n[2]*n_f[2]
    
    return (dtau, dsigma)


@numba.njit
def surface_loading_spread(xx, yy, NN, x0, y0, sx, sy, q):
    
    PSF =np.zeros(NN)
    for i in range(NN):

        PSF[i] = np.exp(-((xx[i] - x0)**2 / (2 * sx**2) + ( yy[i]- y0)**2 / (2 * sy**2)))

    NF = q / np.sum(PSF)


    PSF = PSF * NF

    return PSF


@numba.njit
def surface_load_rate( dt, aa, bb ):

    q = aa + 3 * bb * dt * dt
    return q


@numba.njit
def surface_load( dt, aa, bb ):

    q = aa * dt + bb * dt * dt * dt
    return q


def boussinesq_solution(time, t0, aa,bb, XX, YY, ZZ, DIP_W, x0, y0, NN):
    
    
    if len(XX.shape)>1:
        XX = XX.flatten()
        YY = YY.flatten()
        ZZ = ZZ.flatten()
        DIP_W = DIP_W.flatten()

    dt = time - t0
    
    # if XX.size != NN:
    #     break
    
    
    q = surface_load_rate( dt, aa, bb)
    
    # xx = self.dict['MESH_DICT']['X'] 
    # zz = self.dict['MESH_DICT']['Z']
    # yy = self.dict['MESH_DICT']['Y']
    # dip_w = self.dict['MESH_DICT']['DIP_W']
    # x0 = self.dict['x0']
    # y0 = self.dict['y0']
    
    # PSF = surface_loading_spread( xx, yy, dip_w, NN, x0, y0, sx, sy, q)
    dtau = np.zeros(NN)
    dsigma = np.zeros(NN)
    
    for ii in range(NN):

        (dtau[ii], dsigma[ii]) = boussinesqs1(XX[ii], YY[ii], ZZ[ii], DIP_W[ii], x0, y0, q, 2)
            
    return (dtau, dsigma)




# import matplotlib.pyplot as plt 

# tyr = 365*3600 * 24

# dip_w = 60.0

# W = 3000 
# L = 5000
# DW = 10
# DL = 10

# Z_CORNER = -2900

# W_GRID = np.arange(0,W+DW,DW)
# L_GRID = np.linspace(-L/2-DL/2,L/2+DL/2,DL)

# XX,WW = np.meshgrid(L_GRID, W_GRID)
# NN = WW.size

# YY = np.cos(dip_w * np.pi/180) * WW 
# ZZ = np.sin(dip_w * np.pi/180) * WW + Z_CORNER

# DIP_W = np.ones(NN) * dip_w

# x0 = 0; y0 = -1000

# time = 200
# time *= tyr

# t0 = 0 

# aa = 7.5E-3
# bb = 0.0

# dt = time - t0
# surface_load( dt, aa, bb )

# (dsigma, dtau) = boussinesq_solution(time, t0, aa,bb, XX, YY, ZZ, DIP_W, x0, y0, NN)

# dsigma *= dt 
# dtau *= dt

# # print(t_unique[i]) 
# # print(self.dict['dtt'])

# CFF = (dsigma)
# # CFF = CFF.reshape((self.NW_sample, self.NX_sample)) * 1E-6


# # create the figure
# fig = plt.figure()

# ax = fig.add_subplot(111)

# cntr = ax.contourf(XX,ZZ.reshape(XX.shape), dsigma.reshape(XX.shape) * 1e-6, levels = 200)

# fig.colorbar(cntr)












