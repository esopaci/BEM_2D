#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  7 15:18:21 2022
Kernels
@author: gauss
"""


import numpy as np

import numba
import pyfftw


@numba.njit(numba.float64[:](numba.float64[:]), parallel=True )
def Besseln(z):
    
    n = z.shape[0]
    fz = np.zeros(n)
    
    for i in numba.prange(n):
        if z[i] < 8.:
            t = z[i] / 8.
            fz[i] = (-0.2666949632 * t**14 + 1.7629415168000002 * t**12 + -5.6392305344 * t**10 + 11.1861160576 * t**8 + -14.1749644604 * t**6 + 10.6608917307 * t**4 + -3.9997296130249995 * t**2 + 0.49999791505)
        else:
            t = 8. / z[i]
            eta = z[i] - 0.75 * np.pi
            fz[i] = np.sqrt(2 / (np.pi * z[i])) * ((1.9776e-06 * t**6 + -3.48648e-05 * t**4 + 0.0018309904000000001 * t**2 + 1.00000000195) * np.cos(eta) - (-6.688e-07 * t**7 + 8.313600000000001e-06 * t**5 + -0.000200241 * t**3 + 0.04687499895 * t) * np.sin(eta)) / z[i]
        
    return fz

@numba.njit(numba.float64(numba.float64))
def Bessel1(z):
        
    if z < 8.:
        t = z/ 8.
        fz = z * (-0.2666949632 * t**14 + 1.7629415168000002 * t**12 + -5.6392305344 * t**10 + 11.1861160576 * t**8 + -14.1749644604 * t**6 + 10.6608917307 * t**4 + -3.9997296130249995 * t**2 + 0.49999791505)
    else:
        t = 8. / z
        eta = z - 0.75 * np.pi
        fz = np.sqrt(2 / (np.pi * z)) * ((1.9776e-06 * t**6 + -3.48648e-05 * t**4 + 0.0018309904000000001 * t**2 + 1.00000000195) * np.cos(eta) - (-6.688e-07 * t**7 + 8.313600000000001e-06 * t**5 + -0.000200241 * t**3 + 0.04687499895 * t) * np.sin(eta)) 
    return fz


@numba.njit
def time_kernel(lam, W, N, c, beta, nw):
    
    n = np.arange(-N/2, N/2, 1)
    
    aa = 2.0*np.pi*c/lam
    kn = np.append(np.arange(0,N//2+1), np.arange(-N//2+1,0)) *2.0*np.pi/lam# wave number
    kn[N//2] = 0
    # kn= 2.0 * np.pi * n / lam
    absn = np.abs(n)[:N//2]
    dtmin = beta * lam / N / c
    tw1 = nw *lam / c 
    # kw1 = aa*tw1
    # kwn2 = N/2.0*kw1
    
    # qw = N/2
    qw=4.
    # kw = kw1 + (kwn2-kw1) / (N/2-1.0) * (absn-1.0)
    kw = aa * tw1 * (1.0 + (qw-1.0)/(N/2-1.0) * (absn - 1.0) )
    tw = kw / (absn * aa )
    # tw[N//2] = 0
    return tw, n, kn, dtmin


@numba.njit
def bessel_kernel( lam,W, N, c, beta, nw):
    
    tw, n, kn, dtmin = time_kernel( lam, W, N, c, beta, nw )
    
    kernel = []
    
    thetas = []
    
    for jj in numba.prange(N//2):
        
        twj = tw[jj]
        
        kk = np.abs(kn[jj])

        theta = np.arange(dtmin, twj+dtmin, dtmin)
        
        Nt = theta.size
        
        I = np.zeros(Nt)

        for ii in numba.prange(Nt):
            
            q = theta[ii:] * kk * c
            
            # I[ii] = Besseln(q)/theta[ii]

            I[ii] = np.trapz( Besseln(q), q )
                    
        kernel.append(I)     
        
        thetas.append(theta)
    
    kernel.extend(kernel[::-1])
    
    thetas.extend(thetas[::-1])

            
    return kernel, thetas, tw, n, kn, dtmin


def spring_slider_kernel(zz, N_lam, N_kernel, L, G, nu, W ):

    return np.array([G/L])


@numba.njit
def space_kernel(zz, N_lam, N_kernel, L, G, nu, W ):
    '''
    1D stress disclocation kernel for infinite fault embedded in 2D space .
    The fault is mirrored at the z = 0. This allows the surface reflection. 
    That is why the last element close to the surface should be less than 0.
    By adding the length free surface at z = 0, number of elements to be 
    computed is 2*N+1. Total length of fault is 2L but the computational 
    domain is 1L. Due to the mirroring computational cost is O(N^2)
    The kernel computation without surface reflection is given by Dieterich1992.
    Here we added the mirrored image. The Dieterich 1992 is as follows:  
    # ! Generates kernel stress dislocation kernel
    # ! K is obtained from elastic dislocation solutions. 
    # ! For an array of dislocation segments bounded 
    # ! by edge dislocations in an infinite medium
    # ! ------------------------------------------- 
    # ! K_ij = mu/(2pi*(1-nu)) * (1/X_ij - 1/X_ij_1).
    # ! -------------------------------------------
    # ! where MU and NU are shear modulus and Poisson’s
    # ! ratio, respectively. The terms X_ii and Xij_i are
    # ! distances from the center of segment i to the two
    # ! dislocations bounding segment j.
    # ! (Dieterich, 1992).
    
    
    Parameters
    ----------
    zz : Array 
        Center of the cells. The first elemnt is the deepest cell
        We maniputale it by adding free surface z=0 and mirror the image. 
        The last element must be smaller than 0!
    G : float, optional
        Shear Modulus. The default is 30E9.
    nu : float, optional
        Poisson's ratio. The default is 0.25.

    Returns
    -------
    K : Matrix(z.size,zsize)
        Static stress Kernel.

    '''
               
        
    # For this kernel we need to shift center of the cell to its boundaries.
    # It starts from the bottom corner of the cell.
    # Unlike FFT kernels the length of the cell can be different. 
    # First calculate the cell sizes. 
    
    # zz = np.array([-11500-2000, -2000])
    # G = 30E9
    # nu = 0.25
    
    xx = np.zeros(2*N_lam+1)
    xx[:N_lam] = zz[:]
    xx[N_lam] = 0.0
    xx[N_lam+1:] = -zz[::-1]

    # xx = np.concatenate((zz,[0], -zz[::-1]))
    
    # cell spacing     
    dx = np.diff(xx)
        
    # Intialize the kernel
    kk = np.zeros( (2*N_lam, 2*N_lam) )
        
    # Compute kernel
    for i in range(2*N_lam):
        
        z_i = xx[i] + dx[i]/2

        for j in range(2*N_lam):
            
            X_ij = z_i - xx[j+1] 
            X_ij_1 = z_i - xx[j]
            
            kk[i,j] = 1.0 * G / (2*np.pi * (1 - nu)) * \
            ( 1 / X_ij  - 1 /  X_ij_1  )
            
    K = kk + kk[::1, ::-1]
    K = K[:N_lam:,:N_lam]

    return K


@numba.njit
def fft_kernel(zz, N_lam, N_kernel, L, G, nu, W ):
    '''
    
    Kernel for computing space derivatives in FFT domain. Will be explained in
    more detail
    Parameters
    ----------
    N_lam : TYPE
        DESCRIPTION.
    N_kernel : TYPE
        DESCRIPTION.
    L : TYPE
        DESCRIPTION.
    G : TYPE, optional
        DESCRIPTION. The default is 30e9.
    W : TYPE, optional
        DESCRIPTION. The default is 0.

    Returns
    -------
    KK : TYPE
        DESCRIPTION.

    '''


    # Length of vector f
    # Initialize k vector up to Nyquist wavenumber
    kmax = np.pi/(L/N_kernel)
    dk = kmax/(N_kernel/2)
    k = np.arange(float(N_kernel))
    k[:N_kernel//2] = k[:N_kernel//2] * dk
    k[N_kernel//2:] = k[:N_kernel//2] - kmax

    if W == -1:
        KK = -0.5 * G/(1 - nu) * np.abs(k)
    else: 
        # If depth information is assigned
        KK = -0.5 * G/(1 - nu) * np.sqrt(k*k + (2.0*np.pi/W)**2)
        
    return KK


@numba.njit
def f_FFT_freesurface(KK, a, N_lam, N_kernel):
    
    # Mirroring the values 
    # set the last value near surface 0.
    # a[-1] = 0.0

    # aa = np.concatenate((a, a[::-1]))
    with numba.objmode(D='complex128[:]'):
        AA = pyfftw.empty_aligned(N_lam*2, dtype='complex128')
        AA[:N_lam] = a[:]
        AA[N_lam+1:] = np.flip(a)[:-1]
        AA[N_lam] = 0.0
        fft_object = pyfftw.builders.fft(AA, N_kernel, axis=0)
        D = fft_object()
    
    F = np.multiply( KK, D ) 
    with numba.objmode(f_transfer='complex128[:]'):
        FF = pyfftw.empty_aligned(N_kernel, dtype='complex128')
        FF[:] = F[:]
        fft_object = pyfftw.builders.ifft(FF, axis=0)
        f_transfer = fft_object()
    f_transfer = f_transfer[:N_lam].real * N_lam/N_kernel
    
    return f_transfer

@numba.njit
def f_FFT_periodic(KK, a, N_lam, N_kernel):
    '''
    No mirroring effect is applied pure periodic fault

    Parameters
    ----------
    KK : TYPE
        DESCRIPTION.
    a : TYPE
        DESCRIPTION.
    N_lam : TYPE
        DESCRIPTION.
    N_kernel : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    '''

    # aa = np.concatenate((a, a[::-1]))
    with numba.objmode(D='complex128[:]'):
        AA = pyfftw.empty_aligned(N_lam, dtype='complex128')
        AA[:] = a[:]
        fft_object = pyfftw.builders.fft(AA, N_kernel, axis=0)
        D = fft_object()
    
    F = np.multiply( KK, D ) 
    with numba.objmode(f_transfer='complex128[:]'):
        FF = pyfftw.empty_aligned(N_kernel, dtype='complex128')
        FF[:] = F[:]
        fft_object = pyfftw.builders.ifft(FF, axis=0)
        f_transfer = fft_object()
    f_transfer = f_transfer[:N_lam].real * N_lam/N_kernel
    
    return f_transfer


@numba.njit
def f_FD(KK, a, N_lam, N_kernel):
    
    # aa = np.concatenate((a, a[::-1]))
    # return np.dot(KK, aa)[:N_lam]
    
    # K = np.array(np.float64, 2, 'KK', False, aligned=True), 
         
    # V = np.array(np.float64, 1, 'a', False, aligned=True)
    
    return np.dot(np.ascontiguousarray(KK), np.ascontiguousarray(a))

@numba.njit
def f_spring_slider(KK, vv, N_lam, N_kernel):
    
    return np.dot(KK, vv)


# TEST THE KERNELS ##
##########################################################################

# from scipy.ndimage import gaussian_filter1d


# # PARAMETERS
# N_lam = np.power(2,5)
# N_kernel = N_lam * 4 # kernel is 4 times larger than the number of input
# L = 10E3
# W = 50e3
# dx = L / N_lam
# G = 30E9 
# nu = 0.0

# ## DOMAIN IN DEPTH
# zz2 = np.linspace(-L/2,L/2 ,N_lam*2, endpoint = True)
# zz= zz2[:N_lam]
# zz3 = np.array([zz[0], zz[4* N_lam//5] ])


# # displacement
# d = np.zeros(N_lam)
# d[1*N_lam//3 : 1*N_lam//2] = 1
# d = gaussian_filter1d(d,1)  # To make it smoother
# d2 = np.concatenate( ( d, d[::-1] ) )   # mirrored
# d3 = np.array([d[2* N_lam//5], d[4* N_lam//5] ])

# # %%
# ## This part is 2D OKADA KERNEL
# # Sigma_{12}
# def s12h(x2,x3,y2,y3,W,mu):
#     s12 = mu*( \
#     -(x3-y3)/((x2-y2)**2+(x3-y3)**2)+(x3+y3)/((x2-y2)**2+(x3+y3)**2) \
#     +(x3-y3-W)/((x2-y2)**2+(x3-y3-W)**2)-(x3+y3+W)/((x2-y2)**2+(x3+y3+W)**2) \
#     )/2/np.pi
#     return s12


# # Displacement kernels due to fault slip
# def u1h(x2,x3,y2,y3,W):
#     u1 = (np.arctan((x3-y3)/(x2-y2))-np.arctan((x3+y3)/(x2-y2)) \
#      -np.arctan((x3-y3-W)/(x2-y2))+np.arctan((x3+y3+W)/(x2-y2)) \
#     )/2/np.pi
#     return u1


# """
# Kernels                        
# """
# Kk=np.zeros((N_lam,N_lam))                        # stress kernels 
# y3=np.linspace(0,N_lam-1,N_lam).reshape( (N_lam, ) )*dx     # Top of slip patch
# WW=np.ones((N_lam,1))*dx          # Down-dip width of slip patch

# for i in range(N_lam):
#     # Evaluate the stress at the center of the slip patches
#     # Coordinate from source is top of patch
#     # s1h(x2,x3,y2,y3,W,G)
#     Kk[:,i]=s12h(0,y3+dx/2,0,y3[i],WW[i],G);
    
# tau_okad = np.matmul(Kk,d)


# # %%
# ## This part is alternative spectral computation with numpy fft

# # Define static kernel
# # k = np.append(np.arange(0,N_kernel//2+1), np.arange(-N_kernel//2+1,0)) *2.0*np.pi/L# wave number

# # KK = -0.5 * G * np.abs(k)
# # Dk = np.fft.fft(d3, N_kernel)
# # Fk = np.multiply(KK,Dk)
# # tau_fft2 = np.fft.ifft(Fk).real[N_lam:] * N_lam / N_kernel
# # %%
# ## Computation of kernels
# #CALCULATE STRESS TRANSFER WITH THE SPACE DERIVATIVES
# K_space = space_kernel(zz, N_lam, N_kernel, L, G, nu, W)
# tau_space =  f_FD(K_space, d, N_lam, N_kernel)

# K_space3 = space_kernel(zz3, 2, 2, L, G, nu, W)
# dzz = np.diff(np.concatenate((zz3, [0])))
# tau_space3 =  f_FD(K_space3, d3, 2, 2)

# # # #CALCULATE STRESS TRANSFER WITH THE FOURIER CEOFFICIENTS
# K_spect = fft_kernel(zz2, N_lam, N_kernel, L, G, nu, W )
# tau_fft   =  f_FFT(K_spect, d, N_lam, N_kernel)
# # tau_fft2 = f_transfer_fun(d2,N_kernel,N_kernel,KK)
# # %%

# # PLOTTING###
# import matplotlib.pyplot as plt 

# plt.plot(zz, tau_space, ls = '-', marker = '+', label = 'space',
#          markerfacecolor="None",
#         markeredgecolor='red', alpha = 0.8)
# plt.plot(zz, tau_okad, ls = '-', marker = '.', label = 'space okada',
#          markerfacecolor="None",
#         markeredgecolor='blue', alpha = 0.8)
# plt.plot(zz, tau_fft, ls = '--', label = 'fft', color ='k', alpha = 0.8)

# # zz3 += (dzz/2)
# plt.plot(zz3, [tau_space3[0], tau_space3[0]], color = 'b', label = '2blocks')
# plt.plot([zz3[1],0], [tau_space3[1], tau_space3[1]], color = 'b', )


# # plt.plot(zz, tau_fft2[:N_lam], ls = '-.', label = 'fft2', color = 'green')

# # plt.xlim([-4000, -2000])
# plt.legend()
# plt.xlabel('depth')
# plt.ylabel('$\\tau$')
# plt.savefig('kernel_comparison.jpg')


