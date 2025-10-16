#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 30 16:42:53 2024

@author: sopaci
"""
import pandas as pd 

import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os
import sys

from os.path import expanduser
home = expanduser("~")
simcodepath = os.path.join(home,'simcode')
sys.path.append(os.path.join(simcodepath,'src'))
# from boussinesqs import boussinesq_solution

class Ptool:
    
    t_yr = 365*3600*24
    names_max = ['istep','t','ind_max','v','theta','tau','slip','sigma_n']

    names_ox = ['step', 't', 'x', 'y', 'z', 'v', 'theta', 'tau', 'CFF', 'slip', 'sigma']
    # fmt="%15.0f%24.14E%15.7E%15.7E%15.7E%15.7E%26.16E%20.12E%15.7E%15.7E%20.12E%15.0f"
    vc = 1e-3
    tc = 60
    def __init__(self, d):
    
        self.d = d # root directory to the simulation results
        
        # output file for maximum slip rate
        self.otmax_file = os.path.join(self.d,  
                        [x for x in os.listdir(self.d) if x.endswith('vmax')][0] 
                                       )
        
        # output file for space and time. 
        self.ox_file = os.path.join(self.d, 
                        [x for x in os.listdir(self.d) if x.endswith('ox')][0] 
                                    )
        
        # input parameters as a pickle file
        pars_file = os.path.join(self.d, 
                        'params.pickle'
                                    )
        
        self.pars = pd.read_pickle(r'{}'.format(pars_file))
        
        self.NN_sample = self.pars['mesh']['ww'].size
        
        # Get the number of lines in the output_vmax file
        with open(self.otmax_file, "r") as f:
            self.NN_vmaxtime = sum(1 for _ in f)
        
        self.NN_oxtime = int(self.NN_vmaxtime / self.pars['max_interval'] * self.pars['ot_interval'])
        
        print(f'{self.d}  \t ------->\t\nreading input values completed')   




    def read_ox_1( self ):
        '''
        This function reads the space-time output file. 
        If the output is a dense file, avoid using this function

        Returns
        -------
        df0 : TYPE
            DESCRIPTION.

        '''
        
                
                                    
        df0 = pd.read_csv(self.ox_file, sep='\s+',
                          header = None,
                            comment='#', 
                            )
        df0.columns = self.names_ox
    
        return df0
    
    
    def read_ox_2(self, skip=0, n_snaphots = 1):
        '''
        This function reads the space-time output file as step by step. 
        If the output file is dense, reading this way is more appropriate. 

        Parameters
        ----------
        skip : integer, optional
            number of the snapshot. The default is 0.

        Returns
        -------
        df0 : TYPE
            DESCRIPTION.

        '''
                
        skip0 = int( skip * (self.NN_sample + 1)-1)
                                    
        df0 = pd.read_csv(self.ox_file, nrows=self.NN_sample * n_snaphots,
                          sep='\s+', encoding="utf-8", 
                          skiprows = skip0, comment = '#')
  
        df0.columns = self.names_ox

        return df0
    
    # def compute_slip(self, interval = 100):
    #     '''
    #     Depending on the model sometimes, computing slip values after simulation
    #     ended is more efficient. This code reads the outputs and compute the slip
    #     from slip rate and time. Then it rewrites the ouputfile.

    #     Returns
    #     -------
    #     None.

    #     '''
    #     slip0 = np.zeros(self.NN_sample)

    #     for i in range(1, self.NN_oxtime, interval):
    #         df = self.read_ox_2( skip=i, n_snaphots=interval )
            
    #         t_unique = df.t.iloc[1:]
    #         Nw = self.NN_sample
    #         slip_rate = np.array(df.v.tolist()).reshape(interval,Nw)
    #         slip = np.zeros((interval,Nw))
            
    #         for iw in range(Nw):
    #             slip[:,iw] = cumtrapz(slip_rate[:,iw], t_unique, initial=slip0[iw])
                
    #         slip0 = slip[:,-1]
            
    #         print(slip0[self.NN_sample//2])
    
    
    def read_vmaxfile(self):
        df = pd.read_csv(self.otmax_file, sep='\s+', comment='#',
                            encoding="utf-8", header=None)
        df.columns=self.names_max
        df.dropna(subset=["v"], inplace=True)
        return df
        
    
    
    
    def get_coseismic_instances(self, vc = vc):
        '''
        Finds the slip moments

        Parameters
        ----------dim2_state1_aasp0.0100_dc0.0160_aminbasp-0.0050_aminbbar0.0030_Lasp30_Lbar5_W15_L95
        vc : TYPE, optional
            DESCRIPTION. The default is 0.001.
        tc : TYPE, optional
            DESCRIPTION. The default is 60.

        Returns
        -------
        df2 : TYPE
            DESCRIPTION.

        '''
        df = self.read_vmaxfile()
        df1 = df[df.v > vc].copy()
        list_of_df = np.array_split(df1, np.flatnonzero(np.diff(df1.index) > 10) + 1)
        return list_of_df
        
    def plot_timeseries( self, ot = 'v', warmup = 0 ):
        '''
        This function plots only the each snapshotted maximum slip rate, 
        during run. 

        Parameters
        ----------
        ot : TYPE, optional
            DESCRIPTION. The default is 'v'.

        Returns
        -------
        None.

        '''
        
        
        fig, ax = plt.subplots(1,1, figsize = (8,3), clear = True)
        
        ax.set_xlabel('Time [yr]')
        ax.set_ylabel('{}'.format(ot))
        
        df = self.read_vmaxfile()
        
        df = df[df.t>warmup].copy()
        
        if ot == 'v':
            ax.semilogy(df.t / self.t_yr, df.v)
        else:
            ax.plot(df.t / self.t_yr, df[f'{ot}'])

        
        fig.savefig(os.path.join(self.d, 'time_series_{}.jpg'.format(ot)), dpi=200,
                                 bbox_inches='tight',
                                )
        
        
    def plot_timeseries2( self, ot = 'v', step = 10 ):
        '''


        Parameters
        ----------
        ot : TYPE, optional
            DESCRIPTION. The default is 'v'.

        Returns
        -------
        None.

        '''
        
        if ot == 'v':
            unit = 'm/s'
        elif ot == 'tau':
            unit = 'Pa'
        elif ot == 'theta':
            unit = '-'
            
        
        fig, ax = plt.subplots(1,1, figsize = (8,6), clear = True)
        
        ax.set_xlabel('Time [yr]')
        ax.set_ylabel('{} [{}]'.format(ot, unit))
        
        df = self.read_ox_1()
        
        z_u = df.z.unique()
        
        
        for i in range(0,z_u.size):
        
            if i == 0:
                color = 'b'
                lw = 1
                alpha = 0.5
                ls = ':'
            else:
                color = 'red'
                lw = 1
                alpha = 1
                ls = '--'
                
            df1 = df[(df['z'] == z_u[i])].copy()
            if ot == 'v':
                ax.semilogy((df1['t']) / self.t_yr, df1['v'], 
                        color = color, lw = lw, alpha = alpha, ls = ls, label = f'z={z_u[i]:.0f}'
                        )
            else:
                ax.plot((df1['t']) / self.t_yr, df1[f'{ot}'], 
                        color = color, lw = lw, alpha = alpha, ls = ls, label = f'z={z_u[i]:.0f}'
                        )
            
            # CFF = cumtrapz(df1['CFF'], df1['t'],initial=0)
            
            # ax.plot((df1['t']) / self.t_yr, CFF + df1['tau'].iloc[0], 
            #             color = color, lw = lw, alpha = alpha, ls = '--'
            #             )
        

        ax.legend(loc = 2)
        fig.savefig(os.path.join(self.d, 'time_seriesox_{}.jpg'.format(ot)), dpi=200,
                                      bbox_inches='tight',
                                    )
        

    def plot_slip_profile( self, interval = 10, vc = 1e-5, crit_size=10, tint_steps = (2, 20)  ):

        tfast_step = tint_steps[0]
        tslow_step = tint_steps[1] * self.t_yr

        
        fig, ax = plt.subplots(1,1, figsize = (6,10), clear = True)
        
        ax.set_ylabel('depth [yr]')
        ax.set_xlabel('slip')
        
        ox = self.read_ox_1()
        
        
        # Get unique coordinates and sort them
        x_unique = ox['z'].unique()
        sort_inds = np.argsort(x_unique)
        x_unique = x_unique[sort_inds]
        
        # Get unique times and sort them
        t_vals = np.sort(ox["t"].unique())
        
        # # Check that warm-up time is not greater than simulation time
        # if warm_up > t_vals.max():
        #     print("Warm-up time > simulation time!")
        #     return
        warm_up = 0
        # Determine index for warm-up time
        ind_warmup = np.where(t_vals >= warm_up)[0][0]
        
        # Get the number of unique coordinates and number of time steps
        Nx = len(x_unique)
        Nt = len(t_vals) - 1
        
        # Define the slice and shape for the data
        slice = np.s_[Nx * ind_warmup:Nx * Nt]
        data_shape = (Nt - ind_warmup, Nx)
        
        # Get the data values and reshape them
        x = ox['z'][slice].values.reshape(data_shape)[:, sort_inds]
        slip = ox["slip"][slice].values.reshape(data_shape)[:, sort_inds]
        v = ox["v"][slice].values.reshape(data_shape)[:, sort_inds]
        t = ox["t"][slice].values.reshape(data_shape)[:, sort_inds]

        
        # # Cluster the slip rates > vc0. This is the first clustering 
        # # detecting both creeping slow slips and fast ruptures. 
        # mask = np.array(v>vc,dtype=bool) * 1
        # s = ndimage.generate_binary_structure(2,2)
        # lw, num = ndimage.measurements.label(mask, structure=s)
        
        # # remove the clusters whose sizes are small
        # sizes = ndimage.sum(mask, lw, range(num + 1))
        # mask_size = sizes < crit_size
        # remove_pixel = mask_size[lw]
        # lw[remove_pixel] = 0
        
        # # re-define and sort the labels and detect their position
        # labels = np.unique(lw)
        # lw = np.searchsorted(labels, lw)
        # locs = ndimage.find_objects(lw)
        
        # for loc in locs:
            
        #         # Find fust ruptures
        #         mask = np.array(v[loc]>vc, dtype=bool) * 1

        #         # Cluster the fast ruptures
        #         lw2, num2 = ndimage.label(mask, structure=s)
                                            
        #         # remove the clusters whose sizes are small
        #         sizes = ndimage.sum(mask, lw2, range(num2 + 1))
        #         mask_size = sizes < crit_size
        #         remove_pixel = mask_size[lw2]
        #         lw2[remove_pixel] = 0
                
        #         # re-define and sort the labels and detect their position
        #         labels = np.unique(lw2)
        #         lw2 = np.searchsorted(labels, lw2)
                
        #         # Find the extend of the fast and aftershocks. 
        #         locs2 = ndimage.find_objects(lw2) 
                
        #         for loc2 in locs2:
        #                         # estimate the statics of the rupture instances  
        #             rate2 = v[loc][loc2]
        #             slip2 = slip[loc][loc2]
        #             xx2 = x[loc][loc2]
        #             tt2 = t[loc][loc2]
        #             tt_max, tt_min = tt2.max(), tt2.min()
                
        #             Ntt, Nxx = xx2.shape

        #             time_int=np.arange(tt_min, tt_max, tfast_step)
        #             Ntt = time_int.size    
        #             slip_int = np.empty((time_int.size, Nxx))
        #             rate_int = np.empty_like(slip_int)
        #             xx_int = np.repeat(xx2[0,:],Ntt).reshape(Nxx,Ntt).T

        #             for ix in range(Nxx):
        #                 f = interpolate.interp1d(tt2[:,ix], slip2[:,ix])     
        #                 slip_int[:,ix] = f(time_int)
        #                 f = interpolate.interp1d(tt2[:,ix], rate2[:,ix])     
        #                 rate_int[:,ix] = f(time_int)

        #             for ii in range(time_int.size):
                        
        #                 points = np.array([xx_int[ii,:], slip_int[ii,:]]).T.reshape(-1, 1, 2)
        #                 segments = np.concatenate([points[:-1], points[1:]], axis=1)
        #                 lc = matplotlib.collections.LineCollection(segments, array=rate_int[ii,:], 
        #                                                             cmap='jet', norm=matplotlib.colors.LogNorm(),
        #                                           linewidth=0.4, alpha=1)
        #                 ax.add_collection(lc)
                        
        #                 del points, segments, lc
        
        
        
        for i in range(1, slip.shape[0],interval):
            
            # if v[i,:].any() > vc:
            #     ax.plot( slip[i,:] , x[i,:], color = 'r', lw = 0.5)

            # elif v[i,:].all() <= vc and v[i,:].any()>1e-8:
            #     ax.plot( slip[i,:] , x[i,:], color = 'b', lw = 0.5)
            # else : 
            ax.plot( slip[i,:] , x[i,:], color = 'k', lw = 0.8)

                
                


            
        fig.savefig(os.path.join(self.d, 'slip_profile.jpg'), dpi=200,
                                      bbox_inches='tight',
                                    )


    def plot_slip_profile1( self, warm_up = 0, vmin = -12, vmax = 0 ):

        ox = self.read_ox_1()

        warm_up *= self.t_yr
        
        ox = ox[ox.t>warm_up]
        
        z_unique = ox.z.unique() 
        y_unique = ox.y.unique() 
        x_unique = np.zeros_like(z_unique)

        s_unique = ox.step.unique() 
        
        Nz = z_unique.size 
        Ns = s_unique.size
        
        slip_0 = ox.slip.iloc[0:Nz].to_numpy()
        
        t = ox.t.to_numpy().reshape(Ns,Nz)
        v = ox.v.to_numpy().reshape(Ns,Nz) 
        slip = ox.slip.to_numpy().reshape(Ns,Nz) - slip_0
        z = ox.z.to_numpy().reshape(Ns,Nz)



        from matplotlib.colors import LogNorm

        fig, ax= plt.subplots(1,1, figsize = (8,8))
        ax.set_ylabel('depth [km]')
        ax.set_xlabel('slip [m]')
        
        CS = ax.contourf( slip, z*1e-3, v,
                    levels=np.logspace(vmin,vmax,100), norm = LogNorm(),
                                     cmap="jet", vmin =10**vmin, vmax=10**vmax, extend = 'both')

        Lb = self.pars['G'] * self.pars['mesh']['dc'][0] / self.pars['mesh']['bb'][0] / self.pars['mesh']['sigma_ini'][0] 
        
        z_min = z_unique.min() 
        slip_max = slip.max()
        
        # ax.plot([slip_max-0.02, slip_max-0.02], [z_min*1e-3, (z_min+Lb)*1E-3], lw = 3, color = 'k')        

        # ax.text(slip_max-0.01, (z_min+Lb/2)*1E-3, 'Lb')        
        # ax.plot([slip_max-0.06, slip_max-0.06], [z_min*1e-3, (z_min+4*Lb)*1E-3], lw = 3, color = 'k')        
        # ax.text(slip_max-0.01, (z_min+Lb/2)*1E-3, 'Lb')    
        # ax.text(slip_max-0.06, (z_min+Lb*2)*1E-3, '4Lb')    
        
        CB = fig.colorbar(CS, orientation="vertical", ticks=[10**vmin, 10**((vmax+vmin)//2), 10**vmax],
                          location='right', shrink=0.3, pad = -0.1)

        
        fig.savefig(os.path.join(self.d, 'slip_profile1.jpg'), dpi=200,
                                      bbox_inches='tight',)



    def plot_slip_profile2( self, warm_up = 0, vmin = -12, vmax = 0):

        ox = self.read_ox_1()

        warm_up *= self.t_yr
        
        ox = ox[ox.t>warm_up]
        
        z_unique = ox.z.unique() 
        y_unique = ox.y.unique() 
        x_unique = np.zeros_like(z_unique)
        
        mask = ox['step'] == ox['step'].max()
        ox = ox[~mask]

        s_unique = ox.step.unique() 
        
        V_dyn = 2* np.max((self.pars['mesh']['aa']-self.pars['mesh']['bb'])) * self.pars['mesh']['sigma_ini'].min() * self.pars['c_s']/ self.pars['G']
        print(V_dyn)
        
        Nz = z_unique.size 
        Ns = s_unique.size
        
        slip_0 = ox.slip.iloc[0:Nz].to_numpy()
        
        t = ox.t.to_numpy().reshape(Ns,Nz)
        v = ox.v.to_numpy().reshape(Ns,Nz) 
        
        # The index where the bottom cell has the largest slip rate
        ind0 = v[:,0].argmax() +10000
        
        slip = ox.slip.to_numpy().reshape(Ns,Nz)[:ind0,:] - slip_0
        z = ox.z.to_numpy().reshape(Ns,Nz)[:ind0,:]
        CFF = ox.CFF.to_numpy().reshape(Ns,Nz)[:ind0,:]
        v = v[:ind0,:]
        t = t[:ind0,:]

        indzmax = CFF[-1,:].argmax()
        Ns = t.shape[0]

        
        
        from matplotlib.colors import LogNorm
        from mpl_toolkits.axes_grid1.inset_locator import inset_axes

        fig, (ax,ax1) = plt.subplots(1,2, figsize = (8,8), sharey = True,
                                     gridspec_kw={'width_ratios': [3, 1]})
        ax.set_ylabel('depth [km]')
        ax.set_xlabel('slip [m]')
        
        CS = ax.contourf( slip, z*1e-3, v,
                    levels=np.logspace(vmin,vmax,100), norm = LogNorm(),
                                     cmap="seismic", vmin =10**vmin, vmax=10**vmax, extend = 'both')
        axins1 = inset_axes(
            ax,
            width="4%",  # width: 50% of parent_bbox width
            height="40%",  # height: 5%
            loc=3,
            bbox_to_anchor=(0.5, 0., 0.5, 0.5),
            bbox_transform=ax.transAxes,
            borderpad=0.5,
        )
        
        axins1.xaxis.set_ticks_position("bottom")
        CB = fig.colorbar(CS, cax=axins1, orientation="vertical", ticks=[10**vmin, 10**((vmax+vmin)//2), 10**vmax])

        Lb = self.pars['G'] * self.pars['mesh']['dc'][0] / self.pars['mesh']['bb'][0] / self.pars['mesh']['sigma_ini'][0] 
        
        z_min = z_unique.min() 
        slip_max = slip.max()
        zz = z[0,indzmax]
        
        
        # ax.plot([slip_max/4, slip_max/4], [(zz - Lb/2)*1e-3, (zz + Lb/2)*1e-3], lw = 3, color = 'k')        
        # ax.text(slip_max/4, zz*1E-3, 'Lb')        
        # ax.plot([2*slip_max/4, 2*slip_max/4], [(zz - Lb*2)*1e-3, (zz + Lb*2)*1e-3], lw = 3, color = 'k')        
        # ax.text(slip_max*2/4, zz*1E-3, '4Lb')        

        # for i in range(1, CFF.shape[0]-1, 1000):
            
            # ax1.plot( CFF[i,:] * self.t_yr / 1000, z[-1,:] * 1e-3, color ='b', lw = 0.8)
            
        # ax1.plot( np.min(CFF, axis=0) * self.t_yr / 1000, z[-1,:] * 1e-3, color ='b', lw = 0.8)
        
        ax1.plot( np.mean(CFF, axis=0) * self.t_yr / 1000, z[-1,:] * 1e-3, color ='b', lw = 2)

        # ax1.plot( np.max(CFF, axis=0) * self.t_yr / 1000, z[-1,:] * 1e-3, color ='b', lw = 0.8)

        # ax1.plot( CFF[-1,:] * self.t_yr / 1000, z[-1,:] * 1e-3, color ='b', lw = 0.8)

        
        # ax1.plot([CFF.max()* self.t_yr / 1000, CFF.max()* self.t_yr / 1000], [(zz - Lb/2)*1e-3, (zz + Lb/2)*1e-3], lw = 3, color = 'k')        
        # ax1.plot([CFF.max()/2* self.t_yr / 1000, CFF.max()/2* self.t_yr / 1000], [(zz - Lb*2)*1e-3, (zz + Lb*2)*1e-3], lw = 3, color = 'k')        

        
        ax1.set_xlabel('$\\dot{\\tau_e}$ [kPa/y]')
        
        CB.ax.set_ylabel("slip rate\n[m/s]")
        
        ax2 = inset_axes(ax,
                    width="70%", # width = 30% of parent_bbox
                    height=2., # height : 1 inch
                    loc=5, borderpad = 2)
        
        

        
        # fig, ax= plt.subplots(1,1, figsize = (8,3))
        ax2.semilogy(t[:,0] / self.t_yr, np.max(v, axis = 1), ls='-', color = 'b')

        ax2.axhline(V_dyn, ls='--', color = 'r')
        
        ax2.set_xlabel('time [yr]')
        
        ax2.set_ylabel('max(v) [m/s]')
        
        # ax2.set_xlim(left=0)
        
        ax3 = ax2.twiny()
        ax3.semilogy((t[:,0] - self.pars['mass_removal']['t_onset']) / self.t_yr, np.max(v, axis = 1),
                     ls = '-', lw = 0.7)
        
        ax3.grid() # vertical lines
        
        ax3.set_xlabel('time since quarrying [yr]')

        
        fig.savefig(os.path.join(self.d, 'slip_profile.jpg'), dpi=200,
                                      bbox_inches='tight',)
        
        # fig.savefig(os.path.join(self.d, 'time_series2.jpg'), dpi=200,
        #                               bbox_inches='tight',)
