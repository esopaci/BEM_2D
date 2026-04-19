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
from scipy.interpolate import interp1d
from scipy.integrate import simpson
import traceback


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
    vc = 1e-2
    tc = 60
    def __init__(self, d):
    
        self.d = d # root directory to the simulation results
        self.snapshot_folder = f'{self.d}/snapshots'
        # output file for maximum slip rate
        self.otmax_file = os.path.join(self.d,  
                        [x for x in os.listdir(self.d) if x.endswith('Vmax.txt')][0] 
                                       )
        
        # output file for space and time. 
        ## Find the snapshot steps 
        snapshot_steps = [int(x[5:]) for x in os.listdir(self.snapshot_folder) if x.startswith('step')]
        
        self.snapshot_steps = np.sort(snapshot_steps)
        # self.ox_file = os.path.join(self.d, 
        #                 [x for x in os.listdir(self.d) if x.endswith('ox')][0] 
        #                             )
        
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




    # def read_ox_1( self ):
    #     '''
    #     This function reads the space-time output file. 
    #     If the output is a dense file, avoid using this function

    #     Returns
    #     -------
    #     df0 : TYPE
    #         DESCRIPTION.

    #     '''
        
                
                                    
    #     df0 = pd.read_csv(self.ox_file, sep='\s+',
    #                       header = None,
    #                         comment='#', 
    #                         )
    #     df0.columns = self.names_ox
    
    #     return df0
    
    
    # def read_ox_2(self, skip=0, n_snaphots = 1):
    #     '''
    #     This function reads the space-time output file as step by step. 
    #     If the output file is dense, reading this way is more appropriate. 

    #     Parameters
    #     ----------
    #     skip : integer, optional
    #         number of the snapshot. The default is 0.

    #     Returns
    #     -------
    #     df0 : TYPE
    #         DESCRIPTION.

    #     '''
                
    #     skip0 = int( skip * (self.NN_sample + 1)-1)
                                    
    #     df0 = pd.read_csv(self.ox_file, nrows=self.NN_sample * n_snaphots,
    #                       sep='\s+', encoding="utf-8", 
    #                       skiprows = skip0, comment = '#')
  
    #     df0.columns = self.names_ox

    #     return df0
    
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
        

    def plot_slip_profile( self, Vmin = -9, Vmax = 0):

        
        ## Get the slip isntances 
        # df_slip_inst = self.get_coseismic_instances()
        list_of_steps = self.snapshot_steps.copy()
        
        ox = pd.read_csv( f'{self.snapshot_folder}/step_0', sep = '\\s+')
        Nx = ox.shape[0]
        Nt = len(list_of_steps) - 1
        
        slip_array = np.zeros((Nt,Nx)) 
        V_array = np.zeros((Nt,Nx)) 
        Z_array = np.zeros((Nt,Nx))         
        
        # Loop over the files 
        for i,step in enumerate(list_of_steps[1:]):
            ox = pd.read_csv( f'{self.snapshot_folder}/step_{step}', sep = '\\s+')

            slip_array[i,:] = ox.Slip.to_numpy()
            V_array[i,:] = ox.V.to_numpy()
            Z_array[i,:] = ox.Z.to_numpy()
            
        from matplotlib.colors import LogNorm

        fig, ax= plt.subplots(1,1, figsize = (8,8))
        ax.set_ylabel('Depth [km]')
        ax.set_xlabel('Slip [m]')
        
        CS = ax.contourf( slip_array, -Z_array*1e-3, V_array,
                    levels=np.logspace(Vmin,Vmax,100), norm = LogNorm(),
                                     cmap="jet", vmin =10**Vmin, vmax=10**Vmax, extend = 'both')

        CB = fig.colorbar(CS, orientation="vertical", ticks=[10**Vmin, 10**((Vmax+Vmin)//2), 10**Vmax],
                          location='right', shrink=0.3, pad = 0.01, label = 'Slip rate [m/s]')
        
        
        fig.savefig(os.path.join(self.d, 'slip_profile.jpg'), dpi=200,
                                      bbox_inches='tight',
                                    )
        
        
        
    def extract_slip_info(self, V_dyn = 1e-1, G=32E9, L=15E9):
        '''
        This module reads PyQuake3D results recursively finds the slip events,
        then extract information about the slip event. 
        
        Returns
        -------
        None.

        '''        
        dt_crit = 1e3 # critical time to distinguish slip events

        df = self.read_vmaxfile()
                
        df1 = df[df.v>V_dyn]
        
        dtime = np.diff(df1.t.to_numpy(), prepend=0)
        dtime = np.append(dtime, dt_crit)

        ind_temp = np.argwhere(dtime>=dt_crit).flatten()
        Nevents = ind_temp.size
        
        i_steps = df.istep.to_numpy()
        
        
        event_string=f'{"Evnt":10}{"I_start":10}{"I_finish":10}{"Year":10}{"Day":10}{"Hour":10}{"Min":10}{"Duration":10}{"Nuc_z":10}{"Z_min":10}{"Z_max":10}{"slip_mean":10}{"slip_max":10}{"State_drop":16}{"Stress_drop":16}{"M0":16}{"Mw":16}\n'
        
        ## Loop over events
        for i in range(Nevents-1):
            
            print(f'event {i+1}')
            # Finding indcies of the slip events form maximum slip rate file
            ind1 = int(ind_temp[i])
            ind2 = int(ind_temp[i+1]) - 1  
            
            # Find the iteration step number 
            iter1 = df1.iloc[ind1].istep
            iter2 = df1.iloc[ind2].istep
            
            # earthquake_ind = df[(df.Iteration>=iter1) & (df.Iteration<=iter1)]['slip_v'].argmax()
            
            # Get the indices of the event
            iter_indices = ((i_steps>=iter1) & (i_steps<=iter2))
            step_min = i_steps[iter_indices].min() # Start index of the event
            step_max = i_steps[iter_indices].max() # Finish index of the event
            N_iter = i_steps[iter_indices].size
            
            # X_min = []
            # X_max = []
            # Y_min = []
            # Y_max = []
            Z_min = []
            Z_max = []
            
            
            MO_dot = np.empty(N_iter)
            time = np.empty(N_iter)
            # V_event = np.empty(N_iter)

            try:
                ## Loop during the event
                for ii in range(N_iter):  
                    step = i_steps[iter_indices][ii]
    
                    ox = pd.read_csv( f'{self.snapshot_folder}/step_{step}', sep = '\\s+')
                    V = ox.V.to_numpy()
                    Z = ox.Z.to_numpy()
                    dz = np.diff(Z)[0]
                    
                    # mesh = self.read_mesh(step = step)
    
                    # Get data
                    # cells = mesh.cells.reshape(-1, 4)   # 3 + node IDs for triangles
                    # triangles = cells[:, 1:]         # drop the "3"
                    # points = mesh.points[triangles]  # shape (n_cells, 3, 3)
                    # points = np.mean(points, axis = 1 )
                    
                    # mesh_with_areas = mesh.compute_cell_sizes(area=True, volume=False)
                    
                    # Seismic moment release rate
                                        
                    # Find the index of slip rate exceeds dynamic slip rate
                    ind_Vdyn = (V > V_dyn)
                                        
                    V = V[ind_Vdyn].copy()
                    Z = Z[ind_Vdyn].copy()   
                    # print(V.size, Z.size)
                    # print(V.max(), Z.max())
                    MO_dot[ii] = np.abs(np.sum(dz*dz*2*V*G))
                    time[ii] = ox.Time.iloc[0]


                    # X_min1 = points[ind_Vdyn,0].min()
                    # X_min.append(X_min1)
                    # X_max1 = points[ind_Vdyn,0].max()
                    # X_max.append(X_max1)
                    # Y_min1 = points[ind_Vdyn,1].min()
                    # Y_min.append(Y_min1)
                    # Y_max1 = points[ind_Vdyn,1].max()
                    # Y_max.append(Y_max1)
                    Z_min1 = ox.Z.min()
                    Z_min.append(Z_min1)
                    Z_max1 = ox.Z.max()
                    Z_max.append(Z_max1)
                    
                    if ii == 0:
                        # Nucleation Point
                        Nuc = (Z_min1+Z_max1)*0.5
                        
                        # beginning of slip
                        # Find the index of maximum state. At the end of the 
                        # rupture we will compare the stress drop
    
                        # max_state_ind = mesh.cell_data['state'].argmax()
                        slip_ini  = ox.Slip.to_numpy()
                        shear_ini = ox.Tau_f.to_numpy()
                        state_ini = ox.Theta.to_numpy()
                
                    elif ii == N_iter - 1 :
                        
                        # End of slip
                        slip_end  = ox.Slip.to_numpy()
                        shear_end = ox.Tau_f.to_numpy()
                        state_end = ox.Theta.to_numpy()
                        
                # Seismic moment info
                # Compute seismic moment and 
                M0 = np.abs(simpson(MO_dot, x=time))
                
                Mw = 2/3 * (np.log10(M0) - 9.1)
                # M0_dot_mean = 10**np.mean(np.log10(MO_dot))
                
                
                slip_max = (slip_end - slip_ini).max() 
                slip_mean = (slip_end - slip_ini).mean()                    
                state_drop = (state_end - state_ini).mean()
                stress_drop = (shear_end - shear_ini).mean() 
                
                Z_min = np.min(Z_min)
                Z_max = np.max(Z_max)
                
                
                # ttime = time[vv.argmax()]
                ttime0 = time[0]
                ttime1 = time[-1]
                t_year = ttime0 // self.t_yr 
                t_day  = (ttime0 / 3600 / 24 ) % 365
                t_hour = (ttime0 / 3600 ) % 24
                t_min  = (ttime0 / 60 ) % 60
                t_dur = ttime1 - ttime0
                
                event_string += f'{i:5.0f}{step_min:10.0f}{step_max:10.0f}{t_year:10.0f}{t_day:10.0f}{t_hour:10.0f}{t_min:10.2f}{t_dur:10.2f}{Nuc:10.1f}{Z_min:10.1f}{Z_max:10.1f}{slip_mean:10.3f}{slip_max:10.3f}{state_drop:16.6E}{stress_drop:16.6E}{M0:16.6E}{Mw:16.3f}\n'

                
            except Exception as e:
                print(e)
                print(traceback.print_exc()) # Prints the full traceback to stderr

                pass
            
        with open(os.path.join(self.d, "events.txt"), "w") as file:
            file.write(event_string)

    # def plot_slip_profile1( self, warm_up = 0, vmin = -12, vmax = 0 ):

    #     ox = self.read_ox_1()

    #     warm_up *= self.t_yr
        
    #     ox = ox[ox.t>warm_up]
        
    #     z_unique = ox.z.unique() 
    #     y_unique = ox.y.unique() 
    #     x_unique = np.zeros_like(z_unique)

    #     s_unique = ox.step.unique() 
        
    #     Nz = z_unique.size 
    #     Ns = s_unique.size
        
    #     slip_0 = ox.slip.iloc[0:Nz].to_numpy()
        
    #     t = ox.t.to_numpy().reshape(Ns,Nz)
    #     v = ox.v.to_numpy().reshape(Ns,Nz) 
    #     slip = ox.slip.to_numpy().reshape(Ns,Nz) - slip_0
    #     z = ox.z.to_numpy().reshape(Ns,Nz)



    #     from matplotlib.colors import LogNorm

    #     fig, ax= plt.subplots(1,1, figsize = (8,8))
    #     ax.set_ylabel('depth [km]')
    #     ax.set_xlabel('slip [m]')
        
    #     CS = ax.contourf( slip, z*1e-3, v,
    #                 levels=np.logspace(vmin,vmax,100), norm = LogNorm(),
    #                                  cmap="jet", vmin =10**vmin, vmax=10**vmax, extend = 'both')

    #     Lb = self.pars['G'] * self.pars['mesh']['dc'][0] / self.pars['mesh']['bb'][0] / self.pars['mesh']['sigma_ini'][0] 
        
    #     z_min = z_unique.min() 
    #     slip_max = slip.max()
        
    #     # ax.plot([slip_max-0.02, slip_max-0.02], [z_min*1e-3, (z_min+Lb)*1E-3], lw = 3, color = 'k')        

    #     # ax.text(slip_max-0.01, (z_min+Lb/2)*1E-3, 'Lb')        
    #     # ax.plot([slip_max-0.06, slip_max-0.06], [z_min*1e-3, (z_min+4*Lb)*1E-3], lw = 3, color = 'k')        
    #     # ax.text(slip_max-0.01, (z_min+Lb/2)*1E-3, 'Lb')    
    #     # ax.text(slip_max-0.06, (z_min+Lb*2)*1E-3, '4Lb')    
        
    #     CB = fig.colorbar(CS, orientation="vertical", ticks=[10**vmin, 10**((vmax+vmin)//2), 10**vmax],
    #                       location='right', shrink=0.3, pad = -0.1)


        
    #     fig.savefig(os.path.join(self.d, 'slip_profile1.jpg'), dpi=200,
    #                                   bbox_inches='tight',)



    # def plot_slip_profile2( self, warm_up = 0, vmin = -12, vmax = 0):

    #     ox = self.read_ox_1()

    #     warm_up *= self.t_yr
        
    #     ox = ox[ox.t>warm_up]
        
    #     z_unique = ox.z.unique() 
    #     y_unique = ox.y.unique() 
    #     x_unique = np.zeros_like(z_unique)
        
    #     mask = ox['step'] == ox['step'].max()
    #     ox = ox[~mask]

    #     s_unique = ox.step.unique() 
        
    #     V_dyn = 2* np.max((self.pars['mesh']['aa']-self.pars['mesh']['bb'])) * self.pars['mesh']['sigma_ini'].min() * self.pars['c_s']/ self.pars['G']
    #     print(V_dyn)
        
    #     Nz = z_unique.size 
    #     Ns = s_unique.size
        
    #     slip_0 = ox.slip.iloc[0:Nz].to_numpy()
        
    #     t = ox.t.to_numpy().reshape(Ns,Nz)
    #     v = ox.v.to_numpy().reshape(Ns,Nz) 
        
    #     # The index where the bottom cell has the largest slip rate
    #     ind0 = v[:,0].argmax() +10000
        
    #     slip = ox.slip.to_numpy().reshape(Ns,Nz)[:ind0,:] - slip_0
    #     z = ox.z.to_numpy().reshape(Ns,Nz)[:ind0,:]
    #     CFF = ox.CFF.to_numpy().reshape(Ns,Nz)[:ind0,:]
    #     v = v[:ind0,:]
    #     t = t[:ind0,:]

    #     indzmax = CFF[-1,:].argmax()
    #     Ns = t.shape[0]

        
        
    #     from matplotlib.colors import LogNorm
    #     from mpl_toolkits.axes_grid1.inset_locator import inset_axes

    #     fig, (ax,ax1) = plt.subplots(1,2, figsize = (8,8), sharey = True,
    #                                  gridspec_kw={'width_ratios': [3, 1]})
    #     ax.set_ylabel('depth [km]')
    #     ax.set_xlabel('slip [m]')
        
    #     CS = ax.contourf( slip, z*1e-3, v,
    #                 levels=np.logspace(vmin,vmax,100), norm = LogNorm(),
    #                                  cmap="seismic", vmin =10**vmin, vmax=10**vmax, extend = 'both')
    #     axins1 = inset_axes(
    #         ax,
    #         width="4%",  # width: 50% of parent_bbox width
    #         height="40%",  # height: 5%
    #         loc=3,
    #         bbox_to_anchor=(0.5, 0., 0.5, 0.5),
    #         bbox_transform=ax.transAxes,
    #         borderpad=0.5,
    #     )
        
    #     axins1.xaxis.set_ticks_position("bottom")
    #     CB = fig.colorbar(CS, cax=axins1, orientation="vertical", ticks=[10**vmin, 10**((vmax+vmin)//2), 10**vmax])

    #     Lb = self.pars['G'] * self.pars['mesh']['dc'][0] / self.pars['mesh']['bb'][0] / self.pars['mesh']['sigma_ini'][0] 
        
    #     z_min = z_unique.min() 
    #     slip_max = slip.max()
    #     zz = z[0,indzmax]
        
        
    #     # ax.plot([slip_max/4, slip_max/4], [(zz - Lb/2)*1e-3, (zz + Lb/2)*1e-3], lw = 3, color = 'k')        
    #     # ax.text(slip_max/4, zz*1E-3, 'Lb')        
    #     # ax.plot([2*slip_max/4, 2*slip_max/4], [(zz - Lb*2)*1e-3, (zz + Lb*2)*1e-3], lw = 3, color = 'k')        
    #     # ax.text(slip_max*2/4, zz*1E-3, '4Lb')        

    #     # for i in range(1, CFF.shape[0]-1, 1000):
            
    #         # ax1.plot( CFF[i,:] * self.t_yr / 1000, z[-1,:] * 1e-3, color ='b', lw = 0.8)
            
    #     # ax1.plot( np.min(CFF, axis=0) * self.t_yr / 1000, z[-1,:] * 1e-3, color ='b', lw = 0.8)
        
    #     ax1.plot( np.mean(CFF, axis=0) * self.t_yr / 1000, z[-1,:] * 1e-3, color ='b', lw = 2)

    #     # ax1.plot( np.max(CFF, axis=0) * self.t_yr / 1000, z[-1,:] * 1e-3, color ='b', lw = 0.8)

    #     # ax1.plot( CFF[-1,:] * self.t_yr / 1000, z[-1,:] * 1e-3, color ='b', lw = 0.8)

        
    #     # ax1.plot([CFF.max()* self.t_yr / 1000, CFF.max()* self.t_yr / 1000], [(zz - Lb/2)*1e-3, (zz + Lb/2)*1e-3], lw = 3, color = 'k')        
    #     # ax1.plot([CFF.max()/2* self.t_yr / 1000, CFF.max()/2* self.t_yr / 1000], [(zz - Lb*2)*1e-3, (zz + Lb*2)*1e-3], lw = 3, color = 'k')        

        
    #     ax1.set_xlabel('$\\dot{\\tau_e}$ [kPa/y]')
        
    #     CB.ax.set_ylabel("slip rate\n[m/s]")
        
    #     ax2 = inset_axes(ax,
    #                 width="70%", # width = 30% of parent_bbox
    #                 height=2., # height : 1 inch
    #                 loc=5, borderpad = 2)
        
        

        
    #     # fig, ax= plt.subplots(1,1, figsize = (8,3))
    #     ax2.semilogy(t[:,0] / self.t_yr, np.max(v, axis = 1), ls='-', color = 'b')

    #     ax2.axhline(V_dyn, ls='--', color = 'r')
        
    #     ax2.set_xlabel('time [yr]')
        
    #     ax2.set_ylabel('max(v) [m/s]')
        
    #     # ax2.set_xlim(left=0)
        
    #     ax3 = ax2.twiny()
    #     ax3.semilogy((t[:,0] - self.pars['mass_removal']['t_onset']) / self.t_yr, np.max(v, axis = 1),
    #                  ls = '-', lw = 0.7)
        
    #     ax3.grid() # vertical lines
        
    #     ax3.set_xlabel('time since quarrying [yr]')

        
    #     fig.savefig(os.path.join(self.d, 'slip_profile.jpg'), dpi=200,
    #                                   bbox_inches='tight',)
        
    #     # fig.savefig(os.path.join(self.d, 'time_series2.jpg'), dpi=200,
    #     #                               bbox_inches='tight',)
