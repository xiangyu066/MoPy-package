# -*- coding: utf-8 -*-
"""
Created on Tue Nov 16 22:23:25 2021

@author: XYZ
"""

#%%
print("Running...")
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.simplefilter("ignore")

# 
import MoPy.MSD as msd
import MoPy as mo
print("MoPy package is "+mo.__version__+".")

# define units
um=1
sec=1

#%%
print("------------------------------------------------------------------")
diffusion_const     =67*(um**2/sec)
dt                  =0.1*(sec)
nSteps              =512
nParticles          =1024

print("The default diffusion constant: "+str(diffusion_const)+" [um^2/sec].")
print("The timestep: "+str(dt)+" [sec].")
print("The total simulated steps: "+str(nSteps)+".")
print("The total particles: "+str(nParticles)+".")

# initialization
t,xx,yy=msd.multiple_paths_2d_generateor(diffusion_const,dt,nSteps,nParticles)

#%%
print("------------------------------------------------------------------")
if __name__ == '__main__':
    
    # calculate eMSD by for-loop
    tau,eMSD_for=msd.eMSD_calc(t,xx,yy,'for-loop')
    slope,eMSD_for_f=msd.MSD_fitting(tau,eMSD_for,'MSD_fit')
    polyorder,eMSD_for_f_log=msd.MSD_fitting(tau,eMSD_for,'loglog_fit')
    
    fig,axes=plt.subplots(1,2,figsize=(8,6))
    axes[0].plot(tau,eMSD_for,label="data(nParticles="+str(nParticles)+")")
    axes[0].plot(tau,eMSD_for_f,'r--',lw=1,label='fitting')
    axes[0].grid(linewidth=0.3)
    axes[0].legend()
    axes[0].set_xlabel('$\Delta t$ [sec]',fontweight='bold')
    axes[0].set_ylabel('eMSD [$\mu m^2$]',fontweight='bold')

    axes[1].plot(tau,eMSD_for,Label='Simulation')
    axes[1].plot(tau,eMSD_for_f_log,'--',Label='fitting')
    axes[1].set_xlabel('$\Delta t$ [sec]',fontweight='bold')
    axes[1].grid(lw=0.3)
    axes[1].set_yscale('log')
    axes[1].set_xscale('log')

    print("The diffusion constant is equal to "+str(np.around(slope[0]/4,2))+" [um^2/sec].")
    print('The slope of loglog plot is propotional to '+str(polyorder)+'.')
    print("------------------------------------------------------------------")

    # calculate eMSD by vectorization
    tau,eMSD_vect=msd.eMSD_calc(t,xx,yy,'vectorization')
    slope,eMSD_vect_f=msd.MSD_fitting(tau,eMSD_vect,'MSD_fit')
    polyorder,eMSD_vect_f_log=msd.MSD_fitting(tau,eMSD_vect,'loglog_fit')
    
    fig,axes=plt.subplots(1,2,figsize=(8,6))
    axes[0].plot(tau,eMSD_vect,label="data(nParticles="+str(nParticles)+")")
    axes[0].plot(tau,eMSD_vect_f,'r--',lw=1,label='fitting')
    axes[0].grid(linewidth=0.3)
    axes[0].legend()
    axes[0].set_xlabel('$\Delta t$ [sec]',fontweight='bold')
    axes[0].set_ylabel('eMSD [$\mu m^2$]',fontweight='bold')

    axes[1].plot(tau,eMSD_vect,Label='Simulation')
    axes[1].plot(tau,eMSD_vect_f_log,'--',Label='fitting')
    axes[1].set_xlabel('$\Delta t$ [sec]',fontweight='bold')
    axes[1].grid(lw=0.3)
    axes[1].set_yscale('log')
    axes[1].set_xscale('log')

    print("The diffusion constant is equal to "+str(np.around(slope[0]/4,2))+" [um^2/sec].")
    print('The slope of loglog plot is propotional to '+str(polyorder)+'.')
    print("------------------------------------------------------------------")

    # calculate time-averaged eMSD by for-loop
    tau,time_eMSD_for=msd.time_averaged_eMSD_calc(t,xx,yy,'for-loop')
    slope,time_eMSD_for_f=msd.MSD_fitting(tau,time_eMSD_for,'MSD_fit')
    polyorder,time_eMSD_for_f_log=msd.MSD_fitting(tau,time_eMSD_for,'loglog_fit')
    
    fig,axes=plt.subplots(1,2,figsize=(8,6))
    axes[0].plot(tau,time_eMSD_for,label="data(nParticles="+str(nParticles)+")")
    axes[0].plot(tau,time_eMSD_for_f,'r--',lw=1,label='fitting')
    axes[0].grid(linewidth=0.3)
    axes[0].legend()
    axes[0].set_xlabel('$\Delta t$ [sec]',fontweight='bold')
    axes[0].set_ylabel('time-averaged eMSD [$\mu m^2$]',fontweight='bold')

    axes[1].plot(tau,time_eMSD_for,Label='Simulation')
    axes[1].plot(tau,time_eMSD_for_f_log,'--',Label='fitting')
    axes[1].set_xlabel('$\Delta t$ [sec]',fontweight='bold')
    axes[1].grid(lw=0.3)
    axes[1].set_yscale('log')
    axes[1].set_xscale('log')

    print("The diffusion constant is equal to "+str(np.around(slope[0]/4,2))+" [um^2/sec].")
    print('The slope of loglog plot is propotional to '+str(polyorder)+'.')
    print("------------------------------------------------------------------")

    # calculate time-averaged eMSD by vectorization
    tau,time_eMSD_vect=msd.time_averaged_eMSD_calc(t,xx,yy,'vectorization')
    slope,time_eMSD_vect_f=msd.MSD_fitting(tau,time_eMSD_vect,'MSD_fit')
    polyorder,time_eMSD_vect_f_log=msd.MSD_fitting(tau,time_eMSD_vect,'loglog_fit')
    
    fig,axes=plt.subplots(1,2,figsize=(8,6))
    axes[0].plot(tau,time_eMSD_vect,label="data(nParticles="+str(nParticles)+")")
    axes[0].plot(tau,time_eMSD_vect_f,'r--',lw=1,label='fitting')
    axes[0].grid(linewidth=0.3)
    axes[0].legend()
    axes[0].set_xlabel('$\Delta t$ [sec]',fontweight='bold')
    axes[0].set_ylabel('time-averaged eMSD [$\mu m^2$]',fontweight='bold')

    axes[1].plot(tau,time_eMSD_vect,Label='Simulation')
    axes[1].plot(tau,time_eMSD_vect_f_log,'--',Label='fitting')
    axes[1].set_xlabel('$\Delta t$ [sec]',fontweight='bold')
    axes[1].grid(lw=0.3)
    axes[1].set_yscale('log')
    axes[1].set_xscale('log')

    print("The diffusion constant is equal to "+str(np.around(slope[0]/4,2))+" [um^2/sec].")
    print('The slope of loglog plot is propotional to '+str(polyorder)+'.')
    print("------------------------------------------------------------------")
    
    # calculate time-averaged eMSD by parallel computing
    tau,time_eMSD_parallel=msd.time_averaged_eMSD_calc(t,xx,yy,'parallel')
    slope,time_eMSD_parallel_f=msd.MSD_fitting(tau,time_eMSD_parallel,'MSD_fit')
    polyorder,time_eMSD_parallel_f_log=msd.MSD_fitting(tau,time_eMSD_parallel,'loglog_fit')
    
    fig,axes=plt.subplots(1,2,figsize=(8,6))
    axes[0].plot(tau,time_eMSD_parallel,label="data(nParticles="+str(nParticles)+")")
    axes[0].plot(tau,time_eMSD_parallel_f,'r--',lw=1,label='fitting')
    axes[0].grid(linewidth=0.3)
    axes[0].legend()
    axes[0].set_xlabel('$\Delta t$ [sec]',fontweight='bold')
    axes[0].set_ylabel('time-averaged eMSD [$\mu m^2$]',fontweight='bold')

    axes[1].plot(tau,time_eMSD_parallel,Label='Simulation')
    axes[1].plot(tau,time_eMSD_parallel_f_log,'--',Label='fitting')
    axes[1].set_xlabel('$\Delta t$ [sec]',fontweight='bold')
    axes[1].grid(lw=0.3)
    axes[1].set_yscale('log')
    axes[1].set_xscale('log')

    print("The diffusion constant is equal to "+str(np.around(slope[0]/4,2))+" [um^2/sec].")
    print('The slope of loglog plot is propotional to '+str(polyorder)+'.')
    print("------------------------------------------------------------------")

    # calculate time-averaged eMSD by gpu
    tau,time_eMSD_gpu=msd.time_averaged_eMSD_calc(t,xx,yy,'gpu')
    slope,time_eMSD_gpu_f=msd.MSD_fitting(tau,time_eMSD_gpu,'MSD_fit')
    polyorder,time_eMSD_gpu_f_log=msd.MSD_fitting(tau,time_eMSD_gpu,'loglog_fit')
    
    fig,axes=plt.subplots(1,2,figsize=(8,6))
    axes[0].plot(tau,time_eMSD_gpu,label="data(nParticles="+str(nParticles)+")")
    axes[0].plot(tau,time_eMSD_gpu_f,'r--',lw=1,label='fitting')
    axes[0].grid(linewidth=0.3)
    axes[0].legend()
    axes[0].set_xlabel('$\Delta t$ [sec]',fontweight='bold')
    axes[0].set_ylabel('time-averaged eMSD [$\mu m^2$]',fontweight='bold')

    axes[1].plot(tau,time_eMSD_gpu,Label='Simulation')
    axes[1].plot(tau,time_eMSD_gpu_f_log,'--',Label='fitting')
    axes[1].set_xlabel('$\Delta t$ [sec]',fontweight='bold')
    axes[1].grid(lw=0.3)
    axes[1].set_yscale('log')
    axes[1].set_xscale('log')

    print("The diffusion constant is equal to "+str(np.around(slope[0]/4,2))+" [um^2/sec].")
    print('The slope of loglog plot is propotional to '+str(polyorder)+'.')
    print("------------------------------------------------------------------")


#%%
print("------------------------------------------------------------------")
print("Done.")
