# -*- coding: utf-8 -*-
"""
Created on Tue Nov 16 12:47:53 2021

@author: XYZ
"""

import numpy as np
from scipy import signal
import cupy
import cupy as cp
import cupyx.scipy.signal
import multiprocessing
import functools
import time
        
def single_path_2d_generator(diffusion_const,dt,nSteps):
    t=dt*np.arange(nSteps)
    dx=np.sqrt(2*diffusion_const*dt)*np.random.normal(loc=0,scale=1,size=nSteps)
    dy=np.sqrt(2*diffusion_const*dt)*np.random.normal(loc=0,scale=1,size=nSteps)
    x=np.array([np.sum(dx[0:nStep]) for nStep in range(len(dx))])
    y=np.array([np.sum(dy[0:nStep]) for nStep in range(len(dy))])
    return t,x,y,dx,dy

def multiple_paths_2d_generateor(diffusion_const,dt,nSteps,nParticles):
    print("It's creating "+str(nParticles)+" brownian paths...")
    
    tic=time.time()
    xx=np.zeros((nSteps,nParticles))
    yy=np.zeros((nSteps,nParticles))
    for nParticle in range(0,nParticles):
        t,x,y,_,_=single_path_2d_generator(diffusion_const,dt,nSteps)
        xx[:,nParticle]=x
        yy[:,nParticle]=y
    print("--- %s seconds ---" % (time.time()-tic))
    return t,xx,yy

def dr_square_calc(x,y):
    return np.array([(x[nStep]-x[0])**2+(y[nStep]-y[0])**2 for nStep in range(0,len(x))])

def eMSD_calc(t,xx,yy,calc_mode):
    nSteps,nParticles=xx.shape
    eMSD=[]
    tau=t[1:nSteps]                                                             # lag time
    if calc_mode=='for-loop':
        print("Using for-loop to calculate ensemble MSD.")
        
        tic=time.time()
        eMSD=np.zeros(nSteps)
        for nParticle in range(0,nParticles):
            x=xx[:,nParticle]
            y=yy[:,nParticle]
            eMSD=eMSD+dr_square_calc(x,y)
        eMSD=eMSD/nParticles
        eMSD=np.delete(eMSD,0)
        print("--- %s seconds ---" % (time.time()-tic))
        
    elif calc_mode=='vectorization':
        print("Using vectorization to calculate ensemble MSD.")
        
        tic=time.time()
        # calculate x1-x0, x2-x1, x3-x2, and so on
        kernal=np.zeros((nSteps,nParticles))
        kernal[0,:]=1
        kernal[1,:]=-1
        dx=signal.fftconvolve(xx,kernal,mode='full',axes=0)
        dx=dx[1:nSteps,:]
        dy=signal.fftconvolve(yy,kernal,mode='full',axes=0)
        dy=dy[1:nSteps,:]
        
        # caculate SD, then take ensemble average
        kernel=np.tri(nSteps-1,nSteps-1).T
        dx_square=np.square(np.matmul(dx.T,kernel))
        dy_square=np.square(np.matmul(dy.T,kernel))
        dr_square=dx_square+dy_square
        eMSD=np.mean(dr_square,axis=0)
        print("--- %s seconds ---" % (time.time()-tic))
        
    else:
        print("The ensemble MSD can be calculated by 'for-loop' and 'vectorization'.")
    return tau,eMSD


def time_averaged_eMSD_calc_pool(nTimeStep,xx,yy):                              # nTimeStep is a length of lag-time
    nTimeSteps,nParticles=xx.shape
    kernel=np.zeros((nTimeSteps,nParticles))
    kernel[0,:]=1
    kernel[nTimeStep,:]=-1
    dx_square=np.square(signal.fftconvolve(xx,kernel,mode='full',axes=0))
    dy_square=np.square(signal.fftconvolve(yy,kernel,mode='full',axes=0))
    dr_square=dx_square+dy_square
    time_averaged_eMSD_=np.mean(dr_square[nTimeStep:nTimeSteps,:])
    return time_averaged_eMSD_

def time_averaged_eMSD_calc(t,xx,yy,calc_mode):
    nSteps,nParticles=xx.shape
    tau=t[1:nSteps]                                                             # lag-time
    
    if calc_mode=='for-loop':
        print("Using for-loop to calculate time-averaged ensemble MSD.")
        
        tic=time.time()
        nTimeSteps=nSteps-1
        time_averaged_eMSD=np.zeros(nTimeSteps)
        for nParticle in range(0,nParticles):
            x=xx[:,nParticle]
            y=yy[:,nParticle]
            for nTimeStep in range(1,nTimeSteps+1):
                dr_square=0
                for nStep in range(0,nSteps-nTimeStep):
                    dr_square=dr_square+(x[nStep+nTimeStep]-x[nStep])**2+(y[nStep+nTimeStep]-y[nStep])**2
                time_averaged_eMSD[nTimeStep-1]=time_averaged_eMSD[nTimeStep-1]+dr_square/(nSteps-nTimeStep)
        time_averaged_eMSD=time_averaged_eMSD/nParticles
        print("--- %s seconds ---" % (time.time()-tic))
        
    elif calc_mode=='vectorization':
        print("Using vectorization to calculate time-averaged ensemble MSD.")
        
        tic=time.time()
        nTimeSteps=nSteps-1
        time_averaged_eMSD=np.zeros(nTimeSteps)
        for nStep in range(1,nSteps):
            kernel=np.zeros(xx.shape)
            kernel[0,:]=1
            kernel[nStep,:]=-1
            dx_square=np.square(signal.fftconvolve(xx,kernel,mode='full',axes=0))
            dy_square=np.square(signal.fftconvolve(yy,kernel,mode='full',axes=0))
            dr_square=dx_square+dy_square
            time_averaged_eMSD[nStep-1]=np.mean(dr_square[nStep:nSteps,:])
        print("--- %s seconds ---" % (time.time()-tic))
        
    elif calc_mode=='parallel':
        print("Using vectorization and parallel computing to calculate time-averaged ensemble MSD.")
        print("The number of available processers is "+str(multiprocessing.cpu_count())+".")
        
        tic = time.time()
        pool = multiprocessing.Pool()
        time_averaged_eMSD_=pool.map(functools.partial(time_averaged_eMSD_calc_pool,xx=xx,yy=yy),range(1,nSteps))
        pool.close()
        pool.join()
        time_averaged_eMSD=np.array(time_averaged_eMSD_)
        print("--- %s seconds ---" % (time.time()-tic))
        
    elif calc_mode=='gpu':
        print("Using vectorization and cuda to calculate time-averaged ensemble MSD.")
        
        mempool=cupy.get_default_memory_pool()
        
        tic=time.time()
        SizeOfBatch=4                                                           # next version needs to automatically define batch size according to available ram 
        nBatchs=int(nParticles/SizeOfBatch)
        print("The batch size of particles is "+str(SizeOfBatch)+'. (if necessary, please go to MSD.py to modify the variable "SizeOfBatch".)')
        time_averaged_eMSD_=cp.zeros((nSteps-1,nBatchs))
        
        for nBatch in range(0,nBatchs):
            xx_=xx[:,0+SizeOfBatch*(nBatch):SizeOfBatch*(nBatch+1)]
            yy_=yy[:,0+SizeOfBatch*(nBatch):SizeOfBatch*(nBatch+1)]
            
            kernel_=cp.zeros((nSteps-1,nSteps))
            kernel_[:,0]=1
            kernel_[0:nSteps,1:nSteps+1]=-cp.identity(nSteps-1)
            kernel_=kernel_.reshape((nSteps-1,nSteps,1))
            kernel=cp.tile(kernel_,(1,1,SizeOfBatch))
            rr_=cp.tile(xx_,(nSteps-1,1,1))
            dr_square=cp.square(cupyx.scipy.signal.fftconvolve(rr_,kernel,mode='full',axes=1))
            rr_=cp.tile(yy_,(nSteps-1,1,1))
            dr_square=dr_square+cp.square(cupyx.scipy.signal.fftconvolve(rr_,kernel,mode='full',axes=1))
            kernel_=cp.zeros((nSteps-1,2*nSteps-1))
            kernel_[0::,1:nSteps]=cp.tri(nSteps-1,nSteps-1).T
            kernel_=kernel_.reshape((nSteps-1,2*nSteps-1,1))
            kernel=cp.tile(kernel_,(1,1,SizeOfBatch))
            dr_square=dr_square*kernel
            time_averaged_eMSD_[:,nBatch]=cp.sum(dr_square,axis=(1,2))/cp.sum(kernel,axis=(1,2))
            
        time_averaged_eMSD_=cp.mean(time_averaged_eMSD_,axis=1)
        time_averaged_eMSD=cp.asnumpy(time_averaged_eMSD_)
        print("--- %s seconds ---" % (time.time()-tic))
        
        mempool.free_all_blocks()
        
    else:
        print("The ensemble MSD can be calculated by 'for-loop', 'vectorization', 'parallel', and 'gpu'.")
    return tau,time_averaged_eMSD

def MSD_fitting(tau,MSD,fitting_mode):
    if fitting_mode=='MSD_fit':
        slope,_,_,_=np.linalg.lstsq(tau[0::,np.newaxis],MSD)                    # enforce fitting line passes through origin point
        MSD_f=slope*tau
        return slope, MSD_f
    elif fitting_mode=='loglog_fit':
        p=np.polyfit(np.log(tau), np.log(MSD), 1)
        MSD_f=np.exp(p[1])*np.power(tau,p[0])
        return p[0], MSD_f
    else:
        print("'MSD_fit' and 'loglog_fit' can be selected.")
        
            