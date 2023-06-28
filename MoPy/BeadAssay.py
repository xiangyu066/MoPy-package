# -*- coding: utf-8 -*-
"""
Created on Wed Nov 17 17:18:38 2021

@author: XYZ
"""

import numpy as np
from scipy import optimize

try:
    import cupy
    import cupy as cp
except ModuleNotFoundError:
    print("Module 'cupy' is not installed.")

import multiprocessing
import functools
import time
import matplotlib.pyplot as plt

def pixelFFT(dataset,dt,modulate_amp,isGPU):
    nFrames,height,width=dataset.shape
    if isGPU==False:
        print("Using pixelFFT to fast calculate rotation speed in CPU mode.")
        
        tic=time.time()
        F_=2*np.abs(np.fft.fft(dataset-np.mean(dataset,0),axis=0))/nFrames
        F=F_[0:int(nFrames/2),:,:]
        ff_=np.fft.fftfreq(nFrames,d=dt)
        ff=np.tile(np.reshape(ff_[0:int(nFrames/2)],(-1,1,1)),(height,width))
        mask=(F==np.max(F,0))&(F>modulate_amp)
        mapping=np.sum(ff*mask,0)
        print("--- %s seconds ---" % (time.time()-tic))
    else:
        print("Using pixelFFT to fast calculate rotation speed in GPU mode.")
        
        tic=time.time()
        
        print("--- %s seconds ---" % (time.time()-tic))
    return mapping

def Gaussian2D(xx,yy,amp,xc,yc,sigma_x,sigma_y,offset): 
    theta=0
    a=(np.cos(theta)**2)/(2*sigma_x**2)+(np.sin(theta)**2)/(2*sigma_y**2)
    b=-(np.sin(2*theta))/(4*sigma_x**2)+(np.sin(2*theta))/(4*sigma_y**2)
    c=(np.sin(theta)**2)/(2*sigma_x**2)+(np.cos(theta)**2)/(2*sigma_y**2)
    g=offset+amp*np.exp(-(a*((xx-xc)**2)+2*b*(xx-xc)*(yy-yc)+c*((yy-yc)**2)))
    return g

def _Gaussian2D(Mat,*args):
    xx_,yy_=Mat
    zz_=Gaussian2D(xx_,yy_,*args)
    return zz_

def _Gaussian2Ds(M,*args):
    x,y=M
    arr=np.zeros(x.shape)
    for i in range(len(args)//6):
       arr+=Gaussian2D(x,y,*args[i*6:i*6+6])
    return arr

def Cauchy2D(xx,yy,amplitude,xc,yc,sigma_x,sigma_y,offset):
    xc=float(xc)
    yc=float(yc)
    a=amplitude
    b=sigma_x*sigma_y
    c=1/(sigma_x**2+(xx-xc)**2)
    d=1/(sigma_y**2+(yy-yc)**2)
    f=a*b*c*d+offset
    return f

def _Cauchy2Ds(M,*args):
    x,y=M
    arr=np.zeros(x.shape)
    for i in range(len(args)//6):
        arr+=Cauchy2D(x,y,*args[i*6:i*6+6])
    return arr

def Gaussian2DFit(nFrame,dataset,SizeOfBead,bead_type,eff_pixelsize):
    _,height,width=dataset.shape
    xx,yy=np.meshgrid(np.linspace(0,width-1,width),np.linspace(0,height-1,height) )
    xdata=np.vstack((xx.ravel(),yy.ravel()))
    
    # boundary condition
    if bead_type=='dark':
        bounds=((-np.inf,0,0,0,0,0),
                (0,width,height,(width/2)**2,(height/2)**2,np.inf))
    else:
        bounds=(0,(np.inf,width,height,(width/2)**2,(height/2)**2,np.inf))
 
    zz=dataset[nFrame,:,:]
    ydata=zz.ravel()
    
    # initial condition
    if bead_type=='dark':
        p0=(np.min(zz)-np.mean(zz),round(width/2),round(height/2),SizeOfBead/eff_pixelsize,SizeOfBead/eff_pixelsize,np.mean(zz))
    else:
        p0=(np.max(zz)-np.mean(zz),round(width/2),round(height/2),SizeOfBead/eff_pixelsize,SizeOfBead/eff_pixelsize,np.mean(zz))
    
    # fitting
    try:
        popt,_=optimize.curve_fit(_Gaussian2D,xdata,ydata,p0=p0,bounds=bounds)
    except RuntimeError:
        popt=np.array([np.nan,np.nan,np.nan,np.nan,np.nan,np.nan])   
    return popt    

def Cauchy2DFit(nFrame,dataset,SizeOfBead,bead_type,eff_pixelsize):
    _,height,width=dataset.shape
    xx,yy=np.meshgrid(np.linspace(0,width-1,width),np.linspace(0,height-1,height) )
    xdata=np.vstack((xx.ravel(),yy.ravel()))

    zz=dataset[nFrame,:,:]
    ydata=zz.ravel()
    
    p0=[np.max(zz)-np.mean(zz),round(width/2),round(height/2),SizeOfBead/eff_pixelsize/2,SizeOfBead/eff_pixelsize/2,np.mean(zz)]
    
    # boundary condition
    if bead_type=='dark':
        bounds=((-np.inf,0,0,0,0,0),
                (0,width,height,(width/2)**2,(height/2)**2,np.inf))
    else:
        bounds=(0,(np.inf,width,height,(width/2)**2,(height/2)**2,np.inf))
    
    # fitting
    try:
        popt,_=optimize.curve_fit(_Cauchy2Ds,xdata,ydata,p0=p0,bounds=bounds)
    except RuntimeError:
        popt=np.array([np.nan,np.nan,np.nan,np.nan,np.nan,np.nan])   
    return popt 

def RotateSpeed_calc(dataset,SizeOfBead,bead_type,eff_pixelsize,fit_type,calc_mode):
    nFrames,height,width=dataset.shape
    if calc_mode=='for-loop':
        print("Using 'for-loop' to do the gaussian fitting.")
        
        # gaussian fitting
        if fit_type=='gaussian':
            tic=time.time()
            popts=[Gaussian2DFit(nFrame,dataset,SizeOfBead,bead_type,eff_pixelsize) for nFrame in range(nFrames)]
            print("--- %s seconds ---" % (time.time()-tic))
        elif fit_type=='cauchy':
            tic=time.time()
            popts=[Gaussian2DFit(nFrame,dataset,SizeOfBead,bead_type,eff_pixelsize) for nFrame in range(nFrames)]
            print("--- %s seconds ---" % (time.time()-tic))
    elif calc_mode=='parallel':
        print("Using 'parallel computing' to do the gaussian fitting.")
        print("The number of available processers is "+str(multiprocessing.cpu_count())+".")

        tic = time.time()
        pool = multiprocessing.Pool()
        if fit_type=='gaussian':
            popts=pool.map(functools.partial(Gaussian2DFit,dataset=dataset,SizeOfBead=SizeOfBead,bead_type=bead_type,eff_pixelsize=eff_pixelsize),range(nFrames))
        elif fit_type=='cauchy':
            popts=pool.map(functools.partial(Cauchy2DFit,dataset=dataset,SizeOfBead=SizeOfBead,bead_type=bead_type,eff_pixelsize=eff_pixelsize),range(nFrames))
        pool.close()
        pool.join()
        print("--- %s seconds ---" % (time.time()-tic))
    else:
        print("The calculation modes are only 'for-loop' and 'parallel'.")
        
    # extract gaussian centroids and the frequency of the rotating object
    xcs_=np.array([popts[nFrame][1] for nFrame in range(nFrames)])
    ycs_=np.array([popts[nFrame][2] for nFrame in range(nFrames)])
    xcs=xcs_-np.mean(xcs_[np.isnan(xcs_)==False])
    ycs=ycs_-np.mean(ycs_[np.isnan(ycs_)==False])
    z_=xcs+1j*ycs
    
    # deal with nan-term (Be not fitted)
    for nFrame in range(nFrames):
        if np.isnan(z_[nFrame])==True:
            if nFrame==0:
                z_[nFrame]=0+1j*0
            elif nFrame==nFrames-1:
                z_[nFrame]=0+1j*0
            else:
                z_[nFrame]=(xcs[nFrame-1]+xcs[nFrame+1])/2+1j*(ycs[nFrame-1]+ycs[nFrame+1])/2
    z_[np.isnan(z_)==True]=0+1j*0
    
    # fft
    z=2*np.abs(np.fft.fftshift(np.fft.fft(z_)))/len(z_)
    return popts,xcs,ycs,z

def EllipseFit(x,y):
    # elipse fitting
    X=np.reshape(x,(len(x),1))
    Y=np.reshape(y,(len(y),1))
    K=np.hstack([X**2,X*Y,Y**2,X,Y])
    m=np.ones_like(X)
    coeffs=np.linalg.lstsq(K,m)[0].squeeze()  # solve the least squares problem ||Kx-m||^2
    x_=np.linspace(np.min(X)-0.2*np.abs(np.min(X)),np.max(X)+0.2*np.abs(np.max(X)),len(x))
    y_=np.linspace(np.min(X)-0.2*np.abs(np.min(Y)),np.max(Y)+0.2*np.abs(np.max(Y)),len(x))
    xx,yy=np.meshgrid(x_,y_)
    zz=coeffs[0]*xx**2+coeffs[1]*xx*yy+coeffs[2]*yy**2+coeffs[3]*xx+coeffs[4]*yy
    
    # extract geometry parameter (rotation matrix and fx=fy=0)
    A=coeffs[0]
    B=coeffs[1]
    C=coeffs[2]
    D=coeffs[3]
    E=coeffs[4]
    F=-1
    
    t=np.arctan(B/(A-C))/2
    newA=A*(np.cos(t))**2+B*np.cos(t)*np.sin(t)+C*(np.sin(t))**2
    newC=A*(np.sin(t))**2-B*np.cos(t)*np.sin(t)+C*(np.cos(t))**2
    newD=D*np.cos(t)+E*np.sin(t)
    newE=-D*np.sin(t)+E*np.cos(t)
    newF=F
    newx0=-newD/(2*newA)
    newy0=-newE/(2*newC)
    
    x0=newx0*np.cos(t)-newy0*np.sin(t)
    y0=newx0*np.sin(t)+newy0*np.cos(t)
    a=np.sqrt((-4*newF*newA*newC+newC*newD**2+newA*newE**2)/(4*newC*newA**2))
    b=np.sqrt((-4*newF*newA*newC+newC*newD**2+newA*newE**2)/(4*newA*newC**2))
    phi=np.rad2deg(t)
    
    # calculate eccentricity if the conic section is not a parabola, not a degenerate hyperbola or degenerate ellipse, and not an imaginary ellipse
    eta_=np.linalg.det(np.array([[coeffs[0],coeffs[1]/2,coeffs[3]/2],
                                 [coeffs[1]/2,coeffs[2],coeffs[4]/2],
                                 [coeffs[3]/2,coeffs[4]/2,-1]]))
    if eta_<0:
      eta=1
    elif eta_>0:
      eta=-1

    term1=2*np.sqrt((coeffs[0]-coeffs[2])**2+coeffs[1]**2)
    term2=eta*(coeffs[0]+coeffs[2])+np.sqrt((coeffs[0]-coeffs[2])**2+coeffs[1]**2)
    eccentricity=np.sqrt(term1/term2)  
    
    # collect ellipse parameters
    params=np.array([x0,y0,a,b,phi,eccentricity]) 
    return xx,yy,zz,params

def Gaissian2DFit_Check(nFrame,dataset,popts):
    _,height,width=dataset.shape
    dataset_=dataset[nFrame,:,:]
    popt=popts[nFrame]
    xc=popts[nFrame][1]
    yc=popts[nFrame][2]
    xx,yy=np.meshgrid(np.linspace(0,width-1,width),np.linspace(0,height-1,height) )
    yfit=Gaussian2D(xx,yy,*popt) 
    obs_x=[round(popt[1])]
    obs_y=[round(popt[2])]
    
    fig,axs=plt.subplots(2,2,figsize=(5,5))
    axs[0,0].plot(np.arange(0,width,1),dataset_[obs_y[0],:],'.',label='yy='+str(obs_y))
    axs[0,0].plot(np.arange(0,width,1),yfit[obs_y[0],:],label='fitting')
    axs[0,0].grid(linewidth=0.3)
    axs[0,0].set_xlabel('X [pixel]',weight='bold')
    axs[0,0].xaxis.set_label_position('top')
    axs[0,0].xaxis.tick_top()
    axs[0,0].set_ylabel('Intensity [a.u.]',weight='bold')
    axs[0,0].legend()
    
    axs[0,1].plot(xc,yc,'.')
    axs[0,1].grid(linewidth=0.3)
    axs[0,1].set_xlim(0,width)
    axs[0,1].set_xlabel('X [pixel]',weight='bold')
    axs[0,1].xaxis.set_label_position('top')
    axs[0,1].xaxis.tick_top()
    axs[0,1].set_ylim(0,height)
    axs[0,1].set_ylabel('Y [pixel]',weight='bold')
    axs[0,1].yaxis.set_label_position('right')
    axs[0,1].yaxis.tick_right()
    axs[0,1].invert_yaxis()
    
    axs[1,0].imshow(dataset_)
    axs[1,0].set_title('nFrame = '+str(nFrame)) 
    
    axs[1,1].plot(dataset_[:,obs_x[0]],np.arange(0,height),'.',label='xx='+str(obs_x))
    axs[1,1].plot(yfit[:,obs_x[0]],np.arange(0,height),'-',label='fitting')
    axs[1,1].grid(linewidth=0.3)
    axs[1,1].invert_yaxis()
    axs[1,1].set_xlabel('Intensity [a.u.]',weight='bold')
    axs[1,1].set_ylabel('Y [pixel]',weight='bold')
    axs[1,1].yaxis.set_label_position('right')
    axs[1,1].yaxis.tick_right()
    axs[1,1].legend()
    