# -*- coding: utf-8 -*-
"""
Created on Mon Dec 13 14:51:33 2021

@author: XYZ
"""
try:
    import numpy as np
except ModuleNotFoundError:
    print("Module 'numpy' is not installed.")
 
try:
    import cupy as cp
except ModuleNotFoundError:
    print("Module 'cupy' is not installed.")

try:
    from lmfit import minimize, Parameters
except ModuleNotFoundError:
    print("Module 'lmfit' is not installed.")

#%%
def modified_SuperEllipse(a,b,rx,ry,k,p,s0,x0,y0,phi):
    # s0=0
    
    # the right hand side 
    t=np.linspace(0,a+s0,10000)
    s=0.5*t*np.sqrt(1+(2*k*t)**2)+np.log(np.sqrt(1+(2*k*t)**2)+2*k*t)/(4*k)-s0
    t=t[1-np.abs(s)/a>=0]
    s=s[1-np.abs(s)/a>=0]
    x_right_up=t-((2*k*t)/np.sqrt(1+(2*k*t)**2))*(+b*((1-(np.abs(s)/a)**(2*a/rx))/(1+p*s))**(0.5*ry/b))
    y_right_up=k*t**2+(1/np.sqrt(1+(2*k*t)**2))*(+b*((1-(np.abs(s)/a)**(2*a/rx))/(1+p*s))**(0.5*ry/b))
    x_right_down=t-((2*k*t)/np.sqrt(1+(2*k*t)**2))*(-b*((1-(np.abs(s)/a)**(2*a/rx))/(1+p*s))**(0.5*ry/b))
    y_right_down=k*t**2+(1/np.sqrt(1+(2*k*t)**2))*(-b*((1-(np.abs(s)/a)**(2*a/rx))/(1+p*s))**(0.5*ry/b))
    x_right=np.concatenate((x_right_up,np.flip(x_right_down)))
    y_right=np.concatenate((y_right_up,np.flip(y_right_down)))
    
    # the left hand side
    t=np.linspace(-a-s0,0,10000)
    s=0.5*t*np.sqrt(1+(2*k*t)**2)+np.log(np.sqrt(1+(2*k*t)**2)+2*k*t)/(4*k)+s0
    t=t[1-np.abs(s)/a>=0]
    s=s[1-np.abs(s)/a>=0]
    x_left_up=t-((2*k*t)/np.sqrt(1+(2*k*t)**2))*(+b*((1-(np.abs(s)/a)**(2*a/rx))/(1+p*s))**(0.5*ry/b))
    y_left_up=k*t**2+(1/np.sqrt(1+(2*k*t)**2))*(+b*((1-(np.abs(s)/a)**(2*a/rx))/(1+p*s))**(0.5*ry/b))
    x_left_down=t-((2*k*t)/np.sqrt(1+(2*k*t)**2))*(-b*((1-(np.abs(s)/a)**(2*a/rx))/(1+p*s))**(0.5*ry/b))
    y_left_down=k*t**2+(1/np.sqrt(1+(2*k*t)**2))*(-b*((1-(np.abs(s)/a)**(2*a/rx))/(1+p*s))**(0.5*ry/b))
    x_left=np.concatenate((np.flip(x_left_down),x_left_up))
    y_left=np.concatenate((np.flip(y_left_down),y_left_up))
    
    # rotation
    new_x_right=x_right*np.cos(np.deg2rad(phi))-y_right*np.sin(np.deg2rad(phi));
    new_y_right=x_right*np.sin(np.deg2rad(phi))+y_right*np.cos(np.deg2rad(phi));
    new_x_left=x_left*np.cos(np.deg2rad(phi))-y_left*np.sin(np.deg2rad(phi));
    new_y_left=x_left*np.sin(np.deg2rad(phi))+y_left*np.cos(np.deg2rad(phi));
    
    # shift
    new_x_right=new_x_right+x0;
    new_y_right=new_y_right+y0;
    new_x_left=new_x_left+x0;
    new_y_left=new_y_left+y0;
    
    # collect numerical contour
    numerical_contour_x=np.concatenate((new_x_right,new_x_left))
    numerical_contour_y=np.concatenate((new_y_right,new_y_left))
    return numerical_contour_x,numerical_contour_y

def NumericalSol(imboundary_x,imboundary_y,numerical_x,numerical_y,calc_mode):
    sol_x=[]
    sol_y=[]
    if calc_mode=='vectorization':
        numerical_x_=np.tile(numerical_x.reshape(-1,1),imboundary_x.shape[0])
        imboundary_x_=np.tile(imboundary_x.reshape((1,-1)),(numerical_x.shape[0],1))
        numerical_y_=np.tile(numerical_y.reshape(-1,1),imboundary_y.shape[0])
        imboundary_y_=np.tile(imboundary_y.reshape((1,-1)),(numerical_y.shape[0],1))
        
        dr=np.sqrt((numerical_x_-imboundary_x_)**2+(numerical_y_-imboundary_y_)**2)
        idxs=np.argmin(dr,axis=0)
        sol_x=numerical_x[idxs]
        sol_y=numerical_y[idxs]
    elif calc_mode=='gpu':
        numerical_x_=cp.tile(numerical_x.reshape(-1,1),imboundary_x.shape[0])
        imboundary_x_=cp.tile(imboundary_x.reshape((1,-1)),(numerical_x.shape[0],1))
        numerical_y_=cp.tile(numerical_y.reshape(-1,1),imboundary_y.shape[0])
        imboundary_y_=cp.tile(imboundary_y.reshape((1,-1)),(numerical_y.shape[0],1))
        
        dr=cp.sqrt((numerical_x_-imboundary_x_)**2+(numerical_y_-imboundary_y_)**2)
        idxs_=cp.argmin(dr,axis=0)
        idxs=cp.asnumpy(idxs_)
        sol_x=numerical_x[idxs]
        sol_y=numerical_y[idxs]
    elif calc_mode=='for-loop':
        sol_x=np.zeros(imboundary_x.shape)
        sol_y=np.zeros(imboundary_y.shape)
        
        for nPt in range(len(imboundary_x)):
            dr=np.sqrt((numerical_x-imboundary_x[nPt])**2+ (numerical_y-imboundary_y[nPt])**2)
            idx=np.where(dr==min(dr))
            idx=idx[0]
            sol_x[nPt]=numerical_x[idx]
            sol_y[nPt]=numerical_y[idx]
    else:
        print("Please use 'for-loop', 'vectorization', or 'gpu'.")
    return sol_x,sol_y

def ODR_Calc_(imboundary_x,imboundary_y,sol_x,sol_y):
    dists=np.sqrt((imboundary_x-sol_x)**2+(imboundary_y-sol_y)**2)
    return dists

def ODR_Calc(params,x,y,calc_mode):
    a=params['a'].value
    b=params['b'].value
    m=params['m'].value
    rx=2*a/m
    n=params['n'].value
    ry=2*b/n
    k=params['k'].value
    p=params['p'].value
    s0=params['s0'].value
    x0=params['x0'].value
    y0=params['y0'].value
    phi=params['phi'].value
        
    model_x,model_y=modified_SuperEllipse(a,b,rx,ry,k,p,s0,x0,y0,phi)
    sol_x,sol_y=NumericalSol(x,y,model_x,model_y,calc_mode)
    cost=ODR_Calc_(x,y,sol_x,sol_y)
    return cost

def guess_condition(x,y):
    x0=np.mean(x)
    y0=np.mean(y)
    
    dr=np.sqrt((x-x0)**2+(y-y0)**2)
    max_dr_idx=np.argmax(dr)
    min_dr_idx=np.argmin(dr)
    a=np.sqrt((x[max_dr_idx]-x0)**2+(y[max_dr_idx]-y0)**2)
    b=np.sqrt((x[min_dr_idx]-x0)**2+(y[min_dr_idx]-y0)**2)
    
    ####################
    m=8.0
    n=2.0
    ####################
    
    phi=np.rad2deg(np.arctan((y[max_dr_idx]-y0)/(x[max_dr_idx]-x0)))
    RotM=np.array(((np.cos(np.deg2rad(-phi)),-np.sin(np.deg2rad(-phi))),
                   (np.sin(np.deg2rad(-phi)),np.cos(np.deg2rad(-phi)))))
    Arr_=np.array([x,y])
    Arr_=np.matmul(RotM,Arr_)
    centroid_=np.array([x0,y0])
    centroid_=np.matmul(RotM,centroid_)
    Arr_x,Arr_y=Arr_[0,:],Arr_[1,:]
    Arr_x=Arr_x[Arr_y>centroid_[1]+0.9*b]
    Arr_y=Arr_y[Arr_y>centroid_[1]+0.9*b]
    k=np.polyfit(Arr_x,Arr_y,2)[0]
    
    ####################
    p=0.05
    s0=0.0
    ####################
        
    guess=np.array([a,b,m,n,k,p,s0,x0,y0,phi])
    return guess

def modified_SuperEllipse_Fit(x,y,calc_mode):
    guess=guess_condition(x, y)
    params=Parameters()
    params.add('a',value=guess[0])
    params.add('b',value=guess[1])
    params.add('m',value=guess[2],min=1.5)
    params.add('n',value=guess[3],min=1.1)
    params.add('k',value=guess[4])
    params.add('p',value=guess[5])
    params.add('s0',value=guess[6],min=0,vary=False)
    params.add('x0', value=guess[7])
    params.add('y0', value=guess[8])
    params.add('phi',value=guess[9])

    out=minimize(ODR_Calc,params,args=(x,y,calc_mode),nan_policy='omit')
    return out
