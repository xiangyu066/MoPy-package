#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 10 23:38:36 2021

@author: XYZ
"""

#%%
print("Running...")
import numpy as np
import matplotlib.pyplot as plt
import time
import warnings
warnings.simplefilter("ignore")

# 
import MoPy.BacterialContour as BC
import MoPy as mo
print("MoPy package is "+mo.__version__+".")

# define units
um      =1
sec     =1
Hz      =1/sec
deg     =1
print("------------------------------------------------------------------")

#%% Configuration
# bacterium shape function
semi_major_len                      =2.51 *(um)
semi_minor_len                      =0.77 *(um)
curv_x                              =0.61 *(1/um)
curv_y                              =0.83 *(1/um)
bending                             =-0.07 *(1/um)                              # parabolic coefficient
egg_slope                           =0.1                                        # linear slope
elongation_len                      =0.0

# imaging coordinate
cent_x                              =15.0 *(um)
cent_y                              =-33.0 *(um)
rotating_ang                        =13.0 *(deg)                                       

#%% Generate a virtual contour data
print("--- The configuration of a virtual contour is shown as below ---")
print("The semi-major length: %s [um]" %semi_major_len)
print("The semi-inor length: %s [um]" %semi_minor_len)
print("The curvature in x direction: %s [1/um]" %curv_x)
print("The curvature in y direction: %s [1/um]" %curv_y)
print("The bending coefficient: %s [1/um]" %bending)
print("The egg slope: %s" %egg_slope)
print("The elongation offset: %s [um]" %elongation_len)
print("The centerof x: %s" %cent_x)
print("The centerof y: %s" %cent_y)
print("The rotating angle [deg]: %s" %rotating_ang)
print("------------------------------------------------------------------")

# generate super-ellipse data
print("Generating a virtual bacterial contour...")
tic=time.time()
x_,y_=BC.modified_SuperEllipse(semi_major_len,semi_minor_len,curv_x,curv_y,bending,egg_slope,elongation_len,cent_x,cent_y,rotating_ang)
print("--- %s seconds ---" % (time.time()-tic))

# add some noise and downsampling
print("Adding some noise and sownsampling...")
x=x_+np.random.normal(0,0.05,len(x_))
y=y_+np.random.normal(0,0.05,len(y_))
x=x[0:-1:150]
y=y[0:-1:150]
print("------------------------------------------------------------------")

#%% Fitting
print("Fitting (for-loop)...")
# tic=time.time()
# out=BC.modified_SuperEllipse_Fit(x, y,'for-loop')
# print("--- %s seconds ---" % (time.time()-tic))

# print("Fitting (vectorization)...")
# tic=time.time()
# out=BC.modified_SuperEllipse_Fit(x, y,'vectorization')
# print("--- %s seconds ---" % (time.time()-tic))

print("Fitting (gpu)...")
tic=time.time()
out=BC.modified_SuperEllipse_Fit(x, y,'gpu')
print("--- %s seconds ---" % (time.time()-tic))

# collect fittung results
a=out.params['a'].value
b=out.params['b'].value
m=out.params['m'].value
rx=2*a/m
n=out.params['n'].value
ry=2*b/n
k=out.params['k'].value
p=out.params['p'].value
s0=out.params['s0'].value
x0=out.params['x0'].value
y0=out.params['y0'].value
phi=out.params['phi'].value
fit_x,fit_y=BC.modified_SuperEllipse(a,b,rx,ry,k,p,s0,x0,y0,phi)
print("--- The fitting result is shown as below ---")
print("The semi-major length: %s [um]" %a)
print("The semi-inor length: %s [um]" %b)
print("The curvature in x direction: %s [1/um]" %rx)
print("The curvature in y direction: %s [1/um]" %ry)
print("The bending coefficient: %s [1/um]" %k)
print("The egg slope: %s" %p)
print("The elongation offset: %s [um]" %s0)
print("The centerof x: %s" %x0)
print("The centerof y: %s" %y0)
print("The rotating angle [deg]: %s" %phi)
print("------------------------------------------------------------------")

#%% demonstration
plt.figure(figsize=(6,6))
plt.plot(x,y,'.',label='noisy contour',markersize=1.5)
plt.plot(x_,y_,label='ground-truth contour',lw=1)
plt.plot(fit_x,fit_y,'--',label='fitting',lw=1)
plt.xlabel('X [um]',weight='bold')
plt.ylabel('Y [um]',weight='bold')
plt.legend()
plt.axis('equal')
plt.grid(lw=0.3)

#%%
print("Done.")


