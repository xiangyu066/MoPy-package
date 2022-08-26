#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 10 23:38:36 2021

@author: XYZ
"""

#%%
print("Running...")
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

#%% Comparison of the super-ellipse and the ellipse
x_ellipse,y_ellipse=BC.modified_SuperEllipse(1,0.5,2*1/2,2*0.5/2,0.00001,0,0,0,0,0)
x_super1,y_super1=BC.modified_SuperEllipse(1,0.5,2*1/3.6,2*0.5/2,0.00001,0,0,0,0,0)
x_super2,y_super2=BC.modified_SuperEllipse(1,0.5,2*1/3.6,2*0.5/3.6,0.00001,0,0,0,0,0)

plt.figure(figsize=(6,6))
plt.plot(x_ellipse,y_ellipse,label='(m,n)=(2.0, 2.0)',lw=1)
plt.plot(x_super1,y_super1,label='(m,n)=(3.6, 2.0)',lw=1)
plt.plot(x_super2,y_super2,label='(m,n)=(3.6, 3.6)',lw=1)
plt.xlabel('X',weight='bold',fontsize=16)
plt.ylabel('Y',weight='bold',fontsize=16)
plt.legend(fontsize=12)
plt.title('(a,b)=(1.0, 0.5)',weight='bold',fontsize=12)
plt.axis('equal')
plt.grid(lw=0.3)

#%% Using parabolic bending by the curvilinear transform
x_super1,y_super1=BC.modified_SuperEllipse(1,0.5,2*1/3.6,2*0.5/2,0.00001,0,0,0,0,0)
x_super2,y_super2=BC.modified_SuperEllipse(1,0.5,2*1/3.6,2*0.5/2,0.4,0,0,0,0,0)

plt.figure(figsize=(6,6))
plt.plot(x_super1,y_super1,label='non-bending',lw=1)
plt.plot(x_super2,y_super2,label='parabolic bending (k=0.4)',lw=1)
plt.xlabel('X',weight='bold',fontsize=16)
plt.ylabel('Y',weight='bold',fontsize=16)
plt.legend(fontsize=12)
plt.title('(a,b,m,n)=(1.0, 0.5, 3.6, 2.0)',weight='bold',fontsize=12)
plt.axis('equal')
plt.grid(lw=0.3)

#%% Induce a linear term to describe the egg-shape caps
x_super1,y_super1=BC.modified_SuperEllipse(1,0.5,2*1/3.6,2*0.5/2,0.00001,0,0,0,0,0)
x_super2,y_super2=BC.modified_SuperEllipse(1,0.5,2*1/3.6,2*0.5/2,0.00001,0.3,0,0,0,0)

plt.figure(figsize=(6,6))
plt.plot(x_super1,y_super1,label='equal cap size',lw=1)
plt.plot(x_super2,y_super2,label='non-equal cap size (linear slope=0.3)',lw=1)
plt.xlabel('X',weight='bold',fontsize=16)
plt.ylabel('Y',weight='bold',fontsize=16)
plt.legend(fontsize=12)
plt.title('(a,b,m,n)=(1.0, 0.5, 3.6, 2.0)',weight='bold',fontsize=12)
plt.axis('equal')
plt.grid(lw=0.3)

#%% Division factor
x_super1,y_super1=BC.modified_SuperEllipse(1,0.5,2*1/3.6,2*0.5/2,0.1,0,0.0,0,4,0)
x_super2,y_super2=BC.modified_SuperEllipse(1,0.5,2*1/3.6,2*0.5/2,0.1,0,0.3,0,2,0)
x_super3,y_super3=BC.modified_SuperEllipse(1,0.5,2*1/3.6,2*0.5/2,0.1,0,0.5,0,0,0)
x_super4,y_super4=BC.modified_SuperEllipse(1,0.5,2*1/3.6,2*0.5/2,0.1,0,0.7,0,-2,0)
x_super5,y_super5=BC.modified_SuperEllipse(1,0.5,2*1/3.6,2*0.5/2,0.1,0,0.9,0,-4,0)

plt.figure(figsize=(6,6))
plt.plot(x_super1,y_super1,label='s0=0.0',lw=1)
plt.plot(x_super2,y_super2,label='s0=0.3',lw=1)
plt.plot(x_super3,y_super3,label='s0=0.5',lw=1)
plt.plot(x_super4,y_super4,label='s0=0.7',lw=1)
plt.plot(x_super5,y_super5,label='s0=0.9',lw=1)
plt.xlabel('X',weight='bold',fontsize=16)
plt.ylabel('Y',weight='bold',fontsize=16)
plt.legend(fontsize=12)
plt.title('(a,b,m,n,k)=(1.0, 0.5, 3.6, 2.0, 0.1)',weight='bold',fontsize=12)
plt.axis('equal')
plt.grid(lw=0.3)

#%% Generate a virtual bacterial contour
# bacterium shape configuration
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

# generate super-ellipse data
print("Generating a virtual bacterial contour...")
tic=time.time()
x_,y_=BC.modified_SuperEllipse(semi_major_len,semi_minor_len,curv_x,curv_y,bending,egg_slope,elongation_len,cent_x,cent_y,rotating_ang)
print("--- %s seconds ---" % (time.time()-tic))

plt.figure(figsize=(6,6))
plt.plot(x_,y_,lw=1)
plt.xlabel('X',weight='bold',fontsize=16)
plt.ylabel('Y',weight='bold',fontsize=16)
plt.axis('equal')
plt.grid(lw=0.3)

#%%
print("Done.")


