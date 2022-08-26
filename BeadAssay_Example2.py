# -*- coding: utf-8 -*-
"""
Created on Tue Dec 14 16:01:31 2021

@author: XYZ
"""

#%%
print("Running...")
import matplotlib.pyplot as plt
import pandas as pd
import time
import warnings
warnings.simplefilter("ignore")

# 
import MoPy.BeadAssay as BA
import MoPy as mo
print("MoPy package is "+mo.__version__+".")
print("------------------------------------------------------------------")

#%%
# load data
print("Loading data...")
tic=time.time()
inputfile='.\\Datasets\\BeadAssay\\Test_trajectory (0420 1P2B0T by Sung)\\Test_trajectory (0420 1P2B0T by Sung).csv'
data=pd.read_csv(inputfile).values
xcs,ycs=data[:,0],data[:,1]
print("--- %s seconds ---" % (time.time()-tic))
print("------------------------------------------------------------------")

# fit a general ellipse
print("fit a general ellipse...")
tic=time.time()
xx,yy,zz,ellipse_params=BA.EllipseFit(xcs,ycs)    # paras: (x0,y0,a,b,phi,e)
print("--- %s seconds ---" % (time.time()-tic))
print("------------------------------------------------------------------")

# show result
print("The loaction of cneter (x0,y0): (%s, %s)" % (ellipse_params[0],ellipse_params[1]))
print("The semi lengths (a,b): (%s, %s)" % (ellipse_params[2],ellipse_params[3]))
print("The tilt angle (in deg): %s" % (ellipse_params[4]))
print("The eccentricity: %s" % (ellipse_params[5]))

plt.figure(figsize=(7,7))
plt.plot(xcs,ycs,'.',markersize=1.5)
plt.contour(xx,yy,zz,levels=[1],colors='r',lw=2)
plt.xlabel('X [??]',weight='bold')
plt.ylabel('Y [??]',weight='bold')
plt.grid(linewidth=0.3)
plt.gca().set_aspect('equal')
plt.show()

#%%
print('Done.')

