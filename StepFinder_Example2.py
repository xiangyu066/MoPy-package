# -*- coding: utf-8 -*-
"""
Created on Thu Jan 20 15:46:33 2022

@author: XYZ
"""

#%%
print("Running...")
import numpy as np
import warnings
warnings.simplefilter("ignore")

# 
import MoPy.StepFinder as SF
import MoPy as mo
print("MoPy package is "+mo.__version__+".")

# define units
sec=1
Hz=1
print("------------------------------------------------------------------")

#%%
# step configuration
sampling        =10000 *(Hz)
samples         =10000

# initialization
dt=1/sampling

#%%
print("Test-1: Fixed amplitude and fixed timestep.")
problem=(1,np.nan,np.nan,np.nan,
          0.05*sec,np.nan,np.nan,np.nan)
t,y_=SF.virtual_steps(sampling,samples,problem)
y=y_+np.random.normal(0,1,samples)
y_predicted=SF.StepFit(t,y,'nn')
SF.StepFit_Check(t,y_,y,y_predicted)

print("Test-2: Fixed amplitude and ranged timestep.")
problem=(1,np.nan,np.nan,np.nan,
         np.nan,0.05*sec,0.1*sec,0.01*sec)
t,y_=SF.virtual_steps(sampling,samples,problem)
y=y_+np.random.normal(0,1,samples)
y_predicted=SF.StepFit(t,y,'nn')
SF.StepFit_Check(t,y_,y,y_predicted)

print("Test-3: Ranged amplitude and fixed timestep.")
problem=(np.nan,1,3,0.5,
          0.05*sec,np.nan,np.nan,np.nan)
t,y_=SF.virtual_steps(sampling,samples,problem)
y=y_+np.random.normal(0,1,samples)
y_predicted=SF.StepFit(t,y,'nn')
SF.StepFit_Check(t,y_,y,y_predicted)

print("Test-4: Ranged amplitude and ranged timestep.")
problem=(np.nan,1,3,0.5,
          np.nan,0.05*sec,0.1*sec,0.01*sec)
t,y_=SF.virtual_steps(sampling,samples,problem)
y=y_+np.random.normal(0,1,samples)
y_predicted=SF.StepFit(t,y,'nn')
SF.StepFit_Check(t,y_,y,y_predicted)


#%%
print("Done.")