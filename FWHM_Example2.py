# -*- coding: utf-8 -*-
"""
Created on Wed Nov 17 17:23:40 2021

@author: XYZ
"""

#%%
print("Running...")
import numpy as np
import skimage
from skimage import io
from skimage.measure import label
import matplotlib.pyplot as plt
import math
import warnings
warnings.simplefilter("ignore")

# 
import MoPy.BeadAssay as BA
import MoPy as mo
print("MoPy package is "+mo.__version__+".")

# define units
um=1
sec=1
Hz=1/sec
print("------------------------------------------------------------------")

#%% 
# imaging configuration
pixelsize                       =4.6 *(um)
Mag                             =100
NA                              =1.49
relay_Mag                       =1
eff_pixelsize                   =pixelsize/(Mag*relay_Mag)
SizeOfBead                      =0.1 *(um)  
bead_type                       ='bright'
wavelength                      =0.515 *(um)
DL                              =(0.5*wavelength/NA)/eff_pixelsize

#%%
# load images
inputfile=r'.\Datasets\FWHM\100nm fluo beads\Large field (20220330 by XYZ)\Quest_UQ_200ms3-1.tif'
BF=io.imread(inputfile)
height,width=BF.shape
print("The size of image: ("+str(height)+", "+str(width)+")")

# label individual fluorescence beads
mask_=BF>np.mean(BF)+3*np.std(BF)
mask=skimage.morphology.remove_small_objects(mask_,DL)
L=label(mask)
nSeeds=np.max(L)
print("The number of fluorescence bead: %s" %nSeeds)

FWHM_x=[]
FWHM_y=[]
for nSeed in range(1,nSeeds+1):
    L_n=L==nSeed
    rows,cols=np.where(L_n==True)
    row=int(np.round(np.mean(rows)))
    col=int(np.round(np.mean(cols)))
    
    ROI_=BF[row-16:row+16,col-16:col+16]
    ROI=np.reshape(ROI_,(1,32,32))
    popts,xcs,ycs,z=BA.RotateSpeed_calc(ROI,SizeOfBead,bead_type,eff_pixelsize,'gaussian','for-loop')
    BA.Gaissian2DFit_Check(0,ROI,popts)
    
    # calculate FWHM
    FWHM_x_=2*eff_pixelsize*popts[0][3]*math.sqrt(math.log(4))
    FWHM_y_=2*eff_pixelsize*popts[0][4]*math.sqrt(math.log(4))
    
    FWHM_x.append(FWHM_x_)
    FWHM_y.append(FWHM_y_)
    
    
print('The FWHM in x: %s [um], std: %s [um]' %(np.mean(np.array(FWHM_x)),np.std(np.array(FWHM_x))))
print('The FWHM in y: %s [um], std: %s [um]' %(np.mean(np.array(FWHM_y)),np.std(np.array(FWHM_y))))
print("------------------------------------------------------------------")

#%%
print("Done.")
    