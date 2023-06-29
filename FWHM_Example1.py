# -*- coding: utf-8 -*-
"""
Created on Wed Nov 17 17:23:40 2021

@author: XYZ
"""

#%%
print("Running...")
import numpy as np
from skimage import io
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
pixelsize                       =6.5 *(um)
Mag                             =100
relay_Mag                       =1
eff_pixelsize                   =pixelsize/(Mag*relay_Mag)
SizeOfBead                      =0.1 *(um)  
bead_type                       ='bright'

# load images
inputfile=r'.\dataSets\FWHM\100nm fluo beads\Test crop (20220302 by XYZ)\Stack0.tif'
BF=io.imread(inputfile)

nFrames,height,width=BF.shape
print("The size of image: ("+str(height)+", "+str(width)+")")
print("Total frames: "+str(nFrames))


#%%
if __name__ == '__main__':
    # using 'parallel' to do gaussian fitting
    popts,xcs,ycs,z=BA.RotateSpeed_calc(BF,SizeOfBead,bead_type,eff_pixelsize,'gaussian','for-loop')
    
    checkFrame=0
    print("Check nFrmae = "+str(checkFrame)+".")
    BA.Gaissian2DFit_Check(checkFrame,BF,popts)
    
    # calculate FWHM
    FWHM_x=[2*eff_pixelsize*popts[nFrame][3]*math.sqrt(math.log(4)) for nFrame in range(nFrames)]
    FWHM_y=[2*eff_pixelsize*popts[nFrame][4]*math.sqrt(math.log(4)) for nFrame in range(nFrames)]
    print('The FWHM in x: %s [um], std: %s [um]' %(np.mean(np.array(FWHM_x)),np.std(np.array(FWHM_x))))
    print('The FWHM in y: %s [um], std: %s [um]' %(np.mean(np.array(FWHM_y)),np.std(np.array(FWHM_y))))
    print("------------------------------------------------------------------")
    
    #%%
    print("Done.")
    