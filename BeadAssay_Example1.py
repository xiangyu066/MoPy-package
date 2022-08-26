# -*- coding: utf-8 -*-
"""
Created on Wed Nov 17 17:23:40 2021

@author: XYZ
"""

#%%
print("Running...")
import numpy as np
from skimage import io
import matplotlib.pyplot as plt
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
Mag                             =60
relay_Mag                       =1
sampling                        =36363.6 *(Hz)
eff_pixelsize                   =pixelsize/(Mag*relay_Mag)
dt                              =1/sampling
SizeOfBead                      =1 *(um) 
bead_type                       ='dark'  

# load images
inputfile='.\Datasets\BeadAssay\Test_crop (20220104 by TCK)\MTB24\MyOne T1\Sample5\ss_stack_0-1.tif'
BF=io.imread(inputfile)

nFrames,height,width=BF.shape
print("The size of image: ("+str(height)+", "+str(width)+")")
print("Total frames: "+str(nFrames))

# std projection
plt.figure()
ax=plt.axes()
cax=ax.imshow(np.std(BF,axis=0))
cbar=plt.colorbar(cax)
cbar.set_label('STD',weight='bold')
ax.set_title("The projection of STD.",weight='bold')
plt.show()
print("------------------------------------------------------------------")

#%%
if __name__ == '__main__':
    popts,xcs,ycs,z=BA.RotateSpeed_calc(BF,SizeOfBead,bead_type,eff_pixelsize,'parallel')
    xx,yy,zz,params=BA.EllipseFit(xcs,ycs)
    
    print("The loaction of cneter (x0,y0): (%s, %s)" % (params[0],params[1]))
    print("The semi lengths (a,b): (%s, %s)" % (params[2],params[3]))
    print("The tilt angle (in deg): %s" % (params[4]))
    print("The eccentricity: %s" % (params[5]))
    
    plt.figure(figsize=(12,4))
    plt.subplot(121)
    plt.plot(xcs,ycs,'.',markersize=1)
    plt.contour(xx,yy,zz,levels=[1],colors='r',lw=2)
    plt.xlabel('X [pixel]',weight='bold')
    plt.ylabel('Y [pixel]',weight='bold')
    plt.grid(linewidth=0.3)
    plt.gca().set_aspect('equal')
    
    plt.subplot(122)
    f=np.linspace(-sampling/2,sampling/2,nFrames)
    f_max=f[z==np.max(z)]
    plt.plot(f,z)
    plt.xlabel('Frequency [Hz]',weight='bold')
    plt.ylabel('Amplitude [pixel]',weight='bold')
    plt.title('Maximal f = '+str(np.round(f_max,2))+' Hz')
    plt.grid(linewidth=0.3)
    plt.show()
    
    checkFrame=4444
    print("Check nFrmae = "+str(checkFrame)+".")
    BA.Gaissian2DFit_Check(checkFrame,BF,popts)
    print("------------------------------------------------------------------")
    
    #%%
    print("Done.")
    