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
pixelsize                       =5.5 *(um)
Mag                             =60
relay_Mag                       =1
sampling                        =460 *(Hz)
eff_pixelsize                   =pixelsize/(Mag*relay_Mag)
dt                              =1/sampling
SizeOfBead                      =1 *(um)  
bead_type                       ='bright'

# load images
inputfile=r'.\Datasets\BeadAssay\Test_crop (by KC)\Test_crop.tif'
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
    # pixelFFT 
    spectrum=BA.pixelFFT(BF,dt,modulate_amp=6,isGPU=False)
    
    fig=plt.figure()
    ax=plt.axes()
    cax=ax.imshow(spectrum,cmap='hot')
    cbar=plt.colorbar(cax)
    cbar.set_label('Frequency [Hz]',weight='bold')
    ax.set_title('pixelFFT, f = '+str(np.round(np.sum(spectrum)/np.sum(spectrum>0),2))+' Hz')
    plt.show()
    print("------------------------------------------------------------------")
    
    # using 'for-loop' to do gaussian fitting
    popts1,xcs1,ycs1,z1=BA.RotateSpeed_calc(BF,SizeOfBead,bead_type,eff_pixelsize,'gaussian','for-loop')
    xx1,yy1,zz1,params1=BA.EllipseFit(xcs1,ycs1)
    
    print("The loaction of cneter (x0,y0): (%s, %s)" % (params1[0],params1[1]))
    print("The semi lengths (a,b): (%s, %s)" % (params1[2],params1[3]))
    print("The tilt angle (in deg): %s" % (params1[4]))
    print("The eccentricity: %s" % (params1[5]))

    plt.figure(figsize=(12,4))
    plt.subplot(121)
    plt.scatter(xcs1,ycs1)
    plt.contour(xx1,yy1,zz1,levels=[1],colors='r',lw=2)
    plt.xlabel('X [pixel]',weight='bold')
    plt.ylabel('Y [pixel]',weight='bold')
    plt.grid(linewidth=0.3)
    plt.gca().set_aspect('equal')
    
    plt.subplot(122)
    f=np.linspace(-sampling/2,sampling/2,nFrames)
    f_max=f[z1==np.max(z1)]
    plt.plot(f,z1)
    plt.xlabel('Frequency [Hz]',weight='bold')
    plt.ylabel('Amplitude [pixel]',weight='bold')
    plt.title('Maximal f = '+str(np.round(f_max,2))+' Hz')
    plt.grid(linewidth=0.3)
    plt.show()
    
    checkFrame=13
    print("Check nFrmae = "+str(checkFrame)+".")
    BA.Gaissian2DFit_Check(checkFrame,BF,popts1)
    print("------------------------------------------------------------------")
    
    # using 'parallel' to do gaussian fitting
    popts2,xcs2,ycs2,z2=BA.RotateSpeed_calc(BF,SizeOfBead,bead_type,eff_pixelsize,'gaussian','parallel')
    xx2,yy2,zz2,params2=BA.EllipseFit(xcs2,ycs2)
    
    print("The loaction of cneter (x0,y0): (%s, %s)" % (params2[0],params2[1]))
    print("The semi lengths (a,b): (%s, %s)" % (params2[2],params2[3]))
    print("The tilt angle (in deg): %s" % (params2[4]))
    print("The eccentricity: %s" % (params2[5]))
    
    plt.figure(figsize=(12,4))
    plt.subplot(121)
    plt.scatter(xcs2,ycs2)
    plt.contour(xx2,yy2,zz2,levels=[1],colors='r',lw=2)
    plt.xlabel('X [pixel]',weight='bold')
    plt.ylabel('Y [pixel]',weight='bold')
    plt.grid(linewidth=0.3)
    plt.gca().set_aspect('equal')
    
    plt.subplot(122)
    f=np.linspace(-sampling/2,sampling/2,nFrames)
    f_max=f[z2==np.max(z2)]
    plt.plot(f,z2)
    plt.xlabel('Frequency [Hz]',weight='bold')
    plt.ylabel('Amplitude [pixel]',weight='bold')
    plt.title('Maximal f = '+str(np.round(f_max,2))+' Hz')
    plt.grid(linewidth=0.3)
    plt.show()
    
    checkFrame=44
    print("Check nFrmae = "+str(checkFrame)+".")
    BA.Gaissian2DFit_Check(checkFrame,BF,popts2)
    print("------------------------------------------------------------------")
    
    #%%
    print("Done.")
    