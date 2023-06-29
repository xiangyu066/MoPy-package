# -*- coding: utf-8 -*-
"""
Created on Tue Dec 28 20:54:17 2021

@author: XYZ
"""

#%%
print("Running...")
import os, glob
import numpy as np
from skimage import io
import matplotlib.pyplot as plt
import time
import warnings
warnings.simplefilter("ignore")

# 
import MoPy.BacterialContour as BC
import MoPy.PhaseSeg as PS
import MoPy as mo
print("MoPy package is "+mo.__version__+".")

# define units
um      =1
nm      =0.001*(um)
sec     =1
Hz      =1/sec
deg     =1
print("------------------------------------------------------------------")

#%%
# imaging parameters             
Obj_Mag             =100                                                        # the magnificant of objective
NA                  =1.45                                                                         # the numerical aperture
wavelength          =550*(nm)                                                   # the fluorescence emission wavelength
Relay_Mag           =1.0                                                        # relay lens
pixelsize           =6.5*(um)                                                   # the pixel size of CCD

# digital image processing parameters
thresh_method       ='Customized'                                               # the method of threshold
division_coeff      =0.7                                                        # determinate if division
neighbor_dist       =1*(um)                                                     # safe distance between available cells
border_dist         =0*(um)                                                     # define a safe border distance

# working directory
inputdir=r'.\dataSets\BacterialContour\Vibrio fischeri\ATCC7744 (20210603, from XYZ)'

#%% Initialization
print('Initializing...')
listing=glob.glob(inputdir+'\\*.tif')
nFiles=len(listing)

eff_pixelsize=pixelsize/(Obj_Mag*Relay_Mag)                                     # effective pixelsize
DL=(0.5*wavelength/NA)/eff_pixelsize                                            # diffraction limitation (in pixel)

# Create a single directory
outputdir=inputdir+'\\Analyzd'
if not os.path.exists(outputdir):
    os.mkdir(outputdir)

#%%
tic0=time.time()
data=[]
for nFile in range(nFiles):
    print('Load files...(current file: '+str(nFile+1)+' / total files: '+str(nFiles)+')')
    inputfile=listing[nFile]
    origina=io.imread(inputfile)
    mask=PS.cell_mask(origina,DL,division_coeff,thresh_method)                  # selecting target by Otsu or FWHM           
    mask=PS.cell_filter_neighbors(mask,eff_pixelsize,neighbor_dist)             # filter out intimate cells
    mask=PS.cell_filter_border(mask,eff_pixelsize,border_dist)                  # filter out the cell near the border
    centroids=PS.cell_centroids(mask)                                           # calculate center point of target cell
    label_map,nCells,contours=PS.bwlabel(mask)                                  # label number and find contours
    PS.drawElements(origina,contours,centroids,np.Inf)                          # draw elements for all cell

    # fitting
    for nCell in range(nCells):
        x,y=contours[nCell-1][:,0],contours[nCell-1][:,1]
        x=x*eff_pixelsize
        y=y*eff_pixelsize
        
        try:
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
            
            # save fitting parameters
            data.append([a,b,m,n,k,p])
            
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
            
            # demonstration
            plt.figure(figsize=(6,6))
            plt.plot(x,y,'.',label='data',markersize=3)
            plt.plot(fit_x,fit_y,label='fitting',lw=1)
            plt.xlabel('X [um]',weight='bold',fontsize=24)
            plt.ylabel('Y [um]',weight='bold',fontsize=24)
            plt.legend(fontsize=14)
            plt.xticks(fontsize=14)
            plt.yticks(fontsize=14)
            plt.axis('equal')
            plt.grid(lw=0.3)
            outputname=listing[nFile].replace(inputdir+'\\','')
            outputname=outputname.replace('.tif','')
            plt.savefig(outputdir+'\\'+outputname+'_nCell '+str(nCell)+'.png',bbox_inches='tight')
            plt.show()
        except ValueError:
            print('Something wrong!')
print("--- %s seconds ---" % (time.time()-tic0))

# show histogram
if len(data)>10:
    a_list=[2*data[n][0] for n in range(len(data))]
    b_list=[2*data[n][1] for n in range(len(data))]
    m_list=[data[n][2] for n in range(len(data))]
    n_list=[data[n][3] for n in range(len(data))]
    k_list=[abs(data[n][4]) for n in range(len(data))]
    p_list=[abs(data[n][5]) for n in range(len(data))]

    plt.figure(figsize=(6,6))
    plt.hist(a_list,np.arange(0,5,0.2),weights=np.ones_like(a_list)/len(a_list),histtype='step')
    plt.xlabel('Length [um]',weight='bold',fontsize=24)
    plt.ylabel('Probability',weight='bold',fontsize=24)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.xlim((0,5))
    plt.grid(lw=0.3)
    plt.savefig(outputdir+'\\'+'histogram of length.png',bbox_inches='tight')
    plt.show()
    
    plt.figure(figsize=(6,6))
    plt.hist(b_list,np.arange(0,1,0.02),weights=np.ones_like(b_list)/len(b_list),histtype='step')
    plt.xlabel('Width [um]',weight='bold',fontsize=24)
    plt.ylabel('Probability',weight='bold',fontsize=24)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.xlim((0,1))
    plt.grid(lw=0.3)
    plt.savefig(outputdir+'\\'+'histogram of width.png',bbox_inches='tight')
    plt.show()
    
    plt.figure(figsize=(6,6))
    plt.hist(m_list,np.arange(1,10,0.5),weights=np.ones_like(m_list)/len(m_list),histtype='step')
    plt.xlabel('Geometric order: m',weight='bold',fontsize=24)
    plt.ylabel('Probability',weight='bold',fontsize=24)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.xlim((1,10))
    plt.grid(lw=0.3)
    plt.savefig(outputdir+'\\'+'histogram of geometric order m.png',bbox_inches='tight')
    plt.show()
    
    plt.figure(figsize=(6,6))
    plt.hist(n_list,np.arange(1,3,0.1),weights=np.ones_like(n_list)/len(n_list),histtype='step')
    plt.xlabel('Geometric order: n',weight='bold',fontsize=24)
    plt.ylabel('Probability',weight='bold',fontsize=24)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.xlim((1,3))
    plt.grid(lw=0.3)
    plt.savefig(outputdir+'\\'+'histogram of geometric order n.png',bbox_inches='tight')
    plt.show()
    
    plt.figure(figsize=(6,6))
    plt.hist(k_list,np.arange(0,0.5,0.02),weights=np.ones_like(k_list)/len(k_list),histtype='step')
    plt.xlabel('Bending factor: k [1/um]',weight='bold',fontsize=24)
    plt.ylabel('Probability',weight='bold',fontsize=24)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.xlim((0,0.5))
    plt.grid(lw=0.3)
    plt.savefig(outputdir+'\\'+'histogram of bending factor.png',bbox_inches='tight')
    plt.show()
    
    plt.figure(figsize=(6,6))
    plt.hist(p_list,np.arange(0,0.5,0.02),weights=np.ones_like(p_list)/len(p_list),histtype='step')
    plt.xlabel('Linear factor: p',weight='bold',fontsize=24)
    plt.ylabel('Probability',weight='bold',fontsize=24)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.xlim((0,0.5))
    plt.grid(lw=0.3)
    plt.savefig(outputdir+'\\'+'histogram of cap difference.png',bbox_inches='tight')
    plt.show()


#%%
print("Done.")