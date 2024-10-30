# -*- coding: utf-8 -*-
"""
Created on Tue Dec 28 20:41:19 2021

@author: XYZ
"""

#%%
import numpy as np
from skimage import measure, filters, morphology
from scipy import ndimage
import matplotlib.pyplot as plt

#%%
def cell_mask(Img,DL,division_coeff,method):
    if (method=='Otsu'):
        thresh=filters.threshold_otsu(Img)
        bw=(Img<thresh)
    elif (method=='Customized'):
        # rough definition of background
        bw=(Img<1.5*np.min(Img))                                                # cell area
        bw=ndimage.gaussian_filter(1.0*bw,sigma=3*DL)
        bw=(bw>0)
        dc=np.mean(Img[np.where(bw==False)])                                    # assume that background is a constant value
        
        # the definition of customized extracellular boundary
        thresh=division_coeff*dc+(1-division_coeff)*np.min(Img)
        bw=(Img<thresh)
        bw=ndimage.binary_fill_holes(bw)
        kernal=morphology.disk(np.round(DL))
        bw=morphology.binary_opening(bw,kernal)
    return bw

# remove cells near border ###################################################
def cell_filter_border(bw,eff_pixelsize,R):
    R=np.round(R/eff_pixelsize)                                                 # a border distance
    L=measure.label(bw,connectivity=1)
    N=np.max(L)
    upper=np.shape(L)-np.array([R,R])                                           # upper bound for x- and y-axis
    for n in range(1,N+1):
        L_n=(L==n)
        row,col=np.where(L_n==True)
        checksum=np.sum((row<=R)+(row>=upper[0])+(col<=R)+(col>=upper[1]))      # check whether cell exist on the boundary 
        if (checksum>0):
            L[row,col]=0
    bw=(L>0)
    return bw

# find contour ###############################################################
def cell_outer_contours(bw):
    L=measure.label(bw,connectivity=1)
    N=np.max(L)                                                               
    Cs=[np.roll(measure.find_contours((L==n),0.5)[0],1,axis=1) for n in range(1,N+1)] # contours
    Cs=smooth_contours(Cs)
    return Cs

# smooth binary contour ######################################################
def smooth_contours(Cs):
    for n in range(len(Cs)):
        rho,phi=cart2pol(Cs[n][:,0],Cs[n][:,1])
        T=15
        N=len(phi)
        phi_extend=np.insert(phi,0,phi[N-T:N])
        phi_extend=np.append(phi_extend,phi[0:T])
        phi_extend=np.convolve(phi_extend,np.ones((T,))/T,mode='same')
        rho_extend=np.insert(rho,0,rho[N-T:N])
        rho_extend=np.append(rho_extend,rho[0:T])
        rho_extend=np.convolve(rho_extend,np.ones((T,))/T,mode='same')
        phi=phi_extend[T:T+N]
        rho=rho_extend[T:T+N]
        Cs[n][:,0],Cs[n][:,1]=pol2cart(rho,phi)
    return Cs

# convert cartesian coordinate into polar coordinate #########################
def cart2pol(x,y):
    rho=np.sqrt(x**2+y**2)
    phi=np.arctan2(y,x)
    return rho, phi

# convert polar coordinate into cartesian coordinate #########################
def pol2cart(rho,phi):
    x=rho*np.cos(phi)
    y=rho*np.sin(phi)
    return x, y

# remove intimate cells ######################################################
def cell_filter_neighbors(bw,eff_pixelsize,R):
    R=R/eff_pixelsize                                                           # a distance between bacteria
    L=measure.label(bw,connectivity=1)
    N=np.max(L)
    Cs=cell_outer_contours(bw)                                                  # contours
    idxs=[]                                                                     # a removing listing of intimate cell
    for n in range(N):
        Cn=Cs[n]                                                                # n-th contour
        checkTF=False
        for m in range(N):
            if (n!=m):
                Cm=Cs[m]                                                        # m-th contour
                for i in range(len(Cn)):                                        # i-th point in n-th contour
                    dr=Cn[i]-Cm
                    dist=np.sqrt(dr[:,0]**2+dr[:,1]**2)
                    checksum=np.sum((dist<R))
                    if (checksum>0):
                        idxs.append(n+1)
                        checkTF=True
                        break
            if (checkTF):
                break
    for n in range(len(idxs)):
        L_n=(L==idxs[n])
        row,col=np.where(L_n==True)
        L[row,col]=0
    bw=(L>0)
    return bw

# consistent boundary with Otsu's threshold
def otsu_correct(bw):  
    kernal=morphology.disk(5)
    bw=morphology.binary_dilation(bw,kernal)
    return bw

# label number and find countour #############################################
def bwlabel(bw):
    L=measure.label(bw,connectivity=1)
    N=np.max(L)
    Cs=[]
    for n in range(1,N+1):
        L_n=(L==n)
        # L_n=otsu_correct(L_n)
        row,col=np.where(L_n==True)
        L[row,col]=n
        C=cell_outer_contours(L_n)
        Cs.append(C[0])
    return L, N, Cs

# calculate cell centroids ###################################################
def cell_centroids(bw):
    L=measure.label(bw,connectivity=1)
    N=np.max(L)
    pts=[np.flip(np.mean(np.where((L==n)==True),1),0) for n in range(1,N+1)]    # centroids
    return pts

# draw elements after image process ##########################################
def drawElements(Img,Cs,CMs,label_num_list):                                    # Cs: contours, CMs: centroids
    print('Draw elements...')
    fig, ax = plt.subplots(figsize=(10,10))
    # plt.get_current_fig_manager().window.showMaximized()
    ax.imshow(Img, cmap=plt.cm.gray)   
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    N=len(CMs)
    for n in range(0,N):
        ax.plot(Cs[n][:,0],Cs[n][:,1],lw=0.5)
        if (label_num_list==np.Inf):
            ax.text(CMs[n][0],CMs[n][1],str(n+1),color='white')
        else:
            ax.text(CMs[n][0],CMs[n][1],str(label_num_list[n]),color='white')
    plt.show()