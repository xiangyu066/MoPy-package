# -*- coding: utf-8 -*-
"""
Created on Fri Dec 23 11:36:08 2022

@author: xiangyu066
"""

#%%
import numpy as np
from scipy import ndimage, signal
from scipy.signal import savgol_filter
from scipy.spatial import ConvexHull
from skimage import measure,morphology,filters
import matplotlib.pyplot as plt

#%%
def candidate_peak_finder_row(arr,r0,thresh):
    rows,cols=arr.shape
    arr_edge=np.zeros(arr.shape)
    arr_skel=np.zeros(arr.shape)
    for row in range(rows):
        localmaxs=[]
        localmins=[]
        arr_=arr[row,:]
        arr_=savgol_filter(arr_,19,2)                                           # 19 is characteristic size [px] for e. coli in PH100X and iXON
        peak_candis=[]
        
        # find local maxima
        for peak_candi in range(r0,cols-r0):
            left_mean=np.mean(arr_[peak_candi-r0:peak_candi])
            right_mean=np.mean(arr_[peak_candi+1:peak_candi+1+r0])
            if arr_[peak_candi]-left_mean>thresh and arr_[peak_candi]-right_mean>thresh:
                peak_candis.append(peak_candi)
        
        if len(peak_candis)>0:
            # consecutive numbers cluster
            break_point_=np.where(np.convolve(np.array(peak_candis),np.array([1,-1]),'same')>1)[0]
            for n in range(len(break_point_)):
                if n < len(break_point_)-1:
                    peak_=peak_candis[break_point_[n]:break_point_[n+1]-1]
                else:
                    peak_=peak_candis[break_point_[n]:-1]
                  
                # if only one peak in this range, then it is local maximum
                if len(peak_)==0: 
                    peak_=peak_candis[break_point_[n]]
                    localmaxs.append(peak_)
                else:
                    localmaxs.append(peak_[np.where(arr_[peak_]==np.max(arr_[peak_]))[0][0]])
                                  
            # find local minima
            for n in range(0,len(localmaxs)-1):    
                localmins.append(localmaxs[n]+np.where(arr_[localmaxs[n]:localmaxs[n+1]]==np.min(arr_[localmaxs[n]:localmaxs[n+1]]))[0][0])

            # check finding results ###########################################
            # plt.figure()
            # plt.plot(arr_,'.-',lw=0.5)
            # plt.plot(localmaxs,arr_[localmaxs],'o')
            # plt.plot(localmins,arr_[localmins],'s')
            ###################################################################
        
        ###########
        if len(localmaxs)>0:
            localmaxs[0]=localmaxs[0]+5                                         # shrink outer edge with 5 pixel (e. coli and PH100X + iXON)
            localmaxs[-1]=localmaxs[0-1]-5
        
        # output
        arr_edge[row,localmaxs]=1
        arr_skel[row,localmins]=1
    return arr_edge,arr_skel

def candidate_peak_finder_diag(arr,r0,thresh):
    rows,cols=arr.shape
    arr_edge=np.zeros(arr.shape)
    arr_skel=np.zeros(arr.shape)
    
    for col in range(cols-19):                                                  # 19 is characteristic size [px]
        localmaxs=[]
        localmins=[]
        arr_=np.zeros((rows-col,))
        for row in range(rows-col):
            arr_[row]=arr[row,row+col]
        
        arr_=savgol_filter(arr_,19,2)                                           # 19 is characteristic size [px]
        
        #find local maxima
        peak_candis=[]
        for peak_candi in range(r0,rows-col-r0):
            left_mean=np.mean(arr_[peak_candi-r0:peak_candi])
            right_mean=np.mean(arr_[peak_candi+1:peak_candi+1+r0])
            if arr_[peak_candi]-left_mean>thresh and arr_[peak_candi]-right_mean>thresh:
                peak_candis.append(peak_candi)
        
        if len(peak_candis)>0:
            # consecutive numbers cluster
            break_point_=np.where(np.convolve(np.array(peak_candis),np.array([1,-1]),'same')>1)[0]
            for n in range(len(break_point_)):
                if n < len(break_point_)-1:
                    peak_=peak_candis[break_point_[n]:break_point_[n+1]-1]
                else:
                    peak_=peak_candis[break_point_[n]:-1]
                  
                # if only one peak in this range, then it is local maximum
                if len(peak_)==0: 
                    peak_=peak_candis[break_point_[n]]
                    localmaxs.append(peak_)
                else:
                    localmaxs.append(peak_[np.where(arr_[peak_]==np.max(arr_[peak_]))[0][0]])
            
            # find local minima
            for n in range(0,len(localmaxs)-1):    
                localmins.append(localmaxs[n]+np.where(arr_[localmaxs[n]:localmaxs[n+1]]==np.min(arr_[localmaxs[n]:localmaxs[n+1]]))[0][0])
            
            # check finding results ###########################################
            # plt.figure()
            # plt.plot(arr_,'.-',lw=0.5)
            # plt.plot(localmaxs,arr_[localmaxs],'o')
            # plt.plot(localmins,arr_[localmins],'s')
            ###################################################################
        
        ###########
        if len(localmaxs)>0:
            localmaxs[0]=localmaxs[0]+5                                         # shrink outer edge with 5 pixel
            localmaxs[-1]=localmaxs[0-1]-5
        
        # output
        for n in range(len(localmaxs)): 
            arr_edge[localmaxs[n],localmaxs[n]+col]=1
        
        for n in range(len(localmins)):
            arr_skel[localmins[n],localmins[n]+col]=1
    return arr_edge,arr_skel

def candidate_peak_finder(arr,r0,thresh,is_check):
    # horizental profile
    bw_edge_0,bw_skel_0=candidate_peak_finder_row(arr,r0,thresh)
    
    # vertical profile
    bw_edge_90,bw_skel_90=candidate_peak_finder_row(arr.T,r0,thresh)
    bw_edge_90=bw_edge_90.T
    bw_skel_90=bw_skel_90.T
    
    # 45 tilt profile
    bw_edge_down_45,bw_skel_down_45=candidate_peak_finder_diag(arr,r0,thresh)
    bw_edge_up_45,bw_skel_up_45=candidate_peak_finder_diag(arr.T,r0,thresh)
    bw_edge_up_45=bw_edge_up_45.T
    bw_skel_up_45=bw_skel_up_45.T
    
    # 135 tilt profile
    bw_edge_down_135,bw_skel_down_135=candidate_peak_finder_diag(np.flip(arr,1),r0,thresh)
    bw_edge_down_135=np.flip(bw_edge_down_135,1)
    bw_skel_down_135=np.flip(bw_skel_down_135,1)
    bw_edge_up_135,bw_skel_up_135=candidate_peak_finder_diag(np.flip(arr,1).T,r0,thresh)
    bw_edge_up_135=np.flip(bw_edge_up_135.T,1)
    bw_skel_up_135=np.flip(bw_skel_up_135.T,1)
    
    # first-hand edge and skel
    bw_edge=1.0*bw_edge_0 + 1.0*bw_edge_90 + 1.0*bw_edge_down_45 + 1.0*bw_edge_up_45 + 1.0*bw_edge_down_135+1.0*bw_edge_up_135 >0
    edge_yy,edge_xx=np.where(bw_edge==1)
    bw_skel=1.0*bw_skel_0 + 1.0*bw_skel_90 + 1.0*bw_skel_down_45 + 1.0*bw_skel_up_45 + 1.0*bw_skel_down_135+1.0*bw_skel_up_135 >0
    skel_yy,skel_xx=np.where(bw_skel==1) 
    
    # check
    if is_check==True:
        plt.figure()
        plt.imshow(arr,cmap=plt.cm.gray,origin='lower')
        plt.plot(edge_xx,edge_yy,'r.',markersize=0.25)
        plt.title('local maxima candidat (edge)')
        plt.show()
        
        plt.figure()
        plt.imshow(arr,cmap=plt.cm.gray,origin='lower')
        plt.plot(skel_xx,skel_yy,'r.',markersize=0.25)
        plt.title('local minima candidate (skeleton)')
        plt.show()
        
        plt.figure()
        plt.plot(edge_xx,edge_yy,'.',skel_xx,skel_yy,'.',markersize=0.25)
        plt.xlim([0,arr.shape[1]])
        plt.ylim([0,arr.shape[0]])
        plt.gca().set_aspect('equal')
        plt.title('candidate')
        plt.show()
    return bw_edge,edge_xx,edge_yy,bw_skel,skel_xx,skel_yy

# concave detection ###########################################################
def concave_detector(Cs_,is_check):
    # convex hull
    x,y=Cs_[:,0],Cs_[:,1]

    hull = ConvexHull(Cs_)
    convex_pts = [Cs_[i] for i in hull.vertices]
    Cs_hull=np.zeros((len(convex_pts),2))
    for n in range(len(convex_pts)):
        Cs_hull[n,0]=convex_pts[n][0]
        Cs_hull[n,1]=convex_pts[n][1]
    x_=Cs_hull[:,0]
    x_=np.append(x_,Cs_hull[0,0])
    y_=Cs_hull[:,1]
    y_=np.append(y_,Cs_hull[0,1])
  
    dist=[]
    for n in range(len(x_)-1):
        dx=x_[n]-x_[n+1]
        dy=y_[n]-y_[n+1]
        dist.append(np.sqrt(dx**2+dy**2))
        
    # concave detection
    idxs=np.where(np.array(dist)>15)[0] # 10~20px is a criteria of concave candidate
    
    if is_check==True:
        plt.figure()
        plt.plot(x,y,x_,y_,'ro-')
        for n in range(len(idxs)):
            plt.plot(x_[idxs[n]:idxs[n]+2],y_[idxs[n]:idxs[n]+2],'gs--')
        plt.title('contour (b) & convex hull (r) & the longest hull boundary (gs--)')
        plt.gca().set_aspect('equal')
        plt.show()
    
    idxs_concave=[]
    is_human_checknum=0
    for n in range(len(idxs)):
        idx=idxs[n]
        
        # calculate distance from point to line
        slope=(y_[idx]-y_[idx+1])/(x_[idx]-x_[idx+1])
        const=y_[idx]-slope*x_[idx]
        idx_s=np.where(np.logical_and(x==x_[idx],y==y_[idx]))[0][0]
        idx_e=np.where(np.logical_and(x==x_[idx+1],y==y_[idx+1]))[0][-1]
        dist=[]
        dist_sum=0 # accumulate maximal distance from concave to line of convex hull
  
        for n in range(idx_s,idx_e):
            dist.append(np.abs(slope*x[n]-y[n]+const)/np.sqrt(slope**2+1))
        
        if np.max(dist)>3.2: # an ideal division case; 3~5px (~width/3) is a criteria
            idxs_concave.append(idx_s+np.where(dist==np.max(dist))[0][0])
            is_human_checknum=is_human_checknum+1
        
        dist_sum=dist_sum+np.max(dist) #
        if dist_sum>10: # an over two cells case (e.g. L-shape, or V-shape) 
            is_human_checknum=is_human_checknum+1
    
    if is_check==True:
        plt.figure()
        plt.plot(x,y,x_,y_,'ro-')
        for n in range(len(idxs_concave)):
            plt.plot(x[idxs_concave[n]],y[idxs_concave[n]],'bo')
        plt.title('concave candidates (bo)')
        plt.gca().set_aspect('equal')
        plt.show()
    
    # whether need to take care
    if is_human_checknum>1:
        is_human_check=True
    else:
        is_human_check=False
    
    return is_human_check,idxs_concave

def seg_ecoli_ph100_ixon(arr,thresh,bright_to_bg_factor,is_aux_seg,is_check):
    r0=4                                                                        # ~wudth of cell /4, unit is pixel
    bw_edge,edge_xx,edge_yy,bw_skel,skel_xx,skel_yy=candidate_peak_finder(arr,r0,thresh,is_check)
    
    # refine local maxima edge
    bw=morphology.binary_dilation(bw_edge,morphology.disk(1))
    bw=morphology.area_opening(bw,25)
    bw=morphology.thin(bw)
    if is_check==True:
        yy,xx=np.where(bw==1)
        plt.figure()
        plt.imshow(arr,cmap=plt.cm.gray,origin='lower')
        plt.plot(xx,yy,'r.',markersize=0.25)
        plt.title('refine edge (local maxima)')
        plt.show()
    
    # bright region (outer edge and overlap edge)
    sobel_edge = filters.sobel(arr)
    sobel_edge=sobel_edge>np.mean(sobel_edge)+2*np.std(sobel_edge)              # threshold of bright region
    sobel_edge=morphology.binary_closing(sobel_edge,morphology.disk(2))         # sobel edge enclose 
    sobel_edge=morphology.area_opening(sobel_edge,25)                           # remove small object
    if is_check==True:
        plt.figure()
        plt.imshow(arr,cmap=plt.cm.gray,origin='lower')
        plt.title('original')
        
        plt.figure()
        plt.imshow(sobel_edge,origin='lower')
        plt.title('sobel edge')
    
    bw_cluster=ndimage.binary_fill_holes(sobel_edge)
    bw_cluster=morphology.binary_dilation(bw_cluster,morphology.disk(4*r0))
    bw_cluster=1.0-1.0*bw_cluster>0
    if is_check==True:
        plt.figure()
        plt.imshow(bw_cluster,origin='lower')
        plt.title('cluster mask (from sobel edge)')
    
    bw_=arr>bright_to_bg_factor*np.mean(arr[bw_cluster>0])                                      # 1.2 means greater than 20% of background; default 1.2~1.5
    if is_check==True:
        plt.figure()
        plt.imshow(bw_,origin='lower')
        plt.title('bright parts')
    
    bw= 1.0*bw + 1.0*bw_ >0
    bw=bw*(1.0-1.0*bw_cluster>0)

    # label
    bw=1.0-1.0*bw>0
    if is_check==True:
        plt.figure()
        plt.imshow(bw,origin='lower')
        plt.title('refine contour')
    
    bw=morphology.binary_erosion(bw,morphology.disk(3))                         # enforce break connection 
    if is_check==True:
        plt.figure()
        plt.imshow(bw,origin='lower')
        plt.title('enforce break')
    
    L=measure.label(bw,connectivity=1)
    N=np.max(L)
    seg=[]
    L_ns=[]
    for n in range(2,N+1):
        L_n=(L==n)
        rows,cols=np.where(L_n==True)
        if len(rows)>(((3*r0))**2)/2: # filter out small object; zyla set 3, ixon set 4
            L_n=morphology.binary_dilation(L_n,morphology.disk(4))
            L_n=morphology.binary_erosion(L_n,morphology.disk(2))   
            L_ns.append(L_n)
            Cs=np.roll(measure.find_contours(L_n,0.5)[0],1,axis=1)              # contours
            
            # auxiliary segmentation
            if is_aux_seg==True:
                is_human_check,idxs_concave=concave_detector(Cs,is_check)
                if is_human_check==True:
                    if len(idxs_concave)==2:
                        if idxs_concave[0]<idxs_concave[1]:
                            Cs_1=np.concatenate((Cs[idxs_concave[0]:idxs_concave[1],:],Cs[idxs_concave[0],:].reshape(1,2)))
                            Cs[idxs_concave[0]:idxs_concave[1],:]=np.nan
                            Cs_2=Cs[~np.isnan(Cs)].reshape(-1,2)
                        else:
                            Cs_1=np.concatenate((Cs[idxs_concave[1]:idxs_concave[0],:],Cs[idxs_concave[1],:].reshape(1,2)))
                            Cs[idxs_concave[1]:idxs_concave[0],:]=np.nan
                            Cs_2=Cs[~np.isnan(Cs)].reshape(-1,2)
                        seg.append(Cs_1)
                        seg.append(Cs_2)
                    else:
                       seg.append(Cs) 
                else:
                    seg.append(Cs)
            else:
                seg.append(Cs)
    return seg,L_ns

# create relation between current and previous frame #########################
def cell_conncetion(arr,Cs,old_Cs):
    print('Create cell connection list between frames...')
    L=-np.ones(arr.shape)
    old_L=-np.ones(arr.shape)
    N=len(Cs)
    for n in range(N):
        cols,rows=Cs[n][:,0],Cs[n][:,1]
        cols=np.round(cols).astype('int')
        rows=np.round(rows).astype('int')
        L_=np.zeros(arr.shape)
        for m in range(len(cols)):
            L_[rows[m],cols[m]]=1
        L_=ndimage.binary_fill_holes(L_)
        rows,cols=np.where(L_==True)
        L[rows,cols]=n
    
    old_nums=[]
    new_nums=[]
    if (len(old_Cs)==0):
        for n in range(N):
            L_n=(L==n)
            new_nums.append([n])
        connection_list=np.array(new_nums)
    else:
        old_N=len(old_Cs)
        for old_n in range(old_N):
            cols,rows=old_Cs[old_n][:,0],old_Cs[old_n][:,1]
            cols=np.round(cols).astype('int')
            rows=np.round(rows).astype('int')
            L_=np.zeros(arr.shape)
            for m in range(len(cols)):
                L_[rows[m],cols[m]]=1
            L_=ndimage.binary_fill_holes(L_)
            rows,cols=np.where(L_==True)
            old_L[rows,cols]=old_n
        
        # shift correction
        dx,dy=fast_xcorr2((L>-1),(old_L>-1))                                    # calculate 2d correlation by FFT
        old_L=np.roll(old_L,dy,axis=0)
        old_L=np.roll(old_L,dx,axis=1)
        
        for n in range(N):
            L_n=(L==n)
            m=int(np.median(old_L[np.where(L_n==True)]))                        # more overlap region as evolution candidate 
            old_nums.append(m)
            new_nums.append(n)
        connection_list=np.array(np.transpose([old_nums,new_nums]))             # [previous, current]
    return connection_list

# convert conncetion list into a chain-code array ############################
def cell_connection_chaincode(connection_lists):
    print('Convert the conncetion list into a chain-code array...')
    nSteps=len(connection_lists)                                                # nSteps=nFrames
    nSeeds=np.shape(connection_lists[nSteps-1])[0]
    chaincode_array=np.zeros((nSeeds,nSteps), dtype=np.int)
    brokenCounts=0
    for nSeed in range(0,nSeeds):
        for nStep in range(nSteps-1,-1,-1):
            if (nStep==nSteps-1):
                chaincode_array[nSeed,nStep]=connection_lists[nStep][nSeed,1]
                previous_num=connection_lists[nStep][nSeed,0]
            elif (nStep==0):
                chaincode_array[nSeed,nStep]=previous_num
                idx=(np.where(connection_lists[nStep][:,0]==previous_num))[0]
                if (len(idx)==0):                                               # if the cell lost contact, then break loop to next cell
                    brokenCounts=brokenCounts+1    
                    break
                previous_num=connection_lists[nStep][idx,0]
            else:
                chaincode_array[nSeed,nStep]=previous_num
                idx=(np.where(connection_lists[nStep][:,1]==previous_num))[0]
                if (len(idx)==0):
                    brokenCounts=brokenCounts+1  
                    break
                previous_num=connection_lists[nStep][idx,0]
    return chaincode_array, brokenCounts

# two-dimensional fast cross-correlation #####################################
def fast_xcorr2(bw1,bw2):
    bw2=1.0*bw2
    corr=signal.fftconvolve(1.0*bw1,bw2[::-1,::-1],mode='same')
    y,x=np.unravel_index(np.argmax(corr), corr.shape)
    shift_y=int(y-np.shape(bw1)[0]/2)
    shift_x=int(x-np.shape(bw1)[1]/2)
    return shift_x, shift_y

