# -*- coding: utf-8 -*-
"""
Created on Wed Dec 28 10:34:16 2022

@author: xiangyu066

Notes:
    
    1. The advance analysis (e.g. size measurement) can visit BacterialContour_Example4.py
    2. If labell with red tag, then you have to specially process it.
    
"""

#%%
print("Running...")
import os, glob, pickle
import numpy as np
from skimage import io
import matplotlib.pyplot as plt
import time
import warnings
warnings.simplefilter("ignore")

# 
import MoPy.EvolutionTree as ET
import MoPy as mo
print("MoPy package is "+mo.__version__+".")

print("------------------------------------------------------------------")

#%%
# process parameters
th=75 # peak thershold

# working directory
# inputdir=r'.\dataSets\EvolutionTree\E. coli\HM06 (from CYC)\Gelpad\1220_PC test' 
inputdir=r'.\dataSets\EvolutionTree\V. alginolyicus\VIO5\Gelpad\20170615'


#%% Initialization
print('Initializing...')
listing=glob.glob(inputdir+'\\*.tif')
nFiles=len(listing)

#%%
TrackTrees=[]
TrackLists=[]
for nFile in range(nFiles):
    print('Load files...(current file: '+str(nFile+1)+' / total files: '+str(nFiles)+')')
    inputfile=listing[nFile]
    origina_=io.imread(inputfile)
    nFrames=origina_.shape[0]

    # batch processing
    Cs_list=[]
    branches=[]
    for nFrame in range(112,113):
        print('Processing...(current frame: '+str(nFrame+1)+' / total frames: '+str(nFrames)+')')
        
        # Create a single directory
        outputdir_name=inputfile.replace(inputdir+'\\','')
        outputdir_name=outputdir_name.replace('.tif','')
        outputdir=inputdir+'\\Analyzed\\'+outputdir_name
        if not os.path.exists(outputdir): os.makedirs(outputdir)

        tic=time.time()
        origina=origina_[nFrame,:,:]
        
        # segmentation
        Cs_,L_ns=ET.seg_ecoli_ph100_ixon(origina,th,bright_to_bg_factor=1.2,is_check=True)
        Cs_list.append(Cs_)
        
        print("--- %s seconds ---" % (time.time()-tic))
        
        # show
        plt.figure()
        plt.imshow(origina,cmap=plt.cm.gray,origin='lower')
        for n in range(len(Cs_)):
            Cs=Cs_[n]
            plt.plot(Cs[:,0],Cs[:,1],color=np.random.rand(1,3),lw=0.3)
            
            # auxiliary segmentation
            is_human_check=ET.concave_detector(Cs,is_check=False)
            if is_human_check==True:
                plt.text(np.mean(Cs[:,0]),np.mean(Cs[:,1]),str(n),
                         color='black',fontsize=4,
                         horizontalalignment='center',verticalalignment='center',
                         bbox=dict(boxstyle="square",
                                   ec='none',
                                   fc=(1., 0.8, 0.8),alpha=0.7))
            else:
                plt.text(np.mean(Cs[:,0]),np.mean(Cs[:,1]),str(n),
                         color='white',fontsize=4,
                         horizontalalignment='center',
                         verticalalignment='center')
        plt.title('nFrame = %d'%(nFrame))
        plt.savefig(outputdir+'\\'+str(nFrame)+'.png',bbox_inches='tight')
        plt.show() 
        
        # create connetion between current and previous
        if (nFrame==0):
            branches.append(ET.cell_conncetion(origina,Cs_,[]))                 
        else:
            branches.append(ET.cell_conncetion(origina,Cs_,Cs_list[nFrame-1]))
    
    # build evolution tree
    Tree,_=ET.cell_connection_chaincode(branches)                               # [nCell,ndT]
    TrackTrees.append(Tree)                                                     # [nFile]    
    TrackLists.append(Cs_list)

#%% save database
print('Save analyzed datasets into the computer....')
outputdir=inputdir+'\\Analyzed'
outputfile=outputdir+'\\Seg_label_results'
Results=[TrackTrees,TrackLists]
with open(outputfile,'wb') as f: pickle.dump(Results,f)

#%%
print('Done.')