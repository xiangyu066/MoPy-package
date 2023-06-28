# -*- coding: utf-8 -*-
"""
Created on Tue Nov 16 18:26:22 2021

@author: XYZ
"""

__version__="1.8.5"

"""
Next:
    1. The module of EvolutionTree have to design a process to converge the contour points.


Log:
    < version = 1.8.5 > - 2023.06.28
        1. Rename MSD to MSD2 and modify name of variables in MSD_Example.py.
        2. Add saving fitting results in BacterialContour_Example3.py.
    
    < version = 1.8.4 > - 2023.03.29
        1. Add a parameter "fit_type" of RotateSpeed_calc in the module of BeadAssay. (need to modify some examples)  
    
    < version = 1.8.3 > - 2023.03.28
        1. Add Cauchy_2D in the module of BeadAssay.
    
    < version = 1.8.2 > - 2023.03.04
        1. Define a better bright_to_background factor can improve the segmentaion performance.
        2. Add "auxiliary segmentation" to break connection body. 
    
    < version = 1.8.1 > - 2023.01.09
        1. Auxiliary segmentation (find concave points by convex hull; ref: Concave_detection_test0.py)
    
    < version = 1.8.0 > - 2022.12.23
        1. The module of EvolutionTree has better segemetation than the module of PhaseSeg.
        2. EvolutionTree_Example1.py and BacterialContour_Example4.py is a complete example for size analysis.
    
    < version = 1.7.0 > - 2022.03.30
        1. Automatically pick out fluorescence beads for FWHM measurement.
        2. Deal with nan-term for bead assay fitting. Please visit details for BeadAssay_Example4.py 
    
    < version = 1.6.4 > - 2022.03.02
        1. Add an example for FWHM measurement by fluorescent beads.
        2. Modify BeadAssay_Example1.py
    
    < version = 1.6.3 > - 2022.02.22
        1. Add histogram to BacterialCOntour_Example3.py
        2. Modify the lower bound of the geometric order.
        
    < version = 1.6.2 > - 2022.02.19
        1. A simple example to describe the bacterial equation details.
    
    
"""