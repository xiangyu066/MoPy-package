# MoPy package
MoPy package is a very powerful and efficient analysis toolbox for the bacterial flagellar motor (BFM) research. The most research of BFM contains the mean-square displacement (MSD) calculation, rotation bead in 2-dimensional fitting, image segmentation and bacteria profile measurement. However, when you collect tens to hundreds of thousands of data points or images, the analysis time is at least hours to days. MoPy package is developed with parallel computing, including GPU and vectorization. Researchers can easily use this package to process high-throughput data in minutes to hours.\
\
**[Modules]:**
- [MSD2](https://github.com/xiangyu066/MoPy-package/blob/main/Docs/MSD2.md): MSD calculation (Let the time complexity O(N^3) to O(N).)
- [BeadAssay](https://github.com/xiangyu066/MoPy-package/blob/main/Docs/BeadAssay.md): Bead-assay experiment (rotation speed analysis) and bead profile measurement (FWHM analysis)
- [BacterialContour](https://github.com/xiangyu066/MoPy-package/blob/main/Docs/BacterialContour.md): Bacterial contour analysis (including length, width, bending and cap size)
- [DDM]()
- [PhaseSpatialTracker](https://github.com/xiangyu066/Phase3DTracker): A single particle 3D tracking tool based on the phase-contrast imaging. 

**[dataSets]:** (Due to large size of test dataset, please download from below link and unzip to directory "dataSets".)\
[Download link](https://drive.google.com/drive/folders/1-JJgZDJw9vMe6YCLrdv64CB9P9j4LrH4?usp=sharing)\
\
**[Demo]:**
- Bead-assay:\
  <img src="https://github.com/xiangyu066/MoPy-package/blob/main/Docs/Demo_BA.png" width="70%">
- Evolution Tree:\
  <img src="https://github.com/xiangyu066/MoPy-package/blob/main/Docs/Demo_ET.PNG" width="100%">
- MSD:\
  <img src="https://github.com/xiangyu066/MoPy-package/blob/main/Docs/MSD1.png" width="70%">\
  <img src="https://github.com/xiangyu066/MoPy-package/blob/main/Docs/MSD3.png" width="70%">
- Bacterial contour:\
  <img src="https://github.com/xiangyu066/MoPy-package/blob/main/Docs/Demo_BC.PNG" width="70%">
