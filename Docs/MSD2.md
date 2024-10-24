# Module tutorial of MSD2
1. Simulate the Brownian particles in 2D, then calculate the time-averaged ensemble and ensemble MSD.  Particularly, in the time-averaged ensemble MSD, the computation load is very heavy  (the square of timesteps multiply the number of particle).  Hence, the time-averaged ensemble MSD is a good test example for GPU computation.\
   <img src="https://github.com/xiangyu066/MoPy-package/blob/main/Docs/MSD1.png" width="100%">
2. Before start to GPU computation, we have to modify calculation algorithm by vectorization.\
   part 1: The vectorization of ensemble MSD.\
   <img src="https://github.com/xiangyu066/MoPy-package/blob/main/Docs/MSD2a.png" width="100%">\
   part 2: The vectorization of time-average ensemble MSD.\
   <img src="https://github.com/xiangyu066/MoPy-package/blob/main/Docs/MSD2b.png" width="100%">\
   part 3: The vectorization of time-average MSD without for-loop.\
   <img src="https://github.com/xiangyu066/MoPy-package/blob/main/Docs/MSD2c.png" width="100%">
4. The computing time is an one round test. From below test result, in the computer-1, the parallel computation is 5X faster than for-loop, and then the GPU computation can boost computing efficiency up to 50X comparison by for-loop.\
   <img src="https://github.com/xiangyu066/MoPy-package/blob/main/Docs/MSD3.png" width="70%">
