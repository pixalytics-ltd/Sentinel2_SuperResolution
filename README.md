# Sentinel-2 Super Resolution
This repository included pieces for a Python implementation of the RCNN method described by Latte et al. (_Preprocessing (adapted from section 2.4 of the Latte paper):_
- "S2 Pyrite" and "PS BeautySchoolDropout" collectively handle all of the preprocessing steps for Sentinel-2 and Dove, respectively. 
- "SuperResTraining4x_withMask" and "SuperResTraining8x_withMask" take the input generated by "S2 Pyrite" and "PS BeautySchoolDropout" to train the super resolution neural network. Saves the models to a location of your choice.
- "FortuneTeller" reads in the trained models from the super resolution NN and uses the native resolution imagery (also produced by "S2 Pyrite-no Mask" and "PS BeautySchoolDropout") to predict imagery with S2 spectral characteristics at 2.5m resolution.  Its a beautiful thing...
- To create an Anaconda environment to run all of this: "requirements.txt" provides specific libraries to replicate this env. If you start fresh, create an environment first with TensorFlow (gpu). Then install Arosics per the documentation (https://danschef.git-pages.gfz-potsdam.de/arosics/doc/ , at time of writing, the code is: 'conda install -c conda-forge 'arosics>=1.3.0'). Arosics is a PITA, and I highly recommend you get its neediness out of the way first. Conveniently, the dependencies installed by Arosics basically takes care of most of the rest of the requirements needed to run these notebooks.  The last ones you will need are rasterio and scikit-learn.
