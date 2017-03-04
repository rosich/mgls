# MGLS
Multidimensional Generalized Lomb-Scargle Periodogram

## Basic Usage

### Compile Fortran numerical modules (gfortran compiler must be installed on your system)

    $ bash compile_fortran_modules.sh

#### General syntax

    ./mgls.py <data_file_path> --options
    
### Example
In folder ./data there is a real time series of GJ581, containing 240 points (BJD, RV, RV_err) 

In order to compute genuine GLS periodogram (with matplotlib graphical output)

    ./mgls.py ./data/GJ581_240.dat --gls
![Alt text](mgls/GJ581.png, "GJ581")

To compute MGLS the dimensionality needs to be indicated:

    ./mgls.py ./data/GJ581_240.dat --ndim=4
    
    
