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

    ./mgls.py ./data/HD10180.dat --gls
![Alt text](https://github.com/rosich/mgls/blob/master/HD10180.png "HD10180")

To compute MGLS the dimensionality needs to be indicated:

    ./mgls.py ./data/HD10180.dat --ndim=4
    
    
