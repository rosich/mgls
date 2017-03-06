# MGLS
Multidimensional Generalized Lomb-Scargle Periodogram

## Dependences (tested)

-Python 2.7+
-matplotlib 1.4+
-gfortran compiler
-Lapack library
-

## Basic Usage

### Compile Fortran numerical modules (gfortran compiler must be installed on your system)

    $ bash compile_fortran_modules.sh

#### General syntax

    ./mgls.py <data_file_path> --options
    
### Example
In folder ./data there is a real time series of HD10180, containing 190 points (BJD, RV, RV_err) 

In order to compute genuine GLS periodogram (with matplotlib graphical output)

    ./mgls.py ./data/HD10180.dat --gls
![Alt text](https://github.com/rosich/mgls/blob/master/HD10180.png "HD10180")

To run MGLS (fitting multiple frequencies simultaneously) the dimensionality needs to be indicated:

    ./mgls.py ./data/HD10180.dat --ndim=7
    
Ouput:

    Data points: 190
    Time span: 2428.194514
    Mean sep. 12.7799711263
    Computing candidate frequency tuples...
    Maximizing the candidates...
    Joint power: 0.956656714659
    Periods (d):
            16.3529
            608.64887
            2289.9372
            122.66201
            49.74391
            1.17767
            5.7597


    Fitting coefficients: [  3.55296707e+01  -5.28452074e-05   4.04475461e-04   4.86161909e-04
       2.03763554e-03   2.51690252e-03   3.46886052e-04   3.00662057e-03
      -2.74733012e-03  -1.32863445e-03  -2.94530322e-03  -2.12702714e-03
      -3.51821329e-03   7.16500275e-04  -3.43540288e-03]
    Chi-2_0: 28119.7664143
    Chi-2: 1218.8030594
    K[ 0 ]: 0.00274783831089 5.53183441873e-05
    K[ 1 ]: 0.00138883760499 7.96759776417e-05
    K[ 2 ]: 0.00298515735772 7.31128673726e-05
    K[ 3 ]: 0.00294553951578 8.26984575136e-05
    K[ 4 ]: 0.00432580894197 7.85221878126e-05
    K[ 5 ]: 0.000796054381169 7.37701443339e-05
    K[ 6 ]: 0.00456527766333 7.71905441777e-05

    -641.33975898
    -1203.97415688
    Reduced Chi-2: 7.04510438961
    
    
