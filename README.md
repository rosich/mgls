# MGLS
Multidimensional Generalized Lomb-Scargle Periodogram

## Dependences (tested)

* Python 2.7+
* numpy 1.7+
* matplotlib 1.4+
* gfortran compiler
* Lapack library (fast linear algebra routines)

## Basic Usage

### Compile Fortran numerical modules (gfortran compiler must be installed on your system)

    $ bash compile_fortran_modules.sh

#### General syntax

    ./mgls.py <data_file_path> --option_1 --option_2 --option_n...
    
#### Options available    
    --gls               :: compute and plot unidimensional Generalized Lomb-Scargle periodogram.
    --pmin= / --pmax=   :: set limits in periods to be explored [in units of data file first column]. Prestablished values are 1.5-10000.
    --jitter            :: fit additional jitter (s) in quadrature (e^2 = sigma^ + s^2) [in units of data file third column].
    --period            :: [to be used with gls option] plot GLS periodogram in period log-scale.
    --ndim=             :: number of signals to be fitted.
    --linear_trend      :: fit a linear trend simultaneously.        

#### Multiset 
MGLS admits multiple input data sets. The computation yelds in fitting an additional offset parameter (and jitter) for each set which optimizes the overall fit. The datasets could be either different instruments or distinct observational epochs that need to be treated separately. In this case the syntax is: (regular expresions are allowed)

    ./mgls.py <data_file_path_1>  <data_file_path_2>  <data_file_path_n> --option_1 --option_2 --option_n...

#### MGLS Principles of working
MGLS is a new approach for detecting multiplanetary systems by simultaneous fitting of n-tuple of circular orbits.  

### Examples

#### Simulated data
##### 1-GLS
In folder ./data/synth_hexa_model.dat there is a simulated time series containing 6 signals, in a 240 points file (BJD, RV, RV_err) 

In order to compute genuine GLS periodogram (with matplotlib graphical output)

    ./mgls.py ./data/synth_hexa_model.dat --gls --period --pmin=1.0 --pmax=3000 --jitter 
![Alt text](https://github.com/rosich/mgls/blob/master/hexa_gls.png "hexa")

or plotting in frequency linear-scale
     
     ./mgls.py ./data/synth_hexa_model.dat --gls --pmin=1.0 --pmax=3000 --jitter 
![Alt text](https://github.com/rosich/mgls/blob/master/hexa_gls_f.png "hexa_f")

#### MGLS
To run MGLS (fitting multiple frequencies simultaneously) the dimensionality needs to be indicated:

    ./mgls.py ./data/synth_hexa_model.dat --ndim=6 --pmin=1.0 --pmax=3000 --jitter
    
Ouput:


        ┌Dataset summary──────────────────────┬─────────────┬────────────┬───────────┬─────────────┐
        │ Data set name (full path not shown) │ Data points │ Timespan   │ Mean sep. │ logL (data) │
        ├─────────────────────────────────────┼─────────────┼────────────┼───────────┼─────────────┤
        │ 0/ synth_hexa_model.dat             │ 240         │ 2542.91478 │ 10.59548  │ -666.3606   │
        └─────────────────────────────────────┴─────────────┴────────────┴───────────┴─────────────┘

        Evaluating 0-model...
                logL null model (data + jitter): -608.344674696
                0-freq jitter(s):
                        0/ 2.3055

        Evaluating model...
        Computing candidate frequency tuples...
        Maximizing the candidates...

        Periods (d):
                23.0582
                97.24027
                9.05554
                179.53743
                19.66723
                1.14629

        Fitting coefficients:
                a[ 0 ]: -1.69258
                a[ 1 ]: -0.659809
                a[ 2 ]: 0.450278
                a[ 3 ]: 1.09982
                a[ 4 ]: -1.34973
                a[ 5 ]: -0.492108
                b[ 0 ]: -0.590401
                b[ 1 ]: -0.816941
                b[ 2 ]: -0.73112
                b[ 3 ]: -0.728609
                b[ 4 ]: -1.06838
                b[ 5 ]: -0.756614

        Amplitudes / Uncertainties
                K[ 0 ]: 1.79259622732 +/- 0.234870072349
                K[ 1 ]: 1.05011481604 +/- 0.261906614526
                K[ 2 ]: 0.85865408891 +/- 0.256306518122
                K[ 3 ]: 1.31927189397 +/- 0.262636874708
                K[ 4 ]: 1.72139498255 +/- 0.265989614476
                K[ 5 ]: 0.902571557908 +/- 0.256376972598

        Offsets / Uncertainties
                c[ 0 ]: 7.36772 +/- 0.129196569772

        Jitters:
                set[ 0 ]: 0.000220087673173

        Spectral stats:
                Joint power: 0.17755
                logL_0 (no model): -608.344674696
                logL (model): -500.330890497
                DlogL (model vs. data): 108.013784199

#### Observational data

##### Bidimensional map (HD41248)

The following command plots a bidimensional colour map of a period scan defined in line 324 of file mgls.py 

    ./mgls.py ./data/synth_hexa_model.dat --bidim --jitter --linear_trend
    
![Alt text](https://github.com/rosich/mgls/blob/master/bidim.png "bidim")

This data corresponds to observations of HD41248. The maximum is located at 18.4 d / 25.5 d
    
    


