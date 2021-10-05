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

    $ bash compile_numerical_modules.sh

#### General syntax

    ./mgls.py <data_file_path> --option_1 --option_2 --option_n...
    
#### Options available    
    --gls               :: compute and plot unidimensional Generalized Lomb-Scargle periodogram.
    --pmin= / --pmax=   :: set limits in periods to be explored [in units of data file first column]. Prestablished values are 1.5-10000.
    --jitter            :: fit additional jitter (s) in quadrature (e^2 = sigma^ + s^2) [in units of data file third column].
    --ndim=             :: number of signals to be fitted.
    --linear_trend      :: fit a linear trend simultaneously.  
    --bootstrapping=    :: perform a given number of bootstrap samples to set 10%, 1% and 0.1% FAP levels.

#### Multiset 
MGLS admits multiple input data sets. The computation yelds in fitting an additional offset parameter (and jitter) for each set which optimizes the overall fit. The datasets could be either different instruments or distinct observational epochs that need to be treated separately. In this case the syntax is: (regular expresions are allowed)

    ./mgls.py <data_file_path_1>  <data_file_path_2>  <data_file_path_n> --option_1 --option_2 --option_n...

#### MGLS Principles of working
MGLS is an approach for detecting multiplanetary systems by simultaneous fitting of n-tuple of sinusoidal functions.  

### Examples

#### Simulated data
##### 1-GLS
In folder ./data/synth_hexa_model.dat there is a simulated time-series containing 6 signals, in a 240 points file (e.g. {BJD, RV, RV_err}) 

In order to compute genuine GLS periodogram (with matplotlib graphical output)

    ./mgls.py ./data/synth_hexa_model.dat --gls --period --pmin=1.0 --pmax=3000 --jitter 
![Alt text](https://github.com/rosich/mgls/blob/master/hexa_gls.png "hexa")

#### MGLS
To run MGLS (fitting multiple frequencies simultaneously) the dimensionality needs to be indicated:

    ./mgls.py ./data/synth_hexa_model.dat --ndim=6 --pmin=1.0 --pmax=3000 --jitter
    
Ouput:

                                               
       ███╗   ███╗ ██████╗ ██╗     ███████╗   
       ████╗ ████║██╔════╝ ██║     ██╔════╝   
       ██╔████╔██║██║  ███╗██║     ███████╗   
       ██║╚██╔╝██║██║   ██║██║     ╚════██║   
       ██║ ╚═╝ ██║╚██████╔╝███████╗███████║   
       ╚═╝     ╚═╝ ╚═════╝ ╚══════╝╚══════╝   
     MULTIDIMENSIONAL GENERALIZED LOMB-SCARGLE      
         Albert Rosich (rosich@ice.cat)             


Detected CPUs / using CPUs: 4/4

Reading multiset data

┌Dataset summary─────────────────────┬─────────────┬────────────┬───────────┬─────────────┐
│ Dataset name (full path not shown) │ Data points │ Timespan   │ Mean sep. │ logL (data) │
├────────────────────────────────────┼─────────────┼────────────┼───────────┼─────────────┤
│ 0/ synth_hexa_model.dat            │ 240         │ 2542.91478 │ 10.59548  │ -666.3606   │
└────────────────────────────────────┴─────────────┴────────────┴───────────┴─────────────┘
Max. jitter: 20.0

Evaluating 0-model...                                                                                                                  
        logL null model (data + jitter): -608.3446746953898
        0-freq jitter(s):
                0/ 2.3055

Evaluating model...
Computing candidate frequency tuples...
Maximizing the candidates...

Periods (au):
        23.07071
        9.05676
        180.28757
        6.02493
        97.37801
        18.64276

Fitted coefficients / Uncertainties
        a[ 0 ]: -1.7635815 +/- 0.198413952985
        a[ 1 ]: 0.6723117 +/- 0.19943871229
        a[ 2 ]: 0.6816219 +/- 0.190459772326
        a[ 3 ]: -0.23263012 +/- 0.197732571051
        a[ 4 ]: -0.7047149 +/- 0.200529145816
        a[ 5 ]: 0.02121678 +/- 0.197920853587
        b[ 0 ]: -0.12779249 +/- 0.199068686121
        b[ 1 ]: -1.1055146 +/- 0.19984888947
        b[ 2 ]: -0.93733495 +/- 0.214947848676
        b[ 3 ]: -0.79712594 +/- 0.197484140891
        b[ 4 ]: -0.6766557 +/- 0.207615854684
        b[ 5 ]: -1.2103678 +/- 0.205738421526

Amplitudes / Uncertainties
        K[ 0 ]: 1.7682054968 +/- 0.21228226240280873
        K[ 1 ]: 1.29389554632 +/- 0.2743813900288913
        K[ 2 ]: 1.15896731809 +/- 0.28585766006225655
        K[ 3 ]: 0.830377341498 +/- 0.24497089681917547
        K[ 4 ]: 0.97697800987 +/- 0.2884408108013377
        K[ 5 ]: 1.21055374112 +/- 0.20917568133349385

Offsets / Uncertainties
        c[ 0 ]: 7.28326 +/- 0.138816006399

Jitter(s):
        set[ 0 ]: 0.7855505310907221

Spectral stats:
        Joint P statistic [logL-logL_0/logL_0]: 0.13846
        logL_0 (null-model): -608.3446746953898
        logL (model): -524.1119736428393
        DlogL (model - null_model): 84.2327010525505

#### Observational data

##### Bidimensional map (HD41248)

The following command plots a bidimensional colour map of a period scan defined in line 324 of file mgls.py 

    ./mgls.py ./data/synth_hexa_model.dat --bidim --jitter --linear_trend
    
![Alt text](https://github.com/rosich/mgls/blob/master/bidim.png "bidim")

This data corresponds to observations of HD41248. The maximum is located at 18.4 d / 25.5 d
    
    


