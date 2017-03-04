#!/usr/bin/python
from math import sin, cos, tan, atan, pi, acos, sqrt, exp, log10
import sys, os
import copy
import random
import numpy as np
import multiprocessing as mp
import ConfigParser
sys.path.append('./bin')
import mGLS, mMGLS, mMGLS_AR
sys.path.append('./src')
from EnvGlobals import Globals
import mgls_io

#definitions and constants
to_radians = pi/180.0
to_deg = 1.0/to_radians
#-------------------------

def plot(t, y, t2, y2, err, max_peak, fap_thresholds, trended):
    """
    """
    import matplotlib.pyplot as plt
    y2, err = y2.tolist(), err.tolist()
    #LaTeX env.
    #plt.rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
    ## for Palatino and other serif fonts use:
    #rc('font',**{'family':'serif','serif':['Palatino']})
    #plt.rc('text', usetex=True, fontsize=15)

    fig = plt.figure(1,figsize=(14,4))
    #fig.suptitle(sys.argv[1])
    plt.subplots_adjust(left=0.1, right=0.96, top=0.95, bottom=0.2, hspace = 0.3)
    #fig.patch.set_facecolor('gray')
    #upper plot
    ax1 = fig.add_subplot(111)
    ax1.minorticks_on()
    if max_peak != []:
        plt.ylim((0.0,1.2*max_peak[1]))
            
    ax1.set_xscale('log')
    
    plt.xlim((t[0],t[-1]))
    
    ax1.fill_between(t, y, 0, color='rosybrown', facecolor='rosybrown', alpha=0.75)

    if fap_thresholds != []:
        print "FAP levels encountered"
        colors = ['gray', 'black', 'red', 'green']
        for i in range(len(fap_thresholds)):
            ax1.axhline(fap_thresholds[i],linestyle='-.',color=colors[i])
    else:
        print "No FAP levels"
    
    ax1.set_xlabel(r'P (d)', fontsize=11)
    ax1.set_ylabel(r'Normalized power', rotation='vertical', fontsize=11)

    plt.show()

def _gls_instance(period_range):
    """runs an instance of GLS for a range of periods [p_min, p_max]
    """
    p_step = 0.001
    n_iter = int((period_range[1] - period_range[0])/p_step)
    periods = [period_range[0]+j*p_step for j in range(n_iter)] #period array

    periodogram = mGLS.gls_pow(Globals.time, Globals.rv, Globals.rv_err, periods, Globals.time[0])

    return periodogram
    
def _gls_instance_bootstrapping(n_runs):
    """executes n_runs instances of GLS for with previous data shuffle
    """
    p_step = 0.0035
    n_iter = int((Globals.period_range[1] - Globals.period_range[0])/p_step)
    cpu_periodogram = list()
    for iter in range(n_runs):
        #shuffle RV's and their errors
        comb_rv_err = zip(Globals.rv, Globals.rv_err)
        random.shuffle(comb_rv_err)
        Globals.rv[:], Globals.rv_err[:] = zip(*comb_rv_err)
        periods = [Globals.period_range[0]+j*p_step for j in range(n_iter)] #period array
        periodogram = mGLS.gls_pow(Globals.time, Globals.rv, Globals.rv_err, periods, Globals.time[0])
        out_spectra_sorted = sorted(periodogram, key=lambda l: l[1], reverse=True)
        cpu_periodogram.append(out_spectra_sorted[0][1]) #save the best period determination (highest power)
        del out_spectra_sorted
        
    return cpu_periodogram
    
def bootstrapping(time, rv, rv_err, max_peak):
    """performs a shuffle of data and computes the periodogram. Number of times best peak power is exceeded
       are accounted. Repetition is not allowed in rearrangement.
    """
    #shuffle RV's and their errors
    comb_rv_err = zip(rv, rv_err)
    random.shuffle(comb_rv_err)
    rv[:], rv_err[:] = zip(*comb_rv_err)
    #compute periodogram
    _periodogram = gls_periodogram(period_range, time[0], ncpus)
    out_spectra_sorted = sorted(_periodogram, key=lambda l: l[1], reverse=True)

    return out_spectra_sorted[0]
    
def parallel_bootstrapping(n_bootstrapping):
    """
    """
    n_runs = [n_bootstrapping/Globals.ncpus for i in range(Globals.ncpus)]
    pool = mp.Pool(Globals.ncpus)  #ncpus available
    #run parallell execution
    try:
        out = pool.map_async(_gls_instance_bootstrapping, n_runs).get(1./.00001)
        pool.terminate()
    except KeyboardInterrupt:
            pool.terminate()
            sys.exit()
    #join the output bunches
    out_spectra = list()
    for cpu in range(len(n_runs)):
        out_spectra.extend(out[cpu])
    bootstrapping_stats = list()
    for j in range(len(out_spectra)):
        bootstrapping_stats.append(out_spectra[j])

    return bootstrapping_stats
       
def get_data_vectors(data_in):
    """returns three lists: time, rv, rv uncertainty
    """
    #data_in = read_file(in_file, ' ') #read input file
    time = list()
    rv = list()
    rv_err = list()
    for line in range(len(data_in)):
        time.append(float(data_in[line][0]))
        rv.append(float(data_in[line][1]))
        rv_err.append(float(data_in[line][2]))

    return time, rv, rv_err
    
def fap(bootstrapping_stats, pwr):
    """return FAP for a given pwr. i.e. how many realizations overcome
       a given power, over unit.
    """
    return float(sum(i > pwr for i in bootstrapping_stats))/len(bootstrapping_stats)

def fap_levels(bootstrapping_stats):
    """determines which power a FAP of 1, 0.1, 0.01 % is reached
    """
    FAPs = [1.0,0.1]  #FAPS to compute in %
    n_bs = len(bootstrapping_stats)
    #sort bootstrapping_stats vector ascendently
    sorted_pwr = sorted(bootstrapping_stats)

    return [np.percentile(sorted_pwr,100-FAPs[i]) for i in range(len(FAPs))]
        
def gls_periodogram(p_lims, t_0, ncpus):
    """compute periodogram for a range of periods (period_range: {float, array})
       according to Zechmeister & Kurster 2009
    """
    #parallel branching
    pool = mp.Pool(ncpus)  #ncpus available
    p_range_cpu = (p_lims[1] - p_lims[0])/ncpus
    p_range = [[p_lims[0]+j*p_range_cpu, p_lims[0] + (j+1)*p_range_cpu] for j in range(ncpus)]
    #run parallell execution
    try:
        out = pool.map_async(_gls_instance, p_range).get(1./.0001)
        pool.terminate()
    except KeyboardInterrupt:
            pool.terminate()
            sys.exit()
    #join the output bunches
    out_spectra = list()
    for cpu in range(ncpus):
        out_spectra.extend(out[cpu])

    return out_spectra

def gls_1D():
    """
    """
    #compute periodogram
    _periodogram = gls_periodogram(Globals.period_range, Globals.time[0], Globals.ncpus)
    #sort spectra
    out_spectra_sorted = sorted(_periodogram, key=lambda l: l[1], reverse=True)

    periods = []
    pwr = []
    for j in range(len(_periodogram)):
        periods.append(_periodogram[j][0])
        pwr.append(_periodogram[j][1])

    max_pow = out_spectra_sorted[0]

    chi2 = mGLS.chi2(Globals.time, Globals.rv, Globals.rv_err, Globals.time[0], max_pow)
    chi2_0 = mGLS.chi2_0(Globals.rv, Globals.rv_err)

    print max_pow[0], max_pow[1], max_pow[2], max_pow[3], max_pow[4], \
          chi2, chi2_0

    return periods, pwr, max_pow
    
def orbital_model(time, max_pwr):
    """
    """
    per = max_pwr[0]
    a, b, c = max_pwr[2:]
    x = []
    y = []
    for i in range(int(2.1*(max(time)-min(time)))):
        t = i/2.0
        ph = (2.0*pi/per)*t
        x.append(t+time[0])
        y.append(a*cos(ph) + b*sin(ph) + c)

    return x,y

def phaseFolding(t, y, err, P):
    """obrim fitxer de sortida
    """
    inPhase = []
    phi = []
    data_out = []
    d_0 = float(t[0])
    for i in range(0,len(t)):
        phi_i = ((float(t[i]) - d_0)/P) - int((float(t[i]) - d_0)/P)
        phi.append(phi_i)
        data_out.append([phi_i, y[i], err[i]])
    two_phi = phi[:]
    for i in range(len(phi)):
        two_phi.append(phi[i] + 1.0)
        data_out.append([phi[i] + 1.0, y[i], err[i]])
    #sort phase vector
    data_sorted = sorted(data_out, key=lambda field: field[0])
   
    return phi,y,two_phi,2*y,2*err,data_sorted

def covariance_matrix(sys_matrix):
    """calculate the covariance matrix for params estimation
    """
    from numpy.linalg import inv
    Cov = inv(sys_matrix)
  
    return Cov
    
def print_start_info(ncpus):
    """
    """
    print_message('', 7,94)
    print_message('MULTIDIMENSIONAL GENERALIZED LOMB-SCARGLE PERIODOGRAM FOR RV DATA ', 7, 94)
    print_message('', 7,94)
    
def print_message(str_, index, color):
    """
    """
    #inhibit_msg = False
    if not Globals.inhibit_msg:
        str_out = "\33["+str(index)+";"+str(color)+"m"+ str_ +"\33[1;m"
        print str_out

def conf_init(conf_file):
    """creates an instance of class ConfigParser, and read the .conf file. Returns the object created
    ConfigParser, and read the .conf file. Returns the object created
    """
    conf_file_Object = ConfigParser.ConfigParser()
    if not conf_file_Object.read([conf_file]):
        print_message( str(conf_file) + ": could not be read", 5, 31)
        sys.exit()
    else:
        return conf_file_Object
        
def mgls(freqs):
    """
    """
    if Globals.ar:
        fitting_coeffs, A_matrix = mMGLS_AR.mdim_gls(Globals.time, Globals.rv, Globals.rv_err, freqs, Globals.params, Globals.nar, Globals.time[0])
        model = mMGLS_AR.model(Globals.time, Globals.rv, freqs, fitting_coeffs, Globals.params)
        chi2 = mMGLS_AR.chi2(Globals.time, Globals.rv, Globals.rv_err, model)
        #chi2_0 = mMGLS_AR.chi2_0(Globals.rv, Globals.rv_err)
        pwr = (Globals.chi2_0 - chi2)/Globals.chi2_0
    else:
        fitting_coeffs, A_matrix = mMGLS.mdim_gls(Globals.time, Globals.rv, Globals.rv_err, freqs, Globals.time[0])
        chi2 = mMGLS.chi2(Globals.time, Globals.rv, Globals.rv_err, Globals.time[0], freqs, fitting_coeffs)
        chi2_0 = mMGLS.chi2_0(Globals.rv, Globals.rv_err)
        pwr = (chi2_0 - chi2)/chi2_0
    
    return pwr, fitting_coeffs, A_matrix

def fmgls(freqs):
    """returns MGLS-power for a given frequency tuple.
    """
    return -mgls(freqs)[0]
                        
def model_residuals(residuals):
    """returns residuals of (data - model_k)
    """
    t_res = []
    for i in range(len(Globals.time)):
        t_res.append([Globals.time[i], residuals[i], 2.0*Globals.rv_err[i]])    
    mgls_io.write_file('residuals.dat', t_res, ' ','')
    print_message("File residuals written", 33, 3)
    
    return t_res
 
def bootstrapping_1D(max_pow):
    """
    """
    #iterations
    n_bootstrapping = 1500
    bootstrapping_stats = parallel_bootstrapping(n_bootstrapping)
    print "\n//BOOTSTRAPPING://"
    fap_thresholds = fap_levels(bootstrapping_stats)
    print "FAP Levels:", fap_thresholds
    print "Total bootstapping samples: ", len(bootstrapping_stats)

    return bootstrapping_stats, fap_thresholds        
        
def linear_trend():
    """
    """
    from scipy import stats
    slope, intercept, r_value, p_value, std_err = stats.linregress(Globals.time,Globals.rv)
    print "Linear trend correction:"
    print "\tslope:", slope
    print "\tintercept:", intercept
    print "\tr:", r_value
    print "\tp-value:", p_value
    
    return slope, intercept, r_value, p_value

def y_gauss(x, mu, sigma):
    """
    """
    return 1.0/(sqrt(2*pi)*sigma)*exp((-(x-mu)**2)/(2*sigma**2))
   
    




