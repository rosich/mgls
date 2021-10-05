#!/usr/bin/python
# -*- coding: utf-8 -*-
from math import sin, cos, tan, atan, pi, acos, sqrt, exp, log10, log
import sys, os
import copy
import random
import numpy as np
import scipy.optimize
import multiprocessing as mp
import ConfigParser
sys.path.append('./bin')
import mMGLS
sys.path.append('./src')
from EnvGlobals import Globals
import mgls_io
import time
from numpy.linalg import inv
#import mgls_mc
#from numba import jit

#definitions and constants
to_radians = pi/180.0
to_deg = 1.0/to_radians
#-------------------------

def logL_NullModel():
    """ndim = 0
    """
    Globals.logL_0 = 0.0
    for i in range(Globals.n_sets):
        mean_data = np.mean(Globals.rvs[i])
        inv_sigma2_set = 1.0/(Globals.rv_errs[i]**2) 
        Globals.logL_0 += -0.5*(np.sum(((Globals.rvs[i]-mean_data)**2)*inv_sigma2_set + np.log(2.0*np.pi) - np.log(inv_sigma2_set)) ) 
       
    if Globals.jitter:
        Globals.inhibit_msg = True
        ndim = Globals.ndim
        local_linear_trend = False
        Globals.ndim = 0
        if Globals.linear_trend: 
            local_linear_trend = True
            Globals.linear_trend = False

        param_bounds = [(0.0, Globals.jitter_limit) for iter in range(Globals.n_sets)]
        s_state_init = [random.uniform(0.0, Globals.jitter_limit) for iter in range(Globals.n_sets)]
        
        res = scipy.optimize.minimize(fmgls_multiset, s_state_init, 
                                      method='SLSQP', tol=1e-12, options={'disp': False})
        
        opt_jitters = abs(res.x[:]) #ndim=0
        pwr, fitting_coeffs, A, logL = mgls_multiset(res.x)
        Globals.logL_0 = logL
        #reestablish dimensionality
        Globals.ndim = ndim
        Globals.inhibit_msg = False
        Globals.opt_jitters_0 = opt_jitters

    
    return Globals.logL_0

def plot(t, y, t2, y2, err, max_peak, fap_thresholds, coeffs, peaks=None):
    """
    """
    try:
        import matplotlib.pyplot as plt
        from matplotlib import cm
    except ImportError:
        print "python-matplotlib package must be installed on your box"
             
    def base_round(x, base=20):
        """
        """
        if x >= 10000:
            base = 1000
            return int(base * round(float(x)/base))
        elif x > 1000:
            base = 500
            return int(base * round(float(x)/base))
        elif x > 200:
            base = 100
            return int(base * round(float(x)/base))
        elif x > 100:
            base = 50
            return int(base * round(float(x)/base))
        elif x > 20:
            base = 10
            return int(base * round(float(x)/base))
        elif x > 10:
            base = 5
            return int(base * round(float(x)/base))
        elif x > 5:
            base = 2
            return int(base * round(float(x)/base))
        elif x < 5:
            base = 1
            return int(base * round(float(x)/base))
           
    def tick_function(t):
        P = 1./np.array(t)
        return [ int(round(z)) for z in P]
    
    def make_format(current, other):
        """ current and other are axes
        """
        def format_coord(x, y):
            # x, y are data coordinates
            # convert to display coords
            display_coord = current.transData.transform((x,y))
            inv = other.transData.inverted()
            # convert back to data coords with respect to ax
            ax_coord = inv.transform(display_coord)
            coords = [ax_coord, (1./x, y)]
            return ('Frequency: {:<40}    Period: {:<}'
                    .format(*['({:.3f}, {:.3f})'.format(x, y) for x,y in coords]))
        
        return format_coord
    
    if Globals.inPeriods:
        #plot in frequency
        y2, err = y2.tolist(), err.tolist()
        #LaTeX env.
        plt.rc('font', serif='Helvetica Neue')
        plt.rcParams.update({'font.size': 15.0})
        
        fig, ax1 = plt.subplots(figsize=(15,4.3))
        plt.subplots_adjust(left=0.07, right=0.96, top=0.9, bottom=0.15, hspace=0.3)
    
        #ax1.plot(1./t, y, '-r', color='black', linewidth=0.6)
        ax1.fill_between(1./t, y, min(y), facecolor='steelblue', color='steelblue', alpha=0.99)
        ax1.set_xlim((1./t[-1], 1./t[0]))
        ax1.minorticks_on() 
        ax1.set_ylabel(r'$\Delta$lnL', rotation='vertical', fontsize=15, fontweight='bold')
        ax1.set_xlabel(r'Period (d)', fontsize=15, fontweight='bold')
        ax1.set_ylim((min(y),1.15*max(y))) 
        ax1.set_xscale('log')
        
    else:  
        #plot in frequency
        y2, err = y2.tolist(), err.tolist()
        #LaTeX env.
        plt.rc('font', serif='Helvetica Neue')
        plt.rcParams.update({'font.size': 19.0})
        
        fig, ax1 = plt.subplots(figsize=(14,6.5))
        plt.subplots_adjust(left=0.07, right=0.92, top=0.85, bottom=0.15, hspace=0.3)

        Ks = []
        for ii in range(len(peaks)):
            a,b = peaks[ii][2][1], peaks[ii][2][2]
            Ks.append(sqrt(a**2 + b**2))
        
        Ks_ = []
        for ii in range(len(coeffs)):
            a,b = coeffs[ii][Globals.n_sets], coeffs[ii][Globals.n_sets+1]
            Ks_.append(sqrt(a**2 + b**2))
            
        sc = ax1.scatter(peaks[:,0], peaks[:,1], c=cm.rainbow(Ks), edgecolor=cm.rainbow(Ks), s=30)
        #ax1.fill_between(t, y, min(y), facecolor='steelblue', color='steelblue', alpha=0.99)
        ax1.set_xlim((t[0],t[-1]))
        #fig.colorbar(sc, ax1=ax1)
        ax3 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

        ax3.set_ylabel(r'Amplitude', color='gray', rotation='vertical',fontsize=15, fontweight='bold') 
        ax3.plot(t, Ks_, '-', color='darkgray', linewidth=1.0, zorder=-128)
        ax1.tick_params(axis='y', labelcolor='steelblue')
        ax3.tick_params(axis='y', labelcolor='dimgray')
        ax1.plot(t, y, '-', color='steelblue', linewidth=1.0, zorder=-128)
        
        try:
            ax2 = ax1.twiny()
            #period ticks
            ax2.set_xlim(ax1.get_xlim())
            periods = np.array(1./np.linspace( t[0], t[-1], 9))
            periods = [base_round(period) for period in periods]
            #new_tick_locations = ax1.get_xticks()
            new_tick_locations = 1./np.array(periods)
            ax2.set_xticks(new_tick_locations)
            ax2.set_xticklabels(tick_function(new_tick_locations))
            #mouse coordinates in both scales
            ax2.format_coord = make_format(ax2, ax1)
            ax2.set_xlabel(r'Period (d)', fontsize=15, fontweight='bold')
        except:
            print "Error in period axis"
        
        ax1.minorticks_on() 
        
        ax1.set_ylabel(r'$\Delta$lnL', color='steelblue', rotation='vertical', fontsize=15, fontweight='bold')
        ax1.set_xlabel(r'Frequency (d$^{-1}$)', fontsize=15, fontweight='bold')
        
        ax1.set_ylim((min(y), 1.75*max(y)))  
        #ax1.set_ylim((min(y),48.))
        
        ax3.set_ylim((2.75*max(Ks_), min(Ks_)))
        #ax3.set_ylim((4.,0.))
        #ax1.set_ylim((min(y),45.0))
    
    if fap_thresholds != []:
        print "FAP levels encountered"
        colors = ['silver', 'gray', 'black', 'blue', 'green']
        for i in range(len(fap_thresholds)):
            ax1.axhline(fap_thresholds[i],linestyle='-.',color=colors[i])
    else:
        print "No FAP levels"
    
    #ax1.plot(peaks[:,0], peaks[:,1], 'ro', mfc='none')
    
    y_range = max(y) - min(y)
    x_range = max(t) - min(t)
    
    pv = 0.06*y_range
    ph = 0.01*x_range
    
    for j_ in range(2):
        ax1.annotate(str(1./peaks[j_][0])[:7], xy=(peaks[j_][0], peaks[j_][1]),
                     xytext=(ph+peaks[j_][0], pv+peaks[j_][1]), rotation=0)
    
    #ax3.set_yscale('log')

    #ax1.set_ylim((min(y), 120.0))
    
    
    """
    ax1.axvline(1.25,linestyle='-.',color='red')
            )
    """ 
    
    plt.show()

def peak_counter(freqs, pwrs, coeffs):
    """
    """
    def derivative(freqs, pwrs):
        """
        """
        dpow = []
        for i in range(1,len(pwrs)-1):
            d = (pwrs[i] - pwrs[i-1])/(freqs[i] - freqs[i-1])
            dpow.append([freqs[i], d])
        
        return dpow
    
    peaks = []

    #maxima
    for i in range(1,len(pwrs)-1):
        if pwrs[i-1] < pwrs[i] > pwrs[i+1]:
            peaks.append([freqs[i], pwrs[i], coeffs[i]])
    """
    #minima
    for i in range(1,len(pwrs)-1):
        if pwrs[i-1] > pwrs[i] < pwrs[i+1]:
            peaks.append([freqs[i], pwrs[i], coeffs[i]])

    D_spec = np.array(derivative(freqs, pwrs))
    f_D_spec = D_spec[:,1]
    D_spec_2 = np.array(derivative(freqs, f_D_spec))
    f_D_spec_2 = D_spec_2[:,1]
    
    mgls_io.write_file('derivative.dat', D_spec, ' ', '')
    mgls_io.write_file('derivative_2.dat', D_spec_2, ' ', '')
    
    peaks_2 = []
    #detect extremals via derivative
    for i in range(1,len(D_spec)):
        if f_D_spec[i-1]*f_D_spec[i] < 0.0:
            peaks.append([freqs[i], pwrs[i], coeffs[i]])
    
    peaks_3 = []
    #detect inflection points
    for i in range(1,len(D_spec_2)):
        if f_D_spec_2[i-1]*f_D_spec_2[i] < 0.0:
            peaks.append([freqs[i], pwrs[i], coeffs[i]])
    """
    
    #print "Null derivatives:", len(peaks_2)
    #print "Null 2nd derivatives:", len(peaks_3)
    
    #peaks = np.array(peaks)
    #sort peaks by power
    peaks = sorted(peaks, key=lambda x: x[1], reverse=True)
    
    return np.array(peaks)

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
    """determines which power a FAP of 10, 1, 0.1 % is reached
    """
    FAPs = [10.0,1.0,0.1]  #FAPS to compute in %
    n_bs = len(bootstrapping_stats)
    #sort bootstrapping_stats vector ascendently
    sorted_pwr = sorted(bootstrapping_stats)

    return [np.percentile(sorted_pwr,100-FAPs[i]) for i in range(len(FAPs))]

def mgls_multiset_jitter_search(freq):
    """
    """
    def flatten(_2d_list):
        flat_list = []
        # Iterate through the outer list
        for element in _2d_list:
            if type(element) is list:
                # If the element is of type list, iterate through the sublist
                for item in element:
                    flat_list.append(item)
            else:
                flat_list.append(element)
                
        return flat_list
    
    def jitter_search(jitters):
        """
        """
        non_linear_params = [freq]
        non_linear_params.extend(jitters)
        non_linear_params = flatten(non_linear_params)
        pwr, fitting_coeffs, A, logL = mgls_multiset(non_linear_params)
        
        return -logL
    
    jitter_0 = Globals.opt_jitters_0

    res = scipy.optimize.minimize(jitter_search, jitter_0, 
                                  method='SLSQP', tol=1e-4, options={'disp': False})
    
    opt_jitters = res.x
    non_linear_params = [freq]
    non_linear_params.extend(opt_jitters)
    non_linear_params = flatten(non_linear_params)
    pwr, fitting_coeffs, A, logL = mgls_multiset(non_linear_params)

    return pwr, fitting_coeffs, A, logL
 
def gls_1D():
    """
    """
    Globals.ndim = 1  #unidimensional GLS
    
    f_MAX = max(1./Globals.period_range[1], 1./Globals.period_range[0])
    n_points = 5.*abs(Globals.period_range[1] - Globals.period_range[0])*f_MAX
    freqs = np.linspace(1./Globals.period_range[1], 1./Globals.period_range[0], n_points)
    
    #freqs = np.random.uniform(1./Globals.period_range[1], 1./Globals.period_range[0], 49*10)
    print "Calculating", len(freqs), "frequencies"
    #TimeSpan = Globals.times_seq[-1] - Globals.times_seq[0]
    #N_points = len(Globals.times_seq)
    #freqs = [2.*pi*i/TimeSpan for i in range(int(N_points/2))]
    
    non_linear_params = [0.0]
    if Globals.jitter:
        non_linear_params.extend(Globals.opt_jitters_0)
    
    non_linear_params = np.array(non_linear_params)
    DlogLs = []
    coeffs = []
    max_DlogL = -np.inf
    for i in range(len(freqs)):
        if freqs[i] == non_linear_params[0]:
            freqs[i] = freqs[i-1]
        non_linear_params[0] = freqs[i]
        
        if Globals.jitter:
            pwr, fitting_coeffs, A, logL = mgls_multiset_jitter_search(non_linear_params[0])
        else:
            pwr, fitting_coeffs, A, logL = mgls_multiset(non_linear_params)   
        
        DlogL = logL - Globals.logL_0
        if DlogL < 0.0: DlogL = 0.0
        DlogLs.append(DlogL)
        coeffs.append(fitting_coeffs)
        #find the peak
        if DlogL > max_DlogL:
            max_DlogL = DlogL
    
    return freqs, DlogLs, max_DlogL, coeffs
        
def covariance_matrix(sys_matrix):
    """calculate the covariance matrix for params estimation
    """
    Cov = inv(sys_matrix)
  
    return Cov
    
def print_heading(ncpus):
    """
    """
    """
    print ''
    print_message('                                                                                             ', 7, 90)
    print_message('                MGLS (MULTIDIMENSIONAL GENERALIZED LOMB-SCARGLE) PERIODOGRAM                 ', 7, 97)
    print_message('                                                                                             ', 7, 90)
    """
    k = random.choice([36,90,92,93,94,96,97])
    
    print_message('                                               ', 5, 29)
    print_message('       ███╗   ███╗ ██████╗ ██╗     ███████╗   ', 10,k) 
    print_message('       ████╗ ████║██╔════╝ ██║     ██╔════╝   ', 10,k) 
    print_message('       ██╔████╔██║██║  ███╗██║     ███████╗   ', 10,k) 
    print_message('       ██║╚██╔╝██║██║   ██║██║     ╚════██║   ', 10,k) 
    print_message('       ██║ ╚═╝ ██║╚██████╔╝███████╗███████║   ', 10,k)
    print_message('       ╚═╝     ╚═╝ ╚═════╝ ╚══════╝╚══════╝   ', 10,k)
    
    #self.print_message('                                                  ', 10, 2)
    print_message('     MULTIDIMENSIONAL GENERALIZED LOMB-SCARGLE      ', 29, 7)                                     
    print_message('         Albert Rosich (rosich@ice.cat)             ', 40, 6)
    print_message('', 7,90)

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

def chi2_gls(freqs):
    """
    """
    chi2 = mMGLS.chi2(Globals.time, Globals.rv, Globals.rv_err, Globals.time[0], freqs, fitting_coeffs)
    chi2_0 = mMGLS.chi2_0(Globals.rv, Globals.rv_err)
     
    return (chi2_0 - chi2)/chi2_0
     
def mgls(freqs):
    """
    """
    if Globals.ar:
        fitting_coeffs, A_matrix = mMGLS_AR.mdim_gls(Globals.time, Globals.rv, Globals.rv_err, freqs, Globals.params, Globals.nar, Globals.time[0])
        #model = mMGLS_AR.model(Globals.time, Globals.rv, freqs, fitting_coeffs, Globals.params)
        #chi2 = mMGLS_AR.chi2(Globals.time, Globals.rv, Globals.rv_err, model)
        #chi2_0 = mMGLS_AR.chi2_0(Globals.rv, Globals.rv_err)
        #pwr = (Globals.chi2_0 - chi2)/Globals.chi2_0
        model = mMGLS.model_series(Globals.time, Globals.time[0], freqs, fitting_coeffs, len(Globals.time))
        inv_sigma2 = 1.0/(Globals.rv_err**2 + Globals.s**2)
        logL = -0.5*(np.sum(((Globals.rv-model)**2)*inv_sigma2 + np.log(2.0*np.pi) + np.log(1.0/inv_sigma2)) )     
        pwr = (Globals.logL_0 - logL) / Globals.logL_0 
     
        
    else:
        fitting_coeffs, A_matrix = mMGLS.mdim_gls(Globals.time, Globals.rv, Globals.rv_err, freqs, Globals.time[0])
        #chi2 = mMGLS.chi2(Globals.time, Globals.rv, Globals.rv_err, Globals.time[0], freqs, fitting_coeffs)
        #chi2_0 = mMGLS.chi2_0(Globals.rv, Globals.rv_err)
        
        #compute logL 
        model = mMGLS.model_series(Globals.time, Globals.time[0], freqs, fitting_coeffs, len(Globals.time))
        inv_sigma2 = 1.0/(Globals.rv_err**2 + Globals.s**2)
        logL = -0.5*(np.sum(((Globals.rv-model)**2)*inv_sigma2 + np.log(2.0*np.pi) + np.log(1.0/inv_sigma2)) ) 
            
        pwr = (Globals.logL_0 - logL) / Globals.logL_0 
        #pwr = (chi2_0 - chi2)/chi2_0
    
    return  pwr, fitting_coeffs, A_matrix, logL

def fmgls(non_linear_params):
    """returns MGLS-power for a given frequency tuple.
    """
    return -mgls(non_linear_params)[0]

def gls_multiset_fit(non_linear_params):
    """
    """
    def mgls_jitter(jitter, *args):
        """
        """
        #freqs = non_linear_params[:Globals.ndim]
        #jitters = non_linear_params[Globals.ndim:]
        jitters = jitter
        freqs = args[0]
        
        #fit linear params
        if Globals.linear_trend:
            fitting_coeffs, A_matrix = mMGLS.mdim_gls_multiset_trend(Globals.times_seq, Globals.rvs_seq, Globals.rv_errs_seq, freqs, jitters, Globals.len_sets)
        else:   
            fitting_coeffs, A_matrix = mMGLS.mdim_gls_multiset(Globals.times_seq, Globals.rvs_seq, Globals.rv_errs_seq, freqs, jitters, Globals.len_sets)
    
        if Globals.jitter:
            jitters = np.array(non_linear_params[Globals.ndim:])
        else:
            jitters = np.array([0.0 for iter in range(Globals.n_sets)])

        #fitting coeffs for each dataset
        fitting_coeffs_set = np.array(fitting_coeffs[Globals.n_sets-1:])
        
        logL = 0.0
        for i in range(Globals.n_sets):
            fitting_coeffs_set[0] = fitting_coeffs[i] #append constant
            #print fitting_coeffs_set
            #compute model for (i) dataset
            model = mMGLS.model_series(Globals.times[i], Globals.times[0][0], freqs, fitting_coeffs_set, len(Globals.times[i])) 
            inv_sigma2 = 1.0/(Globals.rv_errs[i]**2.0 + jitters[i]**2.0)
            #add logL for dataset (i)
            logL += -0.5*(np.sum(((Globals.rvs[i] - model)**2.0)*inv_sigma2 + np.log(2.0*np.pi) - np.log(inv_sigma2))) 

        
        return -logL


    freqs = non_linear_params[:Globals.ndim]
    jitters = non_linear_params[Globals.ndim:]
    #fit linear params
    if Globals.jitter:
        #fit jitter @ each period
        param_bounds = [(0.0, Globals.jitter_limit) for iter in range(Globals.n_sets)]
        s_state_init = [random.uniform(0.0, Globals.jitter_limit) for iter in range(Globals.n_sets)]
        
        res = scipy.optimize.minimize(mgls_jitter, s_state_init, 
                                      method='SLSQP', args=(freqs), tol=1e-6, options={'disp': False})
        
        opt_jitters = abs(res.x[Globals.ndim:]) #ndim=1
        pwr, fitting_coeffs, A, logL = mgls_multiset(res.x)
        
    return pwr, fitting_coeffs, A, logL
    
    
    
    """
    if Globals.linear_trend:
        fitting_coeffs, A_matrix = mMGLS.mdim_gls_multiset_trend(Globals.times_seq, Globals.rvs_seq, Globals.rv_errs_seq, freqs, jitters, Globals.len_sets)
    else:   
        fitting_coeffs, A_matrix = mMGLS.mdim_gls_multiset(Globals.times_seq, Globals.rvs_seq, Globals.rv_errs_seq, freqs, jitters, Globals.len_sets)
 
    if Globals.jitter:
        jitters = np.array(non_linear_params[Globals.ndim:])
    else:
        jitters = np.array([0.0 for iter in range(Globals.n_sets)])

    #fitting coeffs for each dataset
    fitting_coeffs_set = np.array(fitting_coeffs[Globals.n_sets-1:])
    
    logL = 0.0
    for i in range(Globals.n_sets):
        fitting_coeffs_set[0] = fitting_coeffs[i] #append constant
        #print fitting_coeffs_set
        #compute model for (i) dataset
        model = mMGLS.model_series(Globals.times[i], Globals.times[0][0], freqs, fitting_coeffs_set, len(Globals.times[i])) 
        inv_sigma2 = 1.0/(Globals.rv_errs[i]**2.0 + jitters[i]**2.0)
        #add logL for dataset (i)
        logL += -0.5*(np.sum(((Globals.rvs[i] - model)**2.0)*inv_sigma2 + np.log(2.0*np.pi) - np.log(inv_sigma2))) 

    #spectral power   
    pwr = (Globals.logL_0 - logL) / Globals.logL_0
 
    return pwr, fitting_coeffs, A_matrix, logL
    """

def mgls_multiset(non_linear_params):
    """
    """
    freqs = non_linear_params[:Globals.ndim]
    jitters = non_linear_params[Globals.ndim:]
    
    #fit linear params
    if Globals.linear_trend:
        fitting_coeffs, A_matrix = mMGLS.mdim_gls_multiset_trend(Globals.times_seq, Globals.rvs_seq, Globals.rv_errs_seq, freqs, jitters, Globals.len_sets)
    else:   
        fitting_coeffs, A_matrix = mMGLS.mdim_gls_multiset(Globals.times_seq, Globals.rvs_seq, Globals.rv_errs_seq, freqs, jitters, Globals.len_sets)
 
    if Globals.jitter:
        jitters = np.array(non_linear_params[Globals.ndim:])
    else:
        jitters = np.array([0.0 for iter in range(Globals.n_sets)])

    #fitting coeffs for each dataset
    fitting_coeffs_set = np.array(fitting_coeffs[Globals.n_sets-1:])

    logL = 0.0
    for i in range(Globals.n_sets):
        fitting_coeffs_set[0] = fitting_coeffs[i] #append constant
        #print fitting_coeffs_set
        #compute model for (i) dataset
        model = mMGLS.model_series(Globals.times[i], Globals.times[0][0], freqs, fitting_coeffs_set, len(Globals.times[i])) 
        inv_sigma2 = 1.0/(Globals.rv_errs[i]**2.0 + jitters[i]**2.0)
        #add logL for dataset (i)
        logL += -0.5*(np.sum(((Globals.rvs[i] - model)**2.0)*inv_sigma2 + np.log(2.0*np.pi) - np.log(inv_sigma2))) 
        
    #spectral power   
    pwr = (Globals.logL_0 - logL) / Globals.logL_0

    return pwr, fitting_coeffs, A_matrix, logL

def fmgls_multiset(freqs):
    """
    """
    return -mgls_multiset(freqs)[3]  #retrun the model logL
           
def linear_trend(times, rvs):
    """
    """
    from scipy import stats
    slope, intercept, r_value, p_value, std_err = stats.linregress(times,rvs)
    #print "Linear trend correction:"    
    
    return slope, intercept, r_value, p_value

def y_gauss(x, mu, sigma):
    """
    """
    return 1.0/(sqrt(2*pi)*sigma)*exp((-(x-mu)**2)/(2*sigma**2))
     
def load_multiset_data():
    """
    """
    #multiset 
    Globals.n_sets = 0
    print_message("\nReading multiset data\n", index=4, color=34)
    #count data files passed as CLI arguments
    Globals.times, Globals.rvs, Globals.rv_errs = [],[],[]
    Globals.dataset_names, DATA = [], []
    Globals.mean_err = []
    
    try:
        from terminaltables import AsciiTable, DoubleTable, SingleTable
        ttable = True
    except:
        ttable = False
    
    s = 0
    for arg in sys.argv[1:]:
        if arg[0] != '-':
            #read data
            try:
                in_data = mgls_io.read_file(arg, ' ')
                if not ttable: print_message("\t" + str(s) + "/ Read data file " + arg, index=3, color=32)
            except IOError:
                print_message("\tData file could not be read", 5, 31)
                sys.exit()       
            
            #assign data vectors
            time, rv, rv_err = mgls_io.get_data_vectors(in_data, Globals.col)
            if Globals.km2m:
                rv *= 1000.0
                rv_err *= 1000.0
            Globals.dataset_names.append( str(s) + '/ ' + arg.split('/')[-1]) #file name. Not full path
            Globals.times.append(time)
            mean_rv = np.mean(rv)
            mean_rv_err = np.mean(rv_err)
            Globals.mean_err.append(mean_rv_err)
            Globals.rvs.append(rv)
            Globals.rv_errs.append(rv_err)
            inv_sigma2 = 1.0/(rv_err**2)
            logL_0 = -0.5*(np.sum(((rv-mean_rv)**2)*inv_sigma2 + np.log(2.0*np.pi) + np.log(1.0/inv_sigma2)) ) 
            Globals.n_sets += 1
            
            #print info
            summ = 0.0
            separation = []
            
            for i in range(len(time)-1):
                summ += time[i+1] - time[i]
                separation.append(time[i+1] - time[i])
            
            if ttable:
                DATA.append([len(time), time[-1] - time[0], summ/len(time), logL_0])
                
            else:
                if not Globals.inhibit_msg:
                    print "\t-----------------Summary of data--------------------"
                    print "\tData points:", len(time)
                    print "\tTime span:", time[-1] - time[0]
                    print "\tMean sep.", summ/len(time)
                    print "\tlogL null model (data, no jitter):", logL_0
                    print "\t----------------------------------------------------"
            #count dataset
            s += 1
       
    #print ttable (if appliable)
    if ttable:
        TABLE_DATA = []
        TABLE_DATA.append(['Dataset name (full path not shown)', 'Data points', 'Timespan', 'Mean sep.', 'logL (data)'])
        for s in range(Globals.n_sets):
            TABLE_DATA.append([str(Globals.dataset_names[s]), str(DATA[s][0]), str(round(DATA[s][1],5)), str(round(DATA[s][2],5)), str(round(DATA[s][3],5))])
        
        TABLE_DATA = tuple(TABLE_DATA)
        table_instance = SingleTable(TABLE_DATA, "Dataset summary")
        table_instance.justify_column = 'right'
        print(table_instance.table)
    
    
    Globals.len_sets = [len(Globals.times[i]) for i in range(len(Globals.times))]
            
    Globals.times_seq, Globals.rvs_seq, Globals.rv_errs_seq = [], [], []
    for i in range(len(Globals.times)):
        Globals.times_seq.extend(Globals.times[i])
        Globals.rvs_seq.extend(Globals.rvs[i])
        Globals.rv_errs_seq.extend(Globals.rv_errs[i])

    Globals.times_seq = np.array(Globals.times_seq)
    Globals.rvs_seq = np.array(Globals.rvs_seq)
    Globals.rv_errs_seq = np.array(Globals.rv_errs_seq)
    
def gen_synthetic_model(periods, fitting_coeffs, opt_jitters_base):
    """create a synthetic model 
    """
    aS, bS = [], []
    linear_trend = 0.0
    
    for j in range(Globals.ndim):
        aS.append(fitting_coeffs[j+Globals.n_sets])
        bS.append(fitting_coeffs[j+Globals.n_sets+Globals.ndim])

    if Globals.linear_trend:
        linear_trend = fitting_coeffs[2*Globals.ndim+Globals.n_sets]

    rvs_concat = []
    rv_errs_concat = []
    phi = 0.0
    X_pre = 0.0
    for k in range(Globals.n_sets):
        model = []
        #offset
        c = fitting_coeffs[k]
        for i in range(len(Globals.times[k])):
            sigma = sqrt(Globals.rv_errs[k][i]**2 )#+ opt_jitters_base[k]**2)
            noise = random.gauss(0.0, sigma) + phi*X_pre
            X_pre = noise
            y = 0.0
            if len(periods) > 0:
                for j in range(len(periods)):
                    y += aS[j]*cos((2.0*pi/periods[j])*(Globals.times[k][i]-Globals.times_seq[0])) + \
                         bS[j]*sin((2.0*pi/periods[j])*(Globals.times[k][i]-Globals.times_seq[0]))   
            else:
                y = c + linear_trend*(Globals.times[k][i]-Globals.times_seq[0]) + noise     
            #model time point
            if len(periods) == 0:
                model.append([Globals.times[k][i], noise, sigma])
            else:
                model.append([Globals.times[k][i], y + c + linear_trend*(Globals.times[k][i]-Globals.times_seq[0]) + noise , sigma])

        model = np.array(model)
        
        Globals.rvs[k], Globals.rv_errs[k] = model[:,1], model[:,2]
    
        rvs_concat.extend(model[:,1])
        rv_errs_concat.extend(model[:,2])
    
    Globals.rvs_seq, Globals.rv_errs_seq = np.array(rvs_concat[:]), np.array(rv_errs_concat[:])
    
    return True

    """
    model = []
    for i in range(len(Globals.times_seq)):
        sigma = sqrt(Globals.rv_errs_seq[i]**2 + opt_jitters[0]**2)
        noise = random.gauss(0.0, sigma)
        y = 0.0
        if len(periods) > 0:
            for j in range(len(periods)):
                y += aS[j]*cos((2.0*pi/periods[j])*(Globals.times_seq[i]-Globals.times_seq[0])) + \
                    bS[j]*sin((2.0*pi/periods[j])*(Globals.times_seq[i]-Globals.times_seq[0]))   
        else:
            y = c + linear_trend*(Globals.times_seq[i]-Globals.times_seq[0]) + noise     
        #model time point
        if len(periods) == 0:
            model.append([Globals.times_seq[i], noise, sigma])
        else:
            model.append([Globals.times_seq[i], y + c + linear_trend*(Globals.times_seq[i]-Globals.times_seq[0]) + noise , sigma])
    
   
    model = np.array(model)
    #global variables
    Globals.times_seq, Globals.rvs_seq, Globals.rv_errs_seq = model[:,0], model[:,1], model[:,2]
    
    Globals.times, Globals.rvs, Globals.rv_errs = [], [], []
    
    Globals.times.append(Globals.times_seq)
    Globals.rvs.append(Globals.rvs_seq)
    Globals.rv_errs.append(Globals.rv_errs_seq)
 
    
    Globals.times = np.array(Globals.times)
    Globals.rvs = np.array(Globals.rvs)
    Globals.rv_errs = np.array(Globals.rv_errs)
    
    return np.array(model)
    """

def compute_residuals(model):
    """
    """
    res = []
    for i in range(len(Globals.times[0])):
        for j in range(len(model)-1):
            if model[j][0] <= Globals.times[0][i] <= model[j+1][0]:
                res.append([Globals.times[0][i], Globals.rvs[0][i] - model[j][1], Globals.rv_errs[0][i] ])
    
    mgls_io.write_file('residuals.dat', res, ' ', '')
  
def multiset_model(opt_freqs, opt_jitters, fitting_coeffs):
    """model
    """ 
    model = list()
    for s in range(Globals.n_sets):
        offset = fitting_coeffs[s]
        #recenter each dataset
        data_file = []
        for j in range(len(Globals.times[s])):
            if Globals.jitter:
                data_file.append( [Globals.times[s][j], Globals.rvs[s][j]-offset, sqrt(Globals.rv_errs[s][j]**2 + opt_jitters[s]**2)] ) 
            else:
                data_file.append( [Globals.times[s][j], Globals.rvs[s][j]-offset, Globals.rv_errs[s][j]] )
        mgls_io.write_file('offset_' + str(s) + '.dat', data_file, ' ', '')
        
        #residuals corresponding to dataset
        data_file_res = []
        
    
        for j in range(len(Globals.times[s])):
            #model
            t = Globals.times[s][j]
            y_model = 0.0
            if Globals.ndim > 0:
                for k in range(Globals.n_sets, len(fitting_coeffs)):
                    if k >= Globals.n_sets and k < Globals.ndim + Globals.n_sets:
                        y_model += fitting_coeffs[k]*cos(2.0*pi*(t-Globals.times[0][0])*opt_freqs[k-Globals.n_sets])
                    elif (k >= (Globals.ndim + Globals.n_sets)) and (k < (2*Globals.ndim+Globals.n_sets)):
                        y_model += fitting_coeffs[k]*sin(2.0*pi*(t-Globals.times[0][0])*opt_freqs[k-(2*Globals.ndim+Globals.n_sets)])    
                    elif k == (2*Globals.ndim + Globals.n_sets):
                        y_model += fitting_coeffs[k]*(t-Globals.times[0][0])

            if Globals.jitter:
                data_file_res.append( [Globals.times[s][j], Globals.rvs[s][j]-offset-y_model, sqrt(Globals.rv_errs[s][j]**2)] ) 
            else:
                data_file_res.append( [Globals.times[s][j], Globals.rvs[s][j]-offset-y_model, Globals.rv_errs[s][j]] )
    
        mgls_io.write_file('residuals_' + str(s) + '.dat', data_file_res, ' ', '')
        
        
        t_0, t_f = Globals.times[s][0] - 10.0, Globals.times[s][-1] + 10.0
        for i in range(12*int(t_f-t_0)):
            t = t_0 + i*(t_f-t_0)/(12*int(t_f-t_0))
            y_model = 0.0
            if Globals.ndim > 0:
                for k in range(Globals.n_sets, len(fitting_coeffs)):
                    if k >= Globals.n_sets and k < Globals.ndim + Globals.n_sets:
                        y_model += fitting_coeffs[k]*cos(2.0*pi*(t-Globals.times[0][0])*opt_freqs[k-Globals.n_sets])
                    elif (k >= (Globals.ndim + Globals.n_sets)) and (k < (2*Globals.ndim+Globals.n_sets)):
                        y_model += fitting_coeffs[k]*sin(2.0*pi*(t-Globals.times[0][0])*opt_freqs[k-(2*Globals.ndim+Globals.n_sets)])    
                    elif k == (2*Globals.ndim + Globals.n_sets):
                        y_model += fitting_coeffs[k]*(t-Globals.times[0][0])
            #write line    
            model.append( [t, y_model, 1.0] )        
        #sort points
        model_s = sorted(model, key=lambda row: row[0])
    #write on file
    mgls_io.write_file('model.dat', model_s, ' ', '')
    
    print_message("Datasets w/ offset written in ./offset_*", 6, 95)
    print_message("Model written in ./model.dat", 6, 95)

    compute_residuals(model_s)

    return model_s

        
    
    
    
  
  
