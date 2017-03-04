#!/usr/bin/python
from math import sin, cos, tan, atan, pi, acos, sqrt, exp, log10
import sys, os
import copy
import random
import numpy as np
import multiprocessing as mp
import ConfigParser
sys.path.append('./bin')
import mGLS, mMGLS
sys.path.append('./src')
from EnvGlobals import Globals
import mgls_io
import mgls_mc
from mgls_lib import *

#definitions and constants
to_radians = pi/180.0
to_deg = 1.0/to_radians
#-------------------------

def _gls_instance_Ndim_bootstrapping(n_runs):
    """executes n_runs instances of MGLS for with previous data shuffle
    """
    cpu_periodogram = list()
    for iter in range(n_runs):
        """
        #shuffle RV's and their errors. Repetition is not allowed
        comb_rv_err = zip(Globals.rv, Globals.rv_err)
        random.shuffle(comb_rv_err)
        Globals.rv[:], Globals.rv_err[:] = zip(*comb_rv_err)
        """
        #allowing repetition
        rv = [0.0]*len(Globals.time)
        rv_err = [0.0]*len(Globals.time)
        for i in range(len(Globals.time)):
            index = int(random.uniform(0,len(Globals.time)))
            rv[i] = Globals.rv[index]
            rv_err[i] = Globals.rv_err[index]
        Globals.rv = rv
        Globals.rv_err = rv_err
        
        opt_state = mgls_mc.optimal(Globals.ndim, msgs = False, temp_steps=20, n_iter=1000)
        pwr_opt, fitting_coeffs, A = mgls(opt_state)
        cpu_periodogram.append(pwr_opt) #save the best period determination (highest power)

    return cpu_periodogram

def fap(bootstrapping_stats, pwr):
    """returns FAP for a given pwr. i.e. how many realizations overcome
       a given power, over unit.
    """
    return float(sum(i > pwr for i in bootstrapping_stats))/len(bootstrapping_stats)

def fap_levels(bootstrapping_stats):
    """determines which power a FAP of 1, 0.1, 0.01 % is reached
    """
    FAPs = [1.0, 0.1, 0.01, 0.001]  #FAPS to compute in %
    n_bs = len(bootstrapping_stats)
    #sort bootstrapping_stats vector ascendently
    sorted_pwr = sorted(bootstrapping_stats)

    return [np.percentile(sorted_pwr,100-FAPs[i]) for i in range(len(FAPs))]

def parallel_Mdim_bootstrapping(n_bootstrapping):
    """
    """
    n_runs = [n_bootstrapping/Globals.ncpus for i in range(Globals.ncpus)]
    pool = mp.Pool(Globals.ncpus)  #ncpus available
    #run parallell execution 
    try:
		out = pool.map_async(_gls_instance_Ndim_bootstrapping, n_runs).get(1./.0001)
		pool.terminate()
    except KeyboardInterrupt:
        pool.terminate()
        sys.exit()
	"""	
	except ZeroDivisionError:
		print "Error: Zero division error. Restarted parallel bootstapping"
	"""		    
    #join the output bunches
    out_spectra = list()
    for cpu in range(len(n_runs)):
        out_spectra.extend(out[cpu])
    bootstrapping_stats = list()
    for j in range(len(out_spectra)):
        bootstrapping_stats.append(out_spectra[j])

    return bootstrapping_stats

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

def Mdim_bootstrapping(max_pow):
    """
    """
    #n_bootstrapping = 500 #iterations
    bootstrapping_stats = parallel_Mdim_bootstrapping(Globals.n_bootstrapping)
    print "\n//BOOTSTRAPPING:// {1.0, 0.1, 0.01, 0.001}%"
    print "FAP Levels:", fap_levels(bootstrapping_stats)
    print "Total bootstapping samples: ", len(bootstrapping_stats)

    return bootstrapping_stats


            
