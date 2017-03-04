#!/usr/bin/env python

#-----------------------------------------------------------------------------
# Project:     Multidimensional Generalized Lomb-Scargle

# Name:        mgls.py

# Purpose:     Multifrequency Lomb-Scargle RV data fitting
#
# Author:      Albert Rosich (rosich@ice.cat)
#
# Created:     2015/01/15

#-----------------------------------------------------------------------------

"""
The MIT License (MIT)

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
"""
from math import sin, cos, pi, acos, sqrt, exp, log, log10
import sys
import random
import numpy as np
import ConfigParser
import multiprocessing as mp
import getopt
sys.path.append('./bin')
import mGLS, mMGLS, mMGLS_AR
import time as tme
sys.path.append('./src')
from mgls_lib import *
import mgls_io
import mgls_mc
import mgls_bootstrapping
from EnvGlobals import Globals

               
if __name__ == '__main__':
    """main program
    """
    # Replace output streams
    stdout, stderr = sys.stdout, sys.stderr

    #option capture
    Globals.inhibit_msg = False  #global inhibit messages is off
    Globals.gls_opt = False
    Globals.bootstrapping = False
    Globals.gls_trended = False
    Globals.analysis = False
    Globals.n_bootstrapping = 50
    Globals.ncpus = mp.cpu_count()
    options, remainder = getopt.gnu_getopt(sys.argv[1:], 'a:i:g:n:d:s:r:c',
                         ['inhibit_msg', 'gls', 'ncpus=', 'ndim=', 'bootstrapping=', 'linear_trend', 'ar=', 'col='])
    #argument parsing
    for opt, arg in options:
        if opt in ('-n', '--ncpus'):
            Globals.ncpus = int(arg)
        elif opt in ('-i', '--inhibit_msg'):
            Globals.inhibit_msg = True
        elif opt in ('-g', '--gls'):
            Globals.gls_opt = True
        elif opt in ('-d', '--ndim'):
            Globals.ndim = int(arg)
        elif opt in ('-s', '--bootstrapping'):
            Globals.bootstrapping = True
            Globals.n_bootstrapping = int(arg)
        elif opt in ('-a', '--linear_trend'):
            Globals.linear_trend = True
        elif opt in ('-r', '--ar'):
            Globals.ar = True
            Globals.nar = int(arg)
        elif opt in ('-c', '--col'):
            Globals.col = int(arg)

 #print init data
    print_start_info(Globals.ncpus)
    
    #read data
    try:
        in_data = mgls_io.read_file(sys.argv[1], ' ')
        print_message("Read data file...............................[OK]", index=3, color=32)
    except IOError:
        print_message("Data file could not be read", 5, 31)
        sys.exit()

    if Globals.ncpus != 0:
        print_message( 'Detected CPUs / using CPUs: ' + str(mp.cpu_count()) + "/" + str(Globals.ncpus), 5, 91)
        pass
    #assign data vectors
    Globals.time, Globals.rv, Globals.rv_err = mgls_io.get_data_vectors(in_data, Globals.col)
    
    #analyze input file 
    summ = 0.0
    separation = []
    for i in range(len(Globals.time)-1):
        summ += Globals.time[i+1] - Globals.time[i]
        separation.append(Globals.time[i+1] - Globals.time[i])
    if not Globals.inhibit_msg:
        print "Data points:", len(Globals.time)
        print "Time span:", Globals.time[-1] - Globals.time[0]
        print "Mean sep.", summ/len(Globals.time)
    #mgls_io.write_file_onecol('t_separation.dat', separation, ' ','')
    
    #periods to scan
    Globals.period_range = [1.0, 5000.0]
    Globals.freq_range = [1./Globals.period_range[1], 1./Globals.period_range[0]]
    #AR process tau
    Globals.params = [1.5]
    #compute the reference chi2 value as a global variable
    Globals.chi2_0 = mMGLS.chi2_0(Globals.rv, Globals.rv_err)
    #compute and subtract a linear trend, if appliable
    if Globals.linear_trend:
            #try to fit a linear trend
            slope, intercept, r, p_value = linear_trend()
            #subtract from data this trend
            Globals.rv -= (slope*Globals.time + intercept)
    #////////////////////////////////////////////////////////////////////////////////////
    #OPTION SELECTION
    #///////////////////////////////////////////////////////////////////////////////////
    if Globals.gls_opt:
        try:
            fap_thresholds = list()   #initialization of FAP list
            periods, pwr, max_pow = gls_1D()
            #bootstrapping stats
            #bootstrapping_stats, fap_thresholds = bootstrapping_1D(max_pow)
            #mgls_io.write_file_onecol('bootstrapping_stats.dat', bootstrapping_stats, ' ', '')
            #print fap(bootstrapping_stats, max_pow[1])
            #plot 1D GLS
            plot(periods,pwr,Globals.time,Globals.rv,Globals.rv_err,max_pow, fap_thresholds, trended=False)
        except:
            sys.stdout, sys.stderr = stdout, stderr
            raise

    else:
        """performs standard analysis
        """
        try:
            #optimize frequency tuple
            opt_state = mgls_mc.parallel_optimization(Globals.ncpus)
            #compute coefficients and A matrix, given the optimal configuration        
            pwr_opt, fitting_coeffs, A = mgls(opt_state)
            print "Joint power:", pwr_opt
            
            print "Periods (d):"
            for i in range(len(opt_state)): print >> stdout, '\t', round(1./opt_state[i],5)
            #fitting_coeffs
            print >> stdout, "\n\nFitting coefficients:", fitting_coeffs
            cov = covariance_matrix(A)
            #for j in range(len(cov)): print sqrt(cov[j][j])
            
            #compute reduced chi2
            if Globals.ar:
                model = mMGLS_AR.model(Globals.time, Globals.rv, opt_state, fitting_coeffs, Globals.params)
                chi2 = mMGLS_AR.chi2(Globals.time, Globals.rv, Globals.rv_err, model)
                chi2_0 = mMGLS_AR.chi2_0(Globals.rv, Globals.rv_err)
                print "Chi-2_0:", chi2_0
                print "Chi-2:", chi2
            else:
                chi2 = mMGLS.chi2(Globals.time, Globals.rv, Globals.rv_err, Globals.time[0], opt_state, fitting_coeffs)
                chi2_0 = mMGLS.chi2_0(Globals.rv, Globals.rv_err)
                print "Chi-2_0:", chi2_0
                print "Chi-2:", chi2
            
            for j in range(len(opt_state)):
                n_dim = Globals.ndim #+ Globals.ar
                a,b = fitting_coeffs[j+1], fitting_coeffs[j+n_dim+1]
                K = sqrt(a**2 + b**2)
                try:
                    da, db = sqrt(cov[j+1][j+1]), sqrt(cov[j+n_dim+1][j+n_dim+1])
                    print "K[",j,"]:", K, (abs(a/K)*da + abs(b/K)*db)
                except ValueError:
                    pass
            print ""
            
            #computing NLL (negative log-likelihood)
            logL = 0.0
            for i in range(len(Globals.time)):
                model = mMGLS.model(Globals.time[i], Globals.time[0], opt_state, fitting_coeffs)
                p = (1./(Globals.rv_err[i]*sqrt(2.*np.pi)))*exp((-(Globals.rv[i] - model)**2) / (2.*(Globals.rv_err[i]**2)))
                logL += log(p)
            #print "Negative log-likelihood:", -logL 
            print -logL
            #print BIC statistic
            print -2*logL + (2*Globals.ndim+1)*log(len(Globals.time))
            
            #print reduced chi2           
            print "Reduced Chi-2:", chi2 / (len(Globals.time) - len(fitting_coeffs) - Globals.nar - 1)
            #residuals of model
            if Globals.ar:
                residuals = mMGLS_AR.residuals(Globals.time, Globals.rv, model)
                model_residuals(residuals)
            else:
                residuals = mMGLS.residuals(Globals.time, Globals.rv, Globals.time[0], opt_state, fitting_coeffs)
                model_residuals(residuals)
                
            if Globals.bootstrapping:
                print "Bootstrapping..."
                bootstrapping_stats = mgls_bootstrapping.Mdim_bootstrapping(pwr_opt)
                print "Times over max. power:", mgls_bootstrapping.fap(bootstrapping_stats, pwr_opt)
            #chi2_histo = mgls_bootstrapping.parametric_bootstrapping(opt_state, fitting_coeffs, 15000)
            #mgls_io.write_file_onecol('chi2.histogram' + str(Globals.ndim), chi2_histo, ' ', '')
          
            #write a file with the model adjusted
            model = list()
            for i in range(len(Globals.time)):
                y_model = mMGLS.model(Globals.time[i], Globals.time[0], opt_state, fitting_coeffs)
                model.append([Globals.time[i], y_model, random.gauss(0.0,1.0)])
            mgls_io.write_file('model.dat', model, ' ', '')
            
        except:
            sys.stdout, sys.stderr = stdout, stderr
            raise
            

    