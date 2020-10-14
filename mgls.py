#! /usr/bin/python

#-----------------------------------------------------------------------------
# Project:     Multidimensional Generalized Lomb-Scargle

# Name:        gls_parallel.py

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
from math import sin, cos, pi, acos, sqrt, exp, log, log10, sqrt, atan
import sys
import random
import copy
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
import mgls_zk
import mgls_genetic
from EnvGlobals import Globals
import scipy.optimize
import copy

def help():
    """Help
    """
    print "===================================================================="
    print "Example of usage:\n"
    print "./mgls_v2.py [data_file_1] [data_file_2] ... [data_file_n] --ndim=2"
    print "===================================================================="
    print "OPTIONS:"
    print ""
    print "--gls               :: Compute and plot unidimensional Generalized Lomb-Scargle periodogram"
    print "--pmin= / --pmax=   :: Set limits in periods to be explored. Prestablished values are 1.25-10000 d"
    print "--jitter            :: Fit additional jitter (s) in quadrature (e^2 = sigma^ + s^2)"
    print "--period            :: [to be used with gls option] plot GLS periodogram in period log-scale"
    print "--ndim=             :: Number of signals to be fitted"
    print "--linear_trend      :: Fit a linear trend simultaneously"
    print ""
    sys.exit()
    
def logL_NullModel():
    """ndim = 0
    """
    print_message("\nEvaluating 0-model...", 6,35) 
    
    Globals.logL_0 = 0.0
    for i in range(Globals.n_sets):
        mean_data = np.mean(Globals.rvs[i])
        inv_sigma2_set = 1.0/(Globals.rv_errs[i]**2) 
        Globals.logL_0 += -0.5*(np.sum(((Globals.rvs[i]-mean_data)**2)*inv_sigma2_set + np.log(2.0*np.pi) + np.log(1.0/inv_sigma2_set)) ) 
       
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
        """
        opt_state = mgls_mc.parallel_optimization_multiset(Globals.ncpus, N_CANDIDATES=6)
        #compute coefficients and A matrix, given the optimal configuration             
        pwr, fitting_coeffs, A, logL = mgls_multiset(opt_state)
        #arrays of frequencies & jitters
        opt_jitters = opt_state[:] #ndim=0
        """
        Globals.logL_0 = logL
        #reestablish dimensionality
        Globals.ndim = ndim
        Globals.inhibit_msg = False
    
        print "\tlogL null model (data + jitter):", Globals.logL_0
        print "\t0-freq jitter(s):" 
        for i in range(len(opt_jitters)): print >> stdout, '\t\t', str(i) + '/', round(opt_jitters[i],5)
        Globals.opt_jitters_0 = opt_jitters
        if local_linear_trend: Globals.linear_trend = True 
        
    else:
        print "\tlogL null model (data, no jitter):", Globals.logL_0
    
    return Globals.logL_0

if __name__ == '__main__':
    """main program
    """
    # Replace output streams
    stdout, stderr = sys.stdout, sys.stderr
    import warnings
    warnings.simplefilter('ignore', np.RankWarning)
  
    #option capture
    Globals.inhibit_msg = False  #global inhibit messages is off
    Globals.gls_opt = False
    Globals.errors = False
    Globals.km2m = False
    Globals.bidim_plot = False
    Globals.testing = False
    Globals.bootstrapping = False
    Globals.logL_0 = -1.0
    Globals.n_bootstrapping = 50
    Globals.multiset = False
    Globals.opt_jitters_0 = []
    Globals.pmin, Globals.pmax = 2.5, 10000.0
    Globals.ncpus = mp.cpu_count()
    
    options, remainder = getopt.gnu_getopt(sys.argv[1:], 'b:i:g:n:d:s:l:r:v:y:j:m:t:p:q:h:x:a:e',\
                         ['bidim','inhibit_msg','gls', 'ncpus=' ,\
                         'ndim=', 'bootstrapping=', 'linear_trend','ar=','col=', \
                         'jitter', 'logL', 'testing', 'pmin=', 'pmax=', 'help', 'period','log_scale','errors','km2m'])
    #argument parsing
    for opt, arg in options:
        if opt in ('-n', '--ncpus'):
            Globals.ncpus = int(arg)
        elif opt in ('-i', '--inhibit_msg'):
            Globals.inhibit_msg = True
        elif opt in ('-g', '--gls'):
            Globals.gls_opt = True
        elif opt in ('-b', '--bidim'):
            Globals.bidim_plot = True
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
        elif opt in ('-m', '--logL'):
            Globals.logL = True
        elif opt in ('-j', '--jitter'):
            Globals.jitter = True
        elif opt in ('-t', '--testing'):
            Globals.testing = True
        elif opt in ('-p', '--pmin'):
            Globals.pmin = float(arg)
        elif opt in ('-q', '--pmax'):
            Globals.pmax = float(arg)
        elif opt in ('-h', '--help'):
            Globals.help = True
        elif opt in ('-x', '--period'):
            Globals.inPeriods = True
        elif opt in ('-a', '--log_scale'):
            Globals.log_scale = True
        elif opt in ('-e', '--errors'):
            Globals.errors = True
        elif opt in ('--km2m'):
            Globals.km2m = True
            
    #print init data
    print_heading(Globals.ncpus)
    #periods to scan
    Globals.period_range = [Globals.pmin, Globals.pmax]  #(days)
    Globals.freq_range = [1./Globals.period_range[1], 1./Globals.period_range[0]]
    
    if Globals.help or len(sys.argv) == 1:
        """usage info
        """
        help()
            
    #CPUs 
    if Globals.ncpus != 0:
        print_message( '\nDetected CPUs / using CPUs: ' + str(mp.cpu_count()) + "/" + str(Globals.ncpus), 5, 31)
        pass
    
    #load data passed thorugh CL
    try:
        load_multiset_data()
       
    except IOError:
        print_message("Some data file could not be read", 5, 31)
        sys.exit()
    
    #jitter limit according to datasets
    Globals.jitter_limit = 10.0*max(Globals.mean_err)
    if Globals.jitter:
        print "Max. jitter:", Globals.jitter_limit
    
    #compute and subtract a linear trend, if appliable
    if Globals.linear_trend:
        """apply a linear trend on data
        """
        print_message("\nLinear trend statistics (previous analysis of data, info purposes)", 6,92)
        #try to fit a linear trend
        
        lt_params = []
        for s in range(Globals.n_sets):
            slope, intercept, r, p_value = linear_trend(Globals.times[s], Globals.rvs[s])
            #subtract from data this trend
            #Globals.rvs[s] -= (slope*Globals.times[s] + intercept)
            #Globals.rvs_seq.extend(Globals.rvs[s])
            lt_params.append([slope, intercept, r, p_value])
        #Globals.rvs_seq = np.array(Globals.rvs_seq)
        
        try:
            from terminaltables import AsciiTable, DoubleTable, SingleTable
            """pip install terminaltables
            """
            TABLE_DATA = []
            TABLE_DATA.append(['Data set', 'slope', 'intercept', 'r', 'p-value'])
            for s in range(Globals.n_sets):
                TABLE_DATA.append([str(Globals.dataset_names[s]), str(round(lt_params[s][0],7)), str(round(lt_params[s][1],5)), str(round(lt_params[s][2],5)), str(lt_params[s][3])])
            
            TABLE_DATA = tuple(TABLE_DATA)
            table_instance = SingleTable(TABLE_DATA, "Linear trend fit")
            table_instance.justify_column = 'right'
            print ""
            print(table_instance.table)
            
        except:
            #if terminaltables is not installed
            for s in range(Globals.n_sets):
                print ""
                print_message('\t/' + str(s) + " Data set:" + str(Globals.dataset_names[s]), 3, 32)
                print "\tslope:", lt_params[s][0]
                print "\tintercept:", lt_params[s][1]
                print "\tr:", lt_params[s][2]
                print "\tp-value:", lt_params[s][3]
   
    #try:
        #compute logL_0 of data (model 0)   
    logL_NullModel()
    #except:
        #print ("Something went wrong when computing logL null model")
        #sys.exit()
        
    
    #////////////////////////////////////////////////////////////////////////////////////
    #OPTION SELECTION
    #///////////////////////////////////////////////////////////////////////////////////
    
    if Globals.gls_opt:
        """unidimensional Lomb-Scargle periodogram
        """
        #GLS (1-D)
        try:
            fap_thresholds = list()  #initialization of FAP list
            file_out = []
            #logL 1-D plot
            freqs, pwr, max_pow = gls_1D()
            for j in range(len(freqs)): file_out.append([freqs[j], pwr[j]])
            #mgls_io.write_file('gls_periodogram.ascii', file_out, ' ', '')
        
        #bootstrapping stats
            if Globals.bootstrapping:
                logL_0 = Globals.logL_0
                #copy data 
                G_rv, G_rv_errs = Globals.rvs_seq, Globals.rv_errs_seq
                #bootstrapping_stats, fap_thresholds = bootstrapping_1D(max_pow)
                #mgls_io.write_file_onecol('bootstrapping_stats.dat', bootstrapping_stats, ' ', '')
                #print fap(bootstrapping_stats, max_pow[1])
                print "Bootstrapping..."
                max_peaks = mgls_mc.bootstrapping_1D(Globals.n_bootstrapping)
                mgls_io.write_file_onecol('FAP_' + str(int(random.uniform(0,10000))) + '.dat', max_peaks, ' ', '')
                
                #print np.mean(max_peaks), np.std(max_peaks), np.mean(max_peaks) + 3.*np.std(max_peaks)
                fap_thresholds = fap_levels(max_peaks)
                print ""
                print "FAP Levels:", fap_thresholds
                print "Total bootstapping samples: ", len(max_peaks)
                
                Globals.rvs_seq, Globals.rv_errs_seq = G_rv, G_rv_errs
                Globals.logL_0 = logL_0
            
            #plot 1D GLS
            peaks = peak_counter(freqs, pwr)
            print "Independent frequencies found (peaks):", len(peaks)
            plot(freqs, pwr, Globals.times_seq, Globals.rvs_seq, Globals.rv_errs_seq, max_pow, fap_thresholds, peaks[:1])
            
        except:
            sys.stdout, sys.stderr = stdout, stderr
            raise
  
    elif Globals.bidim_plot:
        """
        """ 
        p_min_x, p_max_x = 10.0, 30.0 #4.8, 5.1
        p_min_y, p_max_y = 10.0, 30.0#10.6, 12.2
        
        freqs_0 = []
        Globals.ndim = len(freqs_0) + 2
 
        max_logL = -np.inf
        min_pwr = np.inf
        
        #evaluate jitters if selected
        if Globals.jitter:
            #optimize frequency tuple
            opt_state = mgls_mc.parallel_optimization_multiset(Globals.ncpus, N_CANDIDATES=12)
            #compute coefficients and A matrix, given the optimal configuration        
            pwr, fitting_coeffs, A, logL = mgls_multiset(opt_state)
            #arrays of frequencies & jitters
            opt_freqs, opt_jitters = opt_state[:Globals.ndim], opt_state[Globals.ndim:]
        else:
            opt_jitters = [0.0 for iter in range(Globals.n_sets)]
        
        print_message("\nEvaluating model...", 6,35)        
        #bidim plot
        try:           
            import matplotlib as mpl
            if os.environ.get('DISPLAY','') == '':
                print('no display found. Using non-interactive Agg backend')
                mpl.use('Agg')
            import matplotlib.pyplot as plt
            from pylab import figure, show, legend, ylabel, colorbar
            import pylab
            from matplotlib import cm
            from matplotlib.ticker import ScalarFormatter, FormatStrFormatter
            
            #plot it
            # define the grid over which the function should be plotted (xx and yy are matrices)
            """
            aa, bb = pylab.meshgrid(
                     np.concatenate( (pylab.linspace(1./2.0, 1./100.0, 50),pylab.linspace(1./100.0, 1./10000.0, 950)) ), 
                     np.concatenate( (pylab.linspace(1./2.0, 1./100.0, 50),pylab.linspace(1./100.0, 1./10000.0, 950)) ) 
                     )
            """
            aa, bb = pylab.meshgrid( pylab.linspace(1./p_min_x, 1./p_max_x, 700), 
                                     pylab.linspace(1./p_min_x, 1./p_max_y, 700)
                     )
            
            # fill a matrix with the function values
            zz = pylab.zeros(aa.shape)
            
            #several datasets
            if freqs_0 != []:
                inv_sigma2_data = 1.0/(Globals.rv_errs_seq**2)
                fitting_coeffs_0, A_matrix_0 = mMGLS.mdim_gls_multiset(Globals.times_seq, Globals.rvs_seq, Globals.rv_errs_seq, freqs_0, jitters,Globals.len_sets)
                model_0 = mMGLS.model_series_multiset(Globals.times_seq, freqs_0, fitting_coeffs_0, Globals.len_sets)
                logL_0 = -0.5*(np.sum(((Globals.rvs_seq-model_0)**2)*inv_sigma2_data + np.log(2.0*np.pi) + np.log(1.0/inv_sigma2_data)) ) 
                print "\tBase model logL_0:", logL_0
            
            else:
                logL_0 = Globals.logL_0
    
            for i in range(aa.shape[0]):
                a = aa[0][i]
                #print progress control
                if i % 50 == 0: print "current row:", i
                for j in range(bb.shape[0]):
                    b = bb[j][0]
                    #compute only the lower triangle
                    if i < j:
                        #array of frequencies to evaluate
                        freqs = [ a, b ] + freqs_0
                        jitters = opt_jitters
                        if Globals.linear_trend:
                            fitting_coeffs, A_matrix = mMGLS.mdim_gls_multiset_trend(Globals.times_seq, Globals.rvs_seq, Globals.rv_errs_seq, freqs, jitters, Globals.len_sets)
                        else:
                            fitting_coeffs, A_matrix = mMGLS.mdim_gls_multiset(Globals.times_seq, Globals.rvs_seq, Globals.rv_errs_seq, freqs, jitters, Globals.len_sets)                
                        
                        logL = 0.0
                        for s in range(Globals.n_sets):
                            fitting_coeffs_ = []
                            fitting_coeffs_.append(fitting_coeffs[s]) #append constant
                            for k in range(Globals.n_sets, len(fitting_coeffs)):
                                fitting_coeffs_.append(fitting_coeffs[k])
 
                            model = mMGLS.model_series(Globals.times[s], Globals.times[0][0], freqs, fitting_coeffs_, len(Globals.times[s])) 
                            
                            inv_sigma2_set = 1.0/(Globals.rv_errs[s]**2 + opt_jitters[s]**2)
                            logL += -0.5*(np.sum(((Globals.rvs[s] - model)**2)*inv_sigma2_set + np.log(2.0*np.pi) + np.log(1.0/inv_sigma2_set)) )
                    
                        pwr = -(logL_0 - logL)
                        #no negative delta logL are allowed (do to significance of K)
                        if pwr < 0.0: pwr = 0.0
                        
                        if pwr < min_pwr:
                            min_pwr = pwr

                        zz[i,j] = pwr
                        #symmetric part
                        zz[j,i] = pwr
                        
        
            #diagonal            
            for i in range(bb.shape[0]):
                for j in range(bb.shape[0]):
                    if i == j:
                        zz[i,j] = min_pwr
          
             
            #cuts of optimum
            h, v = [], []
            #position of max
            i,j = np.unravel_index(zz.argmax(), zz.shape)
            
            pwrx, pwry, periods = [],[],[]
            for i in range(aa.shape[0]):
                a = aa[0][i]
                p_y = 1./a 
                if i == j:
                    pwr_y = zz[i-1,j]
                else:
                    pwr_y = zz[i,j]
             
                v.append([p_y, pwr_y])
                pwry.append(pwr_y)
                
            for j in range(bb.shape[0]):
                b = bb[j][0]
                p_x = 1./b 
                if i == j:
                    pwr_x = zz[j-1,i]
                else:
                    pwr_x = zz[j,i]
             
                h.append([p_x, pwr_x])
                periods.append(p_x)
                pwrx.append(pwr_x)
                
            mgls_io.write_file('cut_v.dat', v, ' ', '')   
            mgls_io.write_file('cut_h.dat', h, ' ', '')
            
            import warnings
            warnings.filterwarnings("ignore")

            #fig1 = figure()
            #plt.subplots_adjust(bottom=0.11, right=0.76) 
            fig1 = plt.figure(1,figsize=(8,6))
            plt.subplots_adjust(left=0.07, right=0.73, top=0.96, bottom=0.1, hspace=0.05)
            
            plt.rc('font', serif='Helvetica Neue')
            plt.rcParams.update({'font.size': 12.5})
            c_plot = fig1.add_subplot(111)
            # plot the calculated function values
            cplot = c_plot.pcolor(1./aa, 1./bb, zz, vmax=zz.max(), cmap=cm.jet)
            # and a color bar to show the correspondence between function value and color
            cbaxes = fig1.add_axes([0.87, 0.105, 0.03, 0.855])  # This is the position for the colorbar
            cbar = colorbar(cplot, cax = cbaxes, orientation='vertical')
            
            c_plot.yaxis.tick_right()
            cbar.ax.set_ylabel('$\Delta \ln L$', fontsize=16)
            #cbar.ax.set_label_position("left")
            c_plot.yaxis.set_label_position("right")
            c_plot.set_xlabel("$P_1$ (d)", fontsize=16)
            c_plot.set_ylabel("$P_2$ (d)", fontsize=16)
            
            if Globals.log_scale:
                c_plot.set_yscale('log')
                c_plot.set_xscale('log')
            
         
            c_plot.set_xlim(p_min_x,p_max_x)
            c_plot.set_ylim(p_min_y,p_max_y)
            
            c_plot.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
            c_plot.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
            
            plt.savefig("bidim.png", dpi=600)
            plt.show()
           
        except:
            sys.stdout, sys.stderr = stdout, stderr
            raise
                
    elif Globals.testing:
        """testing zone
        """
        
        print_message("\nNoise analysis running...", 3, 31)
        Globals.inhibit_msg = True
        DIM_MAX = 5
   
        rvs_seq, rv_errs_seq = copy.deepcopy(Globals.rvs_seq), copy.deepcopy(Globals.rv_errs_seq)
        rvs_cp, rv_errs_cp = copy.deepcopy(Globals.rvs), copy.deepcopy(Globals.rv_errs)
        
        for init_dim in range(0,DIM_MAX):
            #init dimension
            Globals.ndim = init_dim
            print "Current dim.", Globals.ndim
            #load_multiset_data()
        
            #find the periodicities for a given dimensionality 
            opt_state = mgls_mc.parallel_optimization_multiset(Globals.ncpus, N_CANDIDATES=96)
            print "Periods found:", 1./opt_state[:Globals.ndim]   
            #compute coefficients and A matrix, given the optimal configuration        
            pwr, fitting_coeffs_base, A, logL = mgls_multiset(opt_state)
            
            #arrays of frequencies & jitters
            opt_freqs_base, opt_jitters_base = opt_state[:Globals.ndim], opt_state[Globals.ndim:]
            
            p90, p99, p999 = mgls_mc.noise_analysis(init_dim, opt_freqs_base, fitting_coeffs_base, opt_jitters_base)
            
            print "Dim:", init_dim, "-->", init_dim + 1
            print "DlogL (p = .90):", p90
            print "DlogL (p = .99):", p99
            print "DlogL (p = .999):", p999
            
            Globals.rvs_seq, Globals.rv_errs_seq = rvs_seq[:], rv_errs_seq[:]
            Globals.rvs, Globals.rv_errs = rvs_cp[:], rv_errs_cp[:] 
   
            
    elif Globals.errors:
        """Estimate the uncertainties in period by bootstrap (Efron, 1982) method
        """
        def sigma_clipping(vector, sigma_inf, sigma_sup):
            """
            """
            vector = np.array(vector)
            mean = np.mean(vector, axis=0)
            std = np.std(vector, axis=0)
            print "clipping params:", mean, std
            clipped = []
            for i in range(len(vector)):
                for j in range(len(vector[0])):
                    if not (mean[j]-sigma_inf*std[j] < vector[i][j] < mean[j]+sigma_sup*std[j]):
                        break
                clipped.append(vector[i])
                        
            return clipped
        
        bootstrap_samples = []
        Globals.inhibit_msg = True
        rvs_0 = copy.deepcopy(Globals.rvs)
        
        #mean_sigma = sqrt(np.mean(np.array(Globals.rv_errs)**2))
        mean_sigma = np.mean(np.array(Globals.rv_errs_seq))
        #iterate
        for k in range(500):
            #resample data array
            logL_sample = 0.0
            for j in range(Globals.n_sets):
                for i in range(len(Globals.rvs[j])):
                    Globals.rvs[j][i] += random.gauss(0.0, mean_sigma)

            #optimize frequency tuple
            opt_state = mgls_mc.parallel_optimization_multiset(Globals.ncpus, N_CANDIDATES=8)
            #compute coefficients and A matrix, given the optimal configuration        
            pwr, fitting_coeffs, A, logL = mgls_multiset(opt_state)
            #select the periods
            opt_freqs = opt_state[:Globals.ndim]
            #sort descendent
            opt_freqs = np.sort(opt_freqs)
            
            if Globals.jitter:
                opt_jitters = opt_state[:Globals.n_sets]
                print 1./opt_freqs#, opt_jitters, fitting_coeffs, logL
            else:
                print 1./opt_freqs, logL
                
            #bootstrap_samples.append([1./opt_freqs, opt_jitters, fitting_coeffs, logL])
            bootstrap_samples.append(1./opt_freqs)
            #restablish original values
            Globals.rvs = []
            Globals.rvs = copy.deepcopy(rvs_0)
        
            
        mgls_io.write_file('samples_teegarden_test_visnir' + '.dat', bootstrap_samples, ' ', '')
    
    elif Globals.mcmc:
        """
        """
        
    else:
        """performs multidimensional standard analysis
        """
        
        try:
            print_message("\nEvaluating model...", 6,35)
           
            if not Globals.jitter: 
                jitters = [0.0 for iter in range(Globals.n_sets)]
      
            #optimize frequency tuple
            opt_state = mgls_mc.run_MGLS(Globals.ncpus, N_CANDIDATES=32)
            
            #compute coefficients and A matrix, given the optimal configuration        
            pwr, fitting_coeffs, A, logL = mgls_multiset(opt_state)
            #arrays of frequencies & jitters
            opt_freqs, opt_jitters = opt_state[:Globals.ndim], opt_state[Globals.ndim:]
        
            #covariance matrix
            try:
                cov = covariance_matrix(A)
            except:
                print "Cannot compute the covariance matrix. Singular matrix"
                pass
            
            #print results&info
            print_message("\nPeriods (au):", 6,92)
            for i in range(Globals.ndim): print >> stdout, '\t', round(1./opt_freqs[i],5)
            
            #fitting_coeffs
            print_message("\nFitting coefficients / Uncertainties", 6,92)
            for j in range(Globals.ndim):
                try:
                    da = sqrt(cov[j+Globals.n_sets][j+Globals.n_sets])
                except:
                    print "Value error. delta_a not defined"
                    da = 'N/D'
                    
                print "\t", "a[ " + str(j),"]:", fitting_coeffs[j+Globals.n_sets], '+/-', da
            for j in range(Globals.ndim):
                try:
                    db = sqrt(cov[j+Globals.ndim+Globals.n_sets][j+Globals.ndim+Globals.n_sets])
                except:
                    print "Value error. delta_b not defined"
                    db = 'N/D'
                print "\t", "b[ " + str(j), "]:", fitting_coeffs[j+Globals.n_sets+Globals.ndim], '+/-', db
            
            if Globals.linear_trend:
                #print_message("\nLinear trend:", 6,92)
                print "\t", "linear trend slope", fitting_coeffs[2*Globals.ndim+Globals.n_sets]
                
            print_message("\nAmplitudes / Uncertainties",6,92)
            for j in range(Globals.ndim): 
                a,b = fitting_coeffs[j+Globals.n_sets], fitting_coeffs[j+Globals.ndim+Globals.n_sets]
                K = sqrt(a**2 + b**2)
                try:
                    da, db = sqrt(cov[j+Globals.n_sets][j+Globals.n_sets]), \
                             sqrt(cov[j+Globals.ndim+Globals.n_sets][j+Globals.ndim+Globals.n_sets])
                    print "\tK[",j,"]:", K, "+/-",(abs(a/K)*da + abs(b/K)*db)
                except:
                    print "Value error"
                    pass
            
            print_message("\nPhases / Uncertainties",6,92)
            for j in range(Globals.ndim): 
                a,b = fitting_coeffs[j+Globals.n_sets], fitting_coeffs[j+Globals.ndim+Globals.n_sets]
                Ph = atan(a/b)*180.0/pi
                try:
                    da, db = sqrt(cov[j+Globals.n_sets][j+Globals.n_sets]), \
                             sqrt(cov[j+Globals.ndim+Globals.n_sets][j+Globals.ndim+Globals.n_sets])
                    print "\tPh[",j,"]:", Ph, "+/-", (180./pi)*(abs((1.0/(1.0+(a/b)**2)/b))*da + abs((-a/(1.0+(a/b)**2)/b**2))*db)
                except:
                    print "Value error"
                    pass
                
            #print offsets
            print_message("\nOffsets / Uncertainties",6,92)
            for i in range(Globals.n_sets):
                try:
                    err = sqrt(cov[i][i])
                except:
                    err = ' - '
                    
                print "\tc[ " + str(i), "]:", fitting_coeffs[i], "+/-", err
                
            if Globals.jitter:
                print_message("\nJitter(s):",6,92)
                for i in range(Globals.n_sets):
                    print "\tset[ " + str(i), ']:', opt_state[Globals.ndim+i]
        
            #print "Negative log-likelihood:", -logL 
            print_message("\nSpectral stats:",6,92)
            print "\tJoint P statistic [logL-logL_0/logL_0]:",round(pwr, 5)
            print "\tlogL_0 (null-model):", Globals.logL_0
            print "\tlogL (model): " + str(logL)
            print "\tDlogL (model - null_model): " + str(-Globals.logL_0+logL)
            print ""
            
            #compute and write on disk the fitted model
            FITTED_MODEL = multiset_model(opt_freqs, opt_jitters, fitting_coeffs)
   

        except:
            sys.stdout, sys.stderr = stdout, stderr
            raise
            

