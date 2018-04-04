#!/usr/bin/python
from math import sin, cos, tan, atan, pi, acos, sqrt, exp, log10, log
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
    def base_round(x, base=20):
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
        # current and other are axes
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
    
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print "python-matplotlib package must be installed on your box"
    
    
    y2, err = y2.tolist(), err.tolist()
    #LaTeX env.
    plt.rc('font', serif='Helvetica Neue')
    plt.rcParams.update({'font.size': 10.5})
      
    fig, ax1 = plt.subplots(figsize=(14,4.3))
    plt.subplots_adjust(left=0.07, right=0.96, top=0.85, bottom=0.15, hspace=0.3)
   
    ax1.plot(t, y, '-r', color='black', linewidth=0.6)
    #ax1.fill_between(t, y, min(y), facecolor='steelblue', color='steelblue', alpha=0.99)
    ax1.set_xlim((t[0],t[-1]))
    
    try:
        ax2 = ax1.twiny()
        #period ticks
        ax2.set_xlim(ax1.get_xlim())
        #periods to be 'ticked'
        """
        p_range = (1./t[0] - (1./t[-1]))
        if p_range > 5000.0 and (1./t[-1]) <= 50.0:
            p_range *= 1.9
        elif p_range > 500.0 and p_range < 1500.0 and (1./t[-1]) < 50.0:
            p_range *= 5.0
        elif p_range > 1500.0 and (1./t[-1]) < 50.0:
            p_range *= 1.0    
        
        periods = np.array( [ 1./t[-1], base_round(1./t[-1] + 0.01*p_range),
                            base_round(1./t[-1] + 0.007*p_range), base_round(1./t[-1] + 0.003*p_range),
                            base_round(1./t[-1] + 0.0005*p_range), base_round(1./t[-1] + 0.00005*p_range),
                            base_round(1./t[0]) ] )
        """
        periods = np.array(1./np.linspace( t[0], t[-1], 5))
        periods = [base_round(period) for period in periods]
        #new_tick_locations = ax1.get_xticks()
        new_tick_locations = 1./np.array(periods)
        ax2.set_xticks(new_tick_locations)
        ax2.set_xticklabels(tick_function(new_tick_locations))
        #mouse coordinates in both scales
        ax2.format_coord = make_format(ax2, ax1)
        ax2.set_xlabel(r'Period (d)', fontsize=13, fontweight='bold')
    except:
        print "Error in period axis"
    
    ax1.minorticks_on() 
    ax1.set_ylabel(r'$\Delta$lnL', rotation='vertical', fontsize=13, fontweight='bold')
    ax1.set_xlabel(r'Frequency (d$^{-1}$)', fontsize=13, fontweight='bold')
    ax1.set_ylim((min(y),1.15*max(y)))  
       
    """
    ax1.axvline(1.25,linestyle='-.',color='red')
            )
    ax1.annotate('4.91d', xy=(4.91, 0.08), xytext=(5.5, 0.2)
            )

    #ax1.axhline(max_peak[1],linestyle='-.',color='0.80')

    #Anotations
    ax1.annotate('8.6d', xy=(8.6, 0.4), xytext=(8.0, 0.5)
            )
    """ 
    if fap_thresholds != []:
        print "FAP levels encountered"
        colors = ['gray', 'black', 'red', 'green']
        for i in range(len(fap_thresholds)):
            ax1.axhline(fap_thresholds[i],linestyle='-.',color=colors[i])
    else:
        print "No FAP levels"
    
    plt.show()
   
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

def bootstrapping_1D(max_pow):
    """
    """
    #iterations
    n_bootstrapping = Globals.n_bootstrapping
    print "\n//BOOTSTRAPPING://"
    bootstrapping_stats = parallel_bootstrapping(n_bootstrapping)
    fap_thresholds = fap_levels(bootstrapping_stats)
    print "FAP Levels:", fap_thresholds
    print "Total bootstapping samples: ", len(bootstrapping_stats)

    return bootstrapping_stats, fap_thresholds        
  
def gls_1D():
    """
    """
    Globals.ndim = 1  #unidimensional GLS
    n_points = 5000
    freqs = np.linspace(1./Globals.period_range[1], 1./Globals.period_range[0], n_points) 
    
    non_linear_params = [0.0]
    if Globals.jitter:
        non_linear_params.extend(Globals.opt_jitters_0)
    
    non_linear_params = np.array(non_linear_params)
    DlogLs = []
    max_DlogL = -np.inf
    for i in range(len(freqs)):
        if freqs[i] == non_linear_params[0]:
            freqs[i] = freqs[i-1]
        non_linear_params[0] = freqs[i]
        pwr, fitting_coeffs, A, logL = mgls_multiset(non_linear_params)   
        DlogL = logL - Globals.logL_0
        if DlogL < 0.0: DlogL = 0.0
        DlogLs.append(DlogL)
        #find the peak
        if DlogL > max_DlogL:
            max_logL = DlogL
    
    return freqs, DlogLs, max_DlogL
        
def covariance_matrix(sys_matrix):
    """calculate the covariance matrix for params estimation
    """
    from numpy.linalg import inv
    Cov = inv(sys_matrix)
  
    return Cov
    
def print_heading(ncpus):
    """
    """
    print ''
    print_message('                                                                                            ', 7, 96)
    print_message('           MGLS (MULTIDIMENSIONAL GENERALIZED LOMB-SCARGLE) PERIODOGRAM FOR RV DATA         ', 7, 94)
    print_message('                                                                                            ', 7, 96)
 
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
        #1/(sigma**2 + jitter**2)
        inv_sigma2 = 1.0/(Globals.rv_errs[i]**2.0 + jitters[i]**2.0)
        #add logL for dataset (i)
        logL += -0.5*(np.sum(((Globals.rvs[i] - model)**2.0)*inv_sigma2 + np.log(2.0*np.pi) - np.log(inv_sigma2))) 

    #spectral power   
    pwr = (Globals.logL_0 - logL) / Globals.logL_0
 
    return pwr, fitting_coeffs, A_matrix, logL

def fmgls_multiset(freqs):
    """
    """
    return -mgls_multiset(freqs)[0]
           
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
            Globals.dataset_names.append( str(s) + '/ ' + arg.split('/')[-1]) #file name. Not full path
            Globals.times.append(time)
            mean_rv = np.mean(rv)
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
        TABLE_DATA.append(['Data set name (full path not shown)', 'Data points', 'Timespan', 'Mean sep.', 'logL (data)'])
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

def gen_random_model(dimensionality):
    """generate random multifrequecy circular model using the data base time
    """
    aS, bS, periods, c = [], [], [], random.uniform(0.0,5.0)
   
    for j in range(dimensionality):
        P = random.uniform(2.0, 120.0)
        K = random.uniform(0.25, 2.0)
        a = random.uniform(-3.0, 3.0)
        if K**2 > a**2:
            b = sqrt(K**2 - a**2)
        else:
            a = 0.0
            b = sqrt(K**2 - a**2)
        aS.append(a)
        bS.append(b)
        periods.append(P)
        
    model = []
    for i in range(len(Globals.times_seq)):
        sigma = random.uniform(2.5, 3.5)
        noise = random.gauss(0.0,sigma)
        if len(periods) > 0:
            for j in range(len(periods)):
                y = c + aS[j]*cos((2.0*pi/periods[j])*Globals.times_seq[i]) + \
                    bS[j]*sin((2.0*pi/periods[j])*Globals.times_seq[i]) + noise  
        else:
            y = c + noise     
        #model time point
        model.append([Globals.times_seq[i], y, sigma])
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
                data_file.append( [Globals.times[s][j], Globals.rvs[s][j]-offset, Globals.rv_errs[s][j]**2] )
        mgls_io.write_file('offset_' + str(s) + '.dat', data_file, ' ', '')
        
        t_0, t_f = Globals.times[s][0], Globals.times[s][-1]
        for i in range(3*int(t_f-t_0)):
            t = t_0 + i*(t_f-t_0)/(3*int(t_f-t_0))
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

    return model_s

  
  