#!/usr/bin/python
from math import sin, cos, tan, atan, pi, acos, sqrt, exp, log10
import sys, os
import copy
import random
import numpy as np
import multiprocessing as mp
import ConfigParser
import scipy.optimize
sys.path.append('./bin')
import mGLS, mMGLS
sys.path.append('./src')
from EnvGlobals import Globals
from mgls_lib import *
import mgls_lib
import mgls_io
import time
#from scipy.optimize import dual_annealing, shgo, differential_evolution, basinhopping
#from numba import jit

#definitions and constants
to_radians = pi/180.0
to_deg = 1.0/to_radians
#-------------------------

def print_message(str_, index, color):
    """
    """
    #inhibit_msg = False
    if not Globals.inhibit_msg:
        str_out = "\33["+str(index)+";"+str(color)+"m"+ str_ +"\33[1;m"
        print str_out

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
        
        res = scipy.optimize.minimize(mgls_lib.fmgls_multiset, s_state_init, 
                                      method='SLSQP', tol=1e-12, options={'disp': False})
        
        opt_jitters = abs(res.x[:]) #ndim=0
        pwr, fitting_coeffs, A, logL = mgls_lib.mgls_multiset(res.x)
        Globals.logL_0 = logL
        #reestablish dimensionality
        Globals.ndim = ndim
        Globals.inhibit_msg = False
        Globals.opt_jitters_0 = opt_jitters

    return Globals.logL_0

def _gls_instance_bootstrapping(n):
    """
    """
    #l = mp.Lock()
    max_pows = []
    for _j in range(n):
        rv_0_seq, rv_err_0_seq = Globals.rvs_seq, Globals.rv_errs_seq
        rv_0, rv_err_0 = Globals.rvs, Globals.rv_errs
        t_0 = time.time()
        #shuffle data with thir error
        rv_sets = []
        rv_err_sets = []
        rv_sh = []
        rv_err_sh = []
        for ii in range(Globals.n_sets):
            rv_sets_s, rv_err_sets_s = shuffle_multiset_data(Globals.rvs[ii], Globals.rv_errs[ii])
            rv_sets.append(rv_sets_s)
            rv_err_sets.append(rv_err_sets_s)
            rv_sh.extend(rv_sets_s)
            rv_err_sh.extend(rv_err_sets_s)
        
        Globals.rvs, Globals.rv_errs = rv_sets[:], rv_err_sets[:] 
            
        Globals.rvs_seq, Globals.rv_errs_seq = rv_sh[:], rv_err_sh[:]

        #compute new jitters
        max_logL_0 = -np.inf
        for _ in range(2):
            Globals.logL_0 = logL_NullModel()
            if Globals.logL_0 > max_logL_0:
                max_logL_0 = Globals.logL_0
      
        Globals.logL_0 = max_logL_0
        #compute GLS
        Globals.ndim = 1
        #freqs, pwr, max_pow = gls_1D()
        #if max_pow != 0:
        #   max_pows.append(max_pow)
        Globals.inhibit_msg = True
        
        max_logL = -np.inf
        for _i in range(6):
            #MC optimization
            opt_state = run_MGLS(Globals.ncpus, N_CANDIDATES=4)
            #compute coefficients and A matrix, given the optimal configuration        
            pwr, fitting_coeffs, A, logL = mgls_multiset(opt_state)
            if logL > max_logL:
                max_logL = logL
        
        if (max_logL - Globals.logL_0) > 15.0:
            print "logL > 15 found"
            data_ = zip(Globals.times_seq, Globals.rvs_seq, Globals.rv_errs_seq)
            mgls_io.write_file('_' + str(max_logL - max_logL_0) + '.dat', data_, ' ' , '')
           
        if max_logL - Globals.logL_0 > 0.0:
            max_pows.append(max_logL - Globals.logL_0)
            
        q = (time.time() - t_0)
        chain = "\tIteration: " + str(_j) + "/" + str(n)
        sys.stdout.write('\r' + chain)
        sys.stdout.flush()
        
        #restablish the original timeseries
        Globals.rvs_seq, Globals.rv_errs_seq = rv_0_seq[:], rv_err_0_seq[:]
        Globals.rvs, Globals.rv_errs = rv_0[:], rv_err_0[:]
        
    return max_pows

def shuffle_multiset_data(rv, rv_err):
    """
    """
    #shuffle RV's and their errors
    comb_rv_err = zip(rv, rv_err)
    random.shuffle(comb_rv_err)
    comb_rv_err_ = [random.choice(comb_rv_err) for _ in range(len(comb_rv_err))]
    rv_sh, rv_err_sh = zip(*comb_rv_err_)
    
    return np.array(rv_sh), np.array(rv_err_sh)

def bootstrapping_1D(max_pow):
    """
    """
    #iterations
    n_bootstrapping = Globals.n_bootstrapping
    print "\n//BOOTSTRAPPING://"
    #bootstrapping_stats = parallel_bootstrapping(n_bootstrapping)
    bootstrapping_stats = _gls_instance_bootstrapping(n_bootstrapping)
    #fap_thresholds = fap_levels(bootstrapping_stats)
    #print "FAP Levels:", fap_thresholds
    #print "Total bootstapping samples: ", len(bootstrapping_stats)

    return bootstrapping_stats#, fap_thresholds

def kirkpatrick_cooling(start_temp, alpha):
    """
    """
    T = start_temp
    while True:
        yield T
        T = alpha*T
     
def state_good_(m_state):
    """
    """
    return state_good_mod(m_state, Globals.ndim, Globals.period_range)
    
def state_good_mod(m_state, ndim, period_range):
    """check if new vector accomplishes the required distance between components   
    """
    for i in range(ndim-1):
        for j in range(i+1,ndim):
            #if abs(1./m_state[i] - 1./m_state[j]) < min_dist:
            min_dist = 0.2*m_state[i]
            if abs(m_state[i] - m_state[j]) < min_dist:
                return False
    #check limits
    for i in range(ndim):
        if 1./m_state[i] < period_range[0] or 1./m_state[i] > period_range[1]:
            return False
    return True
    
def gen_modified_state_(old_state):
    """comb = [[mean_1, std_1],[mean_2, std_2], ...] 
    """
    #print "entrada", 1./np.array(old_state)
    counter = 0
    f_limits = Globals.freq_range[:]
    m_state = copy.deepcopy(old_state)
    #select the param to be modified
    if np.random.uniform(0,1) < 0.1:
        param_to_perturb = int(Globals.ndim*np.random.uniform(0,1))
        while True:
            m_state[param_to_perturb] = np.random.uniform(Globals.freq_range[0], Globals.freq_range[1]) 
            counter += 1
            if state_good_(m_state):   
                return m_state
            elif counter == 100:
                return old_state
    else:
        param_to_perturb = int(len(m_state)*np.random.uniform(0,1))
        if param_to_perturb < Globals.ndim:
            while True:
                m_state[param_to_perturb] = np.random.uniform(Globals.freq_range[0], Globals.freq_range[1])  
                counter += 1
                if state_good_(m_state):   
                    return m_state
                elif counter == 100:
                    return old_state
            
        else:
            m_state[param_to_perturb] = np.random.uniform(0.0,Globals.jitter_limit)
            
    return m_state

def compute_metropolis(state, beta, MAX_ITERATIONS, msgs):
    """
    """
    #test initial config
    s_pwr = fmgls_multiset(state)
    s_state = state[:]

    iteration = 0
    while (iteration < MAX_ITERATIONS):
        #perturb the state
        p_state = gen_modified_state_(s_state)
        #run MGLS
        p_pwr = fmgls_multiset(p_state)
      
        delta_pwr = p_pwr - s_pwr
        
        if delta_pwr < 0.0:
            s_state = p_state[:]
            s_pwr = p_pwr
            #os.system('cls' if os.name == 'nt' else 'clear')
            #print_message("Accepted: %f \r" % (-s_pwr), index=3, color=32)
            if msgs:
                sys.stdout.write("\t\t|=====> Accepted PWR: %f \r" % (-s_pwr))
                sys.stdout.flush()
        #otherwise do a random decision
        elif np.random.uniform(0,1) < exp(-beta*delta_pwr):
            #print "Entered with T=", 1.0/beta
            s_state = p_state[:]
            s_pwr = p_pwr
        iteration += 1

    return s_state

def optimal(n_dim, msgs, temp_steps, n_iter):
    """find [f_1,f_2, ..., f_n] tuple that maximizes spectral power
    """
    Globals.ndim = n_dim
    
    if msgs == False:
        Globals.inhibit_msg = True
    print_message("\tSimulated annealing multidimensional logL maximization", index=3, color=43)
    print_message("\tDimensions: " + str(n_dim) + ' ', index=3, color=42)
    #set a temperature cooling schedule. (initial T, alpha)
    temperatures = kirkpatrick_cooling(5000, 0.1)
    #build initial random state
    
    f_min, f_max = Globals.freq_range[:]
    s_state_init = [np.random.uniform(Globals.freq_range[0], Globals.freq_range[1]) for iter in range(Globals.ndim)]
    
    if Globals.jitter:
        if Globals.opt_jitters_0 != []:
            s_state_init.extend(Globals.opt_jitters_0)
        else:
            s_state_init.extend(np.random.uniform(0.0, Globals.jitter_limit) for iter in range(Globals.n_sets))
    
    while not state_good_(s_state_init):
        s_state_init = [np.random.uniform(Globals.freq_range[0], Globals.freq_range[1]) for iter in range(n_dim)]
        if Globals.jitter:
            if Globals.opt_jitters_0 != []:
                s_state_init.extend(Globals.opt_jitters_0)
            else:
                s_state_init.extend(np.random.uniform(0.0, Globals.jitter_limit) for iter in range(Globals.n_sets))
    
    #avoid intermediate temperatures when starting in intermediate steps (if required)
    #Globals.init_step = 0
    for j in range(Globals.init_step): beta = 1.0/temperatures.next()
    #init temperature sweep
    #temp_steps = 35
    print_message("Temperature steps:" + str(temp_steps), index=2, color=37)
   
    for p_temp in range(Globals.init_step, temp_steps):
        MAX_ITERATIONS = n_iter #int(n_iter*(p_temp+1)**1.25)
        beta = 1.0/temperatures.next()
        #print MAX_ITERATIONS, beta
        if msgs:
            print "\t iter [", p_temp, "]"#, "beta:", round(beta,3),"; Iterations:", MAX_ITERATIONS
        #sys.stdout.write("\tTemp. step: %f \r" % (p_temp))
            sys.stdout.flush()
       
        state = compute_metropolis(s_state_init, beta, MAX_ITERATIONS, msgs)[:]
        
        s_state_init = state[:]
  
    return state

def parallel_optimal(instance):
    """
    """
    f_tuple = []
    for i in range(len(instance)):
        if Globals.ndim > 0:
            f_tuple.append(optimal(Globals.ndim, msgs=False, temp_steps=35, n_iter=int(50*(Globals.ndim**1.35 + Globals.n_sets))))
        else: 
            f_tuple.append(optimal(Globals.ndim, msgs=False, temp_steps=15, n_iter=int(2*(Globals.n_sets))))
            
    return f_tuple

def optimal_scipy(n_dim, msgs, temp_steps, n_iter):
    """
    """
    f_min, f_max = Globals.freq_range[:]
    bounds = [(Globals.freq_range[0], Globals.freq_range[1]) for iter in range(Globals.ndim)]
    
    """
    #init_state
    s_state_init = [random.uniform(Globals.freq_range[0], Globals.freq_range[1]) for iter in range(Globals.ndim)]
    """
    if Globals.jitter:
        bounds.extend( [(0.0, Globals.jitter_limit) for iter in range(Globals.n_sets)])
        """
        if Globals.opt_jitters_0 != []:
            #s_state_init.extend(Globals.opt_jitters_0)
            pass
        else:
            #s_state_init.extend(random.uniform(0.0, Globals.jitter_limit) for iter in range(Globals.n_sets))
            bounds.extend( [(0.0, Globals.jitter_limit) for iter in range(Globals.n_sets)])
        """
    """        
    #ret = dual_annealing(fmgls_multiset, bounds, seed=None, no_local_search=False, x0=s_state_init, maxfun=1e4)
    """
    ret = differential_evolution(fmgls_multiset, bounds=bounds, polish=False)
    
    return ret.x

def parallel_optimization_multiset(ncpus, N_CANDIDATES):
    """parallelized optimization of frequency tuple.
       returns: [f_1,f_2, ...., f_n] 
    """
    ERROR_FLAG = False
    #MC-SA approach
    #find a set of optimal candidates
    if N_CANDIDATES < ncpus: N_CANDIDATES = ncpus
    instances = int(N_CANDIDATES / ncpus)
    frequencies_optimal = []
    Globals.init_step = 5
    #parallel pool
    pool = mp.Pool(ncpus)
    
    #init a new seed for random generator
    np.random.seed()
    
    print_message('Computing candidate frequency tuples...', 3, 94)
    instances = [ [instance for instance in range(instances)] for iter in range(ncpus) ] 
    
    try:
        parallel_opt_states = pool.map_async(parallel_optimal, instances).get(9999999)
    except KeyboardInterrupt:
        print ''
        print_message('Parallel pool killed. Bye!', index=3, color=31)
        pool.terminate()
        sys.exit()
    except:
        print "Unexpected error in pool.map_async(parallel_optimal):", sys.exc_info()[0]
        raise
    
    pool.terminate()
   
    for instance in range(len(parallel_opt_states)):
        frequencies_optimal.extend(parallel_opt_states[instance])
        
    print_message('Maximizing the candidates...', 3, 94)
    #maximize the candidates and return the best config.
    max_pow = np.inf
    #print "Instances created:", len(frequencies_optimal)
    best_candidates = []
 
    for j in range(len(frequencies_optimal)):
        #parameter bounds only applies for algorithms SLSQP, 
        #param_bounds = [(frequencies_optimal[j][i]*0.95, frequencies_optimal[j][i]*1.05) for i in range(Globals.ndim)]
        param_bounds = [(Globals.freq_range[0], Globals.freq_range[1]) for i in range(Globals.ndim)]
        if Globals.jitter:
            param_bounds.extend((0.0, Globals.jitter_limit) for iter in range(Globals.n_sets))
            #param_bounds.extend( [(frequencies_optimal[j][i]*0.95, frequencies_optimal[j][i]*1.05) 
            #                       for i in range(Globals.ndim,Globals.ndim+Globals.n_sets)] )
            
        try:                                    
            #res_0 = dual_annealing(fmgls_multiset, bounds=param_bounds, seed=None, 
            #                       no_local_search=False, x0=frequencies_optimal[j], maxfun=1e3)
            #res = differential_evolution(fmgls_multiset, bounds=param_bounds)
            res = scipy.optimize.minimize(fmgls_multiset, frequencies_optimal[j], 
                                          method='SLSQP', bounds=param_bounds, tol=1e-9, options={'disp': False})
        
            #res = shgo(fmgls_multiset, bounds=param_bounds, sampling_method='sobol')
            #minimizer_kwargs = {"method":"SLSQP", "jac":False}
            #res = basinhopping(fmgls_multiset, x0=frequencies_optimal[j], T=0.5, stepsize=0.5, minimizer_kwargs=minimizer_kwargs, niter=200)
        except:
            raise
            print "SLSQP final optimization has failed in candidate", j
            pass
        
        if state_good_(res.x):
            #compute coefficients and A matrix, given the optimal configuration to check if A matrix is not singular        
            pwr, fitting_coeffs, A, logL = mgls_multiset(res.x)
      
            if res.fun < max_pow:
                max_pow = res.fun
                opt_state = res.x
    
        else:
            ERROR_FLAG = True
            #print "Warning:: Solution not compatible with distance constraints:", 1./res.x
        
    try: 
        opt_state
    except NameError:
        opt_state = []
    
    #sort the list of candidates
    #best_candidates_sorted = sorted(best_candidates, key=lambda row: row[1], reverse=True)
    
    #mgls_io.write_file('mgls_candidates.dat', best_candidates_sorted, ' ', '')
    
    return opt_state

def run_MGLS(ncpus, N_CANDIDATES):
    """call parallel_optimization_multiset
    """
    
    #opt_state = dual_annealing_test()
    
    opt_state = parallel_optimization_multiset(ncpus, N_CANDIDATES)
    if opt_state == []:
        print "Warning: Running again the SA searching algorithm"
        
    else:
        return opt_state
    
    #return opt_state
    
def dual_annealing_test():
    """
    """
    def func_(state):
        """
        """
        return fmgls_multiset(state) 
    
    from scipy.optimize import dual_annealing, shgo
    
    f_min, f_max = Globals.freq_range[:]
    s_state_init = [random.uniform(Globals.freq_range[0], Globals.freq_range[1]) for iter in range(Globals.ndim)]
    
    bounds = [(Globals.freq_range[0], Globals.freq_range[1]) for iter in range(Globals.ndim)]
    
    if Globals.jitter:
        bounds.extend( [(0.0, Globals.jitter_limit) for iter in range(Globals.n_sets)] )
        s_state_init.extend([random.uniform(0.0, Globals.jitter_limit) for iter in range(Globals.n_sets)])

    
    ret = shgo(func_, bounds, sampling_method='sobol')
    """
    if Globals.ndim == 2:
        
        temps = [0.05, 1.0, 10.0, 1000.0, 10000.0, 50000.0]
        restarts = [0.0, 1.e-6, 1.e-3, 1.e-1, 0.25, 0.5, 1.0]
        accepts = [-1.e-4, -1.e-3, -1.e-1, -1., -3., -5. ]
        
        for i in range(len(temps)):
            for j in range(len(restarts)):
                for k in range(len(accepts)):
             
                    initial_temp = temps[i]
                    restart_temp_ratio = restarts[j]
                    accept = accepts[k]
                    
                    f_min, f_max = Globals.freq_range[:]
                    s_state_init = [random.uniform(Globals.freq_range[0], Globals.freq_range[1]) for iter in range(Globals.ndim)]
                    bounds = [(Globals.freq_range[0], Globals.freq_range[1]) for iter in range(Globals.ndim)]
                
                    if Globals.jitter:
                        bounds.extend( [(0.0, Globals.jitter_limit) for iter in range(Globals.n_sets)] )
                        s_state_init.extend([random.uniform(0.0, Globals.jitter_limit) for iter in range(Globals.n_sets)])
                    
                    counter = 0
                    for iter in range(10):
                        ret = dual_annealing(func_, bounds, initial_temp, restart_temp_ratio,
                                             accept, no_local_search=False, seed=None, x0=s_state_init, maxfun=1e4)
                        ret.x[:Globals.ndim].sort()
                        
                        #for i in range(Globals.ndim): 
                        diff = 1./ret.x[:Globals.ndim] - [121.0, 48.6]
                        
                        if sum(diff) < 4: 
                            #print 'OK'
                            counter +=1
                        else:
                            #print 1./ret.x[:Globals.ndim]
                            pass
                        
                    print initial_temp, restart_temp_ratio, accept, counter
    
    else:
        ret = dual_annealing(func_, bounds, seed=None, initial_temp=5500, restart_temp_ratio=1e-6, 
                            accept=-500.0, no_local_search=True, x0=s_state_init, maxfun=1e4)
    """   
    return ret.x

def noise_analysis(init_dim, opt_freqs_base, fitting_coeffs_base, opt_jitters_base):
    """
    """
    Globals.ndim = init_dim
    counter = 0
    histogram = []
    NSAMPLES = 50
    
    rvs_seq, rv_errs_seq = copy.deepcopy(Globals.rvs_seq), copy.deepcopy(Globals.rv_errs_seq)
    rvs_cp, rv_errs_cp = copy.deepcopy(Globals.rvs), copy.deepcopy(Globals.rv_errs)
        
    for iter in range(NSAMPLES):
        try:
            #do n times to compute 1% percentile 
            if not Globals.jitter: 
                jitters = [0.0 for iter in range(Globals.n_sets)]
            
            Globals.ndim = init_dim
            #generate a model with the reulting parameters    
            is_model = gen_synthetic_model(1./opt_freqs_base, fitting_coeffs_base, opt_jitters_base)
            #model_0
            max_logL_0 = -np.inf
            for k in range(4):
                opt_state_0 = parallel_optimization_multiset(Globals.ncpus, N_CANDIDATES=4)
                #compute coefficients and A matrix, given the optimal configuration        
                pwr, fitting_coeffs, A, logL_0 = mgls_multiset(opt_state_0)
          
                if logL_0 > max_logL_0:
                    max_logL_0 = logL_0
                    Globals.logL_0 = max_logL_0
           
            #model 
            max_logL = -np.inf
            Globals.ndim += 1
            for k in range(4):
                opt_state = parallel_optimization_multiset(Globals.ncpus, N_CANDIDATES=4)
                #compute coefficients and A matrix, given the optimal configuration        
                pwr, fitting_coeffs, A, logL = mgls_multiset(opt_state)
            
                if logL > max_logL:
                    max_logL = logL
            #print max_logL_0, max_logL
            
            DLog = max_logL - max_logL_0
            if DLog < 0.0: DLog = 0.0
            
            if (max_logL - Globals.logL_0) > 15.0:
                print "logL > 15 found"
                
                data_ = zip(Globals.times_seq, Globals.rvs_seq, Globals.rv_errs_seq)
                mgls_io.write_file('_' + str(max_logL - Globals.logL_0) + '.dat', data_, ' ' , '')
            
            histogram.append(DLog)
            counter += 1
            
            if counter % 5 == 0:
                sys.stdout.write('\r\t                             ')
                sys.stdout.write( '\r\t' + " >> Completed " + str(round((100.*float(counter)/float(NSAMPLES)),2)) + ' %' )
                sys.stdout.flush()
        
            Globals.rvs_seq, Globals.rv_errs_seq = rvs_seq[:], rv_errs_seq[:]
            Globals.rvs, Globals.rv_errs = rvs_cp[:], rv_errs_cp[:]
        
        except KeyboardInterrupt:
            exit()
            
        except:
            print "Exception occurred."
            pass
    
    #histogram = np.array(histogram)
    sys.stdout.write('\r                              ')
    sys.stdout.write('\n')
    mgls_io.write_file_onecol('H_'+ str(Globals.ndim) + '_' + str(time.time()) + '.dat', histogram, ' ', '')

    print "DlogL max.", max(histogram)
    
    return np.percentile(histogram, 90.0), np.percentile(histogram, 99.0), np.percentile(histogram, 99.9)






