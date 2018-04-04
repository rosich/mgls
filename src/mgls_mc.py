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
import mgls_io

#definitions and constants
to_radians = pi/180.0
to_deg = 1.0/to_radians
#-------------------------

def kirkpatrick_cooling(start_temp, alpha):
    """
    """
    T = start_temp
    while True:
        yield T
        T = alpha*T

def state_good_DEPRECATED(m_state):
    """check if new vector accomplishes the required distance between components   
    """
    min_dist = 1.0   #min separation
    for i in range(len(m_state)-1):
        for j in range(i+1,len(m_state)):
            if abs(1./m_state[i] - 1./m_state[j]) < min_dist:
                return False
    #check limits
    for i in range(len(m_state)):
        if 1./m_state[i] < Globals.period_range[0] or 1./m_state[i] > Globals.period_range[1]:
            return False
    return True
        
def gen_modified_state_DEPRECATED(old_state):
    """comb = [[mean_1, std_1],[mean_2, std_2], ...] 
    """
    counter = 0
    f_limits = Globals.freq_range[:]
    m_state = copy.deepcopy(old_state)
    #select the param to be modified
    param_to_perturb = int(len(m_state)*random.uniform(0,1))

    while True:
        m_state[param_to_perturb] = random.uniform(f_limits[0], f_limits[1])
        #m_state[param_to_perturb] = 1./random.gauss(m_state[param_to_perturb], 0.15)
        counter += 1
        if state_good(m_state) or counter == 1000:   
            return m_state
 
    return m_state

def state_good_(m_state):
    """check if new vector accomplishes the required distance between components   
    """
    min_dist = 2.0   #min separation (in period)
    for i in range(Globals.ndim-1):
        for j in range(i+1,Globals.ndim):
            if abs(1./m_state[i] - 1./m_state[j]) < min_dist:
                return False
    #check limits
    for i in range(Globals.ndim):
        if 1./m_state[i] < Globals.period_range[0] or 1./m_state[i] > Globals.period_range[1]:
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
    if random.uniform(0,1) < 0.15:
        param_to_perturb = int(Globals.ndim*random.uniform(0,1))
        while True:
            m_state[param_to_perturb] = random.uniform(Globals.freq_range[0], Globals.freq_range[1]) 
            counter += 1
            if state_good_(m_state):   
                return m_state
            elif counter == 100:
                return old_state
    else:
        param_to_perturb = int(len(m_state)*random.uniform(0,1))
        if param_to_perturb < Globals.ndim:
            while True:
                m_state[param_to_perturb] = random.uniform(Globals.freq_range[0], Globals.freq_range[1])  
                counter += 1
                if state_good_(m_state):   
                    return m_state
                elif counter == 100:
                    return old_state
            
        else:
            m_state[param_to_perturb] = random.uniform(0.0,Globals.jitter_limit)
            
    return m_state

def compute_metropolis(state, beta, MAX_ITERATIONS, msgs):
    """
    """
  
    #test initial config
    s_pwr = -mgls_multiset(state)[0]
    s_state = state[:]

    iteration = 0
    while (iteration < MAX_ITERATIONS):
        #perturb the state
        p_state = gen_modified_state_(s_state)
        #run MGLS
        p_pwr = -mgls_multiset(p_state)[0]
      
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
        elif random.uniform(0,1) < exp(-beta*delta_pwr):
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
    print_message("\tMontecarlo simulated annealing multidimensional khi-2 minimization", index=3, color=43)
    print_message("\tDimensions: " + str(n_dim) + ' ', index=3, color=42)
    #set a temperature cooling schedule. (initial T, alpha)
    temperatures = kirkpatrick_cooling(10000, 0.25)
    #build initial random state
    
    f_min, f_max = Globals.freq_range[:]
    s_state_init = [random.uniform(Globals.freq_range[0], Globals.freq_range[1]) for iter in range(Globals.ndim)]
    
    if Globals.jitter:
        if Globals.opt_jitters_0 != []:
            s_state_init.extend(Globals.opt_jitters_0)
        else:
            s_state_init.extend(random.uniform(0.0,Globals.jitter_limit) for iter in range(Globals.n_sets))
    
    while not state_good_(s_state_init):
        s_state_init = [random.uniform(Globals.freq_range[0], Globals.freq_range[1]) for iter in range(n_dim)]
        if Globals.jitter:
            if Globals.opt_jitters_0 != []:
                s_state_init.extend(Globals.opt_jitters_0)
            else:
                s_state_init.extend(random.uniform(0.0,Globals.jitter_limit) for iter in range(Globals.n_sets))
    
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
            f_tuple.append(optimal(Globals.ndim, msgs=False, temp_steps=20, n_iter=int(30*(Globals.ndim**2.0+Globals.n_sets))))
        else: 
            f_tuple.append(optimal(Globals.ndim, msgs=False, temp_steps=15, n_iter=int(2*(Globals.n_sets))))
            
    return f_tuple

def parallel_optimization_multiset(ncpus, N_CANDIDATES):
    """parallelized optimization of frequency tuple.
       returns: [f_1,f_2, ...., f_n] 
    """ 
    #MC-SA approach
    #find a set of optimal candidates
    if N_CANDIDATES < ncpus: N_CANDIDATES = ncpus
    instances = int(N_CANDIDATES / ncpus)
    frequencies_optimal = []
    Globals.init_step = 5
    #parallel pool
    pool = mp.Pool(ncpus)
    
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
    f_min, f_max = Globals.freq_range[:]
    param_bounds = [ (f_min,f_max) for iter in range(Globals.ndim)]
    if Globals.jitter:
        param_bounds.extend((0.0,Globals.jitter_limit) for iter in range(Globals.n_sets))
 
    for j in range(len(frequencies_optimal)):
        res = scipy.optimize.minimize(fmgls_multiset, frequencies_optimal[j], 
                                      method='SLSQP',bounds=param_bounds, tol=1e-10, options={'disp': False})
 
        best_candidates.append([res.x, res.fun])
        if res.fun < max_pow:
            max_pow = res.fun
            opt_state = res.x
    
    try: 
        opt_state
    except NameError:
        opt_state = []
    
    #sort the list of candidates
    best_candidates_sorted = sorted(best_candidates, key=lambda row: row[1], reverse=True)
    #mgls_io.write_file('mgls_candidates.dat', best_candidates_sorted, ' ', '')
    
    return opt_state








