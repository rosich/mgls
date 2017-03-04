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
import mgls_io
from mgls_lib import *

#definitions and constants
to_radians = pi/180.0
to_deg = 1.0/to_radians
#-------------------------


def kirkpatrick_cooling(start_temp, alpha):
   T = start_temp
   while True:
     yield T
     T = alpha*T

def state_good(m_state):
#check if new vector accomplishes the required distance between components    
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
        
def gen_modified_state(old_state):
    """comb = [[mean_1, std_1],[mean_2, std_2], ...] 
    """
    f_limits = Globals.freq_range[:]
    m_state = copy.deepcopy(old_state)
    #select the param to be modified
    param_to_perturb = int(len(m_state)*random.uniform(0,1))

    if Globals.comb != []:
        m_state[param_to_perturb] = 1./random.gauss(Globals.comb[param_to_perturb], 0.25)
        if param_to_perturb == len(m_state)-1:
            m_state[param_to_perturb] = random.uniform(f_limits[0], f_limits[1])
    else:
        while True:
            #m_state[param_to_perturb] = random.uniform(f_limits[0], f_limits[1])
            m_state[param_to_perturb] = 1./random.gauss(m_state[param_to_perturb], 0.15)
            if state_good(m_state): return m_state

    return m_state

def compute_metropolis(state, beta, MAX_ITERATIONS, msgs):
    """
    """
    #test initial config
    s_pwr = -mgls(state)[0]
    s_state = state[:]
    
    iteration = 0
    while (iteration < MAX_ITERATIONS):
        #perturb the state
        p_state = gen_modified_state(s_state)
        #run MGLS
        p_pwr = -mgls(p_state)[0]
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
    Globals.n_dim = n_dim
    if msgs == False:
        Globals.inhibit_msg = True
    print_message("\tMontecarlo simulated annealing multidimensional khi-2 minimization", index=3, color=43)
    print_message("\tDimensions: " + str(n_dim) + ' ', index=3, color=42)
    #set a temperature cooling schedule. (initial T, alpha)
    temperatures = kirkpatrick_cooling(10000, 0.25)
    #build initial random state
    f_min, f_max = Globals.freq_range[:]
    s_state_init = [random.uniform(2*f_min, f_max/2) for iter in range(n_dim)]
    while not state_good(s_state_init):
        s_state_init = [random.uniform(f_min, f_max) for iter in range(n_dim)]
        
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
        f_tuple.append(optimal(Globals.ndim, msgs=False, temp_steps=20, n_iter=int(15*Globals.ndim**2.3)))

    return f_tuple

def parallel_optimization(ncpus):
    """parallelized optimization of frequency tuple.
       returns: [f_1,f_2, ...., f_n] 
    """
    if Globals.cross_validation: Globals.inhibit_msg = True
    #MC-SA approach
    #find a set of optimal candidates
    N_CANDIDATES = 64
    if N_CANDIDATES < ncpus: N_CANDIDATES = ncpus
    instances = int(N_CANDIDATES / ncpus)
    frequencies_optimal = []
    Globals.init_step = 5
    #parallel pool
    pool = mp.Pool(ncpus)
    print_message('Computing candidate frequency tuples...', 3, 94)
    instances = [[instance for instance in range(instances)] for iter in range(ncpus)] 
  
    try:
        parallel_opt_states = pool.map_async(parallel_optimal, instances).get(9999999)
    except KeyboardInterrupt:
        print ''
        print_message('Parallel pool killed. Bye!', index=3, color=31)
        pool.terminate()
        sys.exit()
      
    for instance in range(len(parallel_opt_states)):
        frequencies_optimal.extend(parallel_opt_states[instance])
    print_message('Maximizing the candidates...', 3, 94)
    #maximize the candidates and return the best config.
    max_pow = 0.0
    #print "Instances created:", len(frequencies_optimal)
    for j in range(len(frequencies_optimal)):
        res = scipy.optimize.minimize(fmgls, frequencies_optimal[j], method='L-BFGS-B', options={'disp': False})
        if -res.fun > max_pow:
            max_pow = -res.fun
            opt_state = res.x
      
    return opt_state











