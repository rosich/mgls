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
from mgls_lib import *

#definitions and constants
to_radians = pi/180.0
to_deg = 1.0/to_radians
#-------------------------

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

def gen_individual(dim):
    """
    """
    
    individual = [1./random.uniform(Globals.period_range[0], Globals.period_range[1]) for iter in range(dim)]
    while not state_good(individual):
        individual = [1./random.uniform(Globals.period_range[0], Globals.period_range[1]) for iter in range(dim)]
    
    return individual

def gen_population(n_individuals, dim):
    """
    """
    population = []
    for j in range(n_individuals):
        population.append(gen_individual(dim))

    return population

def selection(population, n_best):
    """
    """
    return sorted(population, key=fmgls, reverse=False)[:n_best]
   
def cross_individuals(ind_1, ind_2):
    """
    """
    crossed_individual_1, crossed_individual_2 = ind_1, ind_2
    #cut point: index from where we cut the individual cromosomes.
    cut_point = int(random.uniform(1,len(ind_1)))
    cut_point_conjugate =  len(ind_1) - cut_point
    #bulid the new (crossed) individuals 
    crossed_individual_1 = ind_1[:cut_point] 
    crossed_individual_1[cut_point:] = ind_2[:cut_point_conjugate]
    crossed_individual_2 = ind_1[cut_point:] 
    crossed_individual_2[cut_point_conjugate:] = ind_2[cut_point_conjugate:]
    
    return [crossed_individual_1, crossed_individual_2] 
   
def crossover(sub_population):
    """improves the population by adding crossed individuals to the existing ones.
    """
    N_sub_population = len(sub_population) 
    crossed_population = sub_population
    for j in range(N_sub_population/2):
        crossed_pair = cross_individuals(sub_population[2*j], sub_population[2*j+1])
        crossed_population.extend(crossed_pair)
    return crossed_population

def mutate(population):
    """create random small perturbations on random chromosomes
    """    
    N_mutations = int(0.05*len(population)*len(population[0]))  # 5% of population*dimensionality length
    if N_mutations < 1: N_mutations = 1
    
    for j in range(N_mutations):
        individual_to_mutate = int(random.uniform(0,len(population)))
        chromosome_to_mutate = int(random.uniform(0,len(population[0])))
        period_0 = 1./population[individual_to_mutate][chromosome_to_mutate]
        population[individual_to_mutate][chromosome_to_mutate] = 1./random.gauss(period_0, 0.1)
        
        while not state_good(population[individual_to_mutate]):
            population[individual_to_mutate][chromosome_to_mutate] = 1./random.gauss(period_0, 0.3)
            
    return population

def optimize_frequency():
    """
    """   
    candidate_tuples = []
    #generate initial population
    pop_init = gen_population(1500, Globals.ndim)
    print_message("Generated initial population...", index=3, color=34)
    pop_selected_init = selection(pop_init, len(pop_init)/2)

    for iter in range(200):
        crossed_population = crossover(pop_selected_init)
        mutated_population = mutate(crossed_population)
        selected = selection(mutated_population, len(mutated_population)/2)
        #reestablish arrays to iterate
        pop_selected_init = selected
    
    for i in range(50):
        candidate_tuples.append(selected[i])
    print_message("Genetic process terminated...", index=3, color=34)    
    
    return candidate_tuples



