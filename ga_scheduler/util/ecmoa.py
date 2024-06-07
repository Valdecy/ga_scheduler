############################################################################

# Created by: Prof. Valdecy Pereira, D.Sc.
# UFF - Universidade Federal Fluminense (Brazil)
# email:  valdecy.pereira@gmail.com
# Elitist Combinatorial Multiobjective Optimization Algorithm

# Citation: 
# PEREIRA, V. (2024). Project: GA Scheduler, GitHub repository: <https://github.com/Valdecy/GA_Scheduler>

############################################################################

# Required Libraries
import copy
import numpy  as np
import random
import os

############################################################################

# Function 1
def func_1():
    return

# Function 2
def func_2():
    return

############################################################################

# Function: Initial Seed
def seed_function(jobs, list_of_functions):
    seed     = [[]] + [float("inf") for item in list_of_functions]
    sequence = random.sample((range(0, jobs)), jobs)
    seed[0]  = sequence
    for k in range (1, len(list_of_functions) + 1):
        seed[-k] = list_of_functions[-k](seed[0])
    return seed

# Function: Initial Population
def initial_population(population_size, jobs, list_of_functions):
    population = []
    for i in range(0, population_size):
        seed = seed_function(jobs, list_of_functions)
        population.append(seed)
    return population

############################################################################

# Function: Leaders Selection
def selection_leaders(size, M, population):
    
    ################################################
    def ensure_unique_population(population):
        unique_population = set()
        for item in population:
            unique_population.add(tuple(item[0])) 
        unique_population_list = []
        for unique_item in unique_population:
            for original_item in population:
                if (list(unique_item) == original_item[0]):
                    unique_population_list.append(original_item)
                    break  
        return unique_population_list
    ################################################
    
    position = [[] for item in population]
    for m in range(0, len(population)):
        for n in range(1, len(population[m])):
            position[m].append((population[m][n]))
    position = np.array(position)
    idx      = pareto_front_points(position, pf_min = True)
    if (len(idx) > 0):
        leaders = [population[i] for i in range(0, len(idx)) if idx[i] == True]
    leaders = ensure_unique_population(leaders)
    if (size == 1):
        leaders = random.choice(leaders)
        leaders = [leaders]
    if (len(leaders) > size and size > 1):
        leaders = [leaders[i] for i in range(0, size)]  
    return leaders

# Function: TSP Crossover - BCR (Best Cost Route Crossover)
def crossover_tsp_bcr(parent_1, parent_2, list_of_functions):
    
    ################################################
    def ensure_all_numbers_present(lst, range_start, range_end):
        full_set        = set(list(range(range_start, range_end + 1)))
        lst_set         = set(lst)
        missing_numbers = full_set - lst_set
        lst.extend(missing_numbers)
        return lst
    
    def apply_functions_to_list(dist_list, list_of_functions):
        return [ (item, *[list_of_functions[k](item) for k in range(0, len(list_of_functions))]) for item in dist_list ]
    ################################################
   
    p_1  = copy.deepcopy(parent_1)
    p_2  = copy.deepcopy(parent_2)
    M    = len(list_of_functions)
    jobs = len(parent_1[0])
    cut  = random.sample(list(range(0, len(p_1[0]))), int(len(p_1)/2))
    cut  = [ p_1[0][i] for i in cut ]
    best = []
    for j in range(0, len(cut)):
        A         = cut[j]
        p_2[0].remove(A)
        dist_list = []
        for i in range(0, len(p_2) + 1):
            new_list = p_2[0][:i] + [A] + p_2[0][i:]
            if (len(new_list) < jobs):
                new_list = ensure_all_numbers_present(new_list, 0, jobs-1)
            if (new_list not in dist_list):
                dist_list.append(new_list)
        merged = apply_functions_to_list(dist_list, list_of_functions)
        best.extend(merged)
    best = selection_leaders(1, M, best)
    ind  = []
    for m in range(0, len(best)):
        for n in range(0, len(best[m])):
            ind.append(best[m][n])
    return ind

# Function: Breeding
def breeding(leaders, population, list_of_functions):
    offspring = [[] for item in population]
    parent_1  = 0
    parent_2  = 1
    for i in range(0, len(leaders)):
        offspring[i] = leaders[i]
    for i in range (len(leaders), len(offspring)):
        i1, i2, i3, i4 = random.sample(range(0, len(population) - 1), 4)
        rand  = int.from_bytes(os.urandom(8), byteorder = 'big') / ((1 << 64) - 1)
        if (rand > 0.5):
            parent_1 = i1
        else:
            parent_1 = i2  
        rand  = int.from_bytes(os.urandom(8), byteorder = 'big') / ((1 << 64) - 1)
        if (rand > 0.5):
            parent_2 = i3
        else:
            parent_2 = i4
        parent_1 = copy.deepcopy(population[parent_1])  
        parent_2 = copy.deepcopy(population[parent_2])
        rand     = int.from_bytes(os.urandom(8), byteorder = 'big') / ((1 << 64) - 1)  
        if (rand > 0.5):
            offspring[i] = crossover_tsp_bcr(parent_1, parent_2, list_of_functions)
        else:
            offspring[i] = crossover_tsp_bcr(parent_2, parent_1, list_of_functions)
    return offspring

############################################################################

# Function:  Pareto Front  
def pareto_front_points(pts, pf_min = True):
    
    ################################################
    def pareto_front(pts, pf_min):
        pf = np.zeros(pts.shape[0], dtype = np.bool_)
        for i in range(0, pts.shape[0]):
            cost = pts[i, :]
            if (pf_min == True):
                g_cost = np.logical_not(np.any(pts > cost, axis = 1))
                b_cost = np.any(pts < cost, axis = 1)
            else:
                g_cost = np.logical_not(np.any(pts < cost, axis = 1))
                b_cost = np.any(pts > cost, axis = 1)
            dominated = np.logical_and(g_cost, b_cost)
            if  (np.any(pf) == True):
                if (np.any(np.all(pts[pf] == cost, axis = 1)) == True):
                    continue
            if not (np.any(dominated[:i]) == True or np.any(dominated[i + 1 :]) == True):
                pf[i] = True
        return pf
    ################################################
    
    idx     = np.argsort(((pts - pts.mean(axis = 0))/(pts.std(axis = 0) + 1e-7)).sum(axis = 1))
    pts     = pts[idx]
    pf      = pareto_front(pts, pf_min)
    pf[idx] = pf.copy()
    return pf

############################################################################

# ECMO Function
def elitist_combinatorial_multiobjective_optimization_algorithm(size = 15, jobs = 7, list_of_functions = [func_1, func_2], generations = 1500, k = 4, verbose = True):       
    count      = 0
    size       = max(5, size)
    M          = len(list_of_functions)
    k_size     = k*size
    population = initial_population(k_size, jobs, list_of_functions)  
    leaders    = selection_leaders(size, M, population)
    print(' Population Size: ', int(k_size))
    while (count <= generations):       
        if (verbose == True):
            print('Generation = ', count)
        offspring  = breeding(leaders, population, list_of_functions)
        population = offspring + population
        leaders    = selection_leaders(size, M, population)
        rand       = int.from_bytes(os.urandom(8), byteorder = 'big') / ((1 << 64) - 1)  
        if (rand > 0.95):
            population = initial_population(k_size, jobs, list_of_functions) 
        population = leaders + population
        population = population[:k_size]
        count      = count + 1              
    return leaders

############################################################################
