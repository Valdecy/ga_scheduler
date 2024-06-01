############################################################################

# Created by: Prof. Valdecy Pereira, D.Sc.
# UFF - Universidade Federal Fluminense (Brazil)
# email:  valdecy.pereira@gmail.com
# Lesson: pyCombinatorial - Genetic Algorithm
  
# GitHub Repository: <https://github.com/Valdecy>

############################################################################

# Required Libraries
import copy
import numpy  as np
import random
import os

############################################################################

# Function
def target_function():
    return

############################################################################

# Function: 2_opt
def local_search_2_opt(city_tour, recursive_seeding, target_function = target_function):
    if (recursive_seeding < 0):
        count = -2
    else:
        count = 0
    city_list = copy.deepcopy(city_tour)
    distance  = city_list[1]*2
    while (count < recursive_seeding):
        best_route = copy.deepcopy(city_list)
        seed       = copy.deepcopy(city_list)        
        for i in range(0, len(city_list[0]) - 1):
            for j in range(i+1, len(city_list[0])):
                best_route[0][i:j+1] = list(reversed(best_route[0][i:j+1]))               
                best_route[1]        = target_function(best_route[0])                    
                if (city_list[1] > best_route[1]):
                    city_list = copy.deepcopy(best_route)         
                best_route = copy.deepcopy(seed)
        count = count + 1
        if (distance > city_list[1] and recursive_seeding < 0):
             distance          = city_list[1]
             count             = -2
             recursive_seeding = -1
        elif(city_list[1] >= distance and recursive_seeding < 0):
            count             = -1
            recursive_seeding = -2
    return city_list

############################################################################

# Function: Initial Seed
def seed_function(jobs, target_function = target_function):
    seed     = [[],float("inf")]
    sequence = random.sample(list(range(0, jobs)), jobs)
    seed[0]  = sequence
    seed[1]  = target_function(seed[0])
    return seed

# Function: Initial Population
def initial_population(population_size, jobs, target_function = target_function):
    population = []
    for i in range(0, population_size):
        seed = seed_function(jobs, target_function)
        population.append(seed)
    return population

############################################################################

# Function: Fitness
def fitness_function(cost, population_size): 
    fitness = np.zeros((population_size, 2))
    for i in range(0, fitness.shape[0]):
        fitness[i,0] = 1/(1 + cost[i] + abs(np.min(cost)))
    fit_sum      = fitness[:,0].sum()
    fitness[0,1] = fitness[0,0]
    for i in range(1, fitness.shape[0]):
        fitness[i,1] = (fitness[i,0] + fitness[i-1,1])
    for i in range(0, fitness.shape[0]):
        fitness[i,1] = fitness[i,1]/fit_sum
    return fitness

# Function: Selection
def roulette_wheel(fitness): 
    ix     = 0
    random = int.from_bytes(os.urandom(8), byteorder = 'big') / ((1 << 64) - 1)
    for i in range(0, fitness.shape[0]):
        if (random <= fitness[i, 1]):
          ix = i
          break
    return ix

# Function: TSP Crossover - BCR (Best Cost Route Crossover)
def crossover_tsp_bcr(parent_1, parent_2, target_function = target_function):
    individual = copy.deepcopy(parent_2)
    cut        = random.sample(list(range(0, len(parent_1[0]))), int(len(parent_1)/2))
    cut        = [ parent_1[0][i] for i in cut ]
    d_1        = float('+inf')
    for j in range(0, len(cut)):
        best      = []
        A         = cut[j]
        parent_2[0].remove(A)
        dist_list = []
        for i in range(0, len(parent_2) + 1):
            new_list = parent_2[0][:i] + [A] + parent_2[0][i:]
            dist_list.append(new_list)
        fun_list  = [target_function( item ) for item in dist_list]
        d_2       = min(fun_list)
        if (d_2 <= d_1):
            d_1        = d_2
            n          = fun_list.index(d_1)
            best       = dist_list[n]
            individual = [best, d_1]
    return individual

# Function: Breeding
def breeding(population, fitness, elite, target_function = target_function):
    cost = [item[1] for item in population]
    if (elite > 0):
        cost, offspring = (list(t) for t in zip(*sorted(zip(cost, population))))
    for i in range (elite, len(offspring)):
        parent_1, parent_2 = roulette_wheel(fitness), roulette_wheel(fitness)
        while parent_1 == parent_2:
            parent_2 = random.sample(range(0, len(population) - 1), 1)[0]
        parent_1 = copy.deepcopy(population[parent_1])  
        parent_2 = copy.deepcopy(population[parent_2])
        rand     = int.from_bytes(os.urandom(8), byteorder = 'big') / ((1 << 64) - 1)  
        if (rand > 0.5):
            offspring[i] = crossover_tsp_bcr(parent_1, parent_2, target_function)
        else:
            offspring[i] = crossover_tsp_bcr(parent_2, parent_1, target_function)
    return offspring

# Function: Mutation - Swap with 2-opt Local Search
def mutation_tsp_swap(individual, target_function = target_function):
    k                 = random.sample(list(range(0, len(individual[0]))), 2)
    k1                = k[0]
    k2                = k[1]  
    A                 = individual[0][k1]
    B                 = individual[0][k2]
    individual[0][k1] = B
    individual[0][k2] = A
    individual[1]     = target_function(individual[0])
    individual        = local_search_2_opt(individual, -1, target_function)
    return individual

# Function: Mutation
def mutation(offspring, mutation_rate, elite, target_function):
    for i in range(elite, len(offspring)):
        probability = int.from_bytes(os.urandom(8), byteorder = 'big') / ((1 << 64) - 1)
        if (probability <= mutation_rate):
            offspring[i] = mutation_tsp_swap(offspring[i], target_function)
    return offspring

############################################################################

# Function: GA
def genetic_algorithm(jobs, population_size, elite, mutation_rate, generations, target_function = target_function, verbose = True):
    population       = initial_population(population_size, jobs, target_function)
    cost             = [item[1] for item in population]
    cost, population = (list(t) for t in zip(*sorted(zip(cost, population))))
    elite_ind        = population[0] 
    fitness          = fitness_function(cost, population_size)
    count            = 0
    while (count <= generations): 
        if (verbose == True):
            print('Generation: ', count)
        offspring        = breeding(population, fitness, elite, target_function)  
        offspring        = mutation(offspring, mutation_rate, elite, target_function)
        cost             = [item[1] for item in offspring]
        cost, population = (list(t) for t in zip(*sorted(zip(cost, offspring ))))
        elite_child      = population[0]
        fitness          = fitness_function(cost, population_size)
        if(elite_ind[1] > elite_child[1]):
            elite_ind = elite_child 
        count = count + 1  
    route, distance = elite_ind
    return route, distance

############################################################################