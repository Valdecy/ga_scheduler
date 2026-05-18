############################################################################
# GA Scheduler - corrected permutation genetic algorithm utilities
############################################################################

import copy
import os
import random
import numpy as np


def target_function(_sequence=None):
    return 0.0


def _rand01():
    return int.from_bytes(os.urandom(8), byteorder='big') / ((1 << 64) - 1)


def local_search_2_opt(individual, recursive_seeding=-1, target_function=target_function):
    """Permutation 2-opt local improvement for minimization."""
    best = copy.deepcopy(individual)
    best[1] = target_function(best[0])
    improved = True
    passes = 0
    max_passes = float('inf') if recursive_seeding < 0 else recursive_seeding
    while improved and passes < max_passes:
        improved = False
        passes += 1
        route = best[0]
        for i in range(len(route) - 1):
            for j in range(i + 1, len(route)):
                candidate_route = route[:i] + list(reversed(route[i:j + 1])) + route[j + 1:]
                candidate_cost = target_function(candidate_route)
                if candidate_cost < best[1]:
                    best = [candidate_route, candidate_cost]
                    route = best[0]
                    improved = True
    return best


def seed_function(jobs, target_function=target_function):
    sequence = random.sample(list(range(jobs)), jobs)
    return [sequence, target_function(sequence)]


def initial_population(population_size, jobs, target_function=target_function):
    population_size = max(2, int(population_size))
    return [seed_function(jobs, target_function) for _ in range(population_size)]


def fitness_function(cost, population_size=None):
    """Roulette fitness for minimization, robust to negative objective values."""
    cost = np.asarray(cost, dtype=float)
    if population_size is None:
        population_size = len(cost)
    shifted = cost - np.min(cost)
    scores = 1.0 / (1.0 + shifted)
    total = scores.sum()
    if total <= 0 or not np.isfinite(total):
        scores = np.ones_like(scores) / len(scores)
    else:
        scores = scores / total
    cumulative = np.cumsum(scores)
    cumulative[-1] = 1.0
    fitness = np.zeros((population_size, 2), dtype=float)
    fitness[:, 0] = scores
    fitness[:, 1] = cumulative
    return fitness


def roulette_wheel(fitness):
    r = _rand01()
    return int(np.searchsorted(fitness[:, 1], r, side='left'))


def _validate_permutation(route, n):
    return len(route) == n and set(route) == set(range(n))


def repair_permutation(route, n):
    """Repair a route into a valid permutation of 0..n-1."""
    route = list(route)
    missing = [gene for gene in range(n) if gene not in route]
    seen = set()
    repaired = []
    for gene in route:
        if isinstance(gene, int) and 0 <= gene < n and gene not in seen:
            repaired.append(gene)
            seen.add(gene)
        elif missing:
            repaired.append(missing.pop(0))
    repaired.extend(missing)
    return repaired[:n]


def order_crossover(parent_1, parent_2, target_function=target_function):
    """Order crossover (OX) for complete job permutations."""
    p1 = list(parent_1[0])
    p2 = list(parent_2[0])
    n = len(p1)
    if n <= 1:
        child = p1[:]
        return [child, target_function(child)]
    a, b = sorted(random.sample(range(n), 2))
    child = [None] * n
    child[a:b + 1] = p1[a:b + 1]
    fill = [gene for gene in p2 if gene not in child]
    k = 0
    for idx in list(range(b + 1, n)) + list(range(0, b + 1)):
        if child[idx] is None:
            child[idx] = fill[k]
            k += 1
    if not _validate_permutation(child, n):
        child = repair_permutation(child, n)
    return [child, target_function(child)]


def pmx_crossover(parent_1, parent_2, target_function=target_function):
    """Partially mapped crossover (PMX) for complete job permutations."""
    p1 = list(parent_1[0])
    p2 = list(parent_2[0])
    n = len(p1)
    if n <= 1:
        child = p1[:]
        return [child, target_function(child)]
    a, b = sorted(random.sample(range(n), 2))
    child = [None] * n
    child[a:b + 1] = p1[a:b + 1]
    for i in range(a, b + 1):
        gene = p2[i]
        if gene in child:
            continue
        pos = i
        while True:
            mapped_gene = p1[pos]
            pos = p2.index(mapped_gene)
            if child[pos] is None:
                child[pos] = gene
                break
    for i in range(n):
        if child[i] is None:
            child[i] = p2[i]
    if not _validate_permutation(child, n):
        child = repair_permutation(child, n)
    return [child, target_function(child)]


def crossover_tsp_bcr(parent_1, parent_2, target_function=target_function):
    """
    Backward-compatible crossover function name.

    The previous BCR-style implementation evaluated incomplete routes while
    reinserting a cut set of jobs. Scheduling decoders require complete
    chromosomes, so for n >= 4 that design can crash. This replacement uses
    only complete-permutation crossovers; target_function is called only after
    a full child has been built.
    """
    if _rand01() < 0.5:
        return order_crossover(parent_1, parent_2, target_function)
    return pmx_crossover(parent_1, parent_2, target_function)


def breeding(population, fitness, elite, target_function=target_function):
    elite = max(0, min(int(elite), len(population)))
    sorted_population = sorted(population, key=lambda item: item[1])
    offspring = [copy.deepcopy(ind) for ind in sorted_population]

    for i in range(elite, len(offspring)):
        parent_1_idx = roulette_wheel(fitness)
        parent_2_idx = roulette_wheel(fitness)
        while parent_1_idx == parent_2_idx and len(population) > 1:
            parent_2_idx = random.randrange(len(population))
        parent_1 = copy.deepcopy(population[parent_1_idx])
        parent_2 = copy.deepcopy(population[parent_2_idx])
        offspring[i] = crossover_tsp_bcr(parent_1, parent_2, target_function) if _rand01() > 0.5 else crossover_tsp_bcr(parent_2, parent_1, target_function)
    return offspring


def mutation_tsp_swap(individual, target_function=target_function, use_local_search=True):
    individual = copy.deepcopy(individual)
    if len(individual[0]) < 2:
        individual[1] = target_function(individual[0])
        return individual
    k1, k2 = random.sample(range(len(individual[0])), 2)
    individual[0][k1], individual[0][k2] = individual[0][k2], individual[0][k1]
    individual[1] = target_function(individual[0])
    if use_local_search:
        individual = local_search_2_opt(individual, -1, target_function)
    return individual


def mutation(offspring, mutation_rate, elite, target_function=target_function, use_local_search=True):
    elite = max(0, min(int(elite), len(offspring)))
    for i in range(elite, len(offspring)):
        if _rand01() <= mutation_rate:
            offspring[i] = mutation_tsp_swap(offspring[i], target_function, use_local_search)
    return offspring


def genetic_algorithm(jobs, population_size, elite, mutation_rate, generations, target_function=target_function, verbose=True, use_local_search=True):
    if jobs <= 0:
        raise ValueError('jobs must be a positive integer.')
    population = initial_population(population_size, jobs, target_function)
    population = sorted(population, key=lambda item: item[1])
    best = copy.deepcopy(population[0])

    for generation in range(int(generations) + 1):
        if verbose:
            print('Generation:', generation)
        cost = [item[1] for item in population]
        fitness = fitness_function(cost, len(population))
        offspring = breeding(population, fitness, elite, target_function)
        offspring = mutation(offspring, mutation_rate, elite, target_function, use_local_search)
        population = sorted(offspring, key=lambda item: item[1])
        if population[0][1] < best[1]:
            best = copy.deepcopy(population[0])
    return best[0], best[1]

############################################################################
# Multiset / operation-based JSSP genetic algorithm utilities
############################################################################

from collections import Counter


def _normalise_multiset_base(base_sequence):
    base = list(base_sequence)
    if not base:
        raise ValueError('base_sequence must be non-empty for multiset GA.')
    return base, Counter(base)


def _repair_multiset(sequence, required_counts):
    """Repair a candidate so it has exactly the required multiset counts."""
    seq = list(sequence)
    current = Counter(seq)
    surplus_positions = []
    for i, gene in enumerate(seq):
        if current[gene] > required_counts.get(gene, 0):
            surplus_positions.append(i)
            current[gene] -= 1
    missing = []
    for gene, required in required_counts.items():
        missing.extend([gene] * max(0, required - current.get(gene, 0)))
    random.shuffle(missing)
    for pos, gene in zip(surplus_positions, missing):
        seq[pos] = gene
    return seq


def seed_function_multiset(base_sequence, target_function=target_function):
    base, required = _normalise_multiset_base(base_sequence)
    sequence = random.sample(base, len(base))
    sequence = _repair_multiset(sequence, required)
    return [sequence, target_function(sequence)]


def initial_population_multiset(population_size, base_sequence, target_function=target_function):
    population_size = max(2, int(population_size))
    return [seed_function_multiset(base_sequence, target_function) for _ in range(population_size)]


def ppx_crossover_multiset(parent_1, parent_2, required_counts, target_function=target_function):
    """Precedence-preserving crossover for chromosomes with repeated job IDs."""
    p1 = list(parent_1[0])
    p2 = list(parent_2[0])
    used = Counter()
    child = []
    idx1 = idx2 = 0
    n = sum(required_counts.values())

    while len(child) < n:
        prefer_first = random.random() < 0.5
        parents = ((p1, 'p1'), (p2, 'p2')) if prefer_first else ((p2, 'p2'), (p1, 'p1'))
        chosen = None
        for parent, name in parents:
            idx = idx1 if name == 'p1' else idx2
            while idx < len(parent) and used[parent[idx]] >= required_counts[parent[idx]]:
                idx += 1
            if name == 'p1':
                idx1 = idx
            else:
                idx2 = idx
            if idx < len(parent):
                chosen = parent[idx]
                if name == 'p1':
                    idx1 += 1
                else:
                    idx2 += 1
                break
        if chosen is None:
            remaining = [gene for gene, required in required_counts.items() for _ in range(required - used[gene])]
            chosen = random.choice(remaining)
        child.append(chosen)
        used[chosen] += 1

    child = _repair_multiset(child, required_counts)
    return [child, target_function(child)]


def mutate_multiset(individual, mutation_rate, target_function=target_function):
    individual = copy.deepcopy(individual)
    seq = individual[0]
    if len(seq) < 2 or _rand01() > mutation_rate:
        individual[1] = target_function(seq)
        return individual
    operator = random.choice(('swap', 'insert', 'invert'))
    i, j = sorted(random.sample(range(len(seq)), 2))
    if operator == 'swap':
        seq[i], seq[j] = seq[j], seq[i]
    elif operator == 'insert':
        value = seq.pop(j)
        seq.insert(i, value)
    else:
        seq[i:j + 1] = list(reversed(seq[i:j + 1]))
    individual[1] = target_function(seq)
    return individual


def breeding_multiset(population, fitness, elite, required_counts, target_function=target_function):
    elite = max(0, min(int(elite), len(population)))
    sorted_population = sorted(population, key=lambda item: item[1])
    offspring = [copy.deepcopy(ind) for ind in sorted_population]
    for i in range(elite, len(offspring)):
        p1_idx = roulette_wheel(fitness)
        p2_idx = roulette_wheel(fitness)
        while p1_idx == p2_idx and len(population) > 1:
            p2_idx = random.randrange(len(population))
        if _rand01() > 0.5:
            offspring[i] = ppx_crossover_multiset(population[p1_idx], population[p2_idx], required_counts, target_function)
        else:
            offspring[i] = ppx_crossover_multiset(population[p2_idx], population[p1_idx], required_counts, target_function)
    return offspring


def genetic_algorithm_multiset(base_sequence, population_size, elite, mutation_rate, generations, target_function=target_function, verbose=True):
    base, required_counts = _normalise_multiset_base(base_sequence)
    population = initial_population_multiset(population_size, base, target_function)
    population = sorted(population, key=lambda item: item[1])
    best = copy.deepcopy(population[0])
    for generation in range(int(generations) + 1):
        if verbose:
            print('Generation:', generation)
        cost = [item[1] for item in population]
        fitness = fitness_function(cost, len(population))
        offspring = breeding_multiset(population, fitness, elite, required_counts, target_function)
        for i in range(max(0, min(int(elite), len(offspring))), len(offspring)):
            offspring[i] = mutate_multiset(offspring[i], mutation_rate, target_function)
            offspring[i][0] = _repair_multiset(offspring[i][0], required_counts)
            offspring[i][1] = target_function(offspring[i][0])
        population = sorted(offspring, key=lambda item: item[1])
        if population[0][1] < best[1]:
            best = copy.deepcopy(population[0])
    return best[0], best[1]
