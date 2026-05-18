############################################################################
# GA Scheduler - NSGA-III utilities for permutation scheduling
############################################################################

import copy
import itertools
import math
import random
import numpy as np


def _evaluate(sequence, list_of_functions):
    sequence = list(sequence)
    return [sequence] + [float(f(sequence)) for f in list_of_functions]


def initial_population_nsga3(population_size, jobs, list_of_functions):
    population_size = max(2, int(population_size))
    jobs = int(jobs)
    population = []
    seen = set()
    while len(population) < population_size:
        seq = tuple(random.sample(range(jobs), jobs))
        if seq not in seen:
            seen.add(seq)
            population.append(_evaluate(seq, list_of_functions))
        if len(seen) >= math.factorial(jobs):
            break
    return population


def ensure_unique_population(population):
    unique = []
    seen = set()
    for ind in population:
        key = tuple(ind[0])
        if key not in seen:
            seen.add(key)
            unique.append(ind)
    return unique


def dominates(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    return np.all(a <= b) and np.any(a < b)


def non_dominated_sort(population):
    objectives = [np.asarray(ind[1:], dtype=float) for ind in population]
    n = len(population)
    dominated_sets = [[] for _ in range(n)]
    domination_count = [0] * n
    fronts = [[]]
    for p in range(n):
        for q in range(n):
            if p == q:
                continue
            if dominates(objectives[p], objectives[q]):
                dominated_sets[p].append(q)
            elif dominates(objectives[q], objectives[p]):
                domination_count[p] += 1
        if domination_count[p] == 0:
            fronts[0].append(p)
    i = 0
    while fronts[i]:
        next_front = []
        for p in fronts[i]:
            for q in dominated_sets[p]:
                domination_count[q] -= 1
                if domination_count[q] == 0:
                    next_front.append(q)
        i += 1
        fronts.append(next_front)
    return [front for front in fronts if front]


def pareto_front_points(pts, pf_min=True):
    pts = np.asarray(pts, dtype=float)
    mask = np.ones(pts.shape[0], dtype=bool)
    for i in range(pts.shape[0]):
        for j in range(pts.shape[0]):
            if i == j:
                continue
            if pf_min:
                if np.all(pts[j] <= pts[i]) and np.any(pts[j] < pts[i]):
                    mask[i] = False
                    break
            else:
                if np.all(pts[j] >= pts[i]) and np.any(pts[j] > pts[i]):
                    mask[i] = False
                    break
    return mask


def selection_leaders(size, M, population):
    if not population:
        return []
    objectives = np.asarray([ind[1:] for ind in population], dtype=float)
    mask = pareto_front_points(objectives, pf_min=True)
    leaders = [copy.deepcopy(population[i]) for i in range(len(population)) if mask[i]]
    leaders = ensure_unique_population(leaders)
    if len(leaders) > size:
        leaders = nsga3_environmental_selection(leaders, size, M)
    return leaders


def generate_reference_directions(n_objectives, divisions=None):
    n_objectives = int(n_objectives)
    if n_objectives < 1:
        raise ValueError('n_objectives must be positive.')
    if n_objectives == 1:
        return np.ones((1, 1))
    if divisions is None:
        divisions = max(1, min(12, int(math.ceil(24 / n_objectives))))
    refs = []
    for bars in itertools.combinations(range(divisions + n_objectives - 1), n_objectives - 1):
        points = (-1,) + bars + (divisions + n_objectives - 1,)
        vec = []
        for i in range(n_objectives):
            vec.append((points[i + 1] - points[i] - 1) / divisions)
        refs.append(vec)
    refs = np.asarray(refs, dtype=float)
    refs = refs / np.linalg.norm(refs, axis=1, keepdims=True)
    return refs


def normalize_objectives(objectives):
    objectives = np.asarray(objectives, dtype=float)
    ideal = np.min(objectives, axis=0)
    shifted = objectives - ideal
    nadir = np.max(shifted, axis=0)
    nadir[nadir == 0] = 1.0
    return shifted / nadir


def associate_to_reference_directions(normalized_objectives, reference_directions):
    normalized_objectives = np.asarray(normalized_objectives, dtype=float)
    refs = np.asarray(reference_directions, dtype=float)
    associations = []
    distances = []
    for point in normalized_objectives:
        norm_point = np.linalg.norm(point)
        if norm_point == 0:
            perpendicular = np.zeros(len(refs))
        else:
            projection_lengths = refs @ point
            projections = projection_lengths[:, None] * refs
            perpendicular = np.linalg.norm(point - projections, axis=1)
        idx = int(np.argmin(perpendicular))
        associations.append(idx)
        distances.append(float(perpendicular[idx]))
    return np.asarray(associations, dtype=int), np.asarray(distances, dtype=float)


def nsga3_environmental_selection(population, target_size, n_objectives, reference_directions=None):
    population = ensure_unique_population(population)
    if len(population) <= target_size:
        return population
    if reference_directions is None:
        reference_directions = generate_reference_directions(n_objectives)

    fronts = non_dominated_sort(population)
    selected = []
    last_front = []
    for front in fronts:
        front_individuals = [population[i] for i in front]
        if len(selected) + len(front_individuals) <= target_size:
            selected.extend(front_individuals)
        else:
            last_front = front_individuals
            break
    if len(selected) == target_size:
        return selected
    if not last_front:
        return selected[:target_size]

    combined = selected + last_front
    objectives = np.asarray([ind[1:] for ind in combined], dtype=float)
    normalized = normalize_objectives(objectives)
    associations, distances = associate_to_reference_directions(normalized, reference_directions)

    selected_count = np.zeros(len(reference_directions), dtype=int)
    for idx in range(len(selected)):
        selected_count[associations[idx]] += 1

    remaining_slots = target_size - len(selected)
    last_offset = len(selected)
    candidate_indices = list(range(len(last_front)))

    while remaining_slots > 0 and candidate_indices:
        candidate_ref_counts = [(selected_count[associations[last_offset + i]], associations[last_offset + i]) for i in candidate_indices]
        min_count = min(item[0] for item in candidate_ref_counts)
        refs_with_min_count = [ref for count, ref in candidate_ref_counts if count == min_count]
        chosen_ref = random.choice(refs_with_min_count)
        ref_candidates = [i for i in candidate_indices if associations[last_offset + i] == chosen_ref]
        if selected_count[chosen_ref] == 0:
            chosen_local = min(ref_candidates, key=lambda i: distances[last_offset + i])
        else:
            chosen_local = random.choice(ref_candidates)
        selected.append(last_front[chosen_local])
        selected_count[chosen_ref] += 1
        candidate_indices.remove(chosen_local)
        remaining_slots -= 1
    return selected[:target_size]


def order_crossover(parent_1, parent_2):
    p1 = list(parent_1)
    p2 = list(parent_2)
    n = len(p1)
    if n <= 2:
        return p1[:]
    a, b = sorted(random.sample(range(n), 2))
    child = [None] * n
    child[a:b + 1] = p1[a:b + 1]
    fill = [x for x in p2 if x not in child]
    pos = 0
    for i in list(range(0, a)) + list(range(b + 1, n)):
        child[i] = fill[pos]
        pos += 1
    return child


def pmx_crossover(parent_1, parent_2):
    p1 = list(parent_1)
    p2 = list(parent_2)
    n = len(p1)
    if n <= 2:
        return p1[:]
    a, b = sorted(random.sample(range(n), 2))
    child = [None] * n
    child[a:b + 1] = p1[a:b + 1]
    for i in range(a, b + 1):
        if p2[i] not in child:
            value = p2[i]
            pos = i
            while True:
                mapped = p1[pos]
                pos = p2.index(mapped)
                if child[pos] is None:
                    child[pos] = value
                    break
    for i in range(n):
        if child[i] is None:
            child[i] = p2[i]
    return child


def mutate_permutation(sequence, mutation_rate):
    seq = list(sequence)
    if len(seq) < 2 or random.random() > mutation_rate:
        return seq
    operator = random.choice(('swap', 'insert', 'invert'))
    i, j = sorted(random.sample(range(len(seq)), 2))
    if operator == 'swap':
        seq[i], seq[j] = seq[j], seq[i]
    elif operator == 'insert':
        value = seq.pop(j)
        seq.insert(i, value)
    else:
        seq[i:j + 1] = reversed(seq[i:j + 1])
    return seq


def tournament_select(population, rank, rng_size=2):
    candidates = random.sample(range(len(population)), min(rng_size, len(population)))
    return min(candidates, key=lambda idx: rank.get(tuple(population[idx][0]), 10**9))


def make_rank_lookup(population):
    fronts = non_dominated_sort(population)
    rank = {}
    for r, front in enumerate(fronts):
        for idx in front:
            rank[tuple(population[idx][0])] = r
    return rank


def create_offspring(population, target_size, list_of_functions, mutation_rate=0.10, crossover_rate=0.90):
    offspring = []
    rank = make_rank_lookup(population)
    attempts = 0
    while len(offspring) < target_size and attempts < target_size * 20:
        attempts += 1
        p1 = population[tournament_select(population, rank)]
        p2 = population[tournament_select(population, rank)]
        if random.random() <= crossover_rate:
            child_seq = order_crossover(p1[0], p2[0]) if random.random() < 0.5 else pmx_crossover(p1[0], p2[0])
        else:
            child_seq = list(p1[0])
        child_seq = mutate_permutation(child_seq, mutation_rate)
        offspring.append(_evaluate(child_seq, list_of_functions))
    return offspring


def nsga3_algorithm(population_size=50, jobs=7, list_of_functions=None, generations=150, mutation_rate=0.10, crossover_rate=0.90, divisions=None, verbose=True):
    if list_of_functions is None or len(list_of_functions) == 0:
        raise ValueError('list_of_functions must contain at least one objective function.')
    jobs = int(jobs)
    if jobs <= 0:
        raise ValueError('jobs must be positive.')
    n_objectives = len(list_of_functions)
    population_size = max(2, int(population_size))
    reference_directions = generate_reference_directions(n_objectives, divisions)
    population = initial_population_nsga3(population_size, jobs, list_of_functions)
    if verbose:
        print('NSGA-III Population Size:', len(population))
        print('NSGA-III Objectives:', n_objectives)
        print('NSGA-III Reference Directions:', len(reference_directions))
    for generation in range(int(generations) + 1):
        if verbose:
            print('Generation =', generation)
        offspring = create_offspring(population, population_size, list_of_functions, mutation_rate, crossover_rate)
        population = nsga3_environmental_selection(population + offspring, population_size, n_objectives, reference_directions)
    return selection_leaders(population_size, n_objectives, population)


# Backward-compatible alias with old ECMOA name, now implemented by NSGA-III.
def elitist_combinatorial_multiobjective_optimization_algorithm(size=15, jobs=7, list_of_functions=None, generations=1500, k=4, verbose=True):
    # k is ignored intentionally; NSGA-III controls diversity using reference directions.
    return nsga3_algorithm(population_size=size, jobs=jobs, list_of_functions=list_of_functions, generations=generations, verbose=verbose)

############################################################################
# Multiset / operation-based JSSP NSGA-III utilities
############################################################################

from collections import Counter


def _repair_multiset(sequence, required_counts):
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


def initial_population_nsga3_multiset(population_size, base_sequence, list_of_functions):
    population_size = max(2, int(population_size))
    base = list(base_sequence)
    if not base:
        raise ValueError('base_sequence must be non-empty.')
    required_counts = Counter(base)
    population = []
    seen = set()
    attempts = 0
    while len(population) < population_size and attempts < population_size * 100:
        attempts += 1
        seq = _repair_multiset(random.sample(base, len(base)), required_counts)
        key = tuple(seq)
        if key not in seen:
            seen.add(key)
            population.append(_evaluate(seq, list_of_functions))
    while len(population) < population_size:
        seq = _repair_multiset(random.sample(base, len(base)), required_counts)
        population.append(_evaluate(seq, list_of_functions))
    return population


def ppx_crossover_multiset(parent_1, parent_2, required_counts):
    p1 = list(parent_1)
    p2 = list(parent_2)
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
    return _repair_multiset(child, required_counts)


def mutate_multiset_sequence(sequence, mutation_rate):
    seq = list(sequence)
    if len(seq) < 2 or random.random() > mutation_rate:
        return seq
    operator = random.choice(('swap', 'insert', 'invert'))
    i, j = sorted(random.sample(range(len(seq)), 2))
    if operator == 'swap':
        seq[i], seq[j] = seq[j], seq[i]
    elif operator == 'insert':
        value = seq.pop(j)
        seq.insert(i, value)
    else:
        seq[i:j + 1] = list(reversed(seq[i:j + 1]))
    return seq


def create_offspring_multiset(population, target_size, list_of_functions, required_counts, mutation_rate=0.10, crossover_rate=0.90):
    offspring = []
    rank = make_rank_lookup(population)
    attempts = 0
    while len(offspring) < target_size and attempts < target_size * 30:
        attempts += 1
        p1 = population[tournament_select(population, rank)]
        p2 = population[tournament_select(population, rank)]
        if random.random() <= crossover_rate:
            child_seq = ppx_crossover_multiset(p1[0], p2[0], required_counts)
        else:
            child_seq = list(p1[0])
        child_seq = mutate_multiset_sequence(child_seq, mutation_rate)
        child_seq = _repair_multiset(child_seq, required_counts)
        offspring.append(_evaluate(child_seq, list_of_functions))
    return offspring


def nsga3_algorithm_multiset(population_size=50, base_sequence=None, list_of_functions=None, generations=150, mutation_rate=0.10, crossover_rate=0.90, divisions=None, verbose=True):
    if list_of_functions is None or len(list_of_functions) == 0:
        raise ValueError('list_of_functions must contain at least one objective function.')
    if base_sequence is None or len(base_sequence) == 0:
        raise ValueError('base_sequence must be a non-empty multiset chromosome template.')
    base = list(base_sequence)
    required_counts = Counter(base)
    n_objectives = len(list_of_functions)
    population_size = max(2, int(population_size))
    reference_directions = generate_reference_directions(n_objectives, divisions)
    population = initial_population_nsga3_multiset(population_size, base, list_of_functions)
    if verbose:
        print('NSGA-III Population Size:', len(population))
        print('NSGA-III Objectives:', n_objectives)
        print('NSGA-III Reference Directions:', len(reference_directions))
        print('Operation-based JSSP chromosome length:', len(base))
    for generation in range(int(generations) + 1):
        if verbose:
            print('Generation =', generation)
        offspring = create_offspring_multiset(population, population_size, list_of_functions, required_counts, mutation_rate, crossover_rate)
        population = nsga3_environmental_selection(population + offspring, population_size, n_objectives, reference_directions)
    return selection_leaders(population_size, n_objectives, population)
