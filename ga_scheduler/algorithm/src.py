############################################################################
# GA Scheduler - corrected scheduler core
############################################################################

import copy
import itertools
import math
import random
from collections import Counter
import warnings
import plotly.graph_objects as go
import numpy as np

from ga_scheduler.util.ga import genetic_algorithm, genetic_algorithm_multiset
from ga_scheduler.util.nsga3 import nsga3_algorithm, nsga3_algorithm_multiset, selection_leaders


# ---------------------------------------------------------------------------
# Gantt chart styling
# ---------------------------------------------------------------------------
# Curated qualitative palette (Tableau-10-inspired, slightly desaturated for
# print/screen consistency). Cycles when num_jobs > len(palette).
_JOB_PALETTE = (
    '#4C78A8', '#F58518', '#54A24B', '#E45756', '#72B7B2',
    '#EECA3B', '#B279A2', '#FF9DA6', '#9D755D', '#6E89A4',
)
_SETUP_FILL = '#E5E7EB'
_SETUP_LINE = '#9CA3AF'
_AXIS_COLOR = '#374151'
_GRID_COLOR = '#E5E7EB'
_BAR_BORDER = 'rgba(0,0,0,0.18)'
_FONT_FAMILY = (
    'Inter, -apple-system, BlinkMacSystemFont, "Segoe UI", '
    'Helvetica, Arial, sans-serif'
)


class load_ga_scheduler:
    """
    Permutation-priority scheduler plus optional operation-based and flexible-machine decoders.

    Notes
    -----
    Default mode uses a permutation of job IDs. With operation_based_jssp=True,
    the chromosome is an operation sequence where each job ID is repeated once
    per operation; the nth occurrence of job j dispatches operation n of job j.
    This supports repeated machine visits and general operation-level JSSP decoding.

    Passing flexible_sequences enables a separate alternative-machine decoder.
    Its input shape is job -> operation/stage -> [(machine, processing_time), ...].
    This supports flexible flow shop, hybrid flow shop, flexible job shop, and
    general alternative-machine routing while preserving the legacy sequences API.
    """

    def __init__(self, sequences=None, flexible_sequences=None, due_dates=None, setup_time_matrix=None, setup_waste_matrix=None,
                 z_permutations=100, job_weights=None,
                 obj_makespan=True, obj_total_compl_time=True, obj_total_late_jobs=True,
                 obj_max_w_tardiness=True, obj_total_waste=True, obj_setup=True,
                 w_obj_makespan=1, w_obj_total_compl_time=1, w_obj_total_late_jobs=1,
                 w_obj_max_w_tardiness=1, w_obj_total_waste=1, w_obj_setup=1,
                 parallel_machines=False, brute_force=False, pareto_front=False,
                 custom_sequence=None, operation_based_jssp=False,
                 z_mean=None, z_std=None, z_seed=None):
        self.flexible_operations = self._normalize_flexible_sequences(flexible_sequences) if flexible_sequences is not None else None
        self.flexible = self.flexible_operations is not None

        if self.flexible:
            # Keep the legacy ``sequences`` attribute as a fixed first-alternative
            # projection so older helper methods and diagnostics remain meaningful.
            self.sequences = self._first_alternative_sequences(self.flexible_operations)
        else:
            self.sequences = sequences or []

        if not self.sequences:
            raise ValueError('sequences must be non-empty, or provide flexible_sequences as job -> operation/stage -> alternatives.')

        self.due_dates = list(due_dates) if due_dates is not None else []
        self.setup_time_matrix = setup_time_matrix if setup_time_matrix is not None else []
        self.setup_waste_matrix = setup_waste_matrix if setup_waste_matrix is not None else []
        self.parallel = bool(parallel_machines)
        self.brute_force = bool(brute_force)
        self.custom_sequence = copy.deepcopy(custom_sequence) if custom_sequence is not None else None
        self.pareto_front = bool(pareto_front)
        # Flexible-machine problems are operation-dispatch problems by definition.
        self.operation_based_jssp = True if self.flexible else bool(operation_based_jssp)
        self.z_permutations = int(z_permutations)

        # Normalization of the scalarized objective uses random sampling when
        # z_permutations < num_jobs!. Keep a local RNG so users can make the
        # normalization reproducible without perturbing the GA operators.
        self.z_seed = z_seed
        self._z_rng = random.Random(z_seed) if z_seed is not None else random

        self.machine_sequences, self.matrix = self.sequence_inputs()
        self.operations = [[(int(m), int(t)) for m, t in job] for job in self.sequences]
        self.num_jobs = len(self.operations)
        if self.flexible:
            self.num_machines = int(max(m for job in self.flexible_operations for op in job for m, _ in op) + 1)
        else:
            self.num_machines = int(max(m for job in self.operations for m, _ in job) + 1)
        self.operation_counts = [len(job) for job in self.operations]
        self.operation_chromosome = [job_id for job_id, count in enumerate(self.operation_counts) for _ in range(count)]

        self.job_weights = [1.0] * self.num_jobs
        for i, value in enumerate(job_weights or []):
            if i < self.num_jobs:
                self.job_weights[i] = float(value)

        # Disable objectives that lack required data instead of crashing on defaults.
        self.obj_1 = bool(obj_makespan)
        self.obj_2 = bool(obj_max_w_tardiness) and self._has_vector(self.due_dates, self.num_jobs)
        self.obj_3 = bool(obj_total_waste) and self._has_square_matrix(self.setup_waste_matrix, self.num_jobs)
        self.obj_4 = bool(obj_setup) and self._has_square_matrix(self.setup_time_matrix, self.num_jobs)
        self.obj_5 = bool(obj_total_compl_time)
        self.obj_6 = bool(obj_total_late_jobs) and self._has_vector(self.due_dates, self.num_jobs)

        if obj_max_w_tardiness and not self.obj_2:
            warnings.warn('obj_max_w_tardiness disabled: due_dates must contain one value per job.', RuntimeWarning)
        if obj_total_late_jobs and not self.obj_6:
            warnings.warn('obj_total_late_jobs disabled: due_dates must contain one value per job.', RuntimeWarning)
        if obj_total_waste and not self.obj_3:
            warnings.warn('obj_total_waste disabled: setup_waste_matrix must be a num_jobs x num_jobs matrix.', RuntimeWarning)
        if obj_setup and not self.obj_4:
            warnings.warn('obj_setup disabled: setup_time_matrix must be a num_jobs x num_jobs matrix.', RuntimeWarning)

        self.objectives_weights = [
            w_obj_makespan if self.obj_1 else 0,
            w_obj_max_w_tardiness if self.obj_2 else 0,
            w_obj_total_waste if self.obj_3 else 0,
            w_obj_setup if self.obj_4 else 0,
            w_obj_total_compl_time if self.obj_5 else 0,
            w_obj_total_late_jobs if self.obj_6 else 0,
        ]
        self.lst_func = []

        if z_mean is not None or z_std is not None:
            if z_mean is None or z_std is None:
                raise ValueError('z_mean and z_std must be provided together.')
            if len(z_mean) != 6 or len(z_std) != 6:
                raise ValueError('z_mean and z_std must each contain six values.')
            self.z_mean = [float(v) for v in z_mean]
            self.z_std = [float(v) if float(v) != 0 else 1.0 for v in z_std]
        else:
            self.z_mean, self.z_std = self.obj_z_search() if sum(w != 0 for w in self.objectives_weights) > 1 else ([0] * 6, [1] * 6)

        self.objective_functions()
        self.best_sequence = None
        self.last_events = []

    @staticmethod
    def _has_vector(values, n):
        return values is not None and len(values) >= n

    @staticmethod
    def _has_square_matrix(matrix, n):
        try:
            arr = np.asarray(matrix)
            return arr.shape[0] >= n and arr.shape[1] >= n
        except Exception:
            return False

    @staticmethod
    def _is_machine_time_pair(value):
        """Return True for a single (machine, processing_time) pair."""
        if not isinstance(value, (tuple, list)) or len(value) != 2:
            return False
        machine, processing_time = value
        return not isinstance(machine, (tuple, list, dict)) and not isinstance(processing_time, (tuple, list, dict))

    @classmethod
    def _normalize_flexible_sequences(cls, flexible_sequences):
        """Normalize flexible-machine input.

        Expected structure
        ------------------
        flexible_sequences[job][operation] = [(machine, processing_time), ...]

        A fixed operation written as ``(machine, processing_time)`` is also
        accepted and converted to a one-alternative operation. This lets the
        flexible decoder represent single machine, flow shop, job shop,
        flexible flow shop, hybrid flow shop, flexible job shop, and general
        alternative-machine routing with one data model.
        """
        if flexible_sequences is None:
            return None
        normalized = []
        if not isinstance(flexible_sequences, (list, tuple)) or len(flexible_sequences) == 0:
            raise ValueError('flexible_sequences must be a non-empty list of jobs.')
        for job_id, job in enumerate(flexible_sequences):
            if not isinstance(job, (list, tuple)) or len(job) == 0:
                raise ValueError(f'Job {job_id} in flexible_sequences must contain at least one operation/stage.')
            normalized_job = []
            for op_id, operation in enumerate(job):
                if cls._is_machine_time_pair(operation):
                    alternatives = [operation]
                else:
                    if not isinstance(operation, (list, tuple)) or len(operation) == 0:
                        raise ValueError(f'Job {job_id}, operation {op_id} must contain at least one machine alternative.')
                    alternatives = list(operation)
                clean_alternatives = []
                for alt in alternatives:
                    if not cls._is_machine_time_pair(alt):
                        raise ValueError(
                            f'Invalid alternative at job {job_id}, operation {op_id}: expected (machine, processing_time).'
                        )
                    machine, processing_time = alt
                    machine = int(machine)
                    processing_time = int(processing_time)
                    if machine < 0:
                        raise ValueError('Machine ids must be non-negative integers.')
                    if processing_time < 0:
                        raise ValueError('Processing times must be non-negative integers.')
                    clean_alternatives.append((machine, processing_time))
                normalized_job.append(clean_alternatives)
            normalized.append(normalized_job)
        return normalized

    @staticmethod
    def _first_alternative_sequences(flexible_operations):
        """Project flexible input into the legacy fixed-machine representation."""
        return [[operation[0] for operation in job] for job in flexible_operations]

    def _default_machine_choices(self):
        """Return first-alternative choices for all flexible operations."""
        if not self.flexible:
            return []
        return [[0 for _operation in job] for job in self.flexible_operations]

    def _random_machine_choices(self, rng=None):
        """Sample one eligible machine alternative per operation."""
        rng = rng or random
        return [[rng.randrange(len(operation)) for operation in job] for job in self.flexible_operations]

    def _repair_machine_choices(self, machine_choices=None):
        """Repair machine-choice genes so every operation selects an eligible alternative."""
        if not self.flexible:
            return []
        if machine_choices is None:
            machine_choices = self._default_machine_choices()
        repaired = []
        for job_id, job in enumerate(self.flexible_operations):
            given_job = machine_choices[job_id] if job_id < len(machine_choices) and isinstance(machine_choices[job_id], (list, tuple)) else []
            repaired_job = []
            for op_id, alternatives in enumerate(job):
                value = given_job[op_id] if op_id < len(given_job) else 0
                try:
                    value = int(value)
                except Exception:
                    value = 0
                if value < 0 or value >= len(alternatives):
                    value = max(0, min(value, len(alternatives) - 1))
                repaired_job.append(value)
            repaired.append(repaired_job)
        return repaired

    def _repair_operation_sequence(self, sequence):
        """Repair an operation-dispatch sequence to the required job-id multiset."""
        required = Counter(self.operation_chromosome)
        if sequence is None:
            sequence = []
        sequence = list(sequence)
        used = Counter()
        repaired = []
        placeholders = []
        for gene in sequence:
            try:
                gene = int(gene)
            except Exception:
                gene = None
            if gene in required and used[gene] < required[gene]:
                repaired.append(gene)
                used[gene] += 1
            else:
                placeholders.append(len(repaired))
                repaired.append(None)
            if len(repaired) >= len(self.operation_chromosome):
                break
        missing = []
        for gene, count in required.items():
            missing.extend([gene] * (count - used[gene]))
        random.shuffle(missing)
        for idx in placeholders:
            if missing:
                repaired[idx] = missing.pop()
        repaired.extend(missing)
        return repaired[:len(self.operation_chromosome)]

    def _split_flexible_candidate(self, candidate, machine_choices=None):
        """Extract operation sequence and machine choices from flexible candidates.

        Accepted candidate formats
        --------------------------
        dict with keys:
            ``operation_sequence`` or ``sequence``;
            ``machine_choices`` or ``machine_assignment`` or ``choices``.
        tuple/list pair:
            ``(operation_sequence, machine_choices)``.
        plain sequence:
            operation sequence only; first machine alternative is used.
        """
        if machine_choices is not None:
            sequence = candidate
            choices = machine_choices
        elif isinstance(candidate, dict):
            sequence = (candidate.get('operation_sequence') if 'operation_sequence' in candidate
                        else candidate.get('sequence', candidate.get('chromosome')))
            choices = (candidate.get('machine_choices') if 'machine_choices' in candidate
                       else candidate.get('machine_assignment', candidate.get('choices')))
        elif (isinstance(candidate, (tuple, list)) and len(candidate) == 2
              and isinstance(candidate[0], (tuple, list))
              and isinstance(candidate[1], (tuple, list))
              and not all(isinstance(x, (int, np.integer)) for x in candidate)):
            sequence, choices = candidate
        else:
            sequence = candidate
            choices = None
        sequence = self._repair_operation_sequence(sequence)
        choices = self._repair_machine_choices(choices)
        return sequence, choices

    def _make_flexible_individual(self, operation_sequence=None, machine_choices=None, rng=None):
        """Create a valid flexible individual dictionary."""
        rng = rng or random
        if operation_sequence is None:
            base = list(self.operation_chromosome)
            operation_sequence = rng.sample(base, len(base))
        operation_sequence = self._repair_operation_sequence(operation_sequence)
        if machine_choices is None:
            machine_choices = self._random_machine_choices(rng)
        machine_choices = self._repair_machine_choices(machine_choices)
        return {'operation_sequence': list(operation_sequence), 'machine_choices': machine_choices}

    def _flexible_key(self, individual):
        sequence, choices = self._split_flexible_candidate(individual)
        flat_choices = tuple(tuple(job) for job in choices)
        return tuple(sequence), flat_choices

    def _decode_flexible(self, candidate, machine_choices=None):
        """Decode a flexible-machine operation sequence and assignment.

        This decoder supports flexible flow shop, hybrid flow shop, flexible job
        shop, and general alternative-machine routing. Precedence is preserved
        because the nth occurrence of a job dispatches the nth operation of that
        job. Machine eligibility is preserved because the selected alternative is
        always chosen from flexible_sequences[job][operation].
        """
        chromosome, choices = self._split_flexible_candidate(candidate, machine_choices)
        if not self._is_operation_chromosome(chromosome):
            raise ValueError('flexible operation chromosome must contain each job ID repeated exactly as many times as its number of operations.')

        events = []
        machine_available = [0] * self.num_machines
        machine_last_job = [None] * self.num_machines
        job_completion = [0] * self.num_jobs
        next_operation = [0] * self.num_jobs
        total_setup_time = 0
        total_waste = 0.0

        for job_id in chromosome:
            op_index = next_operation[job_id]
            if op_index >= len(self.flexible_operations[job_id]):
                continue
            alternatives = self.flexible_operations[job_id][op_index]
            alt_index = choices[job_id][op_index]
            machine, processing_time = alternatives[alt_index]
            setup = self._setup_time(machine_last_job[machine], job_id)
            setup_start = max(machine_available[machine], job_completion[job_id])
            start = setup_start + setup
            end = start + processing_time
            if setup > 0:
                events.append({'job': None, 'machine': machine, 'start': setup_start, 'end': start, 'kind': 'setup', 'label': f's{machine_last_job[machine]}-{job_id}'})
                total_setup_time += setup
                total_waste += self._setup_waste(machine_last_job[machine], job_id)
            events.append({
                'job': job_id, 'machine': machine, 'operation': op_index,
                'alternative': alt_index, 'start': start, 'end': end,
                'kind': 'job', 'label': f'j{job_id}.o{op_index}@m{machine}'
            })
            machine_available[machine] = end
            machine_last_job[machine] = job_id
            job_completion[job_id] = end
            next_operation[job_id] += 1

        makespan = max(machine_available) if machine_available else 0
        total_completion_time = sum(job_completion)
        max_weighted_tardiness = 0.0
        total_late_jobs = 0
        if self._has_vector(self.due_dates, self.num_jobs):
            for job_id, completion in enumerate(job_completion):
                tardiness = max(0, completion - self.due_dates[job_id])
                max_weighted_tardiness = max(max_weighted_tardiness, self.job_weights[job_id] * tardiness)
                total_late_jobs += int(completion > self.due_dates[job_id])

        return {
            'events': events,
            'completion_times': job_completion,
            'makespan': makespan,
            'total_completion_time': total_completion_time,
            'max_weighted_tardiness': max_weighted_tardiness,
            'total_late_jobs': total_late_jobs,
            'total_setup_time': total_setup_time,
            'total_waste': total_waste,
            'machine_choices': choices,
            'operation_sequence': list(chromosome),
        }

    def _ppx_operation_sequence(self, parent_1, parent_2):
        """Precedence-preserving crossover for repeated job-id operation sequences."""
        p1 = list(parent_1)
        p2 = list(parent_2)
        required_counts = Counter(self.operation_chromosome)
        used = Counter()
        child = []
        idx1 = idx2 = 0
        n = len(self.operation_chromosome)
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
        return self._repair_operation_sequence(child)

    def _crossover_machine_choices(self, choices_1, choices_2):
        child = []
        for job_id, job in enumerate(self.flexible_operations):
            child_job = []
            for op_id, _alternatives in enumerate(job):
                value = choices_1[job_id][op_id] if random.random() < 0.5 else choices_2[job_id][op_id]
                child_job.append(value)
            child.append(child_job)
        return self._repair_machine_choices(child)

    def _mutate_operation_sequence(self, sequence, mutation_rate):
        sequence = list(sequence)
        if len(sequence) < 2 or random.random() > mutation_rate:
            return sequence
        operator = random.choice(('swap', 'insert', 'invert'))
        i, j = sorted(random.sample(range(len(sequence)), 2))
        if operator == 'swap':
            sequence[i], sequence[j] = sequence[j], sequence[i]
        elif operator == 'insert':
            value = sequence.pop(j)
            sequence.insert(i, value)
        else:
            sequence[i:j + 1] = list(reversed(sequence[i:j + 1]))
        return self._repair_operation_sequence(sequence)

    def _mutate_machine_choices(self, machine_choices, mutation_rate):
        machine_choices = self._repair_machine_choices(copy.deepcopy(machine_choices))
        if random.random() > mutation_rate:
            return machine_choices
        mutable = [(j, o) for j, job in enumerate(self.flexible_operations)
                   for o, alternatives in enumerate(job) if len(alternatives) > 1]
        if not mutable:
            return machine_choices
        job_id, op_id = random.choice(mutable)
        n_alts = len(self.flexible_operations[job_id][op_id])
        old_value = machine_choices[job_id][op_id]
        alternatives = [i for i in range(n_alts) if i != old_value]
        machine_choices[job_id][op_id] = random.choice(alternatives) if alternatives else old_value
        return machine_choices

    def _flexible_crossover(self, parent_1, parent_2):
        seq_1, choices_1 = self._split_flexible_candidate(parent_1)
        seq_2, choices_2 = self._split_flexible_candidate(parent_2)
        child_sequence = self._ppx_operation_sequence(seq_1, seq_2)
        child_choices = self._crossover_machine_choices(choices_1, choices_2)
        return self._make_flexible_individual(child_sequence, child_choices)

    def _flexible_mutation(self, individual, mutation_rate):
        seq, choices = self._split_flexible_candidate(individual)
        seq = self._mutate_operation_sequence(seq, mutation_rate)
        choices = self._mutate_machine_choices(choices, mutation_rate)
        return self._make_flexible_individual(seq, choices)

    @staticmethod
    def _tournament_index_by_cost(population, tournament_size=2):
        candidates = random.sample(range(len(population)), min(tournament_size, len(population)))
        return min(candidates, key=lambda idx: population[idx][1])

    def flexible_genetic_algorithm(self, population_size=50, elite=2, mutation_rate=0.10, generations=100, verbose=True):
        """Scalarized GA for flexible-machine scheduling individuals."""
        population_size = max(2, int(population_size))
        elite = max(0, min(int(elite), population_size))
        population = []
        seen = set()
        attempts = 0
        while len(population) < population_size and attempts < population_size * 50:
            attempts += 1
            individual = self._make_flexible_individual()
            key = self._flexible_key(individual)
            if key in seen:
                continue
            seen.add(key)
            population.append([individual, self.target_function(individual)])
        while len(population) < population_size:
            individual = self._make_flexible_individual()
            population.append([individual, self.target_function(individual)])
        population = sorted(population, key=lambda item: item[1])
        best = copy.deepcopy(population[0])
        for generation in range(int(generations) + 1):
            if verbose:
                print('Generation:', generation)
            population = sorted(population, key=lambda item: item[1])
            if population[0][1] < best[1]:
                best = copy.deepcopy(population[0])
            offspring = [copy.deepcopy(ind) for ind in population[:elite]]
            while len(offspring) < population_size:
                p1 = population[self._tournament_index_by_cost(population)][0]
                p2 = population[self._tournament_index_by_cost(population)][0]
                child = self._flexible_crossover(p1, p2)
                child = self._flexible_mutation(child, mutation_rate)
                offspring.append([child, self.target_function(child)])
            population = offspring
        population = sorted(population, key=lambda item: item[1])
        if population[0][1] < best[1]:
            best = copy.deepcopy(population[0])
        self.best_sequence = best[0]
        return best[0], best[1]

    @staticmethod
    def _dominates_objectives(a, b):
        a = np.asarray(a, dtype=float)
        b = np.asarray(b, dtype=float)
        return np.all(a <= b) and np.any(a < b)

    def _flexible_objective_values(self, individual):
        return [float(func(individual)) for func in self.lst_func]

    def _flexible_fronts(self, population):
        n = len(population)
        domination_count = [0] * n
        dominated = [[] for _ in range(n)]
        fronts = [[]]
        objectives = [ind[1:] for ind in population]
        for p in range(n):
            for q in range(n):
                if p == q:
                    continue
                if self._dominates_objectives(objectives[p], objectives[q]):
                    dominated[p].append(q)
                elif self._dominates_objectives(objectives[q], objectives[p]):
                    domination_count[p] += 1
            if domination_count[p] == 0:
                fronts[0].append(p)
        i = 0
        while fronts[i]:
            next_front = []
            for p in fronts[i]:
                for q in dominated[p]:
                    domination_count[q] -= 1
                    if domination_count[q] == 0:
                        next_front.append(q)
            i += 1
            fronts.append(next_front)
        return [front for front in fronts if front]

    def _flexible_environmental_selection(self, population, target_size):
        unique = []
        seen = set()
        for ind in population:
            key = self._flexible_key(ind[0])
            if key not in seen:
                seen.add(key)
                unique.append(ind)
        if len(unique) <= target_size:
            return unique
        selected = []
        fronts = self._flexible_fronts(unique)
        for front in fronts:
            front_individuals = [unique[i] for i in front]
            if len(selected) + len(front_individuals) <= target_size:
                selected.extend(front_individuals)
            else:
                # Simple diversity-preserving truncation: normalize objectives and
                # prefer lower aggregate normalized score inside the split front.
                objs = np.asarray([ind[1:] for ind in front_individuals], dtype=float)
                mins = objs.min(axis=0)
                ranges = objs.max(axis=0) - mins
                ranges[ranges == 0] = 1.0
                scores = ((objs - mins) / ranges).sum(axis=1)
                order = np.argsort(scores)
                remaining = target_size - len(selected)
                selected.extend([front_individuals[i] for i in order[:remaining]])
                break
        return selected[:target_size]

    def flexible_pareto_search(self, population_size=50, elite=2, mutation_rate=0.10, generations=100, verbose=True):
        """Multiobjective evolutionary search for flexible-machine schedules."""
        if not self.lst_func:
            raise ValueError('At least one objective must be active for pareto_front=True.')
        population_size = max(2, int(population_size))
        population = []
        while len(population) < population_size:
            ind = self._make_flexible_individual()
            population.append([ind] + self._flexible_objective_values(ind))
        population = self._flexible_environmental_selection(population, population_size)
        for generation in range(int(generations) + 1):
            if verbose:
                print('Generation =', generation)
            offspring = []
            scalar_population = [[ind[0], float(np.sum(ind[1:]))] for ind in population]
            while len(offspring) < population_size:
                p1 = scalar_population[self._tournament_index_by_cost(scalar_population)][0]
                p2 = scalar_population[self._tournament_index_by_cost(scalar_population)][0]
                child = self._flexible_crossover(p1, p2)
                child = self._flexible_mutation(child, mutation_rate)
                offspring.append([child] + self._flexible_objective_values(child))
            population = self._flexible_environmental_selection(population + offspring, population_size)
        fronts = self._flexible_fronts(population)
        return [population[i] for i in fronts[0]] if fronts else population

    def generate_sequences(self, num_jobs, num_machines, use_all_machines=False, parallel_machines=False,
                           parallel_same_time=False, flow_shop=False, bounds=(1, 10), seed=None):
        def process_lists(lists, pm=False, tm=False):
            processed = []
            for lst in lists:
                if pm:
                    lst = sorted(lst, key=lambda x: x[0])
                if tm and lst:
                    first_time = lst[0][1]
                    lst = [(t[0], first_time) for t in lst]
                processed.append(lst)
            return processed

        if seed is not None:
            random.seed(seed)
        sequences = []
        base_sequence = random.sample(range(num_machines), num_machines) if flow_shop else None
        for _job in range(num_jobs):
            job_sequence = []
            machine_sequence = base_sequence if flow_shop else (random.sample(range(num_machines), num_machines) if use_all_machines else random.sample(range(num_machines), random.randint(1, num_machines)))
            for machine in machine_sequence:
                job_sequence.append((machine, random.randint(bounds[0], bounds[1])))
            sequences.append(job_sequence)
        if parallel_machines or parallel_same_time:
            sequences = process_lists(sequences, pm=parallel_machines, tm=parallel_same_time)
        return sequences

    def _setup_time(self, prev_job, job_id):
        if prev_job is None or not self.obj_4:
            return 0
        return int(self.setup_time_matrix[prev_job][job_id])

    def _setup_waste(self, prev_job, job_id):
        if prev_job is None or not self.obj_3:
            return 0
        return float(self.setup_waste_matrix[prev_job][job_id])

    def _is_operation_chromosome(self, sequence):
        seq = list(sequence)
        return len(seq) == len(self.operation_chromosome) and sorted(seq) == sorted(self.operation_chromosome)

    def _decode_operation_sequence(self, chromosome):
        chromosome = list(chromosome)
        if not self._is_operation_chromosome(chromosome):
            raise ValueError('operation-based JSSP chromosome must contain each job ID repeated exactly as many times as its number of operations.')

        events = []
        machine_available = [0] * self.num_machines
        machine_last_job = [None] * self.num_machines
        job_completion = [0] * self.num_jobs
        next_operation = [0] * self.num_jobs
        total_setup_time = 0
        total_waste = 0.0

        for job_id in chromosome:
            op_index = next_operation[job_id]
            if op_index >= len(self.operations[job_id]):
                # Should not happen after validation, but keeps the decoder safe.
                continue
            machine, processing_time = self.operations[job_id][op_index]
            setup = self._setup_time(machine_last_job[machine], job_id)
            setup_start = max(machine_available[machine], job_completion[job_id])
            start = setup_start + setup
            end = start + processing_time
            if setup > 0:
                events.append({'job': None, 'machine': machine, 'start': setup_start, 'end': start, 'kind': 'setup', 'label': f's{machine_last_job[machine]}-{job_id}'})
                total_setup_time += setup
                total_waste += self._setup_waste(machine_last_job[machine], job_id)
            events.append({'job': job_id, 'machine': machine, 'operation': op_index, 'start': start, 'end': end, 'kind': 'job', 'label': f'j{job_id}.o{op_index}'})
            machine_available[machine] = end
            machine_last_job[machine] = job_id
            job_completion[job_id] = end
            next_operation[job_id] += 1

        makespan = max(machine_available) if machine_available else 0
        total_completion_time = sum(job_completion)
        max_weighted_tardiness = 0.0
        total_late_jobs = 0
        if self._has_vector(self.due_dates, self.num_jobs):
            for job_id, completion in enumerate(job_completion):
                tardiness = max(0, completion - self.due_dates[job_id])
                max_weighted_tardiness = max(max_weighted_tardiness, self.job_weights[job_id] * tardiness)
                total_late_jobs += int(completion > self.due_dates[job_id])

        return {
            'events': events,
            'completion_times': job_completion,
            'makespan': makespan,
            'total_completion_time': total_completion_time,
            'max_weighted_tardiness': max_weighted_tardiness,
            'total_late_jobs': total_late_jobs,
            'total_setup_time': total_setup_time,
            'total_waste': total_waste,
        }

    def _decode(self, permutation):
        if self.flexible:
            return self._decode_flexible(permutation)
        permutation = list(permutation)
        if self.operation_based_jssp or self._is_operation_chromosome(permutation):
            return self._decode_operation_sequence(permutation)
        if sorted(permutation) != list(range(self.num_jobs)):
            raise ValueError('permutation/custom_sequence must contain each job ID exactly once, unless operation_based_jssp=True and the chromosome repeats job IDs by operation count.')

        events = []
        machine_available = [0] * self.num_machines
        machine_last_job = [None] * self.num_machines
        job_completion = [0] * self.num_jobs
        total_setup_time = 0
        total_waste = 0.0

        if self.parallel:
            # Each job is assigned to one alternative machine. Choose minimum completion, not merely minimum start.
            for job_id in permutation:
                best = None
                for machine, processing_time in self.operations[job_id]:
                    setup = self._setup_time(machine_last_job[machine], job_id)
                    start = machine_available[machine] + setup
                    completion = start + processing_time
                    candidate = (completion, start, machine, processing_time, setup)
                    if best is None or candidate < best:
                        best = candidate
                completion, start, machine, processing_time, setup = best
                if setup > 0:
                    setup_start = machine_available[machine]
                    events.append({'job': None, 'machine': machine, 'start': setup_start, 'end': start, 'kind': 'setup', 'label': f's{machine_last_job[machine]}-{job_id}'})
                    total_setup_time += setup
                    total_waste += self._setup_waste(machine_last_job[machine], job_id)
                events.append({'job': job_id, 'machine': machine, 'start': start, 'end': completion, 'kind': 'job', 'label': f'j{job_id}'})
                machine_available[machine] = completion
                machine_last_job[machine] = job_id
                job_completion[job_id] = completion
        else:
            for job_id in permutation:
                current_time = job_completion[job_id]
                for op_index, (machine, processing_time) in enumerate(self.operations[job_id]):
                    setup = self._setup_time(machine_last_job[machine], job_id)
                    setup_start = max(machine_available[machine], current_time)
                    start = setup_start + setup
                    end = start + processing_time
                    if setup > 0:
                        events.append({'job': None, 'machine': machine, 'start': setup_start, 'end': start, 'kind': 'setup', 'label': f's{machine_last_job[machine]}-{job_id}'})
                        total_setup_time += setup
                        total_waste += self._setup_waste(machine_last_job[machine], job_id)
                    events.append({'job': job_id, 'machine': machine, 'operation': op_index, 'start': start, 'end': end, 'kind': 'job', 'label': f'j{job_id}'})
                    machine_available[machine] = end
                    machine_last_job[machine] = job_id
                    current_time = end
                job_completion[job_id] = current_time

        makespan = max(machine_available) if machine_available else 0
        total_completion_time = sum(job_completion)
        max_weighted_tardiness = 0.0
        total_late_jobs = 0
        if self._has_vector(self.due_dates, self.num_jobs):
            for job_id, completion in enumerate(job_completion):
                tardiness = max(0, completion - self.due_dates[job_id])
                max_weighted_tardiness = max(max_weighted_tardiness, self.job_weights[job_id] * tardiness)
                total_late_jobs += int(completion > self.due_dates[job_id])

        return {
            'events': events,
            'completion_times': job_completion,
            'makespan': makespan,
            'total_completion_time': total_completion_time,
            'max_weighted_tardiness': max_weighted_tardiness,
            'total_late_jobs': total_late_jobs,
            'total_setup_time': total_setup_time,
            'total_waste': total_waste,
        }

    def schedule_jobs(self, permutation):
        decoded = self._decode(permutation)
        self.last_events = decoded['events']
        total_length = max(1, int(decoded['makespan']))
        schedule = [['' for _ in range(total_length)] for _ in range(self.num_machines)]
        for event in decoded['events']:
            label = event['label']
            for t in range(int(event['start']), int(event['end'])):
                if 0 <= t < total_length:
                    schedule[event['machine']][t] = label
        return np.array(schedule, dtype=object)

    def _events_from_matrix(self, schedule_matrix):
        """Reconstruct events by scanning runs of identical labels in the schedule matrix.

        Used as a fallback when ``self.last_events`` is not populated (e.g. when a
        caller hands in a schedule_matrix that wasn't produced by ``schedule_jobs``).
        """
        events = []
        if schedule_matrix.ndim != 2:
            return events
        num_machines, total_length = schedule_matrix.shape
        for m in range(num_machines):
            t = 0
            while t < total_length:
                label = schedule_matrix[m][t]
                if label == '':
                    t += 1
                    continue
                start = t
                while t < total_length and schedule_matrix[m][t] == label:
                    t += 1
                end = t
                label_str = str(label)
                if label_str.startswith('s'):
                    events.append({'job': None, 'machine': m, 'start': start,
                                   'end': end, 'kind': 'setup', 'label': label_str})
                else:
                    try:
                        job_id = int(label_str[1:])
                    except ValueError:
                        job_id = -1
                    events.append({'job': job_id, 'machine': m, 'start': start,
                                   'end': end, 'kind': 'job', 'label': label_str})
        return events

    def create_gantt_chart(self, schedule_matrix, size_x=12, size_y=8):
        """Render a Gantt chart for the given schedule using Plotly.

        Parameters
        ----------
        schedule_matrix : np.ndarray
            Schedule grid produced by ``schedule_jobs``. Used to determine which
            machine rows to display.
        size_x, size_y : float
            Approximate figure size hints (legacy parameter names preserved for
            backwards compatibility). Internally converted to pixel dimensions.

        Returns
        -------
        plotly.graph_objects.Figure
            The figure object, in case the caller wants to ``write_html``,
            ``write_image``, or further customise it.
        """
        # Preserve the original behaviour of hiding fully-empty machine rows.
        non_empty_rows = [i for i in range(schedule_matrix.shape[0])
                          if not np.all(schedule_matrix[i] == '')]
        if not non_empty_rows:
            non_empty_rows = list(range(schedule_matrix.shape[0]))

        # Prefer the decoded events cached by schedule_jobs (continuous, exact);
        # fall back to a matrix scan otherwise.
        events = list(self.last_events) if self.last_events else self._events_from_matrix(schedule_matrix)
        active = set(non_empty_rows)
        events = [e for e in events if e['machine'] in active]

        machine_labels = [f'Machine {m}' for m in non_empty_rows]
        machine_to_label = {m: f'Machine {m}' for m in non_empty_rows}

        makespan = max((e['end'] for e in events), default=int(schedule_matrix.shape[1]))

        fig = go.Figure()

        # --- Setup events: single shared trace, hatched grey -----------------
        setup_events = [e for e in events if e['kind'] == 'setup']
        if setup_events:
            fig.add_trace(go.Bar(
                y=[machine_to_label[e['machine']] for e in setup_events],
                x=[e['end'] - e['start'] for e in setup_events],
                base=[e['start'] for e in setup_events],
                orientation='h',
                name='Setup',
                legendgroup='setup',
                marker=dict(
                    color=_SETUP_FILL,
                    line=dict(color=_SETUP_LINE, width=1),
                    pattern=dict(shape='/', fgcolor=_SETUP_LINE, size=6, solidity=0.28),
                ),
                customdata=[[e.get('label', ''), e['start'], e['end'], e['end'] - e['start']]
                            for e in setup_events],
                hovertemplate=(
                    '<b>Setup %{customdata[0]}</b><br>'
                    'Machine: %{y}<br>'
                    'Start: %{customdata[1]:.0f}<br>'
                    'End: %{customdata[2]:.0f}<br>'
                    'Duration: %{customdata[3]:.0f}'
                    '<extra></extra>'
                ),
                width=0.62,
            ))

        # --- Job events: one trace per job id (stable colour + legend entry) -
        job_events = [e for e in events if e['kind'] == 'job']
        job_ids_sorted = sorted({e['job'] for e in job_events if e.get('job') is not None})
        for job_id in job_ids_sorted:
            jobs_of_id = [e for e in job_events if e['job'] == job_id]
            color = _JOB_PALETTE[job_id % len(_JOB_PALETTE)]
            fig.add_trace(go.Bar(
                y=[machine_to_label[e['machine']] for e in jobs_of_id],
                x=[e['end'] - e['start'] for e in jobs_of_id],
                base=[e['start'] for e in jobs_of_id],
                orientation='h',
                name=f'Job {job_id}',
                legendgroup=f'job_{job_id}',
                marker=dict(
                    color=color,
                    line=dict(color=_BAR_BORDER, width=0.5),
                ),
                text=[f'j{job_id}' for _ in jobs_of_id],
                textposition='inside',
                insidetextanchor='middle',
                textfont=dict(color='white', size=11, family=_FONT_FAMILY),
                cliponaxis=False,
                customdata=[[job_id, e['start'], e['end'], e['end'] - e['start']]
                            for e in jobs_of_id],
                hovertemplate=(
                    '<b>Job %{customdata[0]}</b><br>'
                    'Machine: %{y}<br>'
                    'Start: %{customdata[1]:.0f}<br>'
                    'End: %{customdata[2]:.0f}<br>'
                    'Duration: %{customdata[3]:.0f}'
                    '<extra></extra>'
                ),
                width=0.62,
            ))

        # Convert legacy inch-style hints into pixel dimensions.
        width = max(640, int(size_x * 80))
        height = max(280, int(size_y * 55))

        # Render Machine 0 at the top of the chart.
        fig.update_yaxes(categoryorder='array',
                         categoryarray=list(reversed(machine_labels)))

        title_html = (
            '<b>Gantt Chart</b>'
            f'  <span style="font-size:13px;color:#6B7280;font-weight:400;">'
            f'Makespan: {int(makespan)}</span>'
        )

        fig.update_layout(
            title=dict(text=title_html, x=0.02, y=0.97, xanchor='left',
                       font=dict(family=_FONT_FAMILY, size=18, color=_AXIS_COLOR)),
            barmode='stack',
            bargap=0.28,
            plot_bgcolor='white',
            paper_bgcolor='white',
            font=dict(family=_FONT_FAMILY, size=12, color=_AXIS_COLOR),
            width=width,
            height=height,
            margin=dict(l=92, r=24, t=68, b=64),
            showlegend=True,
            legend=dict(
                orientation='h',
                yanchor='bottom', y=-0.22,
                xanchor='left', x=0,
                bgcolor='rgba(0,0,0,0)',
                font=dict(size=11),
                itemsizing='constant',
            ),
            xaxis=dict(
                title=dict(text='Time', font=dict(size=12, color=_AXIS_COLOR)),
                showgrid=True,
                gridcolor=_GRID_COLOR,
                zeroline=False,
                showline=True,
                linecolor=_GRID_COLOR,
                ticks='outside',
                tickcolor=_GRID_COLOR,
                range=[0, makespan * 1.02 + 0.5],
            ),
            yaxis=dict(
                title=None,
                showgrid=False,
                zeroline=False,
                showline=True,
                linecolor=_GRID_COLOR,
                ticks='',
            ),
            hoverlabel=dict(
                bgcolor='white',
                bordercolor=_GRID_COLOR,
                font=dict(family=_FONT_FAMILY, size=12, color=_AXIS_COLOR),
            ),
        )

        fig.show()
        return fig

    def calculate_makespan(self, schedule_matrix=None, permutation=None):
        if permutation is not None:
            return self._decode(permutation)['makespan']
        return schedule_matrix.shape[1]

    def calculate_makespan_p(self, permutation):
        return self._decode(permutation)['makespan']

    def calculate_max_weighted_tardiness(self, schedule_matrix=None, permutation=None):
        if permutation is None:
            raise ValueError('Use calculate_max_weighted_tardiness_p(permutation) for corrected objective evaluation.')
        return self._decode(permutation)['max_weighted_tardiness']

    def calculate_max_weighted_tardiness_p(self, permutation):
        return self._decode(permutation)['max_weighted_tardiness']

    def calculate_total_completion_time(self, schedule_matrix=None, permutation=None):
        if permutation is None:
            raise ValueError('Use calculate_total_completion_time_p(permutation) for corrected objective evaluation.')
        return self._decode(permutation)['total_completion_time']

    def calculate_total_completion_time_p(self, permutation):
        return self._decode(permutation)['total_completion_time']

    def calculate_total_late_jobs(self, schedule_matrix=None, permutation=None):
        if permutation is None:
            raise ValueError('Use calculate_total_late_jobs_p(permutation) for corrected objective evaluation.')
        return self._decode(permutation)['total_late_jobs']

    def calculate_total_late_jobs_p(self, permutation):
        return self._decode(permutation)['total_late_jobs']

    def calculate_total_waste(self, permutation):
        return self._decode(permutation)['total_waste']

    def calculate_total_setup_time(self, permutation):
        return self._decode(permutation)['total_setup_time']

    def calculate_idle_times(self, schedule_matrix):
        total_idle_time = 0
        for row in schedule_matrix:
            last = None
            for i in range(len(row) - 1, -1, -1):
                if row[i] != '':
                    last = i
                    break
            if last is not None:
                total_idle_time += sum(1 for i in range(last) if row[i] == '')
        return total_idle_time

    def calculate_idle_times_p(self, permutation):
        return self.calculate_idle_times(self.schedule_jobs(permutation))

    def sequence_inputs(self):
        num_jobs = len(self.sequences)
        if self.flexible:
            num_machines = max(m for job in self.flexible_operations for op in job for m, _ in op) + 1
        else:
            num_machines = max(max(machine for machine, _ in job) for job in self.sequences) + 1
        matrix = np.zeros((num_jobs, num_machines), dtype=int)
        seen = [set() for _ in range(num_jobs)]
        for job_id, job in enumerate(self.sequences):
            for machine, time in job:
                if machine in seen[job_id]:
                    warnings.warn(f'Job {job_id} visits machine {machine} more than once. The internal matrix stores summed time; exact operation order is preserved separately.', RuntimeWarning)
                matrix[job_id, machine] += int(time)
                seen[job_id].add(machine)
        machine_sequences = [[machine for machine, _ in job] for job in self.sequences]
        return machine_sequences, matrix

    def brute_force_search(self):
        if self.operation_based_jssp:
            raise ValueError('brute_force=True is not supported for operation_based_jssp=True because multiset operation permutations grow very quickly. Use GA or NSGA-III.')
        job_ids = list(range(self.num_jobs))
        best_sequence = None
        best_value = float('inf')
        total = math.factorial(len(job_ids))
        print('\nVerifying', total, 'solutions\n')
        for count, permutation in enumerate(itertools.permutations(job_ids), start=1):
            value = self.target_function(permutation)
            if value < best_value:
                best_value = value
                best_sequence = permutation
                print(f'Searched Space: {count / total * 100:.2f}%; Best Sequence: {best_sequence}; Obj. Function: {best_value:.4f}')
        print('\nBrute Force Search Complete!\n')
        print(f'Best Sequence: {best_sequence}; Obj. Function: {best_value:.4f}')
        self.best_sequence = best_sequence
        return best_sequence, best_value

    def brute_force_search_p(self):
        if self.operation_based_jssp:
            raise ValueError('brute_force=True is not supported for operation_based_jssp=True because multiset operation permutations grow very quickly. Use GA or NSGA-III.')
        merged = []
        job_ids = list(range(self.num_jobs))
        total = math.factorial(len(job_ids))
        print('\nVerifying', total, 'solutions\n')
        for count, permutation in enumerate(itertools.permutations(job_ids), start=1):
            p = list(permutation)
            merged.append([p] + [func(p) for func in self.lst_func])
            if total >= 10 and (count % max(1, int(total / 10)) == 0 or count == total):
                print(f'Searched Space: {count / total * 100:.2f}%')
        print('Brute Force Search Complete!')
        return selection_leaders(total, len(self.lst_func), merged)

    def obj_z_search(self):
        rng = self._z_rng

        def unique_permutations(job_ids, n_perm):
            if n_perm >= math.factorial(len(job_ids)):
                return list(itertools.permutations(job_ids))
            seen = set()
            while len(seen) < n_perm:
                seen.add(tuple(rng.sample(job_ids, len(job_ids))))
            return list(seen)

        if self.flexible:
            sampled = []
            seen = set()
            target = max(1, self.z_permutations)
            attempts = 0
            while len(sampled) < target and attempts < target * 80:
                attempts += 1
                candidate = self._make_flexible_individual(rng=rng)
                key = self._flexible_key(candidate)
                if key not in seen:
                    seen.add(key)
                    sampled.append(candidate)
            if not sampled:
                sampled = [self._make_flexible_individual(rng=rng)]
        elif self.operation_based_jssp:
            base = list(self.operation_chromosome)
            seen = set()
            sampled = []
            target = max(1, self.z_permutations)
            attempts = 0
            while len(sampled) < target and attempts < target * 50:
                attempts += 1
                candidate = tuple(rng.sample(base, len(base)))
                if candidate not in seen:
                    seen.add(candidate)
                    sampled.append(candidate)
            if not sampled:
                sampled = [tuple(base)]
        else:
            job_ids = list(range(self.num_jobs))
            n_perm = max(1, min(self.z_permutations, math.factorial(len(job_ids))))
            sampled = unique_permutations(job_ids, n_perm)
        values = [[] for _ in range(6)]
        for permutation in sampled:
            decoded = self._decode(permutation)
            if self.obj_1: values[0].append(decoded['makespan'])
            if self.obj_2: values[1].append(decoded['max_weighted_tardiness'])
            if self.obj_3: values[2].append(decoded['total_waste'])
            if self.obj_4: values[3].append(decoded['total_setup_time'])
            if self.obj_5: values[4].append(decoded['total_completion_time'])
            if self.obj_6: values[5].append(decoded['total_late_jobs'])
        z_mean, z_std = [], []
        for lst in values:
            if lst:
                z_mean.append(float(np.mean(lst)))
                z_std.append(float(np.std(lst, ddof=1)) if len(lst) > 1 else 1.0)
            else:
                z_mean.append(0.0)
                z_std.append(1.0)
        return z_mean, z_std

    def get_normalization_stats(self):
        """Return the z-score normalization statistics used by target_function."""
        return list(self.z_mean), list(self.z_std)

    def set_normalization_stats(self, z_mean, z_std):
        """Set externally computed z-score normalization statistics."""
        if len(z_mean) != 6 or len(z_std) != 6:
            raise ValueError('z_mean and z_std must each contain six values.')
        self.z_mean = [float(v) for v in z_mean]
        self.z_std = [float(v) if float(v) != 0 else 1.0 for v in z_std]
        return self

    def target_function(self, permutation):
        decoded = self._decode(permutation)
        raw = [decoded['makespan'], decoded['max_weighted_tardiness'], decoded['total_waste'], decoded['total_setup_time'], decoded['total_completion_time'], decoded['total_late_jobs']]
        objective = 0.0
        for i, weight in enumerate(self.objectives_weights):
            if weight != 0:
                z = ((raw[i] - self.z_mean[i]) / (self.z_std[i] + 1e-14)) + 9
                objective += weight * z
        return objective

    def objective_functions(self):
        if self.obj_1: self.lst_func.append(self.calculate_makespan_p)
        if self.obj_2: self.lst_func.append(self.calculate_max_weighted_tardiness_p)
        if self.obj_3: self.lst_func.append(self.calculate_total_waste)
        if self.obj_4: self.lst_func.append(self.calculate_total_setup_time)
        if self.obj_5: self.lst_func.append(self.calculate_total_completion_time_p)
        if self.obj_6: self.lst_func.append(self.calculate_total_late_jobs_p)

    def run_ga_scheduler(self, population_size=5, elite=1, mutation_rate=0.10, generations=100, k=4):
        has_custom = self.custom_sequence is not None
        if self.flexible:
            if self.brute_force:
                raise ValueError('brute_force=True is not supported for flexible_sequences because machine choices and operation multiset permutations grow very quickly. Use GA or pareto_front=True.')
            if has_custom:
                individual = self._make_flexible_individual(*self._split_flexible_candidate(self.custom_sequence))
                obj_fun = self.target_function(individual)
                return individual, self.schedule_jobs(individual), obj_fun
            if self.pareto_front:
                return self.flexible_pareto_search(population_size=population_size, elite=elite, mutation_rate=mutation_rate, generations=generations, verbose=True)
            individual, obj_fun = self.flexible_genetic_algorithm(population_size=population_size, elite=elite, mutation_rate=mutation_rate, generations=generations, verbose=True)
            return individual, self.schedule_jobs(individual), obj_fun

        if not has_custom and not self.brute_force and not self.pareto_front:
            if self.operation_based_jssp:
                job_sequence, obj_fun = genetic_algorithm_multiset(self.operation_chromosome, population_size, elite, mutation_rate, generations, self.target_function, True)
            else:
                job_sequence, obj_fun = genetic_algorithm(self.num_jobs, population_size, elite, mutation_rate, generations, self.target_function, True)
            return job_sequence, self.schedule_jobs(job_sequence), obj_fun
        if not has_custom and not self.brute_force and self.pareto_front:
            if self.operation_based_jssp:
                return nsga3_algorithm_multiset(population_size=population_size, base_sequence=self.operation_chromosome, list_of_functions=self.lst_func, generations=generations, mutation_rate=mutation_rate, verbose=True)
            return nsga3_algorithm(population_size=population_size, jobs=self.num_jobs, list_of_functions=self.lst_func, generations=generations, mutation_rate=mutation_rate, verbose=True)
        if not has_custom and self.brute_force and not self.pareto_front:
            job_sequence, obj_fun = self.brute_force_search()
            return job_sequence, self.schedule_jobs(job_sequence), obj_fun
        if not has_custom and self.brute_force and self.pareto_front:
            return self.brute_force_search_p()
        job_sequence = self.custom_sequence
        obj_fun = self.target_function(job_sequence)
        return job_sequence, self.schedule_jobs(job_sequence), obj_fun
