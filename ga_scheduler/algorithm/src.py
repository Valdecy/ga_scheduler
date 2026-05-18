############################################################################
# GA Scheduler - corrected scheduler core
############################################################################

import itertools
import math
import random
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
    Permutation-priority scheduler plus optional operation-based full JSSP decoder.

    Notes
    -----
    Default mode uses a permutation of job IDs. With operation_based_jssp=True,
    the chromosome is an operation sequence where each job ID is repeated once
    per operation; the nth occurrence of job j dispatches operation n of job j.
    This supports repeated machine visits and general operation-level JSSP decoding.
    """

    def __init__(self, sequences=None, due_dates=None, setup_time_matrix=None, setup_waste_matrix=None,
                 z_permutations=100, job_weights=None,
                 obj_makespan=True, obj_total_compl_time=True, obj_total_late_jobs=True,
                 obj_max_w_tardiness=True, obj_total_waste=True, obj_setup=True,
                 w_obj_makespan=1, w_obj_total_compl_time=1, w_obj_total_late_jobs=1,
                 w_obj_max_w_tardiness=1, w_obj_total_waste=1, w_obj_setup=1,
                 parallel_machines=False, brute_force=False, pareto_front=False,
                 custom_sequence=None, operation_based_jssp=False):
        self.sequences = sequences or []
        if not self.sequences:
            raise ValueError('sequences must be a non-empty list of jobs, where each job is a list of (machine, processing_time) tuples.')

        self.due_dates = list(due_dates) if due_dates is not None else []
        self.setup_time_matrix = setup_time_matrix if setup_time_matrix is not None else []
        self.setup_waste_matrix = setup_waste_matrix if setup_waste_matrix is not None else []
        self.parallel = bool(parallel_machines)
        self.brute_force = bool(brute_force)
        self.custom_sequence = list(custom_sequence) if custom_sequence is not None else []
        self.pareto_front = bool(pareto_front)
        self.operation_based_jssp = bool(operation_based_jssp)
        self.z_permutations = int(z_permutations)

        self.machine_sequences, self.matrix = self.sequence_inputs()
        self.operations = [[(int(m), int(t)) for m, t in job] for job in self.sequences]
        self.num_jobs = len(self.operations)
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
        def unique_permutations(job_ids, n_perm):
            if n_perm >= math.factorial(len(job_ids)):
                return list(itertools.permutations(job_ids))
            seen = set()
            while len(seen) < n_perm:
                seen.add(tuple(random.sample(job_ids, len(job_ids))))
            return list(seen)

        if self.operation_based_jssp:
            base = list(self.operation_chromosome)
            seen = set()
            sampled = []
            target = max(1, self.z_permutations)
            attempts = 0
            while len(sampled) < target and attempts < target * 50:
                attempts += 1
                candidate = tuple(random.sample(base, len(base)))
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
        if not self.custom_sequence and not self.brute_force and not self.pareto_front:
            if self.operation_based_jssp:
                job_sequence, obj_fun = genetic_algorithm_multiset(self.operation_chromosome, population_size, elite, mutation_rate, generations, self.target_function, True)
            else:
                job_sequence, obj_fun = genetic_algorithm(self.num_jobs, population_size, elite, mutation_rate, generations, self.target_function, True)
            return job_sequence, self.schedule_jobs(job_sequence), obj_fun
        if not self.custom_sequence and not self.brute_force and self.pareto_front:
            if self.operation_based_jssp:
                return nsga3_algorithm_multiset(population_size=population_size, base_sequence=self.operation_chromosome, list_of_functions=self.lst_func, generations=generations, mutation_rate=mutation_rate, verbose=True)
            return nsga3_algorithm(population_size=population_size, jobs=self.num_jobs, list_of_functions=self.lst_func, generations=generations, mutation_rate=mutation_rate, verbose=True)
        if not self.custom_sequence and self.brute_force and not self.pareto_front:
            job_sequence, obj_fun = self.brute_force_search()
            return job_sequence, self.schedule_jobs(job_sequence), obj_fun
        if not self.custom_sequence and self.brute_force and self.pareto_front:
            return self.brute_force_search_p()
        job_sequence = self.custom_sequence
        obj_fun = self.target_function(job_sequence)
        return job_sequence, self.schedule_jobs(job_sequence), obj_fun
