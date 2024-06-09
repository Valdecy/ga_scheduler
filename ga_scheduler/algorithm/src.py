############################################################################

# Created by: Prof. Valdecy Pereira, D.Sc.
# UFF - Universidade Federal Fluminense (Brazil)
# email:  valdecy.pereira@gmail.com
# GA Scheduler

# Citation: 
# PEREIRA, V. (2024). Project: GA Scheduler, GitHub repository: <https://github.com/Valdecy/GA_Scheduler>

############################################################################

# Required Libraries
import itertools
import math
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
plt.style.use('ggplot')
import numpy as np
import random

from ga_scheduler.util.ga    import genetic_algorithm
from ga_scheduler.util.ecmoa import elitist_combinatorial_multiobjective_optimization_algorithm, selection_leaders

############################################################################

# GA Scheduler Class
class load_ga_scheduler():
    def __init__(self, sequences = [], due_dates = [], setup_time_matrix = [], setup_waste_matrix = [], z_permutations = 100, job_weights = [], obj_makespan = True, obj_total_compl_time = True, obj_total_late_jobs = True, obj_max_w_tardiness = True, obj_total_waste = True, obj_setup = True, w_obj_makespan = 1, w_obj_total_compl_time = 1, w_obj_total_late_jobs = 1, w_obj_max_w_tardiness = 1, w_obj_total_waste = 1, w_obj_setup = 1, parallel_machines = False, brute_force = False, pareto_front = False, custom_sequence = []): 
      self.obj_1                          = obj_makespan
      self.obj_2                          = obj_max_w_tardiness
      self.obj_3                          = obj_total_waste
      self.obj_4                          = obj_setup
      self.obj_5                          = obj_total_compl_time
      self.obj_6                          = obj_total_late_jobs
      self.lst_func                       = []
      self.z_std                          = []
      self.z_permutations                 = z_permutations
      self.sequences                      = sequences                  # Job Shop Scheduling Input: Mandatory
      self.due_dates                      = due_dates                  # Job Shop Scheduling Input: Only Relevant if Due Date    is a measure
      self.setup_time_matrix              = setup_time_matrix          # Job Shop Scheduling Input: Only Relevant if Setup Time  is a measure
      self.setup_waste_matrix             = setup_waste_matrix         # Job Shop Scheduling Input: Only Relevant if Setup Waste is a measure
      self.parallel                       = parallel_machines
      self.brute_force                    = brute_force
      self.custom_sequence                = custom_sequence
      self.ecmo                           = pareto_front
      self.machine_sequences, self.matrix = self.sequence_inputs()
      self.num_jobs                       = len(self.sequences)
      self.num_machines                   = self.matrix.shape[1]
      self.job_weights                    = [1 for i in range(0, self.num_jobs)]
      self.objectives_weights             = [1, 1, 1, 1, 1, 1] 
      for i in range(0, min(len(job_weights), self.num_jobs)):
          self.job_weights[i] = job_weights[i] 
      if (self.obj_1 == True):
          self.objectives_weights[0] = w_obj_makespan
      else:
          self.objectives_weights[0] = 0
      if (self.obj_2 == True):
          self.objectives_weights[1] = w_obj_max_w_tardiness
      else:
          self.objectives_weights[1] = 0
      if (self.obj_3 == True):
          self.objectives_weights[2] = w_obj_total_waste
      else:
          self.objectives_weights[2] = 0
      if (self.obj_4 == True):
          self.objectives_weights[3] = w_obj_setup
      else:
          self.objectives_weights[3] = 0
      if (self.obj_5 == True):
          self.objectives_weights[4] = w_obj_total_compl_time
      else:
          self.objectives_weights[4] = 0
      if (self.obj_6 == True):
          self.objectives_weights[5] = w_obj_total_late_jobs
      else:
          self.objectives_weights[5] = 0
      if (self.objectives_weights.count(0) == len(self.objectives_weights) -1):
        self.z_mean, self.z_std = [0]*len(self.objectives_weights), [1]*len(self.objectives_weights) 
      else:
        self.z_mean, self.z_std = self.obj_z_search()
      self.objective_functions()
    ###############################################################################   
    
    # Create Instances
    def generate_sequences(self, num_jobs, num_machines, use_all_machines = False, parallel_machines = False, parallel_same_time = False, flow_shop = False, bounds = [1, 10], seed = None):
        
        ################################################
        def process_lists(lists, pm = False, tm = False):
            processed_lists = []
            for lst in lists:
                if (pm):
                    lst = sorted(lst, key = lambda x: x[0])
                if (tm):
                    first_tuple_second_element = lst[0][1]
                    lst = [(t[0], first_tuple_second_element) for t in lst]
                processed_lists.append(lst)
            return processed_lists
        ################################################
        
        if (seed is not None):
            random.seed(seed)
        sequences = []
        if (flow_shop == True):
            base_sequence = random.sample(range(0, num_machines), num_machines)
        for job in range(0, num_jobs):
            job_sequence = []
            if (flow_shop == True):
                for machine in base_sequence:
                    process_time = random.randint(bounds[0], bounds[1])
                    job_sequence.append((machine, process_time))
            else:
                if (use_all_machines):
                    machine_sequence = random.sample(range(0, num_machines), num_machines)
                else:
                    machine_sequence = random.sample(range(0, num_machines), random.randint(1, num_machines))
                for machine in machine_sequence:
                    process_time = random.randint(bounds[0], bounds[1])
                    job_sequence.append((machine, process_time))
            sequences.append(job_sequence)
        if (parallel_machines == True or parallel_same_time == True):
            sequences = process_lists(lists = sequences, pm = parallel_machines, tm = parallel_same_time)
        return sequences
    
    # Schedule
    def schedule_jobs(self, permutation):
        total_length      = np.sum(self.matrix)
        schedule          = [['' for _ in range(0, total_length)] for _ in range(0, self.num_machines)]
        machine_end_times = [0] * self.num_machines
        job_end_times     = [0] * self.num_jobs
        if (self.parallel == False):
            for job_id in permutation:
                operations = self.machine_sequences[job_id]
                for op_index, machine in enumerate(operations):
                    time_required = self.matrix[job_id, machine]
                    start_time    = job_end_times[job_id]
                    while any(schedule[machine][start_time:start_time + time_required]):
                        start_time = start_time + 1
                    end_time = start_time + time_required
                    for t in range(start_time, end_time):
                        schedule[machine][t] = f"j{job_id}"
                    job_end_times[job_id]      = end_time
                    machine_end_times[machine] = end_time
        else:        
            for job_id in permutation:
                earliest_start_time = float('inf')
                best_machine        = None
                for machine, time_required in self.sequences[job_id]:
                    start_time = machine_end_times[machine]
                    while any(schedule[machine][start_time:start_time + time_required]):
                        start_time = start_time + 1
                    if (start_time < earliest_start_time):
                        earliest_start_time = start_time
                        best_machine        = machine
                time_required = self.matrix[job_id, best_machine]
                end_time      = earliest_start_time + time_required
                for t in range(earliest_start_time, end_time):
                    schedule[best_machine][t]   = f"j{job_id}"
                machine_end_times[best_machine] = end_time
                job_end_times[job_id]           = end_time
        max_time = max(len(row) for row in schedule)
        for col in range(max_time - 1, -1, -1):
            if (all(row[col] == '' for row in schedule)):
                for row in schedule:
                    row.pop()
            else:
                break
        return np.array(schedule)
    
    ###############################################################################
    
    # Plot
    def create_gantt_chart(self, schedule_matrix, size_x = 12, size_y = 8):
        non_empty_rows = [i for i in range(schedule_matrix.shape[0]) if not np.all(schedule_matrix[i] == '')]
        schedule       = schedule_matrix[non_empty_rows, :]
        num_machines   = schedule.shape[0]
        total_length   = schedule.shape[1]
        fig, ax        = plt.subplots(figsize = (size_x, size_y))
        job_color_dict = {}
        idle_patches   = []
        job_colors     = [
                            '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', 
                            '#7f7f7f', '#bcbd22', '#17becf', '#bf77f6', '#ff9408', '#d1ffbd', '#c85a53', 
                            '#3a18b1', '#ff796c', '#04d8b2', '#ffb07c', '#aaa662', '#0485d1', '#fffe7a', 
                            '#b0dd16', '#6f7be3', '#12e193', '#82cafc', '#ac9362', '#f8481c', '#c292a1', 
                            '#c0fa8b', '#ca7b80', '#f4d054', '#fbdd7e', '#ffff7e', '#cd7584', '#f9bc08', 
                            '#c7c10c'
                         ]
        for machine in range(0, num_machines):
            for time in range(0, total_length):
                job = schedule[machine][time]
                if (job):
                    if (job not in job_color_dict):
                        job_index           = int(job[1:])  
                        job_color_dict[job] = job_colors[job_index % len(job_colors)]
                    color = job_color_dict[job]
                    ax.add_patch(mpatches.Rectangle((time, num_machines - machine - 1), 1, 1, edgecolor = 'black', facecolor = color))
                    ax.text(time + 0.5, num_machines - machine - 0.5, job, ha = 'center', va = 'center', color = 'black', fontsize = 8, weight = 'bold')
                else:
                    left_hatch = mpatches.Rectangle((time, num_machines - machine - 1), 1, 1, edgecolor = 'black', facecolor = 'none', hatch = '//')
                    right_hatch = mpatches.Rectangle((time, num_machines - machine - 1), 1, 1, edgecolor = 'black', facecolor = 'none', hatch = '\\')
                    ax.add_patch(left_hatch)
                    ax.add_patch(right_hatch)
                    idle_patches.append((time, machine, left_hatch, right_hatch))
        for machine in range(0, num_machines):
            for time in range(total_length - 1, -1, -1):
                if (schedule[machine][time] == ''):
                    for patch in idle_patches:
                        if (patch[0] == time and patch[1] == machine):
                            patch[2].remove()
                            patch[3].remove()
                else:
                    break
        ax.set_xlim(0, total_length)
        ax.set_ylim(0, num_machines)
        ax.set_xticks(np.arange(0, total_length + 1, 1))
        ax.set_xticklabels(np.arange(0, total_length + 1, 1))
        ax.set_yticks(np.arange(0, num_machines, 1))
        ax.set_yticklabels([])
        for i in range(0, num_machines):
            ax.text(-0.5, num_machines - i - 0.5, f'Machine {i}', va = 'center', ha = 'right', fontsize = 10)
        ax.set_xlabel('Time')
        ax.set_title('Gantt Chart')
        ax.grid(True, linestyle = '--', alpha = 0.7)
        plt.show()
        
    ###############################################################################
    
    # Objectives
    def calculate_makespan(self, schedule_matrix):
        total_length = schedule_matrix.shape[1]
        makespan     = 0
        for machine in range(0, self.num_machines):
            for time in range(total_length - 1, -1, -1):
                if (schedule_matrix[machine][time] != ''):
                    makespan = max(makespan, time + 1)
                    break
        return makespan

    def calculate_makespan_p(self, permutation):
        schedule_matrix = self.schedule_jobs(permutation)
        total_length    = schedule_matrix.shape[1]
        makespan        = 0
        for machine in range(0, self.num_machines):
            for time in range(total_length - 1, -1, -1):
                if (schedule_matrix[machine][time] != ''):
                    makespan = max(makespan, time + 1)
                    break
        return makespan
    
    def calculate_max_weighted_tardiness(self, schedule_matrix):
        completion_times = {f'j{job}': 0 for job in range(0, self.num_jobs)}
        total_length     = schedule_matrix.shape[1]
        for machine in range(0, self.num_machines):
            for time in range(0, total_length):
                job = schedule_matrix[machine][time]
                if (job):
                    completion_times[job] = time + 1
        max_weighted_tardiness = 0
        for job in range(0, self.num_jobs):
            tardiness              = max(0, completion_times[f'j{job}'] - self.due_dates[job])
            weighted_tardiness     = self.job_weights[job] * tardiness
            max_weighted_tardiness = max(max_weighted_tardiness, weighted_tardiness)
        return max_weighted_tardiness

    def calculate_max_weighted_tardiness_p(self, permutation):
        schedule_matrix  = self.schedule_jobs(permutation)
        completion_times = {f'j{job}': 0 for job in range(0, self.num_jobs)}
        total_length     = schedule_matrix.shape[1]
        for machine in range(0, self.num_machines):
            for time in range(0, total_length):
                job = schedule_matrix[machine][time]
                if (job):
                    completion_times[job] = time + 1
        max_weighted_tardiness = 0
        for job in range(0, self.num_jobs):
            tardiness              = max(0, completion_times[f'j{job}'] - self.due_dates[job])
            weighted_tardiness     = self.job_weights[job] * tardiness
            max_weighted_tardiness = max(max_weighted_tardiness, weighted_tardiness)
        return max_weighted_tardiness

    def calculate_total_completion_time(self, schedule_matrix):
        completion_times = {f'j{job}': 0 for job in range(0, self.num_jobs)}
        total_length     = schedule_matrix.shape[1]
        for machine in range(0, self.num_machines):
            for time in range(0, total_length):
                job = schedule_matrix[machine][time]
                if (job):  
                    completion_times[job] = time + 1  
        total_completion_time = sum(completion_times[f'j{job}'] for job in range(0, self.num_jobs))
        return total_completion_time  

    def calculate_total_completion_time_p(self, permutation):
        schedule_matrix  = self.schedule_jobs(permutation)
        completion_times = {f'j{job}': 0 for job in range(0, self.num_jobs)}
        total_length     = schedule_matrix.shape[1]
        for machine in range(0, self.num_machines):
            for time in range(0, total_length):
                job = schedule_matrix[machine][time]
                if (job):  
                    completion_times[job] = time + 1  
        total_completion_time = sum(completion_times[f'j{job}'] for job in range(0, self.num_jobs))
        return total_completion_time  

    def calculate_total_late_jobs(self, schedule_matrix):
        completion_times = {f'j{job}': 0 for job in range(0, self.num_jobs)}
        total_length     = schedule_matrix.shape[1]
        for machine in range(0, self.num_machines):
            for time in range(0, total_length):
                job = schedule_matrix[machine][time]
                if (job): 
                    completion_times[job] = time + 1
        total_late_jobs = 0
        for job in range(0, self.num_jobs):
            if (completion_times[f'j{job}'] > self.due_dates[job]):  
                total_late_jobs = total_late_jobs + 1  
        return total_late_jobs  
    
    def calculate_total_late_jobs_p(self, permutation):
        schedule_matrix  = self.schedule_jobs(permutation)
        completion_times = {f'j{job}': 0 for job in range(0, self.num_jobs)}
        total_length     = schedule_matrix.shape[1]
        for machine in range(0, self.num_machines):
            for time in range(0, total_length):
                job = schedule_matrix[machine][time]
                if (job): 
                    completion_times[job] = time + 1
        total_late_jobs = 0
        for job in range(0, self.num_jobs):
            if (completion_times[f'j{job}'] > self.due_dates[job]):  
                total_late_jobs = total_late_jobs + 1  
        return total_late_jobs
    
    def calculate_total_waste(self, permutation):
        total_waste = 0
        for i in range(0, len(permutation) - 1):
            total_waste = total_waste + self.setup_waste_matrix[permutation[i]][permutation[i+1]]
        return total_waste

    def calculate_idle_times(self, schedule_matrix):
        total_idle_time = 0
        for row in schedule_matrix:
            row_idle_time  = 0
            last_job_index = None
            for i in range(len(row) - 1, -1, -1):
                if (row[i] != ''):
                    last_job_index = i
                    break
            if (last_job_index is not None):
                for i in range(0, last_job_index):
                    if (row[i] == ''):
                        row_idle_time =  row_idle_time  + 1
            total_idle_time = total_idle_time + row_idle_time
        return total_idle_time
    
    def calculate_idle_times_p(self, permutation):
        schedule_matrix = self.schedule_jobs(permutation)
        total_idle_time = 0
        for row in schedule_matrix:
            row_idle_time  = 0
            last_job_index = None
            for i in range(len(row) - 1, -1, -1):
                if (row[i] != ''):
                    last_job_index = i
                    break
            if (last_job_index is not None):
                for i in range(0, last_job_index):
                    if (row[i] == ''):
                        row_idle_time =  row_idle_time  + 1
            total_idle_time = total_idle_time + row_idle_time
        return total_idle_time
    
    def calculate_total_setup_time(self, permutation):
        total_setup_time = 0
        for i in range(0, len(permutation) - 1):
            total_setup_time = total_setup_time + self.setup_time_matrix[permutation[i]][permutation[i+1]]
        return total_setup_time
    
    ###############################################################################
    
    # Inputs
    def sequence_inputs(self):
        num_jobs     = len(self.sequences)
        num_machines = max(max(machine for machine, _ in job) for job in self.sequences) + 1
        matrix       = np.zeros((num_jobs, num_machines), dtype = int)
        for job_id, job in enumerate(self.sequences):
            for machine, time in job:
                matrix[job_id, machine] = time
        machine_sequences = []
        for job_id, job in enumerate(self.sequences):
            machine_sequence = [machine for machine, _ in job]
            machine_sequences.append(machine_sequence)
        return machine_sequences, matrix
    
    ###############################################################################
    
    # Find Solution
    def brute_force_search(self):
        job_ids                 = list(range(0, len(self.sequences)))
        best_sequence           = None
        minimal_objective_value = float('inf')
        count                   = 1
        total                   = math.factorial(len(job_ids))
        print('')
        print('Verifying', total, 'solutions')
        print('')
        for permutation in itertools.permutations(job_ids):
            objective_value = self.target_function(permutation)
            count           = count + 1
            searched_perc   = round(count / total * 100, 4)
            if (objective_value < minimal_objective_value):
                minimal_objective_value = objective_value
                best_sequence           = permutation
                print(f"Searched Space: {searched_perc:.2f}%; Best Sequence: {best_sequence}; Obj. Function: {minimal_objective_value:.4f}")
        print('')
        print('Brute Force Search Complete!')
        print('')
        print(f"Best Sequence: {best_sequence}; Obj. Function: {minimal_objective_value:.4f}")
        self.best_sequence = best_sequence
        return best_sequence, minimal_objective_value
    
    def brute_force_search_p(self):
        merged  = [] 
        job_ids = list(range(0, len(self.sequences)))
        count   = 1
        total   = math.factorial(len(job_ids))
        print('')
        print('Verifying', total, 'solutions')
        print('')
        for permutation in itertools.permutations(job_ids):
            m = []
            p = [item for item in permutation]
            m.append(p)
            for k in range(0, len(self.lst_func)):
                m.append(self.lst_func[k](p))
            merged.append(m)
            count         = count + 1
            searched_perc = round(count / total * 100, 4)
            if (count % (int(total/10)) == 0 or count == total):
                print(f"Searched Space: {searched_perc:.2f}%")
        print('Brute Force Search Complete!')
        leaders = selection_leaders(total, len(self.lst_func), merged)
        return leaders
    
    def obj_z_search(self):
        
        ################################################
        def unique_permutations(job_ids, num_permutations):
            seen = set()
            while (len(seen) < num_permutations):
                perm = tuple(random.sample(job_ids, len(job_ids)))
                seen.add(perm)
            return list(seen)
        ################################################
        
        job_ids              = list(range(0, len(self.sequences)))
        num_permutations     = min(self.z_permutations, math.factorial(len(job_ids)))
        sampled_permutations = unique_permutations(job_ids, num_permutations)
        obj_1_lst            = []
        obj_2_lst            = []
        obj_3_lst            = []
        obj_4_lst            = []
        obj_5_lst            = []
        obj_6_lst            = []
        z_mean               = []
        z_std                = []
        for permutation in sampled_permutations:
            schedule_matrix = self.schedule_jobs(permutation)
            if (self.obj_1 == True):
                makespan = self.calculate_makespan(schedule_matrix)
                obj_1_lst.append(makespan)
            if (self.obj_2 == True):
                max_weighted_tardiness = self.calculate_max_weighted_tardiness(schedule_matrix)
                obj_2_lst.append(max_weighted_tardiness)
            if (self.obj_3 == True):
                total_waste = self.calculate_total_waste(permutation)
                obj_3_lst.append(total_waste)
            if (self.obj_4 == True):
                total_setup_time = self.calculate_total_setup_time(permutation)
                obj_4_lst.append(total_setup_time)
            if (self.obj_5 == True):
                total_compl_time = self.calculate_total_completion_time(schedule_matrix)
                obj_5_lst.append(total_compl_time)
            if (self.obj_6 == True):
                total_late_jobs = self.calculate_total_late_jobs(schedule_matrix)
                obj_6_lst.append(total_late_jobs)
        if (obj_1_lst):
            z_mean.append(np.mean(obj_1_lst))
            #z_std.append(np.std(obj_1_lst, ddof = 1) / np.sqrt(len(obj_1_lst)))
            z_std.append(np.std(obj_1_lst, ddof = 1))
        else:
            z_mean.append(None)
            z_std.append(None)
        if (obj_2_lst):
            z_mean.append(np.mean(obj_2_lst))
            #z_std.append(np.std(obj_2_lst, ddof = 1) / np.sqrt(len(obj_2_lst)))
            z_std.append(np.std(obj_2_lst, ddof = 1))
        else:
            z_mean.append(None)
            z_std.append(None)
        if (obj_3_lst):
            z_mean.append(np.mean(obj_3_lst))
            #z_std.append(np.std(obj_3_lst, ddof = 1) / np.sqrt(len(obj_3_lst)))
            z_std.append(np.std(obj_3_lst, ddof = 1))
        else:
            z_mean.append(None)
            z_std.append(None)
        if (obj_4_lst):
            z_mean.append(np.mean(obj_4_lst))
            #z_std.append(np.std(obj_4_lst, ddof = 1) / np.sqrt(len(obj_4_lst)))
            z_std.append(np.std(obj_4_lst, ddof = 1))
        else:
            z_mean.append(None)
            z_std.append(None)
        if (obj_5_lst):
            z_mean.append(np.mean(obj_5_lst))
            #z_std.append(np.std(obj_5_lst, ddof = 1) / np.sqrt(len(obj_5_lst)))
            z_std.append(np.std(obj_5_lst, ddof = 1))
        else:
            z_mean.append(None)
            z_std.append(None)
        if (obj_6_lst):
            z_mean.append(np.mean(obj_6_lst))
            #z_std.append(np.std(obj_6_lst, ddof = 1) / np.sqrt(len(obj_6_lst)))
            z_std.append(np.std(obj_6_lst, ddof = 1))
        else:
            z_mean.append(None)
            z_std.append(None)
        return z_mean, z_std
    
    def target_function(self, permutation): 
        schedule_matrix = self.schedule_jobs(permutation)
        objective_value = 0.0 / 1.0
        if (self.objectives_weights[0] != 0):
            makespan                 = self.calculate_makespan(schedule_matrix)
            z_makespan               = ( (makespan - self.z_mean[0]) / (self.z_std[0] + 1e-14)) + 9
            objective_value          = objective_value + self.objectives_weights[0] * z_makespan
        if (self.objectives_weights[1] != 0):
            max_weighted_tardiness   = self.calculate_max_weighted_tardiness(schedule_matrix)
            z_max_weighted_tardiness = ( (max_weighted_tardiness - self.z_mean[1]) /  (self.z_std[1] + 1e-14) ) + 9
            objective_value          = objective_value + self.objectives_weights[1] * z_max_weighted_tardiness
        if (self.objectives_weights[2] != 0):
            total_waste              = self.calculate_total_waste(permutation)
            z_total_waste            = ( (total_waste - self.z_mean[2]) /  (self.z_std[2] + 1e-14) ) + 9
            objective_value          = objective_value + self.objectives_weights[2] * z_total_waste 
        if (self.objectives_weights[3] != 0):
            total_setup_time         = self.calculate_total_setup_time(permutation)
            z_total_setup_time       = ( (total_setup_time - self.z_mean[3]) /  (self.z_std[3] + 1e-14) ) + 9
            objective_value          = objective_value + self.objectives_weights[3] * z_total_setup_time 
        if (self.objectives_weights[4] != 0):
            total_compl_time         = self.calculate_total_completion_time(schedule_matrix)
            z_total_compl_time       = ( (total_compl_time - self.z_mean[4]) /  (self.z_std[4] + 1e-14) ) + 9
            objective_value          = objective_value + self.objectives_weights[4] * z_total_compl_time 
        if (self.objectives_weights[5] != 0):
            total_late_jobs          = self.calculate_total_late_jobs(schedule_matrix)
            z_total_late_jobs        = ( (total_late_jobs - self.z_mean[5]) /  (self.z_std[5] + 1e-14) ) + 9
            objective_value          = objective_value + self.objectives_weights[5] * z_total_late_jobs 
        return objective_value
    
    def objective_functions(self):
        if (self.obj_1 == True):
            self.lst_func.append(self.calculate_makespan_p)
        if (self.obj_2 == True):
            self.lst_func.append(self.calculate_max_weighted_tardiness_p)
        if (self.obj_3 == True): 
            self.lst_func.append(self.calculate_total_waste) 
        if (self.obj_4 == True):
            self.lst_func.append(self.calculate_total_setup_time)
        if (self.obj_5 == True):
            self.lst_func.append(self.calculate_total_completion_time_p)
        if (self.obj_6 == True):
            self.lst_func.append(self.calculate_total_late_jobs_p)
        return 
    
    ###############################################################################
    
    # GA Scheduler
    def run_ga_scheduler(self, population_size = 5, elite = 1, mutation_rate = 0.10, generations = 100, k = 4):
        if (self.brute_force == False and len(self.custom_sequence) == 0 and self.ecmo == False):
            job_sequence, obj_fun = genetic_algorithm(self.num_jobs, population_size, elite, mutation_rate, generations, self.target_function, True)
            schedule_matrix       = self.schedule_jobs(job_sequence) 
            return job_sequence, schedule_matrix, obj_fun
        elif (self.brute_force == False and len(self.custom_sequence) == 0 and self.ecmo == True):
            leaders = elitist_combinatorial_multiobjective_optimization_algorithm(population_size, self.num_jobs, self.lst_func, generations, k, True)
            return leaders
        if (self.brute_force == True and len(self.custom_sequence) == 0 and self.ecmo == False):
            job_sequence, obj_fun = self.brute_force_search()
            schedule_matrix       = self.schedule_jobs(job_sequence)
            return job_sequence, schedule_matrix, obj_fun
        elif (self.brute_force == True and len(self.custom_sequence) == 0 and self.ecmo == True):
            leaders = self.brute_force_search_p()
            return leaders
        if (len(self.custom_sequence) != 0):
            job_sequence    = self.custom_sequence
            schedule_matrix = self.schedule_jobs(job_sequence) 
            obj_fun         = self.target_function(self.custom_sequence)
            return job_sequence, schedule_matrix, obj_fun
    
############################################################################
