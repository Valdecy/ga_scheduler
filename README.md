# GA Scheduler

## Overview

**GA Scheduler** is a Python library for modeling, optimizing, and visualizing machine scheduling problems using genetic algorithms and multiobjective evolutionary optimization. It supports a broad range of scheduling environments, including **single-machine**, **parallel-machine**, **flow shop**, **flexible flow shop / hybrid flow shop**, **job shop**, **flexible job shop**, and **alternative-machine routing problems**. The library can optimize several performance criteria, including **makespan**, **weighted tardiness**, **sequence-dependent setup time**, **sequence-dependent setup waste**, **total completion time**, and **number of late jobs**. It also provides **interactive Gantt chart visualizations** to help users inspect, interpret, and communicate scheduling solutions.


##  Citation

Yigit F.; Basilio M.P; Pereira V. (2024). A Hybrid Approach for the Multi-Criteria-Based Optimization of Sequence-Dependent Setup-Based Flow Shop Scheduling. Mathematics. 12(13):2007. doi: https://doi.org/10.3390/math12132007

## Features

- **Scheduling Machine Environments**: Supports single-machine, parallel-machine, flow shop, flexible flow shop / hybrid flow shop, job shop, flexible job shop, and alternative-machine routing problems.
- **Many-Objective / Multi-Objective Optimization**: Supports optimization for multiple objectives including makespan, weighted tardiness, sequence-dependent setup time, sequence-dependent setup waste, total completion time, and number of late jobs.
- **Genetic Algorithm Integration**: Utilizes a Genetic Algorithm (GA) to efficiently explore the solution space and find optimal or near-optimal job sequences.
- **Brute Force Search**: For small problem instances, brute force search can be used to find the optimal job sequence. It is intentionally disabled for flexible sequences because the search space combines operation-dispatch permutations and machine-choice combinations.
- **Customizability**: Allows customization of job sequences, machine alternatives, setup times, setup waste, due dates, job weights, objective weights, and other scheduling parameters.
- **Sequence-Dependent Job Setup Times**: Supports setup times between jobs through a `setup_time_matrix`, where the setup time depends on the previously processed job and the next job.
- **Sequence-Dependent Setup Waste**: Supports setup waste between jobs through a `setup_waste_matrix`, allowing the model to penalize material loss, cleaning waste, scrap, or other changeover-related waste.
- **Machine-to-Machine Setup Times**: Supports setup or transfer times between machines through a `machine_setup_time_matrix`. This is useful when moving a job from one machine to another requires transportation, preparation, tooling change, calibration, cleaning, or other machine-transition activities.
- **Machine Blocking Constraints**: Supports machine blocking groups through `machine_block_groups`. Machines in the same blocking group cannot operate simultaneously. This is useful for modeling shared operators, shared tools, shared cranes, shared physical space, shared power sources, or other shared resources.
- **Machine Maintenance Constraints**: Supports predefined machine maintenance requirements through `machine_maintenance`. Maintenance activities can be assigned to specific machines with an earliest start time, duration, and optional label.
- **Flexible Maintenance Scheduling**: Maintenance does not need to start exactly at its earliest time. The scheduler can delay maintenance to the next feasible time if the machine is busy, as long as the maintenance requirement is respected.
- **Conditional Maintenance Visibility**: If a maintenance activity is only required after a certain time and the final schedule finishes before that point, the maintenance may be omitted from the Gantt chart while still being reported in the maintenance summary.
- **Alternative Machine Routing**: Each operation can have multiple possible machines with different processing times, allowing the GA to optimize both the job sequence and the machine choices.
- **Custom Sequence Evaluation**: Allows the user to provide a custom job sequence instead of relying only on the GA-generated sequence. This is useful for testing, benchmarking, validating heuristics, or comparing human-designed schedules.
- **Objective Weighting**: Allows the user to assign different weights to each objective, making it possible to emphasize makespan, tardiness, setup time, waste, completion time, or late jobs according to the decision-maker's priorities.
- **Pareto Front Generation**: When enabled, the scheduler can return a set of non-dominated solutions instead of a single weighted solution, supporting trade-off analysis among conflicting objectives.
- **Visualization**: Generates Gantt charts to visualize the scheduling of jobs, setup times, machine-to-machine setups, maintenance periods, and machine usage across the planning horizon.

## Usage

1. Install

```bash
pip install ga_scheduler

```

2. Try it in **Colab**:

a) Multiobjective - Weighted 

- Single Machine Scheduling     - Brute Force ( [ Colab Demo ](https://colab.research.google.com/drive/1f8j9R3vClF9kmJGrS8ODGDCy_JWnh7lL?usp=sharing)) 
- Single Machine Scheduling     - Genetic Algorithm ( [ Colab Demo ](https://colab.research.google.com/drive/1EevgVgIl0g9ELvUdMKRy38hKItrrPwI7?usp=sharing)) 
<!-- -->
- Parallel Machines Scheduling  - Brute Force ( [ Colab Demo ](https://colab.research.google.com/drive/1qQmvkkNliPAVlTk2ShvM0Di9JAKzkqmL?usp=sharing)) 
- Parallel Machines Scheduling  - Genetic Algorithm ( [ Colab Demo ](https://colab.research.google.com/drive/1yyfWNei8JNWpsOuy3UBB-pm0MIW5uxQO?usp=sharing)) 
<!-- -->
- Flow Shop Machines Scheduling - Brute Force ( [ Colab Demo ](https://colab.research.google.com/drive/1Fiq5JB9jNXjc_HSDUhEWvujILse2QdfD?usp=sharing)) 
- Flow Shop Machines Scheduling - Genetic Algorithm ( [ Colab Demo ](https://colab.research.google.com/drive/1CqcoXxyBypo_maEE7_-55s64_dsnJ42w?usp=sharing))
<!-- -->
- Job Shop Machines Scheduling  - Brute Force ( [ Colab Demo ](https://colab.research.google.com/drive/1MCo3msB8cVbjg-fT9FV5QBmTKFM6km3a?usp=sharing)) 
- Job Shop Machines Scheduling  - Genetic Algorithm ( [ Colab Demo ](https://colab.research.google.com/drive/1etJc3z0JMVx4FQBLZCZbCgtQsyt1pjQJ?usp=sharing)) 
 
b) Multiobjective - Pareto Front

- Single Machine Scheduling     - Brute Force ( [ Colab Demo ](https://colab.research.google.com/drive/1VOufI5VzOgzheTjODWNxKCWt3bPOfEtt?usp=sharing)) 
- Single Machine Scheduling     - NSGA3 ( [ Colab Demo ](https://colab.research.google.com/drive/1ex07yTxUPzZiGomtR1PepKYyJhzI7vdF?usp=sharing)) 
<!-- -->
- Parallel Machines Scheduling  - Brute Force ( [ Colab Demo ](https://colab.research.google.com/drive/1xPqmIEaIwmYrJNIFb9BSSM1XEqFpyMKL?usp=sharing)) 
- Parallel Machines Scheduling  - NSGA3 ( [ Colab Demo ](https://colab.research.google.com/drive/140ZoIMwzQizsRz6TefsbxWG9d4mBatfC?usp=sharing)) 
<!-- -->
- Flow Shop Machines Scheduling - Brute Force ( [ Colab Demo ](https://colab.research.google.com/drive/1Vom9tdoDzX0D9qdCe3wBErUVtN0nGko3?usp=sharing)) 
- Flow Shop Machines Scheduling - NSGA3 ( [ Colab Demo ](https://colab.research.google.com/drive/1DXeifEbk2XeQocG81WRGKtVDFoxdB6RW?usp=sharing))
<!-- -->
- Job Shop Machines Scheduling  - Brute Force ( [ Colab Demo ](https://colab.research.google.com/drive/13LK2Ckc8XoftJosNQn7f7bNa0U56Zy4p?usp=sharing)) 
- Job Shop Machines Scheduling  - NSGA3 ( [ Colab Demo ](https://colab.research.google.com/drive/1PYV0afWMVYREwtRQRbNm1u8McDczAb0k?usp=sharing)) 

c) Flexible Sequences (GA = Enabled; NSGA3 = Enabled; Brute Force = Disabled)

- Flexible Flow Shop Machines Scheduling  - ( [ Colab Demo ](https://colab.research.google.com/drive/1WdFqNmNelAp1p97m79m88e5Ozm6qOJGK?usp=sharing)) 
- Flexible Job Shop Machines Scheduling   - ( [ Colab Demo ](https://colab.research.google.com/drive/1RJLfAxkfGIMX9ct2Md08GIHGuaEx-p59?usp=sharing)) 
- Alternative-Machine Routing Scheduling  - ( [ Colab Demo ](https://colab.research.google.com/drive/1WiAcadhp68-190hCljU9C7fLnUIYqPzN?usp=sharing)) 

d) Machine Setup Time, Machine Block Groups, Machine Maintenance

- Machine Setup Time, Block Groups & Maintenance - ( [ Colab Demo ](https://colab.research.google.com/drive/17Xx26PsbE2_Y9TfIGIYNITUWnvpoW2IK?usp=sharing)) 
