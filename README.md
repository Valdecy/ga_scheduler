# GA Scheduler

## Overview

**GA Scheduler** is a Python library for modeling, optimizing, and visualizing machine scheduling problems using genetic algorithms and multiobjective evolutionary optimization. It supports a broad range of scheduling environments, including **single-machine**, **parallel-machine**, **flow shop**, **flexible flow shop / hybrid flow shop**, **job shop**, **flexible job shop**, and **alternative-machine routing problems**. The library can optimize several performance criteria, including **makespan**, **weighted tardiness**, **sequence-dependent setup time**, **sequence-dependent setup waste**, **total completion time**, and **number of late jobs**. It also provides **interactive Gantt chart visualizations** to help users inspect, interpret, and communicate scheduling solutions.



##  Citation

Yigit F.; Basilio M.P; Pereira V. (2024). A Hybrid Approach for the Multi-Criteria-Based Optimization of Sequence-Dependent Setup-Based Flow Shop Scheduling. Mathematics. 12(13):2007. doi: https://doi.org/10.3390/math12132007

## Features

- **Scheduling Machine Environments**: Supports single-machine, parallel-machine, flow shop, flexible flow shop / hybrid flow shop, job shop, flexible job shop, and alternative-machine routing problems.
- **Multi-Objective Optimization**: Supports optimization for multiple objectives including makespan, weighted tardiness, sequence-dependent setup time, sequence-dependent setup waste, total completion time, and number of late jobs.
- **Genetic Algorithm Integration**: Utilizes a GA to efficiently explore the solution space and find optimal or near-optimal job sequences.
- **Many or Multiobjective Algorithm Integration**: Alternatively, the multiobjective problem can be solved using the NSGA3, which returns the Pareto Front as the solution.
- **Brute Force**: For small problem instances, the brute force search can be used to find the optimal job sequence. It is intentionally disabled for flexible sequences because the search space combines operation-dispatch permutations and machine-choice combinations.
- **Customizability**: Allows customization of job sequences, setup times, due dates, and more.
- **Visualization**: Generates Gantt charts to visualize the scheduling of jobs across machines.

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
