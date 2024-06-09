# GA Scheduler

## Overview

GA Scheduler is an advanced scheduling tool that leverages genetic algorithms to optimize **single**, **parallel**, **flow shop**, and **job shop** machines scheduling problems. Multiple objectives can be addressed such as makespan, weighted tardiness, total waste changeover between jobs, total setup times changeover between jobs, total completion time and total of late jobs. Additionally, the library provides a comprehensive way to visualize scheduling results through Gantt charts.

## Features

- **Scheduling Machine Environments**: Supports single machine, parallel machines, flow shop, and job shop scheduling problems.
- **Multi-Objective Optimization**: Supports optimization for multiple objectives including makespan, weighted tardiness, total waste changeover between jobs, setup times changeover between jobs, total completion time and total of late jobs.
- **Genetic Algorithm Integration**: Utilizes a GA to efficiently explore the solution space and find optimal or near-optimal job sequences.
- **Many or Multiobjective Algorithm Integration**: Alternatively, the multiobjective problem can be solved using the ECMOA (Elitist Combinatorial Multiobjective Optimization Algorithm), which returns the Pareto Front as the solution.
- **Brute Force**: For small problem instances, the brute force search can be used to find the optimal job sequence.
- **Customizability**: Allows customization of job sequences, setup times, due dates, and more.
- **Visualization**: Generates Gantt charts to visualize the scheduling of jobs across machines.

## Usage

1. Install

```bash
pip install ga_scheduler

```

2. Try it in **Colab**:

a) Multiobjective - Weighted 

- Single Machine Scheduling - Brute Force ( [ Colab Demo ](https://colab.research.google.com/drive/1f8j9R3vClF9kmJGrS8ODGDCy_JWnh7lL?usp=sharing)) 
- Single Machine Scheduling - Genetic Algorithm ( [ Colab Demo ](https://colab.research.google.com/drive/1EevgVgIl0g9ELvUdMKRy38hKItrrPwI7?usp=sharing)) 
<!-- -->
- Parallel Machines Scheduling - Brute Force ( [ Colab Demo ](https://colab.research.google.com/drive/1qQmvkkNliPAVlTk2ShvM0Di9JAKzkqmL?usp=sharing)) 
- Parallel Machines Scheduling - Genetic Algorithm ( [ Colab Demo ](https://colab.research.google.com/drive/1yyfWNei8JNWpsOuy3UBB-pm0MIW5uxQO?usp=sharing)) 
<!-- -->
- Flow Shop Machines Scheduling - Brute Force ( [ Colab Demo ](https://colab.research.google.com/drive/1Fiq5JB9jNXjc_HSDUhEWvujILse2QdfD?usp=sharing)) 
- Flow Shop Machines Scheduling - Genetic Algorithm ( [ Colab Demo ](https://colab.research.google.com/drive/1CqcoXxyBypo_maEE7_-55s64_dsnJ42w?usp=sharing))
<!-- -->
- Job Shop Machines Scheduling - Brute Force ( [ Colab Demo ](https://colab.research.google.com/drive/1MCo3msB8cVbjg-fT9FV5QBmTKFM6km3a?usp=sharing)) 
- Job Shop Machines Scheduling - Genetic Algorithm ( [ Colab Demo ](https://colab.research.google.com/drive/1etJc3z0JMVx4FQBLZCZbCgtQsyt1pjQJ?usp=sharing)) 
 
b) Multiobjective - Pareto Front

- Single Machine Scheduling - Brute Force ( [ Colab Demo ](https://colab.research.google.com/drive/1QF7eG4JYdl1BjlIBRvmNGAw3r-QzPuR4?usp=sharing)) 
- Single Machine Scheduling - ECMOA ( [ Colab Demo ](https://colab.research.google.com/drive/1ex07yTxUPzZiGomtR1PepKYyJhzI7vdF?usp=sharing)) 
<!-- -->
- Parallel Machines Scheduling - Brute Force ( [ Colab Demo ](https://colab.research.google.com/drive/1xPqmIEaIwmYrJNIFb9BSSM1XEqFpyMKL?usp=sharing)) 
- Parallel Machines Scheduling - ECMOA ( [ Colab Demo ](https://colab.research.google.com/drive/140ZoIMwzQizsRz6TefsbxWG9d4mBatfC?usp=sharing)) 
<!-- -->
- Flow Shop Machines Scheduling - Brute Force ( [ Colab Demo ](https://colab.research.google.com/drive/1Vom9tdoDzX0D9qdCe3wBErUVtN0nGko3?usp=sharing)) 
- Flow Shop Machines Scheduling - ECMOA ( [ Colab Demo ](https://colab.research.google.com/drive/1DXeifEbk2XeQocG81WRGKtVDFoxdB6RW?usp=sharing))
<!-- -->
- Job Shop Machines Scheduling - Brute Force ( [ Colab Demo ](https://colab.research.google.com/drive/13LK2Ckc8XoftJosNQn7f7bNa0U56Zy4p?usp=sharing)) 
- Job Shop Machines Scheduling - ECMOA ( [ Colab Demo ](https://colab.research.google.com/drive/1PYV0afWMVYREwtRQRbNm1u8McDczAb0k?usp=sharing)) 
