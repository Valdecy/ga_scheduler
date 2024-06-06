# GA Scheduler

## Overview

GA Scheduler is an advanced scheduling tool that leverages genetic algorithms to optimize **job shop**, **flow shop**, and **parallel** machines scheduling problems. Multiple objectives can be addressed such as makespan, weighted tardiness, total waste changeover between jobs, and total setup times changeover between jobs. Additionally, the library provides a comprehensive way to visualize scheduling results through Gantt charts.

## Features

- **Scheduling Machine Environments**: Supports job shop, flow shop, and parallel machines scheduling problems
- **Multi-Objective Optimization**: Supports optimization for multiple objectives including makespan, weighted tardiness, total waste changeover between jobs, and setup times changeover between jobs.
- **Genetic Algorithm Integration**: Utilizes a GA to efficiently explore the solution space and find optimal or near-optimal job sequences.
- **Brute Force**: For small problem instances, the brute force search can be used to find the optimal job sequence.
- **Customizability**: Allows customization of job sequences, setup times, due dates, and more.
- **Visualization**: Generates Gantt charts to visualize the scheduling of jobs across machines.

## Usage

1. Install

```bash
pip install ga_scheduler

```

2. Try it in **Colab**:

- Job Shop Machines Scheduling - Brute Force ( [ Colab Demo ](https://colab.research.google.com/drive/1MCo3msB8cVbjg-fT9FV5QBmTKFM6km3a?usp=sharing)) 
- Job Shop Machines Scheduling - Genetic Algorithm ( [ Colab Demo ](https://colab.research.google.com/drive/1etJc3z0JMVx4FQBLZCZbCgtQsyt1pjQJ?usp=sharing)) 
<!-- -->
- Flow Shop Machines Scheduling - Brute Force ( [ Colab Demo ](https://colab.research.google.com/drive/1Fiq5JB9jNXjc_HSDUhEWvujILse2QdfD?usp=sharing)) 
- Flow Shop Machines Scheduling - Genetic Algorithm ( [ Colab Demo ](https://colab.research.google.com/drive/1CqcoXxyBypo_maEE7_-55s64_dsnJ42w?usp=sharing))
 <!-- -->
- Parallel Machines Scheduling - Brute Force ( [ Colab Demo ](https://colab.research.google.com/drive/1qQmvkkNliPAVlTk2ShvM0Di9JAKzkqmL?usp=sharing)) 
- Parallel Machines Scheduling - Genetic Algorithm ( [ Colab Demo ](https://colab.research.google.com/drive/1yyfWNei8JNWpsOuy3UBB-pm0MIW5uxQO?usp=sharing)) 
