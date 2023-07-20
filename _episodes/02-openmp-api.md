---
title: Introduction to OpenMP
slug: dirac-intro-to-openmp-openmp-api
teaching: 0
exercises: 0
questions:
- What is OpenMP?
- How does it work?
objectives:
- Learn what OpenMP is
- Understand how to use the OpenMP API
- Learn how to compile and run OpenMP applications
keypoints:
---
## What is OpenMP?

OpenMP is an industry-standard API specifically designed for parallel programming in shared memory environments. It supports programming in languages such as C, C++, and Fortran. OpenMP is an open source, industry-wide initiative that benefits from collaboration among hardware and software vendors, governed by the OpenMP Architecture Review Board ([OpenMP ARB](https://www.openmp.org/)).
> ## An OpenMP Timeline
> 
> <a href="{{ page.root }}/fig/OpenMP_Timeline.png" target="new">
> <img src="{{ page.root }}/fig/OpenMP_Timeline.png" alt="OpenMP-history" width="40%" height="40%">
> </a>
> 
> In 2018 OpenMP 5.0 was released, the current version is 5.2, released in November 2021.
{: .solution}

## How does it work? 

OpenMP allows programmers to identify and parallelize sections of code, enabling multiple threads to execute them concurrently. It follows a shared-memory model, where all threads have access to a common memory space, and they communicate through shared variables.

To parallelise code using OpenMP, programmers use special directives, which are typically added as **#pragma** statements in the code. 
~~~
#pragma omp <name_of_directive>
~~~


These directives inform the compiler about the portions of the code that should be executed in parallel. The compiler then generates the necessary code to distribute the workload among multiple threads.

When an OpenMP program runs, it typically starts with a single master thread. The master thread encounters parallel regions marked by directives and creates a team of multiple slave threads. Each slave thread executes a separate instance of the parallelized section of code. After completing their work, the slave threads synchronize with the master thread, which continues the program's execution.

<img src="{{ page.root }}/fig/how_it_works.svg" alt="How it works?" width="50%" height="50%" />

OpenMP directives provide control over various aspects of parallel execution, including:

- **Parallel regions:** Specifies which parts of the code should be executed in parallel. These regions can be nested, allowing for different levels of parallelism.
- **Loop parallelization:** Enables parallel execution of loops by distributing loop iterations among threads. It helps to speed up computation-intensive tasks.
- **Variable scope:** Defines how variables are shared or made private to each thread. It ensures proper data access and prevents data races.
- **Thread synchronization:** Allows synchronization between threads using constructs like barriers and critical sections. It helps to coordinate access to shared resources.
- **Work distribution:** Specifies how work is divided among threads, such as static or dynamic scheduling of tasks. It helps balance the workload and optimize resource utilization.

## Using OpenMP

Let's start with a simple C code that prints "Hello World!" to the console. Save the 
following code in a file named **`hello-omp.c`**.

~~~
#include <stdio.h>

int main (int argc, char *argv[]) {
    printf("Hello World!\n");
}
~~~
{: .language-c}
