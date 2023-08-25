---
title: Introduction to OpenMP
slug: dirac-intro-to-openmp-openmp-api
teaching: 0
exercises: 0
math: true
questions:
- What is OpenMP?
- How does OpenMP work?
- Why the researches should prefer OpenMP over POSIX?
objectives:
- Learn what OpenMP is
- Understand how to use the OpenMP API
- Learn how to compile and run OpenMP applications
- Understand the difference between OpenMP and low level threading APIs
keypoints:
- OpenMP is an industry-standard API for parallel programming in shared memory environments.
- It supports C, C++, and Fortran and is governed by the OpenMP ARB.
- OpenMP follows the fork-join model, using master and slave threads for parallel tasks.
- Compiler directives guide the compiler to create parallel code, e.g., `#pragma omp parallel`.
- Runtime library routines offer predefined functions for thread control and synchronization.
- Environment variables fine-tune OpenMP runtime behavior.
- OpenMP is user-friendly, automating thread distribution across systems.
- Both OpenMP and Low-Level Threading APIs provide effective parallel programming options. The choice depends on factors like ease of use, control, and performance optimization.
---
## What is OpenMP?

OpenMP is an industry-standard API specifically designed for parallel programming in shared memory environments. It supports programming in languages such as C, C++, and Fortran. OpenMP is an open source, industry-wide initiative that benefits from collaboration among hardware and software vendors, governed by the OpenMP Architecture Review Board ([OpenMP ARB](https://www.openmp.org/)).

> ## An OpenMP Timeline
> 
> <a href="{{ page.root }}/fig/OpenMP_Timeline.png" target="new">
> <img src="{{ page.root }}/fig/OpenMP_Timeline.png" alt="OpenMP-history" width="40%" height="40%">
> </a>
> 
> The timeline provides an overview of OpenMP's evolution until 2014, with significant advancements 
> occurring thereafter. Notably, OpenMP 5.0 marked a significant step in 2018, followed by the latest 
> iteration, OpenMP 5.2, which was released in November 2021.
{: .solution}


## How does it work? 

OpenMP allows programmers to identify and parallelize sections of code, enabling multiple threads to execute them concurrently. This concurrency is achieved using a shared-memory model, where all threads can access a common memory space and communicate through shared variables.

To understand how OpenMP orchestrates this parallel execution, let's explore the **fork-join model** it employs. Think of your program as a team with a leader (the master thread) and workers (the slave threads). When your program starts, the leader thread takes the lead. It identifies parts of the code that can be done at the same time and marks them. These marked parts are like tasks to be completed by the workers. The leader then gathers a group of helper threads, and each helper tackles one of these marked tasks. Each worker thread works independently, taking care of its task. Once all the workers are done, they come back to the leader, and the leader continues with the rest of the program.

In simpler terms, when your program finds a special "parallel" section, it's like the leader telling the workers to split up and work on different tasks together (that's the "fork" part). After finishing their tasks, these workers come back to the leader, allowing the leader to move forward (that's the "join" part). This teamwork approach helps OpenMP speed up tasks and get things done faster.

<img src="{{ page.root }}/fig/fork-join.png" alt="How it works?" width="60%" height="60%" />

## OpenMP API Overview
The OpenMP API consists of three key components that enable parallel programming:
> <p style="text-align: left; font-size:18px"> <b>1. Compiler
Directives </b> </p>
> {: .checklist}
> OpenMP directives are special comments indicated by `#pragma omp` 
> statements that guide the compiler in creating parallel code. They 
> mark sections of code to be executed concurrently by multiple 
> threads. In C/C++, the syntax for pragma directives is as follows:
>
> ~~~c
> #pragma omp <name_of_directive> <optional_clause>
> ~~~
> {: .language-c}
> For example: 
> - The `#pragma omp parallel` directive specifies a block of code for concurrent execution.
> - The `#pragma omp for` directive parallelizes loops by distributing loop iterations among threads.
> - The `#pragma omp sections` directive divides sections of code among threads for concurrent execution.
> <p style="text-align: left; font-size:18px"> <b>2. Runtime Library Routines </b> </p>
> {: .checklist}
> These are predefined functions provided by the OpenMP runtime 
> library. They allow you to control the behavior of threads, manage 
> synchronization, and handle parallel execution. For example, in the 
> following code the runtime routine `omp_get_thread_num()` returns 
> the unique identifier of the calling thread.
> ~~~c
> #include <stdio.h>
> #include <omp.h>
> int main() {
>     int num_threads, thread_id;
>     #pragma omp parallel private(thread_id)
>     {
>         thread_id = omp_get_thread_num();
>         printf("Hello from thread %d\n", thread_id);
>     }
> }
> ~~~
> {: .language-c}
>
> > ## Header Dependency 
> > Note that to use of OpenMP directives and functions in your code, it's essential to 
> > include the `<omp.h>` header file. This header file provides the necessary definitions for utilizing 
> > OpenMP constructs, such as parallel regions, work-sharing directives, and synchronization mechanisms.
> > {: .output}
> {: .caution}
>
> <p style="text-align: left; font-size:18px"> <b>3. Environment Variables </b> </p>
> {: .checklist}
> These are settings that can be adjusted to influence the behavior of the OpenMP runtime. They provide 
> a way to fine-tune the parallel execution of your program. Setting OpenMP environment variables is 
> typically done similarly to other environment variables for your system. For instance, you can adjust 
> the number of threads as follows:
> ~~~bash
> export OMP_NUM_THREADS=4
> ./my_openmp_program
> ~~~
> {: .language-bash}
{: .callout}

## OpenMP vs. Low-Level Threading APIs (POSIX Threads)
When it comes to parallel programming with threads, there are two main ways to tackle it: the 
user-friendly OpenMP and the more intricate Low-Level Threading APIs. In this context, '***Low-Level 
Threading APIs***', refer to interfaces like the Portable Operating System Interface (POSIX), which 
defines a set of standard functions and interfaces for interacting with operating systems. Each 
approach has its own advantages and considerations, so let's break them down in simpler terms:

OpenMP serves as a gentle introduction to parallel programming. It offers an easy way to parallelize 
your code without delving into complex technicalities. This makes it ideal for beginners or anyone 
seeking a straightforward approach. It's like choosing a reliable car that gets you from point A to 
point B comfortably. 

**Advantages of OpenMP:**
- ***User-Friendly:*** OpenMP requires minimal code adjustments, making it a friendly choice for 
 newcomers.
- ***Automated Work Distribution:*** It divides tasks among threads, ensuring balanced workloads.
- ***Wider Applicability:*** OpenMP functions across different systems and programming languages.

For instance, consider a scenario where you have a task that involves doing the same thing over and 
over again, like processing a bunch of images. With OpenMP, you can split up the work among different 
threads, so they each handle a part of the task at the same time.

Now, imagine you're a master chef who wants complete control over every ingredient and spice in a 
recipe. That's what Low-Level Threading APIs, like POSIX, offer – a lot of control over threads and 
how they work together. But this kind of control requires a bit more knowledge and effort.

**Benefits of Low-Level Threading APIs: **
- ***Full Control:*** These APIs allow you to customize every aspect of threads, but it requires a 
deeper understanding.
- ***Better Performance:*** If you know exactly what you're doing, you can make things run faster and 
more efficiently.
- ***Flexible Solutions:*** You can create solutions that perfectly match your unique needs.

Let's say you're building a game where players from different parts of the world can interact with 
each other in real-time. Here, using POSIX threading would give you the control you need to manage 
these interactions smoothly and avoid any weird glitches.

**Choosing the Right Framework** 
When deciding between OpenMP and Low-Level Threading APIs like POSIX for your parallel programming 
needs, several factors come into play:

| Aspect                  | OpenMP                                   | Low-Level Threading APIs like POSIX                    |
 |-------------------------|------------------------------------------|------------------------------------------------------|
| **Ease of Use**             | User-friendly, higher-level abstractions | Lower-level, more manual management of threads       |
| **Parallelism Scope**       | Parallel regions, loops, tasks            | Usually limited to thread creation and synchronization |
| **Portability**             | Portable across different systems        | Depends on system-specific implementation           |
| **Abstraction Level**       | Higher-level constructs and directives   | Direct interaction with system-level threading       |
| **GPU Offloading Support**  | Supports offloading work to GPUs         | Typically lacks built-in support for GPU offloading  |
| **Maintenance Complexity**  | Easier maintenance due to higher abstractions | Requires more low-level management                |
| **Performance Optimization** | May offer automatic optimization         | Requires manual tuning for performance               |
| **Common Usage**            | Widely used in research and industry      | Less common due to lower-level complexities         |

In summary, OpenMP offers a more user-friendly and productive experience, especially for researchers 
who want to focus on problem-solving rather than thread management. Additionally, its support for GPU #
offloading enhances performance in certain scenarios. On the other hand, Low-Level Threading APIs like 
POSIX provide greater control and are suitable for those with a solid grasp of system-level 
programming.
