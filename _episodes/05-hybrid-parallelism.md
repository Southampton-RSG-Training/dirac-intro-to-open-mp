---
title: Introduction to Hybrid Parallelism
slug: dirac-intro-to-openmp-hybrid-parallelism
math: true
teaching: 0
exercises: 0
questions:
    - What is hybrid parallelism?
    - How could hybrid parallelism benefit my software?
objectives:
    - Learn what hybrid parallelism is
    - Understand the advantages of disadvantages of hybrid parallelism
    - Learn how to use OpenMP and MPI together
keypoints:
    -
---

At this point in the lesson, we've introduced the basics you need to get out there and start writing parallel code using
OpenMP! There is one thing still worth being brought to your attention, and that is *hybrid parallelism*.

> ## The message passing interface
>
> In this episode, we will assume you have some knowledge about the Message Passing Interface (MPI) and that you have a
> basic understand of how to paralleise code using MPI.
>
> If you're not sure, you can think of MPI as being like an OpenMP program where everything is in a
> `pragma omp parallel` directive.
>
{: .callout}


## What is hybrid parallelism?

When we talk about hybrid paralleism, what we're really talking about is writing parallel code using more than one
parallelisation paradigm. The reason we want to do this is to take advantage of the strengths of each paradigm to
improve the performance, scaling and efficiency of our parallel core. The most common form of hybrid parallelism in
research is *MPI+X*. What this means is that an application is *mostly* parallelised using the Message Passing Interface
(MPI), which has been extended using some +X other paradigm. A common +X is OpenMP, creating MPI+OpenMP.

> ## Heterogeneous Computing
>
> An MPI+OpenMP scheme is known as homogenous computing, meaning all the processing units involved are of the sme type.
> The opposite is heterogeneous computing, where different types of processing architectures are used such as CPUs,
> GPUs (graphics processing units), TPUs (tensor processing units) and FGPAs (field-programmable gate arrays).The goal
> of heterogeneous computing is to leverage the strengths of each processor type to achieve maximum performance and
> efficiency. The most common in research will be CPU and GPU.
>
{: .callout}

![A diagram showing MPI+OpenMP](fig/mpi+openmp.png)

In an MPI+OpenMP application, one or multiple MPI processes/ranks are created each of which spawn their own set of
OpenMP threads as shown in the diagram above. In this setup, the MPI processes can still freely communicate with one
another. Threads within the same MPI process, obviously do not need to communicate with one another but, maybe not so
obvious is that threads in one processes cannot communicate with the threads in another -- not unless we are very
careful and explicitly set up communication between specific threads using the parent MPI processes.

As an example of how resources could be split using an MPI+OpenMP approach, consider a HPC cluster with some number of
compute nodes with each having 64 CPU cores. One approach would be to spawn one MPI process per rank which spawns 64
OpenMP threads, or 2 MPI processes which both spawn 32 OpenMP threads, and so on and so forth.

### Advantages

* Improved memory efficiency
* Better scaling and load balancing
* Flexibility for different architectures

### Disadvantages

* Slower due to more overheads
* Difficult to write and maintain
* More limited portability

## When do I need to use hybrid parallelism?

## Writing a hybrid parallel application

To demonstrate how to use MPI+OpenMP, we are going to write a program which computes an approximation for $\pi$ using a
[Riemann sum](https://en.wikipedia.org/wiki/Riemann_sum). This is not a great example to extol the virtues of hybrid
parallelism, as it is only a small problem. However, it is a simple problem which can be easily extended and
parallelised. Specifically, we will write a program to solve to integral to compute the value of $\pi$,

$$ \int_{0}^{1} \frac{4}{1 + x^{2}} ~ \mathrm{d}x = 4 \tan^{-1}(x) = \pi $$

There are a plethora of methods available to numerically evaluate this integral. To keep the problem simple, we will
re-cast the integral into a easier-to-code summation. How we got here isn't that important for our purposes, but what we
will be implementing in code is the follow summation,

$$ \pi = \lim_{n \to \infty} \sum_{i = 0}^{n} \frac{1}{n} ~ \frac{4}{1 + x_{i}^{2}} $$

where $x_{i}$ is the the midpoint of the $i$-th rectangle. To get an accurate approximation of $\pi$, we'll need to
split the domain into a large number of smaller rectangles.

### A simple parallel implementation using OpenMP

To begin, let's first write a serial implementation as in the code example below.

```c
#include <stdio.h>
#include <time.h>
#include <unistd.h>

#define PI 3.141592653589793238462643

int main(void)
{
    struct timespec begin;
    clock_gettime(CLOCK_MONOTONIC_RAW, &begin);

    /* Initialise parameters. N is the number of rectangles we will sum over,
       and h is the width of each rectangle (1 / N) */
    const long N = (long)1e10;
    const double h = 1.0 / N;
    double sum = 0.0;

    /* Compute the summation. At each iteration, we calculate the position x_i
       and use this value in 4 / (1 + x * x). We are not including the 1 / n
       factor, as we can multiply it once at the end to the final sum */
    for (long i = 0; i <= N; ++i) {
        const double x = h * (double)i;
        sum += 4.0 / (1.0 + x * x);
    }

    /* To attain our final value of pi, we multiply by h as we did not include
       this in the loop */
    const double pi = h * sum;

    struct timespec end;
    clock_gettime(CLOCK_MONOTONIC_RAW, &end);
    printf("Calculated pi %18.6f error %18.6f\n", pi, pi - PI);
    printf("Total time = %f seconds\n", (end.tv_nsec - begin.tv_nsec) / 1000000000.0 + (end.tv_sec - begin.tv_sec));

    return 0;
}
```

In the above, we are using $N = 10^{10}$ rectangles (using this number of rectangles is overkill, but is used to
demonstrate the performance increases from parallelisation. If we save this (as `pi.c`), compile and run we should get
output as below,

```bash
$ gcc pi.c -o pi.exe
$ ./pi.exe
Calculated pi           3.141593 error           0.000000
Total time = 34.826832 seconds
```

You should see that we've compute an accurate approximation of $\pi$, but it also took a very long time at 35 seconds!
To speed this up, let's first parallelise this using OpenMP. All we need to do, for this simple application, is to use a
parallel for to split the loop between OpenMP threads as shown below.

```c
/* Parallelise the loop using a parallel for directive. We will set the sum
   variable to be a reduction variable. As it is marked explicitly as a reduction
   variable, we don't need to worry about any race conditions corrupting the
   final value of sum */
#pragma omp parallel for shared(N, h), reduction(+:sum)
for (long i = 0; i <= N; ++i) {
    const double x = h * (double)i;
    sum += 4.0 / (1.0 + x * x);
}

/* For diagnostics purposes, we are also going to print out the number of
   threads which were spawned by OpenMP */
printf("Calculated using %d OMP threads\n", omp_get_max_threads());
```

Once we have made these changes (you can find the completed implementation [here](code/examples/05-pi-omp.c)), compiled
and run the program, we can see there is a big improvement to the performance of our program. It takes just 5 seconds to
complete, instead of 35 seconds, to get our approximate value of $\pi$, e.g.:

```bash
$ gcc -fopenmp pi-omp.c -o pi.exe
$ ./pi.exe
Calculated using 8 OMP threads
Calculated pi           3.141593 error           0.000000
Total time = 5.166490 seconds
```

### A hybrid implementation using MPI and OpenMP

Now that we have a working parallel implementation using OpenMP, we can now expand our code to a hybrid parallel code by
implementing MPI. In this example, we can porting an OpenMP code to a hybrid MPI+OpenMP application but we could have
also done this the other way around by porting an MPI code into a hybrid application. Neither *"evolution"* is more
common or better than the other, the route each code takes toward becoming hybrid is different.

So, how do we split work using a hybrid approach? One approach for an embarrassingly parallel problem, such as the one
we're working on is to can split the problem size into smaller chunks *across* MPI ranks, and to use OpenMP to
parallelise the work. For example, consider a problem where we have to do a calculation for 1,000,000 input parameters.
If we have four MPI ranks each of which will spawn 10 threads, we could split the work evenly between MPI ranks so each
rank will deal with 250,000 input parameters. We will then use OpenMP threads to do the calculations in parallel. If we
use a sequential scheduler, then each thread will do 25,000 calculations. Or we could use OpenMP's dynamic scheduler to
automatically balance the workload. We have implemented this situation in the code example below.

```c
/* We have an array of input parameters. The calculation which uses these parameters
   is expensive, so we want to split them across MPI ranks */
struct input_par_t input_parameters[total_work];

/* We need to determine how many input parameters each rank will process (we are
   assuming total_work is cleanly divisible) and also lower and upper indices of
   the input_parameters array the rank will work on */
int work_per_rank = total_work / num_ranks;
int rank_lower_limit = my_rank * work_per_rank;
int rank_upper_limit = (my_rank + 1) * work_per_rank;

/* The MPI rank now knows which subset of data it will work on, but we will use
   OpenMP to spawn threads to execute the calculations in parallel. We'll make sure
   of the auto scheduler to best determine how to balance the work */
#pragma omp parallel for schedule(auto)
for (int i = rank_lower_limit; i < rank_upper_limit; ++i) {
    some_expensive_calculation(input_parameters[i]);
}
```

> ## Still not sure about MPI?
>
> If you're still a bit unsure of how MPI is working, you can basically think of it as wrapping large parts of  your
> code in a `pragma omp parallel` region as we saw in an earlier episode. We can re-write the code example above in the
> same way, but using OpenMP thread IDs instead.
>
> ```c
> struct input_par_t input_parameters[total_work];
>
> #pragma omp parallel
> {
>     int num_threads = omp_get_num_threads();
>     int thread_id = omp_get_thread_num();
>
>     int work_per_thread = total_work / num_threads;
>     int thread_lower = thread_id * work_per_thread;
>     int thread_upper = (thread_id + 1) * work_per_thread;
>
>     for(int i = thread_lower; i < thread_upper; ++i) {
>         some_expensive_calculation(input_parameters[i]);
>     }
> }
> ```
>
{: .hidden-callout}

In the above example, we have only included the parallel region of code. It is unfortunately not as simple as this,
because we have to deal with the additional complexity from using MPI. We need to initialise MPI, as well as communicate
and receive data and deal with the other complications which come with MPI. When we include all of this, our code
example becomes much more complicated than before. In the next code block, we have implemented a hybrid MPI+OpenMP
approximation of $\pi$ using the same Riemann sum method.

```c
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <unistd.h>

#include <mpi.h>
#include <omp.h>

#define PI 3.141592653589793238462643
#define ROOT_RANK 0

int main(void)
{
    struct timespec begin;
    clock_gettime(CLOCK_MONOTONIC_RAW, &begin);

    /* We have to initialise MPI first and determine the number of ranks and
       which rank we are to be able to split work with MPI */
    int my_rank;
    int num_ranks;
    MPI_Init(NULL, NULL);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_ranks);

    /* We can leave this part unchanged, the parameters per rank will be the
       same */
    double sum = 0.0;
    const long N = (long)1e10;
    const double h = 1.0 / N;

    /* The OpenMP parallelisation is almost the same, we are still using a
       parallel for to do the loop in parallel. To parallelise using MPI, we
       have each MPI rank do every num_rank-th iteration of the loop. Each rank
       will do N / num_rank iterations split between it's own OpenMP threads */
#pragma omp parallel for shared(N, h, my_rank, num_ranks), reduction(+:sum)
    for (long i = my_rank; i <= N; i = i + num_ranks) {
        const double x = h * (double)i;
        sum += 4.0 / (1.0 + x * x);
    }

    /* The sum we compute is per rank now, but only includes N / num_rank
       elements so is not the final value of pi */
    const double rank_pi = h * sum;

    /* To get the final value, we will use a reduction across ranks to sum up
       the contributions from each MPI rank */
    double reduced_pi;
    MPI_Reduce(&rank_pi, &reduced_pi, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    if (my_rank == ROOT_RANK) {
        struct timespec end;
        clock_gettime(CLOCK_MONOTONIC_RAW, &end);
        /* Let's check how many ranks are running and how many threads each rank
           is spawning  */
        printf("Calculated using %d MPI ranks each spawning %d OMP threads\n", num_ranks, omp_get_max_threads());
        printf("Calculated pi %18.6f error %18.6f\n", reduced_pi, reduced_pi - PI);
        printf("Total time = %f seconds\n", (end.tv_nsec - begin.tv_nsec) / 1000000000.0 + (end.tv_sec - begin.tv_sec));
    }

    MPI_Finalize();

    return 0;
}
```

So you can see that it's much longer and more complicated; although not much more than a [pure MPI
implementation](code/examples/05-pi-mpi.c). To compile our hybrid program, we use the MPI compiler command `mpicc` with
the argument `-fopenmp`. We can then either run our compiled program using `mpirun`.

```bash
$ mpicc -fopenmp 05-pi-omp-mpi.c -o pi.exe
$ mpirun pi.exe
Calculated 8 MPI ranks each spawning 8 OMP threads
Calculated pi           3.141593 error           0.000000
Total time = 5.818889 seconds
```

Ouch, this took longer to run than the pure OpenMP implementation (although only marginally longer in this example! You
may have noticed that we have 8 MPI ranks, each of which are spawning 8 of their own OpenMP threads. This is an
important thing to realise. When you specify the number of threads for OpenMP to use, this is the number of threads
*each* MPI process will spawn. So why did it take longer? With each of the 8 MPI ranks spawning 8 threads, 64 threads
threads were in flight. More threads means more overheads and if, for instance, we have 8 CPU Cores, then contention
arises as each thread competes for access to a CPU core.

Let's improve this situation by using a combination of rank and threads so that $N_{\mathrm{ranks}} N_{\mathrm{threads}}
\le 8$. One way to do this is by setting the `OMP_NUM_THREADS` environment variable and by specifying the number of
processes we want to spawn with `mpirun`. For example, we can spawn two MPI processes which will both spawn 4 threads
each.

```bash
$ export OMP_NUM_THREADS 4
$ mpirun -n 2 pi.exe
Calculated using 4 OMP threads and 2 MPI ranks
Calculated pi           3.141593 error           0.000000
Total time = 5.078829 seconds
```

This is better now, as threads aren't fighting for access to a CPU core. If we change the number of ranks to 1 and
the number of threads to 8, will it take the same amount of time to run as the pure OpenMP implementation?

```bash
$ export OMP_NUM_THREADS 8
$ mpirun -n 1 pi.exe
Calculated using 8 OMP threads and 1 MPI ranks
Calculated pi           3.141593 error           0.000000
Total time = 5.328421 seconds
```

In this case, we don't. It takes *slightly* longer to run because of the overhead associated with MPI. Even when we use
one MPI rank, we still have to initialise MPI, set the rank number, and so on. All of this takes time. The same happens
with using 8 ranks and 1 thread per rank, as there is still a slight overhead related to scheduling the work for that
single OpenMP thread.

```bash
$ export OMP_NUM_THREADS 1
$ mpirun -n 8 pi.exe
Calculated using 1 OMP threads and 8 MPI ranks
Calculated pi           3.141593 error           0.000000
Total time = 5.377609 seconds
```

> ## How many ranks and threads should I use?
>
> How many ranks and threads you should use depends on lots of parameters, such as the size of your problem (e.g. do you
> need a large number of threads but a smaller number of ranks to improve memory efficiency?), the hardware you are
> using and the design/structure of your code. It's unfortunately very difficult to predict the best combination of
> ranks and threads. Often we won't know until *after* we've run lots of tests and gained some intuition. It's a
> delicate balance of balancing overheads associated with thread synchronisation in OpenMP and data communication in
> MPI. As mentioned earlier, a hybrid implementation will typically be slower than a "pure" MPI implementation for example.
>
{: .callout}

> ## Optimum combination of threads and ranks for approximating $\pi$
>
> Try various combinations of the number of OpenMP threads and number of MPI processes. For this program, what's faster?
> Only using [MPI](code/examples/05-pi-mpi.c), only using [OpenMP](code/examples/05-pi-omp.c) or a
> [hybrid](code/examples/05-pi-omp-mpi.c) implementation? Why do you think this is the fastest method of
> parallelisation?
>
> Note that there will be some level of variance in the run time each time you run the program, due to factors such as
> other programs using your CPU at the same time. You should run each thread/rank combination multiple time to get an
> average.
>
> > ## Solution
> >
> > There is not really a right answer here, as the best combination will depend on lots of factors such as the hardware
> > you are running the program on. On a MacBook Pro with a 6-core M1 Pro, the best combination of ranks and threads
> > was, rather naturally, when either $N_{\mathrm{ranks}} = 1$, $N_{\mathrm{threads}} = 6$ and $N_{\mathrm{ranks}} = 6$,
> > $N_{\mathrm{threads}} = 1$ with the former being slightly faster. Otherwise, we found the best balance was
> > $N_{\mathrm{ranks}} = 2$, $N_{\mathrm{threads}} = 3$.
> {: .solution}
>
{: .challenge}

## Submitting a hybrid application to Slurm

There isn't that much you have to do different when it comes to submitting a MPI+OpenMP application to Slurm. In
addition to the standard parameters for controlling the length of the job, the resources it requires and etc., we also
need to set additional parameters to tell Slurm how to distribute the threads and ranks across the hardware available.
For example, instead of only settings the number of ranks, we need to set the number of tasks (or MPI ranks) and how
many CPUs to assign to each task:

```bash
#!/bin/bash

#SBATCH --time=00:05:00        # Walltime limit for job
#SBATCH --nodes=2              # Number of nodes to use
#SBATCH --tasks-per-node=2     # Number of MPI ranks/tasks to create
#SBATCH --cpus-per-task=20     # Number of CPUs available to each MPI rank/task
#SBATCH --partition=standard   # Partition/queue name

# Export the number of threads as cpus-per-task
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Use srun to run the program (but could also use mpirun/mpiexec)
srun ./pi.exe
```

> ## Which thread goes where?
>
> It's not necessarily always clear how threads and ranks will be distributed across cores, and you can accidentally
> overlap or oversubscribe CPU cores rather easily! An excellent tool for seeing how threads and ranks get distributed
> is a program called [xthi](https://git.ecdf.ed.ac.uk/dmckain/xthi).
>
> In this exercise, we will use `xthi` to experiment with how OpenMP threads and MPI ranks are distributed across the
> CPU cores. The first thing to do is download and compile `xthi` onto your HPC system. Then using this slurm script as
> your starting point, experiment with changing the number of tasks per node and cpus per node and seeing how threads
> and ranks are distributed.
>
> ```bash
> #!/bin/bash
>
> #SBATCH --time=00:01:00
> #SBATCH --nodes=1
> #SBATCH --ntasks=4
> #SBATCH --tasks-per-node=2
> #SBATCH --cpus-per-task=2
> #SBATCH --partition=YOUR_PARTITION
>
> export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
>
> mpirun -np $SLURM_NTASKS xthi
> ```
>
{: .challenge}
