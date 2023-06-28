---
title: Synchronisation and Race Conditions
slug: dirac-intro-to-openmp-synchronisation
teaching: 0
exercises: 0
questions:
- Why do I need to worry about thread synchronisation?
- What is a race condition?
objectives:
- Understand what thread synchronisation is
- Understand what a race condition is
- Learn how to control thread synchronisation
- Learn how to avoid errors caused by race conditions
keypoints:
    - Blah
---

In the previous episode, we saw how to use parallel regions, and the shortcut parallel for, to split work across
multiple threads. In this episode, we will learn how to synchronise threads and how to avoid data inconsistencies caused
by unsynchronised threads.

## Synchronisation and race conditions

In the previous episodes, we've seen just how easy it is to write parallel code using OpenMP. However, to make sure that
the code we're writing is both *efficient* and *correct*, we need some understanding of thread synchronisation and  race
conditions. In the context of parallel computing, thread  or rank (in the case of MPI) synchronisation plays a crucial
role in guaranteeing the *correctness* of our program, particularly in regard to data consistency and integrity.

> ## What is code correctness?
>
> Code correctness in parallel programming is the guarantee that a program operates as expected in multi-threaded
> and multi-process environments, providing both consistent and valid results. This is usually about how a parallel
> algorithm deals with data accesses and modification, minimizing the occurrence of data inconsistencies creeping in.
>
{: .callout}

So what is thread synchronisation? Thread synchronisation is the coordination of threads, which is usually done to avoid
conflicts caused by multiple threads accessing the same piece of shared data. It's important to synchronise threads
properly so they are able to work together, and not interfere with data another thread is accessing or modifying.A
Synchronisation is also important for data dependency, to make sure that a thread has access to the data it requires.
This is particularly important for algorithms which are iterative.

The synchronisation mechanisms are, in general, incredibly important tools, as they help us avoid race conditions. Race
conditions are very important to avoid, as they result in data inconsistencies if we don't get rid of them in our
program. A race  condition happens when two (or more) threads access and modify the same piece of shared data at the
same time. To illustrate this, consider the diagram below:

![Race conditions](fig/race_condition.png)

Two threads access the same shared variable (with the value 0) and increment it by 1. Intuitively, we might except the
final value of the shared variable to be 2. However, due to the potential for concurrent access and modification, we
can't actually guarantee what it will be. If both threads access and modify the variable concurrently, then the
final value will be 1. That's because both variables read the initial value of 0, increment it by 1, and write to the
shared variable.

In this case, it doesn't actually matter if the writes don't happen concurrent. The inconsistency here lies with the
value read by each thread. However, if one thread manages to access and modify the variable before the other thread can
read its value, then we'll get the value we expect. For example, if thead 0 increment the variable before thread 1 reads
it, then thread 1 will read a value of 1 and increment that by 1 resulting in a final value of 2. This illustrates why
it's called a race condition! Threads compete to access and modify variables before other threads do the same.

> ## Analogy: editing a document
>
> Imagine two people trying to update the same document at the same time. If they don't communicate what they're doing,
> they might edit the same part of the document and overwrite each others changes, ending up with a messy and
> inconsistent document. This is just like what happens with a race condition in OpenMP. Different thread access and
> modifying the same part of memory, which results in messy and inconsistent memory access and the wrong result.
>
{: .callout}

## Synchronisation mechanisms

Synchronisation in OpenMP is all about coordinating the execution of threads, especially when there is data dependency
in your program or when uncoordinated data access will result in a race condition. The synchronisation mechanisms in
OpenMP allow us to control the order of access to shared data, coordinate data dependencies (e.g. waiting for
calculations to be complete) and tasks (if one task needs to be done before other tasks can continue), and to
potentially limit access to tasks or data to certain threads.

### Barriers

Barriers are a synchronisation mechanism which are used to create a waiting point in our program. When a thread reaches
a barrier, it is forced to wait until all other threads have reached the same barrier before it can continue to do work.
To add a barrier, we use the `#pragma omp barrier` directive. In the example below, we have used a barrier to
synchronise threads such that they don't start on their main calculation until a look up table has been initialised (in
parallel), as the calculation depends on this data.

```c
#pragma omp parallel
{
    int thread_id = omp_get_num_thread();

    /* The initialisation of the look up table is done in parallel */
    initialise_lookup_table(thread_id);

#pragma omp barrier  /* As all threads depend on the table, we have to wait until all threads
                        are done and have reached the barrier */

    do_main_calculation(thread_id);
}
```

We can also put a barrier into a parallel for loop. In the next example, a barrier is used to ensure that the
calculation for `new_matrix` is done before it is copied into `old_matrix`.

```c
float old_matrix[NX][NY];
float new_matrix[NX][NY];

#pragma omp parallel for
for (int i = 0; i < NUM_ITERATIONS; ++i) {
    int thread_id = omp_get_thread_num();
    iterate_matrix_solution(old_matrix, new_matrix, thread_id);
#pragma omp barrier  /* wait until new_matrix has been updated by all threads */
    copy_matrix(new_matrix, old_matrix);
}
```

Barriers introduce additional overhead into our parallel algorithms, as some threads will be idle whilst waiting for
other threads to catch up. There are no way around this synchronisation overhead, so we have to be careful not to
overuse barriers. This overhead increases with the number of threads in use, and becomes even worse if the workload is
uneven.

> ## Blocking thread execution and `nowait`
>
> Most parallel constructs in OpenMP will synchronise threads before they exit the parallel region. For example,
> consider a parallel for loop. If one thread finishes its work before the others, it doesn't leave the parallel region
> and start on its next bit of code. It's forced to wait around for the other threads to finish, because OpenMP forces
> threads to be synchronised.
>
> This isn't ideal if the next bit of work is independent of the previous work just finished. To avoid any wasted CPU
> effort due to waiting around to be synchronisation we can use the `nowait` clause which overrides the synchronisation
> that would typically occur and allow a "finished" thread to continue to its next chunk of work.. In the example below,
> a `nowait` clause is used with a parallel for.
>
> ```c
> #pragma omp parallel
> {
>     #pragma omp for nowait  /* with nowait  the loop executes as normal, but... */
>     for (int i = 0; i < NUM_ITERATIONS; ++i) {
>         parallel_function();
>     }
>
>     /* ...if a thread finishes its work in the loop, then it can move on immediately
>        to this function without waiting for the other threads to finish */
>     next_function();
> }
> ```
>
{: .callout}

### Synchronisation regions

A common challenge in shared memory programming is coordinating threads to prevent multiple threads from concurrently
modifying the same piece of data. One mechanism in OpenMP to coordinate thread access are *synchronisation regions*,
which are used to prevent multiple threads from executing the same piece of code at the same time. When a thread reaches
one of these regions, they are queue up and wait their turn to access the data and execute the code within the region.

| Region | Description | Label |
| - | - | - |
| critical | Critical regions are used to prevent race conditions when threads need to access and modify the same data. Only one thread is allowed in the critical region at the same time, so threads have to queue up to take their turn. | `#pragma omp critical` |
| single | Single regions are used for code which needs to be executed only by a single thread, such as for I/O operations. The first thread to reach the region will execute the code, whilst the other threads will behave as if they've reached a barrier until the executing thread is finished. | `#pragma omp single` |
| master | The master region is identical to the single, except execution is done by the designated master thread. | `#pragma omp master` |

The next example builds on the previous example which included a lookup table. In the the modified code, the lookup
table is written to disk after it has been initialised. This happens in a single region, as only one thread needs to
write the result to disk.

```c
#pragma omp parallel
{
    int thread_id = omp_get_num_thread();
    initialise_lookup_table(thread_id);
#pragma omp barrier  /* Add a barrier to ensure the lookup table is ready to be written to disk */
#pragma omp single   /* We don't want multiple threads trying to write to file -- this could also be master */
    {
        write_table_to_disk();
    }
    do_main_calculation(thread_id);
}
```

If we wanted to sum up something in parallel, we can (or need to) use a critical region to prevent a race condition
where threads try to update the same sum variable at once. For example,

```c
int sum = 0;
#pragma omp parallel for
for (int i = 0; i < NUM_THINGS; ++i) {
    #pragma omp critical
    {
        sum += i;
    }
}
```

Due to the critical region, only one thread can access and increment `sum`. This prevents a race condition from
happening. But an even better way to do a reduction like this, is to use the [reduction
clause](https://www.intel.com/content/www/us/en/docs/advisor/user-guide/2023-0/openmp-reduction-operations.html) in the
parallel for construct.

> ## Reporting progress
>
> Create a program that updates a shared counter to track the progress of a parallel loop. Which critical region would
> you use? Can you think of any problems with what you have implemented?
>
> ```c
> #include <math.h>
> #include <omp.h>
>
> #define NUM_ELEMENTS 10000
>
> int main(int argc, char **argv) {
>   int progress = 0;
>   int array[NUM_ELEMENTS] = {0};
>   int output_frequency = NUM_ELEMENTS / 10; /* output every 10% */
>
> #pragma omp parallel for schedule(static)
>   for (int i = 0; i < NUM_ELEMENTS; ++i) {
>     array[i] = log(i) * cos(3.142 * i);
>   }
>
>   return 0;
> }
> ```
>
> > ## Solution
> >
> > To implement a progress bar, we increment `progress` and display the output within a critical a region region. If we
> > used either a master or single region, then `progress` would not be incremented by all threads and will not reflect
> > the true progress.
> >
> > ```c
> > #include <math.h>
> > #include <omp.h>
> > #include <stdio.h>
> >
> > #define NUM_ELEMENTS 1000
> >
> > int main(int argc, char **argv) {
> >   int progress = 0;
> >   int array[NUM_ELEMENTS] = {0};
> >   int output_frequency = NUM_ELEMENTS / 10; /* output every 10% */
> >
> > #pragma omp parallel for schedule(static)
> >   for (int i = 0; i < NUM_ELEMENTS; ++i) {
> >     int thread_id = omp_get_thread_num();
> >
> >     array[i] = log(i) * cos(3.142 * i);
> >
> > #pragma omp critical
> >     {
> >       progress++;
> >       if (progress % output_frequency == 0) {
> >         printf("Thread %d: overall progress %3.0f%%\n", thread_id,
> >                (double)progress / NUM_ELEMENTS * 100.0);
> >       }
> >     }
> >   }
> >
> >   return 0;
> > }
> > ```
> >
> {: .solution}
{: .challenge}

## Preventing race conditions

### Data dependency

### Critical region

### Atomic operations

In OpenMP, atomic operations are operations which are done without interference from other threads. If we make modifying
some value in an array atomic, then it's guaranteed, by the compiler, that no other thread can read or modify that array
until the atomic operation is finished. You can think of it as a thread having, temporary, exclusive access to something
in our program. Sort of like a "one at a time" rule for accessing and modifying parts of the program.

To make something atomic, we use the `omp atomic` pragma.

```c
int shared_variable = 0;
int shared_array[4] = {0, 0, 0, 0};

/* Put the pragma before the shared variable */
#pragma omp parallel
{
    #pragma omp atomic
    shared_variable += 1;

}

/* Can also use in a parallel for */
#pragma omp parallel for
for (int i = 0; i < 4; ++i) {
    #pragma omp atomic
    shared_array[i] += 1;
}
```

If you have too many atomic regions, then performance is degraded as threads stop and start to access the atomic region.
Atomic operations are also not very good for complex data structures, really only good on primitive types and pointers.

### Locks

Locks are like critical regions, but more flexible.

```c
omp_lock_t lock;
omp_init_lock(&lock);

int shared_variable = 0;

#pragma omp parallel
{
    omp_set_lock(&lock);
    shared_variable += 1;
    omp_unset_lock(&lock);
}

omp_destroy_lock(&lock);
```

> ## Remove the race condition
>
> In the following program, an array of values is created and then summed together using a parallel for loop.
>
> ```c
> #include <math.h>
> #include <omp.h>
> #include <stdio.h>
>
> #define ARRAY_SIZE 524288
>
> int main(int argc, char **argv) {
>   float sum = 0;
>   float array[ARRAY_SIZE];
>
>   omp_set_num_threads(4);
>
> #pragma omp parallel for schedule(static)
>   for (int i = 0; i < ARRAY_SIZE; ++i) {
>     array[i] = cos(M_PI * i);
>   }
>
> #pragma omp parallel for schedule(static)
>   for (int i = 0; i < ARRAY_SIZE; i++) {
>     sum += array[i];
>   }
>
>   printf("Sum: %f\n", sum);
>
>   return 0;
> }
> ```
>
> When we run the program multiple times, the output we expect sum to have the value of `0.000000`. But if we run the
> program multiple times, we sometimes get the wrong output:
>
> ```c
> 1. Sum: 1.000000
> 2. Sum: -1.000000
> 3. Sum: 2.000000
> 4. Sum: 0.000000
> 5. Sum: 2.000000
> ```
>
> Find and fix the race condition in the program. Try using both an atomic operation and by using locks.
>
> > ## Solution
> >
> {: .solution}
{: .challenge}
