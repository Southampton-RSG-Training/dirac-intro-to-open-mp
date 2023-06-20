---
title: Writing Parallel Applications with OpenMP
slug: dirac-intro-to-openmp-parallel-code
teaching: 0
exercises: 0
questions:
objectives:
- Learn how to parallelise work using OpenMP
- Learn how to use common OpenMP pragma directives
keypoints:
---

TODO

## Parallel Loops

### Using schedulers

Whenever we use a parallel for, the iterations have to be split into smaller chunks so each thread has something to do.
In most OpenMP implementations, the default behaviour is to split the iterations into equal sized chunks,

```c
int CHUNK_SIZE = NUM_ITERATIONS / omp_get_num_threads();
```

If the amount of time it takes to compute each iteration is the same, or nearly the same, then this is a perfectly
efficient way to parallelise the work. Each thread will finish its chunk at roughly the same time as the other thread.
But if the work is imbalanced, even if just one thread takes longer per iteration, then the threads become out of
sync and some will finish before others. This not only means that some threads will finish before others and have to
wait until the others are done before the program can continue, but it's also an inefficient use of resources to have
threads/cores idling rather than doing work.

Fortunately, we can use other types of "scheduling" to control how work is divided between threads. In simple terms, a
scheduler is an algorithm which decides how to assign chunks of work to the threads. We can controller the scheduler we
want to use with the `schedule` directive:

```c
#pragma omp parallel for schedule(SCHEDULER_NAME, OPTIONAL_ARGUMENT)
for (int i = 0; i < NUM_ITERATIONS; ++i) {
    ...
}
```

`schedule` takes too arguments: the name of the scheduler and an optional argument.

| Scheduler | Description | Argument |  Uses |
| - | - | - | - |
| static |  The work is divided into equal-sized chunks, and each thread is assigned a chunk to work on at compile time. | The chunk size to use. | Best used when the workload is balanced across threads, where each iteration takes roughly the same amount of time. |
| dynamic | The work is divided into lots of small chunks, and each thread is dynamically assigned a new chunk with it finishes its current work. | The chunk size to use. |  Useful for loops with a workload imbalance, or variable execution time per iteration. |
| guided |  The chunk sizes start large and decreases in size gradually. | The smallest chunk size to use. | Most useful when the workload is unpredictable, as the scheduler can adapt the chunk size to adjust for any imbalance. |
| auto | The best choice of scheduling is chosen at run time. | - | Useful in all cases, but can introduce additional overheads whilst it decides which scheduler to use. |
| runtime | Determined at runtime by the `OMP_SCHEDULE` environment variable or `omp_schedule` pragma. | - | - |

> ## Try out the different schedulers
>
> WIP: loop is not unbalanced enough to demonstrate schedulers well
>
> ```c
> #define NUM_ITERATIONS 100
>
> double start = omp_get_wtime();
>
> #pragma omp parallel for
> for (i = 0; i < NUM_ITERATIONS; i++) {
>   unbalanced_loop();
> }
>
> double end = omp_get_wtime();
>
> printf("Total time for %d reps = %f\n", NUM_ITERATIONS, end - start);
> ```
>
> Try out the different schedulers, with different block sizes and see how long it takes to finish the loop. Which
> scheduler was best?
>
> You should use this [code](code/solutions/04-schedulers-start.c), which includes the unbalanced loop, for this
> exercise.
>
> > ## Solution
> >
> > ```
> > Static:  Total time per rep = 0.000691
> > Dynamic: Total time per rep = 0.000615
> > Guided:  Total time per rep = 0.000618
> > ```
> >
> {: .solution}
{: .challenge}
