---
title: Writing Parallel Applications with OpenMP
slug: dirac-intro-to-openmp-parallel-code
teaching: 0
exercises: 0
questions:
- How can I use OpenMP within a program?
objectives:
- Learn how to parallelise work in a program using OpenMP
- Describe two major OpenMP pragma directives
- Define and use a parallel region in our code
- Use OpenMP library functions to obtain the number of available threads and the current thread identifier
- Describe the classes of visibility (or scoping) of variables between threads
- Parallelise a for loop using OpenMP
- Describe the different schedulers available for how OpenMP assigns loop iterations to threads
- Change the scheduling behaviour for an example program
keypoints:
- Use `#pragma omp parallel` to define a parallel code section
- There are two types of variable scoping for parallel regions - *shared* (variables are shared across threads) and *private* (threads have their own copy of a variable separate to those of other threads).
- To avoid ambiguous code behaviour, it is good practice to explicitly default to a `none` variable sharing policy between thread, and define exceptions explicitly.
- Using `#pragma omp parallel for` is a shorter way of defining an `omp parallel` section and a `omp parallel for` within it.
- Using the library functions `omp_get_num_threads()` and `omp_get_thread_num()` outside of a parallel region will return 1 and 0 respectively.
- There are 5 different scheduling methods - *static*, *dynamic*, *guided*, *auto*, and *runtime*.
- We can use the `OMP_SCHEDULE` environment variable to define a scheduler and chunk size that is used by the `runtime` scheduler.
---

## Using OpenMP in a Program

As we introduced in the last episode,
OpenMP directives are special comments indicated by `#pragma omp` statements that guide the compiler in creating parallel code. 
They mark sections of code to be executed concurrently by multiple threads.
At a high level, the C/C++ syntax for pragma directives is as follows:

~~~
#pragma omp <name_of_directive> [ <optional_clause> ...]
~~~
{: .language-c}

Following a directive are multiple optional clauses, which are themselves C expressions and may contain other clauses,
with any arguments to both directives and clauses enclosed in parentheses and separated by commas. For example:

~~~
#pragma omp a-directive a-clause(argument1, argument2)
~~~
{: .language-c}

OpenMP offers a number of directives for parallelisation, although the two we'll focus on in this episode are:

- The `#pragma omp parallel` directive specifies a block of code for concurrent execution.
- The `#pragma omp for` directive parallelizes loops by distributing loop iterations among threads.


### Our First Parallelisation

For example, amending our previous example,
in the following we specify a specific block of code to run parallel threads,
using the OpenMP runtime routine `omp_get_thread_num()` to return
the unique identifier of the calling thread:

~~~
#include <stdio.h>
#include <omp.h>
int main() {
    #pragma omp parallel
    {
        printf("Hello from thread %d\n", omp_get_thread_num());
    }
}
~~~
{: .language-c}

So assuming you've specified `OMP_NUM_THREADS` as `4`:

~~~
Hello from thread 0
Hello from thread 1
Hello from thread 3
Hello from thread 2
~~~
{: .output}

Although the output may not be in the same order,
since the order and manner in which these threads (and their `printf` statements) run is not guaranteed.

So in summary, simply by adding this directive we have accomplished a basic form of parallelisation.

### What about Variables?

So how do we make use of variables across, and within, our parallel threads?
Of particular importance in parallel programs is how memory is managed and how and where variables can be manipulated,
and OpenMP has a number of mechanisms to indicate how they should be handled.
Essentially, OpenMP provided two ways to do this for variables:

- **Shared**: holds a single instance for all threads to share
- **Private**: creates and hold a separate copy of the variable for each thread

For example, what if we wanted to hold the thread ID and the total number of threads within variables in the code block?
Let's start by amending the parallel code block to the following:

~~~
        int num_threads = omp_get_num_threads();
        int thread_id = omp_get_thread_num();
        printf("Hello from thread %d out of %d\n", thread_id, num_threads);
~~~
{: .language-c}

Here, `omp_get_num_threads()` returns the total number of available threads.
If we recompile and re-run we should see:

~~~
Hello from thread 0 out of 4
Hello from thread 1 out of 4
Hello from thread 3 out of 4
Hello from thread 2 out of 4
~~~
{: .output}

> ## OpenMP and C Scoping
> 
> Try printing out `num_threads` at the end of the program, after the `#pragma` code block, and recompile.
> What happens? Is this what you expect?
> 
> > ## Solution
> > 
> > Since the variable is scoped only to the code block within the curly braces,
> > as with any C code block, `num_threads` is no longer in scope and cannot be read.
> {: .solution}
{: .challenge}

Now by default, variables declared within parallel regions are private to each thread.
But what about declarations outside of this block? For example:

~~~
    ...
    int num_threads, thread_id;

    #pragma omp parallel
    {
        num_threads = omp_get_num_threads();
        thread_id = omp_get_thread_num();
        printf("Hello from thread %d out of %d\n", thread_id, num_threads);
    }
~~~
{: .language-c}

Which may seem on the surface to be correct.
However this illustrates a critical point about why we need to be careful.
Now the variables declarations are outside of the parallel block,
by default, variables are *shared* across threads, which means these variables can be changed at any time by
any thread, which is potentially dangerous.
So here, `thread_id` may hold the value for another thread identifier when it's printed,
since there is an opportunity between it's assignment and it's access within `printf` to be changed in another thread.
This could be particularly problematic with a much larger data set and complex processing of that data,
where it might not be obvious that incorrect behaviour has happened at all,
and lead to incorrect results.
This is known as a *race condition*, and we'll look into them in more detail in the next episode.

> ## Observing the Race Condition
> 
> We can observe the race condition occurring by adding a sleep command between the `thread_id` assignment
> and use.
> Add `#include <unistd.h>` to the top of your program, and after `thread_id`'s assignment, add `sleep(2);`
> which will force the code to wait for 2 seconds before the variable is accessed,
> providing more opportunity for the race condition to occur.
> Hopefully you'll then see the unwanted behaviour emerge, for example:
> 
> ~~~
> Hello from thread 2 out of 4
> Hello from thread 2 out of 4
> Hello from thread 2 out of 4
> Hello from thread 2 out of 4
> ~~~
> {: .output}
{: .callout}

But with our code, this makes variables potentially *unsafe*, since within a single thread,
we are unable to guarantee their expected value.
One approach to ensuring we don't do this accidentally is to specify that there is no default behaviour for variable
classification.
We can do this by changing our directive to:

~~~
    #pragma omp parallel default(none)
~~~
{: .language-c}

Now if we recompile, we'll get an error mentioning that these variables aren't specified for use within the parallel region:

~~~
hello_world_omp.c: In function 'main':
hello_world_omp.c:10:21: error: 'num_threads' not specified in enclosing 'parallel'
   10 |         num_threads = omp_get_num_threads();
      |         ~~~~~~~~~~~~^~~~~~~~~~~~~~~~~~~~~~~
hello_world_omp.c:8:13: note: enclosing 'parallel'
    8 |     #pragma omp parallel default(none)
      |             ^~~
hello_world_omp.c:11:19: error: 'thread_id' not specified in enclosing 'parallel'
   11 |         thread_id = omp_get_thread_num();
      |         ~~~~~~~~~~^~~~~~~~~~~~~~~~~~~~~~
hello_world_omp.c:8:13: note: enclosing 'parallel'
    8 |     #pragma omp parallel default(none)
      |             ^~~
~~~
{: .output}

So we now need to be explicit in every case for which variables are accessible within the block,
and whether they're private or shared:

~~~
    #pragma omp parallel default(none) private(num_threads, thread_id)
~~~
{: .language-c}

So here, we ensure that each thread has its own private copy of these variables,
which is now thread safe.

### Parallel `for` Loops

A typical program uses `for` loops to perform many iterations of the same task,
and fortunately OpenMP gives us a straightforward way to parallelise them,
which builds on the use of directives we've learned so far.

~~~
    ...
    int num_threads, thread_id;
    
    omp_set_num_threads(4);
    
    #pragma omp parallel for default(none) private(num_threads, thread_id)
    for (int i = 1; i <= 10; i++)
    {
        num_threads = omp_get_num_threads();
        thread_id = omp_get_thread_num();
        printf("Hello from iteration %i from thread %d out of %d\n", i, thread_id, num_threads);
    }
    
    printf("%d",i);
}
~~~
{: .language-c}

So essentially, very similar format to before, but here we use `for` in the pragma preceding a loop definition,
which will then assign 10 separate loop iterations across the 4 available threads.
Later in this episode we'll explore the different ways in which OpenMP is able to schedule iterations from loops across these threads,
and how to specify different scheduling behaviours.

> ## A Shortcut for Convenience
> 
> The `#pragma omp parallel for` is actually equivalent to using two separate directives.
> For example:
> 
> ~~~
> #pragma omp parallel
> {
>     #pragma omp for
>     for (int 1 = 1; 1 <=10; i++)
>     {
>         ...
>     }
> }
> ~~~
> {: .language-c}
> 
> ...is equivalent to:
> 
> ~~~
> #pragma omp parallel for
> for (int 1 = 1; 1 <=10; i++)
> {
>     ...
> }
> ~~~
> {: .language-c}
> 
> In the first case, `#pragma omp parallel` spawns a group of threads,
> whilst `#pragma omp for` divides the loop iterations between them.
> But if you only need to do parallelisation within a single loop,
> the second case has you covered for convenience.
{: .callout}

Note we also explicitly set the number of desired threads to 4, using the OpenMP `omp_set_num_threads()` function,
as opposed to the environment variable method.
Use of this function will override any value set in `OMP_NUM_THREADS`.

You should see something (but perhaps not exactly) like:

~~~
Hello from iteration 1 from thread 0 out of 4
Hello from iteration 2 from thread 0 out of 4
Hello from iteration 3 from thread 0 out of 4
Hello from iteration 4 from thread 1 out of 4
Hello from iteration 5 from thread 1 out of 4
Hello from iteration 6 from thread 1 out of 4
Hello from iteration 9 from thread 3 out of 4
Hello from iteration 10 from thread 3 out of 4
Hello from iteration 7 from thread 2 out of 4
Hello from iteration 8 from thread 2 out of 4
~~~
{: .output}

So with careful attention to variable scoping,
using OpenMP to parallelise an existing loop is often quite straightforward.
However, particularly with more complex programs, there are some aspects and potential pitfalls with OpenMP parallelisation 
we need to be aware of - such as race conditions - which we'll explore in the next episode.

> ## Calling Thread Numbering Functions Elsewhere?
>
> Write, compile and run a simple OpenMP program that calls both `omp_get_num_threads()` and `omp_get_thread_num()` outside of a parallel region,
> and prints the values received.
> What happens?
> 
> > ## Solution
> > 
> > `omp_get_num_threads()` will return 1 as you might expect, since there is only the primary thread active.
> >
> > `omp_get_thread_num()` will return 0, which refers to the identifier for the primary thread, which is zero.
> {: .solution}
{: .challenge}

### Using Schedulers

Whenever we use a parallel for, the iterations have to be split into smaller chunks so each thread has something to do.
In most OpenMP implementations, the default behaviour is to split the iterations into equal sized chunks,

~~~
int CHUNK_SIZE = NUM_ITERATIONS / omp_get_num_threads();
~~~
{: .language-c}

If the amount of time it takes to compute each iteration is the same, or nearly the same, then this is a perfectly
efficient way to parallelise the work. Each thread will finish its chunk at roughly the same time as the other thread.
But if the work is imbalanced, even if just one thread takes longer per iteration, then the threads become out of
sync and some will finish before others. This not only means that some threads will finish before others and have to
wait until the others are done before the program can continue, but it's also an inefficient use of resources to have
threads/cores idling rather than doing work.

Fortunately, we can use other types of "scheduling" to control how work is divided between threads. In simple terms, a
scheduler is an algorithm which decides how to assign chunks of work to the threads. We can controller the scheduler we
want to use with the `schedule` directive:

~~~
#pragma omp parallel for schedule(SCHEDULER_NAME, OPTIONAL_ARGUMENT)
for (int i = 0; i < NUM_ITERATIONS; ++i) {
    ...
}
~~~
{: .language-c}

`schedule` takes too arguments: the name of the scheduler and an optional argument.

| Scheduler | Description | Argument |  Uses |
| - | - | - | - |
| static | The work is divided into equal-sized chunks, and each thread is assigned a chunk to work on at compile time. | The chunk size to use (default: divides iterations into chunks of approx. equal size). | Best used when the workload is balanced across threads, where each iteration takes roughly the same amount of time. |
| dynamic | The work is divided into lots of small chunks, and each thread is dynamically assigned a new chunk with it finishes its current work. | The chunk size to use (default: 1). |  Useful for loops with a workload imbalance, or variable execution time per iteration. |
| guided |  The chunk sizes start large and decreases in size gradually. | The smallest chunk size to use (default: 1). | Most useful when the workload is unpredictable, as the scheduler can adapt the chunk size to adjust for any imbalance. |
| auto | The best choice of scheduling is chosen at run time. | - | Useful in all cases, but can introduce additional overheads whilst it decides which scheduler to use. |
| runtime | Determined at runtime by the `OMP_SCHEDULE` environment variable or `omp_schedule` pragma. | - | - |

> ## Try Out Different Schedulers
>
> Try each of the static and dynamic schedulers on the code below,
> which uses `sleep` to mimic processing iterations that take increasing amounts of time to complete as the loop increases.
> `static` is already specified, so replace this next with `dynamic`.
> Which scheduler is fastest?
>
> ~~~
> #include <unistd.h>
> #include <stdlib.h>
> #include <stdio.h>
> #include <omp.h>
> 
> #define NUM_THREADS 4
> #define NUM_ITERATIONS 8
> 
> int main ( ) {
>     int i;
>     double start = omp_get_wtime();
> 
>     #pragma omp parallel for num_threads(NUM_THREADS) schedule(static)
>     for (i = 0; i < NUM_ITERATIONS; i++) {
>         sleep(i);
>         printf("Thread %d finished iteration %d\n", omp_get_thread_num(), i);
>     }
> 
>     double end = omp_get_wtime();
>     printf("Total time for %d reps = %f\n", NUM_ITERATIONS, end - start);
> }
> ~~~
> {: .language-c}
>
> Try out the different schedulers and see how long it takes to finish the loop.
> Which scheduler was best?
>
> > ## Solution
> >
> > You should see something like:
> >
> > ~~~
> > Static:  Total time for 8 reps = 13.003299
> > Dynamic: Total time for 8 reps = 10.007052
> > ~~~
> > 
> > Here we can see that `dynamic` is the fastest,
> > which is better with iterations taking differing amounts of time.
> > But note there is an overhead to using dynamic scheduling,
> > threads that complete need to stop and await a new value to process from a next iteration.
> {: .solution}
> 
> With a dynamic scheduler, the default chunk size is 1.
> What happens if specify a chunk size of 2,
> i.e. `scheduler(dynamic, 2)`?
> 
> > ## Solution
> >
> > ~~~
> > Dynamic: Total time for 16 reps = 13.004029
> > ~~~
> > 
> > So here, we now see approximately the same results.
> > By increasing the chunk size, the dynamic scheduler behaves more like the static one,
> > since the workload for static would have the same chunk size calculated to be 2
> > (`NUM_ITERATIONS / NUM_THREADS = CHUNK_SIZE`, `8 / 4 = 2`).
> {: .solution}
{: .challenge}

> ## A Matter of Convenience
> 
> We've seen that we can amend our code directly to use different schedulers,
> but when testing our code with each of them editing and recompiling can become tedious.
> Fortunately we can use the `OMP_SCHEDULE` environment variable to specify the scheduler instead,
> as well as the chunk size, so we don't need to recompile.
> 
> Edit your code to specify `runtime` as the scheduler, i.e. `scheduler(runtime)`,
> recompile, then set the environment variable in turn to each scheduler, e.g.
> 
> ~~~
> export OMP_SCHEDULE=dynamic
> ~~~
> {: .language-bash}
> 
> Then rerun. Try it with different chunk sizes too, e.g.:
> 
> ~~~
> export OMP_SCHEDULE=static,1
> ~~~
> {: .language-bash}
> 
> So much more convenient!
{: .challenge}

