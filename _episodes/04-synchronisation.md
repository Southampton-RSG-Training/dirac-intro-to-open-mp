---
title: Loop Scheduling and Thread Synchronisation
slug: dirac-intro-to-openmp-synchronisation
teaching: 0
exercises: 0
questions:
-
objectives:
- Understand what thread synchronisation is
- Learn how to control thread synchronisation
- Understand the difference between different thread schedulers and regions
keypoints:
    - Blah
usemathjax: true
---

## Adding barriers

- Introduce reduction operations?

```c
int sum = 0;
#pragma omp parallel for reduction(+:sum)
for (int i = 0; i < 10; i++) {
    sum += i;
}

// Synchronize threads after the parallel loop
#pragma omp barrier

printf("Sum: %d\n", sum);
```

## Critical regions

Blah blah, this section is for talkin g about regions.

### `master`

### `single`

### `critical`
