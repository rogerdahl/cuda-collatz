# CUDA Collatz

See [Wikipedia - Collatz Conjecture](http://en.wikipedia.org/wiki/Collatz_conjecture) and [On the 3x + 1 problem](http://www.ericr.nl/wondrous/index.html) for more information about this intriguing mathematical problem.

I think this might be the world's fastest single chip Collatz [Delay Records](#delay-record) calculator. With starting numbers in the 64 bit range, this app checks around 8 billion numbers per second for Delay Records on a GTX570 graphics card.

## Glossary

### N

A positive number for which the Delay is to be calculated. In theory, it can be infinitely large, but this implementation limits starting N to 64 bits and N that may be reached during the [Trajectory](#trajectory) to 128 bits.

### Delay

The number of iterations of the Collatz rules on a given N until the N reaches the value 1.

### Trajectory

The path a specific N follows from its first value until it reaches the value 1.

### Delay Class

All numbers N that have a specific [Delay](#delay).

### Class Record

The lowest [N](#n) in a given Delay Class.

The first N that we find that has a previously unencountered [Delay](#delay) is a Class Record. If the Delay is also in the highest [Delay Class](#delay-class), it is also a [Delay Record](#delay-record).

### Delay Record

The lowest N that has a Delay higher than the Delay of the previous Delay Record.


## Overview of implemented optimizations

The performance of this app is achieved by a combination of high level and low level optimizations. What follows is brief overview of each optimization.

### High level optimizations

These optimizations are described on the [Wikipedia - Collatz Conjecture](http://en.wikipedia.org/wiki/Collatz_conjecture) page.

#### Skipping N

Many [N](#n) can be ruled out as [Delay Records](#delay-record) without calculating their [Delays](#delay).

* All even N are skipped because any even N Delay Record can be derived directly from the previous odd numbered Delay Record. (Skips 50% of all N.)

* All N on the form of 3k+2 (N is congruent 2 modulo 3) are skipped because these numbers are not potential [Delay Records](#delay-record). (Skips 33% of all remaining N.)

* A table, called a sieve, is used to skip checking of many N. The sieve checks whether paths come together. If two paths join then the upper one can never yield a [Delay Record](#delay-record) (or [Class Record](#class-record)) and can be skipped. (Skips approximately 80% of all remaining N.)

    Example: Any N of the form 8k+4 will reach 6k+4 after 3 steps, and so will any N of the form 8k+5. Therefore no N of the form 8k+5 can be a Class Record (5 itself being the only exception). So any N of the form 8k+5 does not need to be checked, and all positions of the form 8k+5 in the sieve contain a zero.

After these optimization have been performed, Less than 7% of all N remain to actually be calculated. So, while the app checks around 8 billion Ns, it calculates the Delay of around 560 million N s.


#### Delay calculation optimizations

The [Delays](#delay) for the N that were not skipped must be calculated. The following optimizations are done when calculating Delays.

* Lookup tables are used to perform multiple steps of the Collatz calculation per iteration of the calculation loop.

* A table, called a "tail", is used to prevent having to calculate the Delays of small Ns. Once N drops below a certain value, the final delay is calculated by looking the remaining Delay up in this table and adding it to the current Delay.

* In addition, an early break technique has been tested. In this technique, N is compared with known, earlier [Delay Records](#delay-record). Calculation is ended early when it is determined that N cannot possibly become a Delay Record. Unfortunately, the speed increase from ending calculations early was outweighed by the overhead of continually checking N against a table of [Delay Records](#delay-record), resulting in a net decrease of calculation speed. So, the early break optimization has been left in the code, but has been disabled.


### Low Level optimizations

These are specific to my CUDA implementation for the GPU.

* The sieve technique that is described on Wikipedia yields a table that contains a power-of-two number of entries. For instance, a 20 bit sieve contains 2\^20 = 1,048,576 entries. The table is applicable to all Ns, to infinity. Each entry is a true/false value that describes if the corresponding N modulus X can be skipped. In a straightforward GPU implementation, one would create 1,048,576 threads, have each check its true/false value in the table, and abort if the value is false. This would abort around 80% of the values, and calculations would proceed on 20%. On the GPU, threads are run in warps. A warp has to keep running as long as at least 1 thread is active. In general, each warp would have a few threads active after sieving, so there would be almost no speed advantage on the GPU. The most fun I had with this project was with finding out how to optimize this on the GPU. The solution turned out to be simple. All I had to do was to transform the table of true/false values to a table of offsets to all the true values. In this way, only the same number of threads as there were true values had to be started and no threads had to be aborted. Each thread determines which N it should calculate by using its index to look up the corresponding offset in the sieve offsets table and adding it to the base N.

* Instead of individually performing the three steps described above that filter out N I rolled those filters into a combined table. Because one of the filters remove all numbers on the form 3k+2 (one number out of three), this was accomplished by creating three variations of the table, each filtering a different set of every three numbers and, for each block of N select the one that filters out the correct numbers for that N base.

* The step algorithm requires two tables called c and d. It also requires that 3 to-the-power of the lookup index be calculated for each lookup. Because the indexes into each of the tables and the index used in the 3 to-the-power-of calculation is the same for a given round in the loop, I created a table for the exp3 values and interleaved the three tables so that a single lookup could be used for finding both the c and d values and the 3exp value. I found that a step size of 19 is the largest step size in which none of the values in the tables overflow 32 bit values. The size of the step tables doubles for each additional step. 19 steps takes 2 \^ 19 \* 4 \* 4 = 8,388,608 bytes.

* The Delay calculation loop was simplified by making sure that the step table is wider than the sieve bits.

* In C, there is no efficient way of doing math operations with higher bit width than what is natively supported by the machine (because C does not support an efficient way of capturing the carry flag and including it in new calculations.) The target GPU, GF110, is a 32 bit machine and this calculator does 128 bit calculations while calculating the Delay, so it was written in PTX (A VM assembly language for NVIDIA GPUs). This helped speed up other operations as well.


## Sieve generator

As described above, the sieve is a precomputed table that specifies N for which no [Delay Record](#delay-record) are possible and thus, can be skipped.

A 19 bit wide sieve turned out to be the optimal size in my GPU implementation. Initially, I thought that the optimal size for the sieve would be the widest sieve that would fit in GPU memory, so I went about creating an app that could create an arbitrarily wide sieve.

Generating a small sieve is simple. To generate a sieve, say 10 bits wide, 1024k + i is calculated, where i loops from 0 to 1023. 10 steps of x/2 or (3x+1)/2 are done. After that a number on the form 3\^p + r is obtained. If some of those numbers end up with the same p and r, all of them can be skipped, except the lowest one.

However, this method does not work for generating a large sieve. The reason is that the algorithm is slowed down by a [Schlemiel the Painter's algorithm](http://en.wikipedia.org/wiki/Schlemiel_the_Painter%27s_algorithm). For each new entry in the table, the algorithm has to revisit all the previously generated entries. As the number of entries increases, the algorithm keeps slowing down, until it virtually grinds to a halt.

By analyzing the algorithm, I found that it could be implemented in a way that does not require revisiting all the previously generated entries for each new entry. The new algorithm makes it feasible to create large sieves. It works by creating entries that can be sorted in such a way that only a single pass over all the records is necessary.

A sieve that would use 2GB of memory covers 2 (because we remove even numbered bits in the end) \* 2GB \* 8 (bits per byte) = 32gbit = 2\^35 = 34 359 738 368 bits. To generate this sieve, it is necessary to have a sortable table with the same number of entries. Each entry is 16 bytes (optimized using bitfields). 16 bytes \* 34 359 738 368 entry = 512GB of temporary storage.

Unless one has a supercomputer with TBs of RAM, it is necessary to use disks for storage. I found a library called STXXL that implements STL for large datasets and includes algorithms that are efficient when using disk based storage. [STXXL](http://stxxl.sourceforge.net/) enabled me to easily create an app that manipulates the data in much the same way as I would with regular STL. The stxxl::sort is not in-place. It requires the same amount of disk space as the size of the data being sorted, to store the sorted runs during sorting. So another 512GB is required during the step that sorts the entries.

The same number of index records is also required, each is 64 bits + 8 bits = 9 bytes. This is less than the extra memory used by sorting the Collatz records, so the peak disk usage is 1TB.

Adding 20% for overhead, I determined that around 1.2TB of disk space was required to generate a 2\^35 sieve. At the time when I did this project, disks weren't that large, so I set up several of my largest disks in a JBOD configuration to hold the temporary data. The single file on there, that was over 1TB at one point, is still the biggest file I've seen. It took around two weeks to run the app, during which time the disks were working continuously.


## Technologies

CUDA, PTX, C++, Boost


## To  do

* There is one unused 32 bit word used for padding in the interleaved step table. It might be worth it to extend the exp3 to this word, so that more steps can be done in one iteration.

