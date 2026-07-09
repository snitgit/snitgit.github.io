---
layout: post
title:  "The Cluster and the Bound: What HPC Actually Buys, and Where Physics Says Stop"
date:   2026-07-09 16:00:00 +0700
categories: HPC InformationTheory
tags: [hpc, simulation, parallel-computing, information-theory, shannon]
---

What high-performance computing actually buys you when you simulate the world — and why every engineering field, pushed far enough, runs into a limit that belongs to information theory.

<svg viewBox="0 0 720 280" role="img" aria-label="Several performance curves rising with effort, each saturating beneath one dashed capacity bound" style="max-width:100%;height:auto;font-family:monospace;">
<line x1="46" y1="24" x2="46" y2="246" stroke="currentColor" stroke-opacity="0.4"/>
<line x1="46" y1="246" x2="706" y2="246" stroke="currentColor" stroke-opacity="0.4"/>
<line x1="46" y1="190.5" x2="706" y2="190.5" stroke="currentColor" stroke-opacity="0.12"/>
<line x1="46" y1="135.0" x2="706" y2="135.0" stroke="currentColor" stroke-opacity="0.12"/>
<line x1="46" y1="79.5" x2="706" y2="79.5" stroke="currentColor" stroke-opacity="0.12"/>
<line x1="46" y1="55.1" x2="706" y2="55.1" stroke="currentColor" stroke-width="1.4" stroke-dasharray="6 5"/>
<polyline points="46.0,246.0 59.8,193.0 73.5,154.7 87.2,127.1 101.0,107.1 114.8,92.7 128.5,82.2 142.2,74.7 156.0,69.3 169.8,65.3 183.5,62.5 197.2,60.4 211.0,58.9 224.8,57.9 238.5,57.1 252.2,56.5 266.0,56.1 279.8,55.8 293.5,55.6 307.2,55.5 321.0,55.4 334.8,55.3 348.5,55.2 362.2,55.2 376.0,55.2 389.8,55.1 403.5,55.1 417.2,55.1 431.0,55.1 444.8,55.1 458.5,55.1 472.2,55.1 486.0,55.1 499.8,55.1 513.5,55.1 527.2,55.1 541.0,55.1 554.8,55.1 568.5,55.1 582.2,55.1 596.0,55.1 609.8,55.1 623.5,55.1 637.2,55.1 651.0,55.1 664.8,55.1 678.5,55.1 692.2,55.1 706.0,55.1" fill="none" stroke="currentColor" stroke-opacity="0.45" stroke-width="1.3"/>
<polyline points="46.0,246.0 59.8,212.4 73.5,184.7 87.2,161.8 101.0,143.0 114.8,127.5 128.5,114.8 142.2,104.3 156.0,95.6 169.8,88.5 183.5,82.6 197.2,77.7 211.0,73.7 224.8,70.5 238.5,67.8 252.2,65.5 266.0,63.7 279.8,62.2 293.5,60.9 307.2,59.9 321.0,59.0 334.8,58.3 348.5,57.8 362.2,57.3 376.0,56.9 389.8,56.6 403.5,56.3 417.2,56.1 431.0,55.9 444.8,55.8 458.5,55.7 472.2,55.6 486.0,55.5 499.8,55.4 513.5,55.3 527.2,55.3 541.0,55.3 554.8,55.2 568.5,55.2 582.2,55.2 596.0,55.2 609.8,55.1 623.5,55.1 637.2,55.1 651.0,55.1 664.8,55.1 678.5,55.1 692.2,55.1 706.0,55.1" fill="none" stroke="currentColor" stroke-opacity="0.45" stroke-width="1.3"/>
<polyline points="46.0,246.0 59.8,224.6 73.5,205.6 87.2,188.8 101.0,173.8 114.8,160.5 128.5,148.7 142.2,138.2 156.0,128.9 169.8,120.6 183.5,113.3 197.2,106.8 211.0,101.0 224.8,95.9 238.5,91.3 252.2,87.2 266.0,83.6 279.8,80.4 293.5,77.6 307.2,75.1 321.0,72.8 334.8,70.8 348.5,69.1 362.2,67.5 376.0,66.1 389.8,64.9 403.5,63.8 417.2,62.8 431.0,61.9 444.8,61.2 458.5,60.5 472.2,59.9 486.0,59.4 499.8,58.9 513.5,58.4 527.2,58.1 541.0,57.7 554.8,57.4 568.5,57.2 582.2,56.9 596.0,56.7 609.8,56.5 623.5,56.4 637.2,56.2 651.0,56.1 664.8,56.0 678.5,55.9 692.2,55.8 706.0,55.7" fill="none" stroke="currentColor" stroke-opacity="1" stroke-width="2"/>
<polyline points="46.0,246.0 59.8,232.8 73.5,220.4 87.2,209.0 101.0,198.3 114.8,188.4 128.5,179.1 142.2,170.5 156.0,162.5 169.8,155.1 183.5,148.1 197.2,141.7 211.0,135.7 224.8,130.1 238.5,124.9 252.2,120.0 266.0,115.5 279.8,111.3 293.5,107.4 307.2,103.8 321.0,100.4 334.8,97.3 348.5,94.4 362.2,91.6 376.0,89.1 389.8,86.7 403.5,84.5 417.2,82.5 431.0,80.6 444.8,78.8 458.5,77.2 472.2,75.6 486.0,74.2 499.8,72.9 513.5,71.7 527.2,70.5 541.0,69.4 554.8,68.4 568.5,67.5 582.2,66.7 596.0,65.9 609.8,65.1 623.5,64.4 637.2,63.8 651.0,63.2 664.8,62.6 678.5,62.1 692.2,61.6 706.0,61.1" fill="none" stroke="currentColor" stroke-opacity="0.45" stroke-width="1.3"/>
<polyline points="46.0,246.0 59.8,237.8 73.5,230.0 87.2,222.5 101.0,215.3 114.8,208.5 128.5,201.9 142.2,195.6 156.0,189.6 169.8,183.9 183.5,178.3 197.2,173.1 211.0,168.0 224.8,163.2 238.5,158.6 252.2,154.1 266.0,149.9 279.8,145.8 293.5,141.9 307.2,138.2 321.0,134.7 334.8,131.3 348.5,128.0 362.2,124.9 376.0,121.9 389.8,119.0 403.5,116.3 417.2,113.7 431.0,111.2 444.8,108.8 458.5,106.5 472.2,104.3 486.0,102.2 499.8,100.1 513.5,98.2 527.2,96.4 541.0,94.6 554.8,92.9 568.5,91.3 582.2,89.7 596.0,88.3 609.8,86.8 623.5,85.5 637.2,84.2 651.0,82.9 664.8,81.7 678.5,80.6 692.2,79.5 706.0,78.5" fill="none" stroke="currentColor" stroke-opacity="0.45" stroke-width="1.3"/>
<circle cx="706.0" cy="55.1" r="2.5" fill="currentColor" fill-opacity="0.45"/>
<circle cx="706.0" cy="55.1" r="2.5" fill="currentColor" fill-opacity="0.45"/>
<circle cx="706.0" cy="55.7" r="3.5" fill="#d05a4a"/>
<circle cx="706.0" cy="61.1" r="2.5" fill="currentColor" fill-opacity="0.45"/>
<circle cx="706.0" cy="78.5" r="2.5" fill="currentColor" fill-opacity="0.45"/>
<text x="54" y="47.1" font-size="11" fill="currentColor" fill-opacity="0.7">capacity bound</text>
<text x="576" y="268" font-size="11" fill="currentColor" fill-opacity="0.7">effort / resource &#8594;</text>
</svg>

*Fig. 1 — The shape this essay is about. Different systems, different engineering effort, one dashed line none of them cross. Clever implementations move you up a curve; only more physical resource moves the line.*

## Part I — What a cluster actually buys

"Run it on the cluster" sounds like a single action, but there is no single mechanism by which a cluster makes a simulation faster. Every simulator sits somewhere on a spectrum of four parallelism types, and knowing which one applies tells you — before you submit a single job — what the machine can and cannot do for you.

| Level | What runs in parallel | Scaling | What it costs |
|---|---|---|---|
| **A** | **Parameter sweep.** Independent full runs: geometries, random seeds, design corners | Near-linear, "embarrassingly parallel" | Nothing but a scheduler |
| **B** | **Domain decomposition.** Pieces of one big problem, exchanging boundary data | Good, until communication dominates | The solver must support it |
| **C** | **Solver kernels.** Threaded or GPU linear algebra inside one solve | Flattens past ~8–32 cores | Memory bandwidth, not cores |
| **D** | **Time & event parallelism.** Causally ordered events or timesteps | Poor — causality is serial | Rollback and synchronization machinery |

Amdahl's law is the referee: speedup can never exceed one over the serial fraction. Simulators whose inner loop is *causal* — event queues, cycle-by-cycle state machines, car-following updates — carry a large serial fraction per run. For those, the cluster's real gift is level A: running *many* runs at once, not one run faster. Consider how differently the common simulator families behave.

### Full-wave field solvers (FEM, electromagnetics)

A finite-element electromagnetic solve assembles a large sparse complex-valued system from the mesh, factors it, adaptively refines, and repeats. The bottleneck is the sparse factorization — memory-hungry and bandwidth-bound, so threading (level C) helps up to a modest core count and then flattens. The structural gift is that *frequency points are independent*: a swept analysis is a level-A ensemble hiding inside one project, and geometry sweeps stack a second layer of level A on top. Domain decomposition (level B) exists for electrically enormous problems, but most research-scale work never needs it. The cluster's value here is throughput: fifty geometries overnight, not one geometry in a tenth of the time.

### Hardware description simulation (event-driven RTL)

Event-driven logic simulation is the classic worst case for parallelism: a signal change schedules dependent evaluations in causal order, fine-grained and irregular — level D, where hard-won single-run speedups top out around 2–4×. Industry accelerates around the problem instead: regression farms running thousands of independent testbenches and seeds (level A, and where nearly all verification cycles actually go), compiled-code simulators that trade event semantics for raw speed, and emulation — mapping the design onto FPGAs, which is parallel because the *circuit itself* is parallel. The simulator isn't sped up; it's sidestepped.

### Agent-based microsimulation (traffic, crowds)

Car-following and lane-change models update every agent each timestep with local interactions. The step loop is cheap but stubbornly sequential, and spatial partitioning (level B) forces boundary synchronization every step. The practical pattern is scenario ensembles: demand seeds, control policies, calibration variants — hundreds of independent processes, a dispatcher, a collector. Level A again.

### Cycle-accurate architecture simulation (the most serial of all)

Simulating a processor cycle by cycle is one long dependency chain — typically well under a few million instructions per second, one host thread per simulated system. Here even level A gets an assist from statistics: design-space exploration runs hundreds of configurations as independent jobs, while sampling techniques simulate only representative slices of a workload. That shortcut is quietly an information argument — most simulated cycles carry redundant information about the statistic you actually want.

### Message passing on graphs (the happy exception)

Belief propagation inverts the pattern: in each iteration, every edge-message depends only on the *previous* iteration's messages, so all of them can be computed at once. Regular structure, dense arithmetic — GPUs devour it, with almost no serial fraction. This is why iterative decoders run at gigabits per second in hardware: the algorithm's graph *is* the parallel circuit.

The meta-lesson is worth stating plainly. For research computing, the highest-value skill is not making one simulation faster. It is shaping the scientific question into an *ensemble of independent runs*, then automating dispatch and collection. That architecture — a sweep dispatcher over a farm of solvers — transfers unchanged across field solvers, traffic models, architecture studies, and decoder analysis, because it exploits the one parallelism level that every simulator offers.

## Part II — The bound that keeps coming back

Now ask a different question of any of those fields: not "how fast can we simulate it," but "what is the *best possible* performance, independent of implementation cleverness?" The answer, remarkably often, arrives in the same form — an information or entropy argument over the field's physical degrees of freedom:

**physical resource → count degrees of freedom → apply Shannon → a bound no engineering can beat → measure the gap**

| Field | Scarce resource | The reopened bound |
|---|---|---|
| **Photolithography** | Photons per exposure | Shot noise sets the bits of pattern information a dose can carry; the stochastic defect floor is a capacity statement about the photon budget |
| **Antennas** | Aperture volume (electrical size) | Classical Q-factor limits; the number of significant radiating modes is a count of spatial channels, hence a Shannon capacity of a *structure*, not just a link |
| **Digital circuits** | Energy per switching event | Landauer's kT·ln 2 per erased bit; long interconnects behave as noisy channels — which is why memory buses and serial links now carry error-correcting codes |
| **Computer architecture** | Memory and interconnect bandwidth | The roofline model is a capacity line; communication lower bounds — how many bits *must* move — cap any algorithm's speed regardless of core count |
| **Road networks** | Network flow | The macroscopic fundamental diagram is a capacity curve; congestion is what operating past it looks like, much as a channel behaves beyond its rate limit |
| **Coding theory** | Channel uses | The original 1948 bound — and iterative decoding on graphs is the algorithm that finally approached it, fifty years later |

> Shannon's theorem is not about radios. It is a counting theorem about distinguishable states under a resource constraint.

That is why the same shape keeps appearing. Any field with a scarce physical resource, some noise or uncertainty, and a notion of "useful outcome" is a channel in disguise. The bound gets *reopened* — rediscovered inside the field's own vocabulary — at the moment its engineering matures enough to push against physics rather than against implementation. Early on, better engineering yields order-of-magnitude gains and nobody asks about limits. Late in the curve, every gain is small, and the interesting question flips from "how do we improve" to "how far from the ceiling are we, and what sets the ceiling?"

The two halves of this essay are one method. The bounds in the table are only computable if you know the physical degrees of freedom of a *real* system — the actual mode spectrum of a structure, the actual flow states of a network, the actual noise statistics of a process. Closed-form answers exist only for textbook geometries; everything else requires full-scale simulation, and full-scale simulation across a family of designs is affordable only as the level-A ensembles of Part I. The cluster is the microscope. The information bound is what you point it at.
