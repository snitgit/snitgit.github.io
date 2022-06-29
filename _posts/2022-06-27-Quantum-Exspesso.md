---
layout: post
title:  "Quantum ESPRESSO on exascale.mahidol.ac.th Multi-Node-Multi-GPUs MNMG Testing"
date:   2022-06-27 04:40:29 +0700
categories: Physics Chemistry 
---
Quantum ESPRESSO is an integrated suite of Open-Source computer codes for electronic-structure calculations and materials modeling at the nanoscale based on density-functional theory, plane waves, and pseudopotentials.



What can PWscf do ?

PWscf performs many different kinds of self-consistent calculations of electronic-structure properties within Density-Functional Theory (DFT), using a Plane-Wave (PW) basis set and pseudopotentials (PP).[3]

In particular:

+ ground-state energy and one-electron (Kohn-Sham) orbitals, atomic forces, stresses;

+ structural optimization, also with variable cell;

+ molecular dynamics on the Born-Oppenheimer surface, also with variable cell;

+ macroscopic polarization (and orbital magnetization) via Berry Phases;

+ various forms of finite electric fields, with a sawtooth potential or with the modern theory of polarization; 

+ Effective Screening Medium (ESM) method

# Download Dataset:

The environment variable BENCHMARK_DIR will be used throughout the example to refer to the directory containing the AUSURF112 input files.
```console
cd ~
mdkir qe
mkdir qe/ausurf
cd qe/ausurf
wget https://repository.prace-ri.eu/git/UEABS/ueabs/-/raw/master/quantum_espresso/test_cases/small/Au.pbe-nd-van.UPF
wget https://repository.prace-ri.eu/git/UEABS/ueabs/-/raw/master/quantum_espresso/test_cases/small/ausurf.in
```
# Run QE on Compute Node
```console
$ cd ~/qe/ausurf

$ module unload *
$ module load quantum_espresso/v7.0
$ mpirun -n 1  pw.x -npool 1 -ntg 1 -ndiag 1 -input ausurf.in 
```


References:
1. [MaX school][Max school] on Advanced Materials and Molecular Modelling with Quantum ESPRESSO, 

[Max school]: http://www.max-centre.eu/news-events/max-school-advanced-materials-and-molecular-modelling-quantum-espresso