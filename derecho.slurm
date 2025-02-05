#!/bin/bash -l

### Job Name
#PBS -N myjob
#PBS -A URIC0009                                        # Charging account
#PBS -l select=1:ncpus=1:ngpus=1:mpiprocs=1:mem=32GB    # Request two resource chunks, each with 4 CPUs, GPUs, MPI ranks, and 40 GB of memory
#PBS -l gpu_type=a100                                   # gp100|v100|a100
#PBS -l job_priority=regular                            # regular|premium|economy>
#PBS -l walltime=03:00:00                               # Allow job to run up to 12 hour
#PBS -q regular                                         # Route the job to the casper queue
#PBS -j oe                                              # Join output and error streams into single file


module load cuda
source /glade/u/home/kjakhar/.bashrc
conda activate py39

python -u -m 2DTurbulenceUV.py ${yName} ${filterType} ${N} ${Nfilter} ${libraryDerivativeOrder} ${thresholdAlpha} ${tolerance} ${T} ${skipSnapshot} ${subtractG2} ${SAVE_DIR}