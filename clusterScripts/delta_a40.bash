#!/bin/bash

for run in {1000..1020}
do
    # Submit the SLURM job with the parameters
    sbatch delta_a40.slurm "$run"

    # # Print the input string
    echo "Submitted run_num : $run"
done