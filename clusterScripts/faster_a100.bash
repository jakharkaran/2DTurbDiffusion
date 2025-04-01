#!/bin/bash

for run in {0..20}
do
    # Submit the SLURM job with the parameters
    sbatch faster_a100.slurm "$run"

    # # Print the input string
    echo "Submitted run_num : $run"
done

