#!/bin/bash

for run in {0..2}
do
    # Submit the SLURM job with the parameters
    sbatch bridges.slurm "$run"

    # # Print the input string
    echo "Submitted run_num : $run"
done

