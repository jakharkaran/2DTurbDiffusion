#!/bin/bash

for run in {1000..1020}
do
    # Submit the SLURM job with the parameters
    sbatch faster_t4.slurm "$run"

    # # Print the input string
    echo "Submitted run_num : $run"
done

