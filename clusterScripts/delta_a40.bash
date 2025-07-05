#!/bin/bash

for run in {1000..1020}
do
    # Submit the SLURM job with the parameters
    sbatch delta_a40.slurm "$run"

    # # Print the input string
    echo "Submitted run_num : $run"
done


# for filepath in ../config/*.yaml; do
#     # extract just the filename
#     filename=${filepath#*/}

#     # submit the job using the full path
#     sbatch delta_a40.slurm "$filename"

#     # echo only the config name
#     echo "Submitted job for config: $filename"
# done

