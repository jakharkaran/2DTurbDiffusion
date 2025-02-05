#!/bin/bash

for run in 0
do
    qsub -v run=$run derecho.pbs
    echo $i
done