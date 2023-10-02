#!/bin/bash

for A in {0.1,0.2,0.3,0.4}; do
#for a in {0..4000..100}; do
    #A=$( bc -l <<< $a*0.0001 )
    sbatch submit.polariton ${A}
done
