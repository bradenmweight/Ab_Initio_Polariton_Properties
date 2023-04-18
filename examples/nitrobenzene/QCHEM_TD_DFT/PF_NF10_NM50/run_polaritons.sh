#!/bin/bash

for j in {0..2000..5}; do
    eta=$( bc -l <<< $j*0.0001 )
    #echo $eta
    sbatch submit.polariton ${eta}
done
