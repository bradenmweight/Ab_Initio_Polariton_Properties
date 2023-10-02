#!/bin/bash

# Braden M. Weight, 2023

# Note: For single-point calculation, set MIN = MAX

A0_MIN="0.0"
A0_MAX="1.0"
dA0="0.05"

WC_MIN="0.0"
WC_MAX="20.0"
dWC="1.0"

E_POL="001" # Electric field polarization. 
            # Input as three positive integers. 
            # Will normalize later.

NM=50 # Number of electronic states
NF=5 # Number of photonic Fock states

NA0=$(printf %.0f $(echo "($A0_MAX-$A0_MIN)/$dA0" | bc -l))
NWC=$(printf %.0f $(echo "($WC_MAX-$WC_MIN)/$dWC" | bc -l))

for (( a=0; a<=NA0; a++ )); do
        A0=$( bc -l <<< $a*$dA0+$A0_MIN )
        sbatch submit.polariton ${A0} ${NM} ${NF} ${NWC} ${WC_MIN} ${WC_MAX} ${dWC} ${E_POL}
done



#### FOR THETA/PHI SCAN ####
for A in {0.1,0.2,0.3,0.4}; do
    sbatch submit.polariton ${A}
done


