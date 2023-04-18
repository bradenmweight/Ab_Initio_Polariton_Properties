# __*Ab Initio* Molecular Polaritons__

## Proposed directory structure:
```
QCHEM_TD_DFT/ (Location for LR-TD-DFT using QCHEM)
QCHEM_TD_DFT/PF_NF5_NM10/ (Location for solving the Pauli-Fierz Hamiltonian)
QCHEM_TD_DFT/QCHEM.plots/ (Location of dens* and trans* Gaussian-type cube files)
```

## QCHEM
--To be run in QCHEM_TD_DFT/:
1. ```QCHEM.in``` (Example input file)
2. ```get_HAM_and_DIP_Matrix.py```

## Pauli-Fierz Hamiltonian
--To be run in QCHEM_TD_DFT/PF_NF5_NM10/:
1. ```Pauli-Fierz_DdotE.py``` (Modify Number of Fock/Electronic States) \
    a. Standalone Syntax: Pauli-Fierz_DdotE.py $A_0$ $\omega_c$ \
    b. One can also use the following scripts to scan {$A_0$} and {$\omega_c$} using and HPC cluster \
    c. ```run_polaritons.sh``` (Use to submit $N_{A_0}$ jobs for each coupling strength $\omega_c$) \
    d. ```submit.polariton``` (Call PF code $N{\omega_c}$ times for each cavity frequency $\omega_c$)

## Compute Properties
--To be run in QCHEM_TD_DFT/PF_NF5_NM10/
1. ```compute_properties.py```
2. ```submit.compute``` (To submit an HPC job)






