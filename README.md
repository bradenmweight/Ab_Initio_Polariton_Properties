# ```Ab Initio Polaritons```

## Proposed directory structure:
```
QCHEM_TD_DFT/
QCHEM_TD_DFT/PF_NF5_NM10/
QCHEM_TD_DFT/PF_NF5_NM10/data_PF/
```

## QCHEM
--To be run in QCHEM_TD_DFT/:
1. QCHEM.in (Example input file)
2. get_HAM_and_DIP_Matrix.py
(also can use Gaussian16 with Multiwfn -- to be described later)

## Pauli-Fierz Hamiltonian
--To be run in QCHEM_TD_DFT/PF_NF5_NM10/:
1. Pauli-Fierz_DdotE.py (Modify Number of Fock/Electronic States)
2. run_polaritons.sh (Use to submit $N{\omega_c}$ jobs for each cavity frequency $\omega_c$)
3. submit.polariton (Call Pauli-Fierz_DdotE.py $N_{A_0}$ times for each coupling strength $A_0$)

## Compute Properties
--To be run in QCHEM_TD_DFT/PF_NF5_NM10/
1. compute_properties.py






