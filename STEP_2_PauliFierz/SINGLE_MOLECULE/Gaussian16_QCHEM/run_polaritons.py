import numpy as np
import os

WC   = 3.154 # eV
EPOL = "111" # Cavity polarization -- "111", "001", "110", "101", "011"
NM   = 5 # Number of electronic states (including ground state)
NF   = 5 # Number of Fock states

A0MIN = 0.0
A0MAX = 0.5
dA0   = 0.1

A0_LIST = np.arange( A0MIN, A0MAX+dA0, dA0 )

for A0 in A0_LIST:
    os.system(f"python3 PF.py {NM} {NF} {A0} {WC} {EPOL}")


