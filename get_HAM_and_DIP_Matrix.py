import numpy as np
import subprocess as sp
from matplotlib import pyplot as plt
import os, sys


RPA = True # TDA is always printed... if RPA=True, look for RPA results
DATA_DIR = "PLOTS_DATA"

def plot_dipole(DIP,label):

    xyz_dict = { 0:'x', 1:'y', 2:'z' }
    for dim in range(3):
        plt.imshow( np.abs( DIP[:,:,dim] ),origin='lower', cmap='afmhot_r')
        plt.colorbar(pad=0.01)
        plt.xlabel("Electronic State Index",fontsize=15)
        plt.ylabel("Electronic State Index",fontsize=15)
        plt.title(f"Electronic Dipole Matrix, |$\mu^{xyz_dict[dim]}$"+"$_{\\alpha\\beta}$| (a.u.)",fontsize=15)
        plt.savefig(f"{DATA_DIR}/DIP_{label}_{xyz_dict[dim]}.jpg", dpi=600)
        plt.clf()

    plt.imshow( np.sqrt( DIP[:,:,0]**2 + DIP[:,:,1]**2 + DIP[:,:,2]**2 ),origin='lower', cmap='afmhot_r')
    plt.colorbar(pad=0.01)
    plt.xlabel("Electronic State Index",fontsize=15)
    plt.ylabel("Electronic State Index",fontsize=15)
    plt.title("Electronic Dipole Matrix, |$\mu^{Tot.}_{\\alpha\\beta}$| (a.u.)",fontsize=15)
    plt.savefig(f"{DATA_DIR}/DIP_{label}_Total.jpg", dpi=600)
    plt.clf()

def plot_energies(E_TDA,E_RPA):
    dRPA = (E_RPA - E_RPA[0]) * 27.2114 # a.u. to eV
    dTDA = (E_TDA - E_TDA[0]) * 27.2114 # a.u. to eV
    plt.plot( np.arange(len(E_TDA)), dRPA, c="black", label="RPA" )
    plt.plot( np.arange(len(E_TDA)), dTDA, c="red", label="TDA" )
    plt.xlim(0,len(E_TDA))
    plt.xlabel("Electronic State Index",fontsize=15)
    plt.ylabel("Excitation Energy Difference, RPA - TDA (eV)",fontsize=15)
    plt.title("Electronic Dipole Matrix, |$\mu^{Tot.}_{\\alpha\\beta}$| (a.u.)",fontsize=15)
    plt.savefig(f"{DATA_DIR}/E_TDA_RPA_COMP.jpg", dpi=600)
    plt.clf()

    plt.plot( np.arange(len(E_TDA)), dRPA - dTDA )
    plt.xlabel("Electronic State Index",fontsize=15)
    plt.ylabel("Excitation Energy Difference, RPA - TDA (eV)",fontsize=15)
    plt.xlim(0,len(E_TDA))
    plt.title("Electronic Dipole Matrix, |$\mu^{Tot.}_{\\alpha\\beta}$| (a.u.)",fontsize=15)
    plt.savefig(f"{DATA_DIR}/E_TDA_RPA_DIFF.jpg", dpi=600)
    plt.clf()





sp.call(f"mkdir -p {DATA_DIR}", shell=True)

##### READ QCHEM OUTPUT #####

# Look for ground state total SCF energy
GS_ENERGY = float( sp.check_output("grep 'SCF   energy in the final basis set' QCHEM.out | awk '{print $9}'", shell=True).decode().split()[0] )

# Look for excitation energies from TDA and RPA
sp.call("grep 'Excited state' QCHEM.out > "+f"{DATA_DIR}/EXC_ENERGIES.dat", shell=True)
NROOTS = int( sp.check_output(f"wc -l {DATA_DIR}/EXC_ENERGIES.dat "+ "| awk '{print $1}'", shell=True).decode() )

if ( RPA == True ): 
    NROOTS //= 2
    print( f"\tThere are {NROOTS} TDDFT/TDA roots." )
    print( f"\tThere are {NROOTS} TDDFT/RPA roots." )
    sp.call(f"head -n {NROOTS} {DATA_DIR}/EXC_ENERGIES.dat" + " | awk '{print $8}' > "+f"{DATA_DIR}/EXC_TDA.dat", shell=True)
    sp.call(f"tail -n {NROOTS} {DATA_DIR}/EXC_ENERGIES.dat" + " | awk '{print $8}' > "+f"{DATA_DIR}/EXC_RPA.dat", shell=True)
else:
    print( f"\tThere are {NROOTS} TDDFT/TDA roots." )
    sp.call(f"head -n {NROOTS} {DATA_DIR}/EXC_ENERGIES.dat" + " | awk '{print $8}' > "+f"{DATA_DIR}/EXC_TDA.dat", shell=True)
sp.call(f"rm {DATA_DIR}/EXC_ENERGIES.dat",shell=True)

##### Create arrays to store all data #####
EAD = np.zeros(( NROOTS+1 )) # Add ground state
DIP = np.zeros(( NROOTS+1, NROOTS+1, 3 )) # Add ground state
###########################################
EAD[0] = GS_ENERGY # Already in a.u.
if ( RPA == True ):
    EAD[1:] = GS_ENERGY + np.loadtxt(f"{DATA_DIR}/EXC_TDA.dat") / 27.2114 # eV to a.u.
    np.savetxt(f"{DATA_DIR}/ADIABATIC_ENERGIES_TDA.dat",EAD)
    E_TDA = EAD * 1.0
    EAD[1:] = GS_ENERGY + np.loadtxt(f"{DATA_DIR}/EXC_RPA.dat") / 27.2114 # eV to a.u.
    np.savetxt(f"{DATA_DIR}/ADIABATIC_ENERGIES_RPA.dat",EAD)
    E_RPA = EAD * 1.0
    plot_energies(E_TDA,E_RPA)
else:
    EAD[1:] = GS_ENERGY + np.loadtxt(f"{DATA_DIR}/EXC_TDA.dat") / 27.2114 # eV to a.u.
    np.savetxt(f"{DATA_DIR}/ADIABATIC_ENERGIES_TDA.dat",EAD)


##### Look for dipoles from TDA #####
DIP[0,0,:] = np.array( sp.check_output("grep -A 4 'Electron Dipole Moments of Ground State' QCHEM.out | tail -n 1", shell=True).decode().split()[1:] ).astype(float)
DIP[np.arange(1,NROOTS+1), np.arange(1,NROOTS+1),:] = np.array( sp.check_output(f"grep -m 1 -A {3+NROOTS} 'Electron Dipole Moments of Singlet Excited State' QCHEM.out | tail -n {NROOTS}", shell=True).decode().split() ).astype(float).reshape(NROOTS,4)[:,1:]
DIP[0, np.arange(1,NROOTS+1),:] = np.array( sp.check_output(f"grep -m 1 -A {3+NROOTS} 'Transition Moments Between Ground and Singlet Excited States' QCHEM.out | tail -n {NROOTS}", shell=True).decode().split() ).astype(float).reshape(NROOTS,6)[:,2:5]
DIP[np.arange(1,NROOTS+1),0,:] = np.array( sp.check_output(f"grep -m 1 -A {3+NROOTS} 'Transition Moments Between Ground and Singlet Excited States' QCHEM.out | tail -n {NROOTS}", shell=True).decode().split() ).astype(float).reshape(NROOTS,6)[:,2:5]

# Excited-to-excited needs more work to be done...need a loop
NTERMS = NROOTS * (NROOTS-1) // 2 # Upper triangle without diagonal terms
tmp = np.array( sp.check_output(f"grep -m 1 -A {3+NTERMS} 'Transition Moments Between Singlet Excited States' QCHEM.out | tail -n {NTERMS}", shell=True).decode().split() ).astype(float).reshape(NTERMS,6)[:,:5]
for J,K,dx,dy,dz in tmp:
    DIP[int(J),int(K),:] = np.array([dx,dy,dz])
    DIP[int(K),int(J),:] = np.array([dx,dy,dz])

np.save(f"{DATA_DIR}/DIPOLE_TDA.dat.npy", DIP)
plot_dipole(DIP,"TDA")

if ( RPA == True ):

    DIP = np.zeros(( NROOTS+1, NROOTS+1, 3 )) # Add ground state

    ##### Look for dipoles from RPA #####
    DIP[0,0,:] = np.array( sp.check_output("grep -A 4 'Electron Dipole Moments of Ground State' QCHEM.out | tail -n 1", shell=True).decode().split()[1:] ).astype(float)
    DIP[np.arange(1,NROOTS+1), np.arange(1,NROOTS+1),:] = np.array( sp.check_output(f"grep -m 2 -A {3+NROOTS} 'Electron Dipole Moments of Singlet Excited State' QCHEM.out | tail -n {NROOTS}", shell=True).decode().split() ).astype(float).reshape(NROOTS,4)[:,1:]
    DIP[0, np.arange(1,NROOTS+1),:] = np.array( sp.check_output(f"grep -m 2 -A {3+NROOTS} 'Transition Moments Between Ground and Singlet Excited States' QCHEM.out | tail -n {NROOTS}", shell=True).decode().split() ).astype(float).reshape(NROOTS,6)[:,2:5]
    DIP[np.arange(1,NROOTS+1),0,:] = np.array( sp.check_output(f"grep -m 2 -A {3+NROOTS} 'Transition Moments Between Ground and Singlet Excited States' QCHEM.out | tail -n {NROOTS}", shell=True).decode().split() ).astype(float).reshape(NROOTS,6)[:,2:5]

    # Excited-to-excited needs more work to be done...need a loop
    NTERMS = NROOTS * (NROOTS-1) // 2 # Upper triangle without diagonal terms
    tmp = np.array( sp.check_output(f"grep -m 2 -A {3+NTERMS} 'Transition Moments Between Singlet Excited States' QCHEM.out | tail -n {NTERMS}", shell=True).decode().split() ).astype(float).reshape(NTERMS,6)[:,:5]
    for J,K,dx,dy,dz in tmp:
        DIP[int(J),int(K),:] = np.array([dx,dy,dz])
        DIP[int(K),int(J),:] = np.array([dx,dy,dz])

    np.save(f"{DATA_DIR}/DIPOLE_RPA.dat.npy", DIP)

    plot_dipole(DIP,"RPA")





