import numpy as np
import subprocess as sp
from matplotlib import pyplot as plt
import os
import glob

# Created by Braden M. Weight, 2022

"""
Note: Requires the use of Multwfn to get Gaussian dipole matrix elements
Multiwfn Homepage -- http://sobereva.com/multiwfn/
Multiwfn Download -- http://sobereva.com/multiwfn/download.html
Dipoles are computed at TDA level regardless of whether Gaussian did TDA or RPA.

Example Gaussian Input:
    #p B3LYP/6-31G*
    #p TD=(singlets,nstates=50) IOp(6/8=3) IOp(9/40=4)

After TD calculation, formchk the .chk file to get the .fchk file
$ formchk geometry.chk geometry.fchk
"""

def get_Globals():
    global MULTIWFN
    MULTIWFN   = "$HOME/Multiwfn_3.7_bin_Linux_noGUI/Multiwfn"
    
    # Automatically detect .out and .fchk files
    # If naming convention is weird, then manually set these variables as shown in the comments below
    global OUT_FILE, FCHK_FILE
    OUT_FILE  = glob.glob("*.out")[0]  # "geometry_test.out"
    FCHK_FILE = glob.glob("*.fchk")[0] # "geometry_test.fchk"

    # Automatically detect the number of excited states in the TD-DFT calculation
    # One can also use less than the total number of excited states by manually setting NExcStates
    NExcStates = int( sp.check_output("grep 'Excited State' %s | tail -n 1 | awk '{print $3}'" % (OUT_FILE), shell=True).decode().split(":")[0] )

    # Do not change below here
    global DATA_DIR, NSTATES
    DATA_DIR   = "PLOTS_DATA"
    NSTATES    = NExcStates + 1
    sp.call(f"mkdir {DATA_DIR}", shell=True)

def run_Multiwfn():
    # Makes permanent dipoles
    STRING = f'{MULTIWFN} << EOF\n{FCHK_FILE}\n18\n5\n{OUT_FILE}\n2\nEOF'
    sp.call(STRING,shell=True)

    # Makes transition dipoles
    STRING = f'{MULTIWFN} << EOF\n{FCHK_FILE}\n18\n5\n{OUT_FILE}\n4\nEOF'
    sp.call(STRING,shell=True)

def get_Energies_Dipoles():
    # Need to run Multiwfn ( *.fchk 18, 5, *.out 2 ) ( 18, 5, *.out, 4 )
    run_Multiwfn()

    DIP_MAT     = np.zeros(( NSTATES, NSTATES, 3 )) # All dipole matrix elements
    E_ADIABATIC = np.zeros( NSTATES ) # Adiabatic molecular energies

    # Read in permanent dipole moments (in a.u.)
    permFile = open("dipmom.txt","r").readlines()
    DIP_MAT[0,0] = np.array( permFile[1].split()[6:9] )
    permFile = np.array( permFile[5:NSTATES+4] )
    exc_state_perm_dips = np.array( [ permFile[j].split()[1:4] for j in range(NSTATES-1) ] ).astype(float)
    for j in range( NSTATES-1 ):
        DIP_MAT[1+j,1+j] = exc_state_perm_dips[j] # CHECK THIS WITH POLAR MOLECULAR SYSTEM

    # Read in transition energies (in eV)
    E_TRANSITION = np.array([ permFile[j].split()[4] for j in range(NSTATES-1) ]).astype(float) # These are in eV

    # Get ground state energy from SCF cycle
    sp.call(f" grep 'SCF Done' {OUT_FILE} > GS_Total_Energy.dat ",shell=True) # FOR HF/DFT etc.
    #sp.call(" grep 'Wavefunction amplitudes converged' {OUT_FILE} > GS_Total_Energy.dat ",shell=True) # FOR CCSD/MP2/etc.
    GS_Energy = float(open("GS_Total_Energy.dat","r").readlines()[0].split()[4]) * 27.2114 # Convert to eV
    sp.call(" rm GS_Total_Energy.dat ",shell=True)

    E_ADIABATIC[0] = GS_Energy # Shift states by total energy of ground state at this R
    for j in range( NSTATES-1 ):
        E_ADIABATIC[1+j] = GS_Energy + E_TRANSITION[j] # In eV
    
    # Read in transition dipole moments
    transDipFile = open("transdipmom.txt","r").readlines()
    for line in transDipFile:
        t = line.split()
        if (  len(t) == 7 and t[0] != "i" and t[0] != "Transition" ):
            i = int( t[0] )
            j = int( t[1] )
            if ( i <= NSTATES-1 and j <= NSTATES-1 and i != j ): # Exclude terms from ground state and permanant dipoles
                dxyz = np.array( t[2:5] )
                DIP_MAT[ i, j ] = dxyz
                DIP_MAT[ j, i ] = dxyz

    """
    # Read in EOM-CCSD Ground-to-excited dipole moments. These should be exact, but we neglect excited-to-ground transition dipole moments.
    sp.call(''' grep "Ground to excited state transition electric dipole moments" geometry.out -A 31 | tail -n 30 > G2E_TdDip.dat ''', shell=True )
    transDipFile = np.loadtxt("G2E_TdDip.dat")[:,1:4] # Only <g|r|ej> terms
    for j in range(NExcStates):
        DIP_MAT[ 0, j+1 ] = transDipFile[j]
        DIP_MAT[ j+1, 0 ] = DIP_MAT[ 0, j+1 ]
    """

    for d in [0,1,2]:
        plt.imshow( np.abs( DIP_MAT[:,:,d] ), origin='lower' )
        plt.colorbar()
        plt.title("Dipole (a.u.)")
        if ( d == 0 ):
            plt.savefig(f"{DATA_DIR}/DIPOLE_RPA_x.jpg")
        if ( d == 1 ):
            plt.savefig(f"{DATA_DIR}/DIPOLE_RPA_y.jpg")
        if ( d == 2 ):
            plt.savefig(f"{DATA_DIR}/DIPOLE_RPA_z.jpg")
        plt.clf()

    outDipFile = open(f"{DATA_DIR}/DIPOLE_RPA_ssddd.dat","w")
    for i in range( NSTATES ):
        for j in range( i, NSTATES ):
            outDipFile.write(f"{i} {j} {DIP_MAT[i,j,0]} {DIP_MAT[i,j,1]} {DIP_MAT[i,j,2]}\n")
    outDipFile.close()
    np.savetxt(f"{DATA_DIR}/DIPOLE_RPA_x.dat",DIP_MAT[:,:,0])
    np.savetxt(f"{DATA_DIR}/DIPOLE_RPA_y.dat",DIP_MAT[:,:,1])
    np.savetxt(f"{DATA_DIR}/DIPOLE_RPA_z.dat",DIP_MAT[:,:,2])
    np.save(f"{DATA_DIR}/DIPOLE_RPA.dat.npy",DIP_MAT[:,:,:])

    np.savetxt(f"{DATA_DIR}/ADIABATIC_ENERGIES_RPA.dat", E_ADIABATIC/27.2114)
    np.savetxt(f"{DATA_DIR}/ADIABATIC_ENERGIES_RPA_TRANSITION.dat", (E_ADIABATIC-E_ADIABATIC[0])/27.2114)

if ( __name__ == "__main__" ):
    get_Globals()
    get_Energies_Dipoles()