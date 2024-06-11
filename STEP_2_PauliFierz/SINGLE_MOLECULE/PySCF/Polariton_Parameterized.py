import numpy as np
import subprocess as sp
from matplotlib import pyplot as plt

import H_EL

DATA_DIR = "Polariton_DATA/"
sp.call(f"mkdir -p {DATA_DIR}", shell=True)

def get_Pauli_Fierz_Hamiltonian( CAV_DICT ):
    NEL       = CAV_DICT["NEL"]
    NF        = CAV_DICT["NF"]
    WC        = CAV_DICT["WC"]
    LAMBDA    = CAV_DICT["LAMBDA"]
    SUBSPACE  = CAV_DICT["SUBSPACE"]
    ENERGY_AD = CAV_DICT["ENERGY_AD"][:NEL]
    DIPOLE_AD = CAV_DICT["DIPOLE_AD"][:NEL,:NEL]

    if ( SUBSPACE.upper() == "FULL" ):
        print( "dim(H) = (%1.0f,%1.0f)" % (NEL*NF,NEL*NF) )
        Iel  = np.eye( NEL )
        Iph  = np.eye( NF )
        a_op = np.diag(np.sqrt(np.arange(1,NF)), k=1)
        H   = np.zeros( (NEL*NF, NEL*NF) )
        H  += np.kron( np.diag( ENERGY_AD ), Iph ) # Electronic Part
        H  += np.kron( Iel, WC * np.diag(np.arange(NF)) ) # Photonic Part
        H  += np.sqrt( WC / 2 ) * LAMBDA * np.kron( DIPOLE_AD, a_op.T + a_op ) # Bi-linear Coupling
        H  += 0.500 * LAMBDA**2 * np.kron( DIPOLE_AD @ DIPOLE_AD, Iph ) # Dipole self-energy
    elif ( SUBSPACE.upper() == "TRUNCATED" ):
        if ( NF >= 3 ):
            print ( "For 01 subspace, NF can only be 2." )
            print ("Setting NF = 2")
            NF = 2
        print( "dim(H) = (%1.0f,%1.0f)" % (1 + (NEL - 1) + 1, 1 + (NEL - 1) + 1) )
        H           = np.zeros( (1 + (NEL - 1) + 1, 1 + (NEL - 1) + 1) )
        H[(np.arange(NEL),np.arange(NEL))] = ENERGY_AD[:]
        H[-1,-1]    = H[0,0] + WC
        H[0,-1]     = np.sqrt( WC / 2 ) * LAMBDA * DIPOLE_AD[0,0]
        H[-1,0]     = np.sqrt( WC / 2 ) * LAMBDA * DIPOLE_AD[0,0]
        H[-1,1:-1]  = np.sqrt( WC / 2 ) * LAMBDA * DIPOLE_AD[0,1:]
        H[1:-1,-1]  = np.sqrt( WC / 2 ) * LAMBDA * DIPOLE_AD[0,1:]
        DSE = 0.500 * LAMBDA**2 * DIPOLE_AD @ DIPOLE_AD
        for j in range( NEL ):
            for k in range( j, NEL ):
                H[j,k] += DSE[j,k]
                H[k,j] += DSE[j,k]
    else:
        print(f"Subspace not implemented: {SUBSPACE}")
        exit()

    return H

def get_Jaynes_Cummings_Hamiltonian( CAV_DICT ):
    NEL       = CAV_DICT["NEL"]
    NF        = CAV_DICT["NF"]
    WC        = CAV_DICT["WC"]
    LAMBDA    = CAV_DICT["LAMBDA"]
    SUBSPACE  = CAV_DICT["SUBSPACE"]
    ENERGY_AD = CAV_DICT["ENERGY_AD"][:NEL]
    DIPOLE_AD = CAV_DICT["DIPOLE_AD"][:NEL,:NEL]

    if ( SUBSPACE.upper() == "FULL" ):
        print( "dim(H) = (%1.0f,%1.0f)" % (NEL*NF,NEL*NF) )
        Iel  = np.eye( NEL )
        Iph  = np.eye( NF )
        a_op = np.diag(np.sqrt(np.arange(1,NF)), k=1)
        DIPOLE_AD = DIPOLE_AD - np.diag(np.diagonal(DIPOLE_AD)) # Remove permanent dipole
        DIP_MINUS = np.triu( DIPOLE_AD )
        DIP_PLUS  = np.tril( DIPOLE_AD )
        H   = np.zeros( (NEL*NF, NEL*NF) )
        H  += np.kron( np.diag( ENERGY_AD ), Iph ) # Electronic Part
        H  += np.kron( Iel, WC * np.diag(np.arange(NF)) ) # Photonic Part
        H  += np.sqrt( WC / 2 ) * LAMBDA * np.kron( DIP_MINUS, a_op.T ) # Bi-linear Coupling
        H  += np.sqrt( WC / 2 ) * LAMBDA * np.kron( DIP_PLUS , a_op   ) # Bi-linear Coupling
    elif ( SUBSPACE.upper() == "TRUNCATED" ):
        if ( NF >= 3 ):
            print ( "For 01 subspace, NF can only be 2." )
            print ("Setting NF = 2")
            NF = 2
        print( "dim(H) = (%1.0f,%1.0f)" % (1 + (NEL - 1) + 1, 1 + (NEL - 1) + 1) )
        H           = np.zeros( (1 + (NEL - 1) + 1, 1 + (NEL - 1) + 1) )
        H[(np.arange(NEL),np.arange(NEL))] = ENERGY_AD[:] # Diagonal energies
        H[-1,-1]    = H[0,0] + WC # Photonic Part
        H[-1,1:-1]  = np.sqrt( WC / 2 ) * LAMBDA * DIPOLE_AD[0,1:] # Bi-linear Coupling
        H[1:-1,-1]  = np.sqrt( WC / 2 ) * LAMBDA * DIPOLE_AD[0,1:] # Bi-linear Coupling
    else:
        print(f"Subspace not implemented: {SUBSPACE}")
        exit()
    
    return H

def solve_Polariton_Hamiltonian( CAV_DICT ):

    if ( len(CAV_DICT["DIPOLE_AD"].shape) == 3 ):
        # Do dipole projection
        EPOL = CAV_DICT["EPOL"]
        CAV_DICT["DIPOLE_AD"] = np.einsum('JKd,d->JK', CAV_DICT["DIPOLE_AD"], EPOL ) / np.linalg.norm( EPOL )

    if ( CAV_DICT["HAM"].upper() == "PF" ):
        H = get_Pauli_Fierz_Hamiltonian( CAV_DICT )
    elif ( CAV_DICT["HAM"].upper() == "JC" ):
        H = get_Jaynes_Cummings_Hamiltonian( CAV_DICT )
    else:
        print(f"Hamiltonian not implemented: {CAV_DICT['HAM'].upper()}")
        exit()

    E, U = np.linalg.eigh( H )
    np.savetxt("%s/E_LAM_%1.4f_WC_%1.4f_EPOL_%s_HAM_%s_NEL_%1.0f_NF_%1.0f_SUBSPACE_%s.dat" % (DATA_DIR,CAV_DICT["LAMBDA"],CAV_DICT["WC"],"".join(map(str,CAV_DICT["EPOL"])), CAV_DICT["HAM"], CAV_DICT["NEL"], CAV_DICT["NF"], CAV_DICT["SUBSPACE"]), E, fmt="%1.8f")
    np.save   ("%s/E_LAM_%1.4f_WC_%1.4f_EPOL_%s_HAM_%s_NEL_%1.0f_NF_%1.0f_SUBSPACE_%s.npy" % (DATA_DIR,CAV_DICT["LAMBDA"],CAV_DICT["WC"],"".join(map(str,CAV_DICT["EPOL"])), CAV_DICT["HAM"], CAV_DICT["NEL"], CAV_DICT["NF"], CAV_DICT["SUBSPACE"]), E)
    return E, U

if ( __name__ == "__main__"):

    do_HEL = False # Calculate Electronic Energies and Dipoles OR read from file if already done

    if ( do_HEL == True ):
        # Do electronic calculation to get the energies and dipoles (i.e., adiabatic basis)
        atom       = [("Li", 0, 0, 0,), ("F", 2.0, 0, 0)]
        basis      = '321g'
        unit       = 'Angstrom'
        functional = "wB97XD"
        nstates    = 10 # Number of excited states
        ENERGY_AD, DIPOLE_AD = H_EL.get_ELECTRONIC_ENERGY_DIPOLE( atom=atom, unit=unit, basis=basis, functional=functional, nstates=nstates)
    else:
        PATH_TO_DATA = "H_electronic_DATA/"
        ENERGY_AD = np.loadtxt(f"{PATH_TO_DATA}/ENERGY.dat")
        DIPOLE_AD = np.load(f"{PATH_TO_DATA}/DIPOLE.npy")

    # Contruct polaritonic Hamiltonian
    CAV_DICT = {}
    CAV_DICT["ENERGY_AD"] = ENERGY_AD
    CAV_DICT["DIPOLE_AD"] = DIPOLE_AD
    CAV_DICT["read_HEL"]  = True # True or False # Whether to do electronic calculation or read from file
    CAV_DICT["NEL"]       = 2     # Number of electronic states (including ground state)
    CAV_DICT["NF"]        = 2      # Fock States
    CAV_DICT["EPOL"]      = np.array([1,1,1]) # Cavity polarization vector



    # Example 1: Excited State Splitting
    LAM_LIST = np.arange( 0,0.0105,0.0005 )
    HAM_LIST = ["JC", "PF"]
    SUB_LIST = ["TRUNCATED", "FULL"]
    color_list = ["black", "red", "blue", "green"]
    line_list = ["-", "o", "-", "o"]
    ENERGY   = np.zeros( (len(HAM_LIST), len(SUB_LIST), len(LAM_LIST), 100) )
    icount   = 0
    for HAMi,HAM in enumerate( HAM_LIST ):
        print( HAM )
        for SUBi,SUB in enumerate( SUB_LIST ):
            print( SUB )
            for LAMi,LAM in enumerate( LAM_LIST ):
                CAV_DICT["HAM"]       = HAM
                CAV_DICT["SUBSPACE"]  = SUB
                CAV_DICT["LAMBDA"]    = LAM
                CAV_DICT["WC"]        = CAV_DICT["ENERGY_AD"][1] - CAV_DICT["ENERGY_AD"][0]    # Cavity Frequency, a.u.
                E_TMP, _ = solve_Polariton_Hamiltonian( CAV_DICT )
                ENERGY[HAMi,SUBi,LAMi,:len(E_TMP)] = E_TMP[:] * 27.2114 # a.u. to eV
            for state in range( 3 ):
                if ( state == 0 ):
                    plt.plot( LAM_LIST, ENERGY[HAMi,SUBi,:,state] - ENERGY[HAMi,SUBi,0,0], line_list[icount], c=color_list[icount], label="%s %s" % (HAM, SUB) )
                else:
                    plt.plot( LAM_LIST, ENERGY[HAMi,SUBi,:,state] - ENERGY[HAMi,SUBi,0,0], line_list[icount], c=color_list[icount] )
            icount += 1
    plt.legend()
    plt.xlabel("Coupling Strength, $\lambda$ (a.u.)", fontsize=15)
    plt.ylabel("Energy (eV)", fontsize=15)
    plt.xlim(LAM_LIST[0],LAM_LIST[-1])
    plt.ylim(3.07,3.175)
    plt.tight_layout()
    plt.savefig(f"{DATA_DIR}/Example1_ExcitedStateSplitting.jpg", dpi=300)
    plt.clf()

    # Example 1: Excited State Splitting
    LAM_LIST = np.arange( 0,0.205,0.005 )
    HAM_LIST = ["JC", "PF"]
    SUB_LIST = ["TRUNCATED", "FULL"]
    color_list = ["black", "red", "blue", "green"]
    line_list = ["-", "o", "-", "o"]
    ENERGY   = np.zeros( (len(HAM_LIST), len(SUB_LIST), len(LAM_LIST), 100) )
    icount   = 0
    for HAMi,HAM in enumerate( HAM_LIST ):
        print( HAM )
        for SUBi,SUB in enumerate( SUB_LIST ):
            print( SUB )
            for LAMi,LAM in enumerate( LAM_LIST ):
                CAV_DICT["HAM"]       = HAM
                CAV_DICT["SUBSPACE"]  = SUB
                CAV_DICT["LAMBDA"]    = LAM
                CAV_DICT["WC"]        = CAV_DICT["ENERGY_AD"][1] - CAV_DICT["ENERGY_AD"][0]    # Cavity Frequency, a.u.
                E_TMP, _ = solve_Polariton_Hamiltonian( CAV_DICT )
                ENERGY[HAMi,SUBi,LAMi,:len(E_TMP)] = E_TMP[:] * 27.2114 # a.u. to eV
            for state in range( 3 ):
                if ( state == 0 ):
                    plt.plot( LAM_LIST, ENERGY[HAMi,SUBi,:,state] - ENERGY[HAMi,SUBi,0,0], line_list[icount], c=color_list[icount], label="%s %s" % (HAM, SUB) )
                else:
                    plt.plot( LAM_LIST, ENERGY[HAMi,SUBi,:,state] - ENERGY[HAMi,SUBi,0,0], line_list[icount], c=color_list[icount] )
            icount += 1
    plt.legend()
    plt.xlabel("Coupling Strength, $\lambda$ (a.u.)", fontsize=15)
    plt.ylabel("Energy (eV)", fontsize=15)
    plt.xlim(LAM_LIST[0],LAM_LIST[-1])
    plt.ylim(-0.1,6)
    plt.tight_layout()
    plt.savefig(f"{DATA_DIR}/Example2_GroundStateModifications.jpg", dpi=300)
    plt.clf()

