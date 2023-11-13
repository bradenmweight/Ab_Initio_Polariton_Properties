import numpy as np
from numpy import kron as kron_prod
import sys
import subprocess as sp
import itertools
import more_itertools
from matplotlib import pyplot as plt
from numba import jit
from time import time
from scipy.special import binom as binomial_coeff
import psutil

#from distinct_permutations import distinct_permutations # Speed-up was not noticable

##### Braden M. Weight #####
#####  November 7, 2023  #####

# SYNTAX: python3 Pauli-Fierz_DdotE.py A0

def get_globals():
    global NEL, NF, A0, WC_eV, WC_AU
    global EVEC_INTS, EVEC_NORM, EVEC_OUT
    global RPA, WRITE_WFNS, SCALE_A0, SUBSPACE

    SCALE_A0 = True # A0 --> A0 / sqrt(NMOL)

    if ( len(sys.argv) > 2 ):
        SUBSPACE   = int(sys.argv[2])  # None = Use Largest Possible Hilbert, 1 = Up to First Excited Subspace, 2 = Up to Second Excited Subspace, etc.
    else:
        SUBSPACE = None
    NEL        = 2  # Number of Electronic States (including ground state)
    NF         = 2  # Number of Fock Basis States
    RPA        = True                # If True, look for TD-DFT/RPA data rather than TD-DFT/TDA data
    WRITE_WFNS = False                # If True, writes <alpha,n|\Phi_j> -- All polaritonic wfns in full adiabatic-Fock basis
                                     # (NEL=50,NF=50) --> WF ~ 48 MB
    A0         = float(sys.argv[1]) # a.u.
    WC_eV      = 0.111526 * 27.2114 # eV
    EVEC_INTS  = [ 1,0,0 ]  # Cavity Polarization Vector (input as integers without normalizing)
    
    WC_AU     = WC_eV / 27.2114
    EVEC_NORM = EVEC_INTS / np.linalg.norm(EVEC_INTS)
    EVEC_OUT = "_".join(map(str,EVEC_INTS))

def get_a_op(nf):
    a = np.zeros((nf,nf))
    for m in range(1,nf):
        a[m,m-1] = np.sqrt(m)
    return a.T

def get_H_PF(EAD, MU):
    """
    Input: HAD,   Adiabatic hamiltonian (diagonal) energies from electronic structure
    Output: H_PF, Pauli-Fierz Hamiltonian using Adiabatic-Fock basis states
    KRON ORDER: MOL1, MOL2, MOL3, ..., MOLN, PHOTON1
    """

    print (f"Dimension = {(NEL**NMOL * NF)}")
    I_el   = np.identity(NEL)
    I_ph = np.identity(NF)
    a_op   = get_a_op(NF)
    q_op   = a_op.T + a_op
    MU     = np.einsum("d,mJKd->mJK", EVEC_NORM[:], MU[:,:,:,:] )

    H_PF = np.zeros( (NEL**NMOL * NF, NEL**NMOL * NF) ) # Single cavity mode
    TMP  = 0 # Make TMP exist throughout

    #### H_EL ####
    for A in range( NMOL ):
        # FIRST TERM
        if ( A == 0 ):
            TMP = np.diag( EAD[A,:] )
        else:
            TMP = I_el
        # N-1 TERMS
        for B in range( 1, NMOL ):
            if ( A == B ):
                TMP = np.kron( TMP, np.diag( EAD[A,:] ) )
            else:
                TMP = np.kron( TMP, I_el )
        H_PF += np.kron( TMP, I_ph )

    #### H_PH ####
    for A in range( NMOL ):
        # FIRST TERM
        TMP = I_el
        # N-1 TERMS
        for B in range( 1, NMOL ):
            TMP = np.kron( TMP, I_el )
    H_PF += np.kron( TMP, np.diag( np.arange( NF ) * WC_AU ) )

    #### H_INT ####
    for A in range( NMOL ):
        # FIRST TERM
        if ( A == 0 ):
            TMP = MU[A]
        else:
            TMP = I_el
        # N-1 TERMS
        for B in range( 1, NMOL ):
            if ( A == B ):
                TMP = np.kron( TMP, MU[A] )
            else:
                TMP = np.kron( TMP, I_el )
        H_PF += kron_prod(TMP, q_op) * WC_AU * A0_SCALED

    # #### H_DSE ####
    # Diagonal Terms (MOL A <--> MOL A)
    for A in range( NMOL ): # First Term
        #print( "DSE (Diag.): A", A )
        if ( A == 0 ):
            TMP = MU[A] @ MU[A]
        else:
            TMP = I_el
        # N-1 TERMS
        for B in range( 1, NMOL ):
            if ( A == B ):
                TMP = np.kron( TMP, MU[B] @ MU[B] )
            else:
                TMP = np.kron( TMP, I_el )

        H_PF += np.kron( TMP, I_ph ) * WC_AU * A0_SCALED**2

    # # Off-diagonal Terms (MOL A < MOL B)
    if ( NMOL >= 2 ):
        for A in range( NMOL ): # First Term
            for B in range( A+1, NMOL ):
                TMP = 1

                # Identites up to first position
                for C in range( A ):
                    if ( C == 0 ): 
                        TMP = I_el
                    else:
                        TMP = np.kron( TMP, I_el )

                TMP = np.kron( TMP, MU[A] ) # Dipole for molecule A

                # Identites up to second position
                for C in range( A+1,B ):
                    TMP = np.kron( TMP, I_el )
                
                TMP = np.kron( TMP, MU[B] ) # Dipole for molecule B
                
                # Identites for the rest of the molecules
                for C in range( B+1, NMOL ):
                    TMP = np.kron( TMP, I_el )
            
                # Factor "2" is for double counting A != B terms
                H_PF += 2 * np.kron( TMP, I_ph ) * WC_AU * A0_SCALED ** 2

    if ( H_PF.size * H_PF.itemsize * 10 ** -9 > 0.5 ): # If H_PF larger than 0.5 GB, tell the user.
        print(f"\tWARNING!!! Large Matrix" )
        print(f"\tMemory size of numpy array in (MB, GB): ({round(H_PF.size * H_PF.itemsize * 10 ** -6,2)},{round(H_PF.size * H_PF.itemsize * 10 ** -9,2)})" )

    #print( H_PF )

    return H_PF

def get_H_PF_noDSE(EAD, MU):
    """
    Input: HAD,   Adiabatic hamiltonian (diagonal) energies from electronic structure
    Output: H_PF, Pauli-Fierz Hamiltonian using Adiabatic-Fock basis states
    KRON ORDER: MOL1, MOL2, MOL3, ..., MOLN, PHOTON1
    """

    print (f"Dimension = {(NEL**NMOL * NF)}")
    I_el   = np.identity(NEL)
    I_ph = np.identity(NF)
    a_op   = get_a_op(NF)
    q_op   = a_op.T + a_op
    MU     = np.einsum("d,mJKd->mJK", EVEC_NORM[:], MU[:,:,:,:] )

    H_PF = np.zeros( (NEL**NMOL * NF, NEL**NMOL * NF) ) # Single cavity mode
    TMP  = 0 # Make TMP exist throughout

    #### H_EL ####
    for A in range( NMOL ):
        # FIRST TERM
        if ( A == 0 ):
            TMP = np.diag( EAD[A,:] )
        else:
            TMP = I_el
        # N-1 TERMS
        for B in range( 1, NMOL ):
            if ( A == B ):
                TMP = np.kron( TMP, np.diag( EAD[A,:] ) )
            else:
                TMP = np.kron( TMP, I_el )
        H_PF += np.kron( TMP, I_ph )

    #### H_PH ####
    for A in range( NMOL ):
        # FIRST TERM
        TMP = I_el
        # N-1 TERMS
        for B in range( 1, NMOL ):
            TMP = np.kron( TMP, I_el )
    H_PF += np.kron( TMP, np.diag( np.arange( NF ) * WC_AU ) )

    #### H_INT ####
    for A in range( NMOL ):
        # FIRST TERM
        if ( A == 0 ):
            TMP = MU[A]
        else:
            TMP = I_el
        # N-1 TERMS
        for B in range( 1, NMOL ):
            if ( A == B ):
                TMP = np.kron( TMP, MU[A] )
            else:
                TMP = np.kron( TMP, I_el )
        H_PF += kron_prod(TMP, q_op) * WC_AU * A0_SCALED


    if ( H_PF.size * H_PF.itemsize * 10 ** -9 > 0.5 ): # If H_PF larger than 0.5 GB, tell the user.
        print(f"\tWARNING!!! Large Matrix" )
        print(f"\tMemory size of numpy array in (MB, GB): ({round(H_PF.size * H_PF.itemsize * 10 ** -6,2)},{round(H_PF.size * H_PF.itemsize * 10 ** -9,2)})" )

    #print( H_PF )

    return H_PF

def get_H_PF_RWA(EAD, MU):
    """
    Input: HAD,   Adiabatic hamiltonian (diagonal) energies from electronic structure
    Output: H_PF, Pauli-Fierz Hamiltonian using Adiabatic-Fock basis states
    KRON ORDER: MOL1, MOL2, MOL3, ..., MOLN, PHOTON1
    """

    print (f"Dimension = {(NEL**NMOL * NF)}")
    I_el   = np.identity(NEL)
    I_ph = np.identity(NF)
    a_op   = get_a_op(NF)
    q_op   = a_op.T + a_op
    MU     = np.einsum("d,mJKd->mJK", EVEC_NORM[:], MU[:,:,:,:] )

    a_plus  = a_op.T
    a_minus = a_op

    MU_plus  = np.zeros_like( MU )
    MU_minus = np.zeros_like( MU )
    for j in range( NEL ):
        for k in range( NEL ):
            if ( j < k ):
                MU_minus[:,j,k] = MU[:,j,k] # Check this direction
                MU_plus[:,k,j]  = MU[:,k,j] # Check this direction

    H_PF = np.zeros( (NEL**NMOL * NF, NEL**NMOL * NF) ) # Single cavity mode
    TMP  = 0 # Make TMP exist throughout

    #### H_EL ####
    for A in range( NMOL ):
        # FIRST TERM
        if ( A == 0 ):
            TMP = np.diag( EAD[A,:] )
        else:
            TMP = I_el
        # N-1 TERMS
        for B in range( 1, NMOL ):
            if ( A == B ):
                TMP = np.kron( TMP, np.diag( EAD[A,:] ) )
            else:
                TMP = np.kron( TMP, I_el )
        H_PF += np.kron( TMP, I_ph )

    #### H_PH ####
    for A in range( NMOL ):
        # FIRST TERM
        TMP = I_el
        # N-1 TERMS
        for B in range( 1, NMOL ):
            TMP = np.kron( TMP, I_el )
    H_PF += np.kron( TMP, np.diag( np.arange( NF ) * WC_AU ) )


    #### H_INT ####
    # MATTER PLUS -- PHOTON MINUS
    for A in range( NMOL ):
        # FIRST TERM
        if ( A == 0 ):
            TMP = MU_plus[A]
        else:
            TMP = I_el
        # N-1 TERMS
        for B in range( 1, NMOL ):
            if ( A == B ):
                TMP = np.kron( TMP, MU_plus[A] )
            else:
                TMP = np.kron( TMP, I_el )
        H_PF += kron_prod(TMP, a_minus) * WC_AU * A0_SCALED

    # MATTER PLUS -- PHOTON MINUS
    for A in range( NMOL ):
        # FIRST TERM
        if ( A == 0 ):
            TMP = MU_minus[A]
        else:
            TMP = I_el
        # N-1 TERMS
        for B in range( 1, NMOL ):
            if ( A == B ):
                TMP = np.kron( TMP, MU_minus[A] )
            else:
                TMP = np.kron( TMP, I_el )
        H_PF += kron_prod(TMP, a_plus) * WC_AU * A0_SCALED

    # #### H_DSE ####
    # Diagonal Terms (MOL A <--> MOL A)
    for A in range( NMOL ): # First Term
        #print( "DSE (Diag.): A", A )
        if ( A == 0 ):
            TMP = MU[A] @ MU[A]
        else:
            TMP = I_el
        # N-1 TERMS
        for B in range( 1, NMOL ):
            if ( A == B ):
                TMP = np.kron( TMP, MU[B] @ MU[B] )
            else:
                TMP = np.kron( TMP, I_el )

        H_PF += np.kron( TMP, I_ph ) * WC_AU * A0_SCALED**2

    # # Off-diagonal Terms (MOL A < MOL B)
    if ( NMOL >= 2 ):
        for A in range( NMOL ): # First Term
            for B in range( A+1, NMOL ):
                TMP = 1

                # Identites up to first position
                for C in range( A ):
                    if ( C == 0 ): 
                        TMP = I_el
                    else:
                        TMP = np.kron( TMP, I_el )

                TMP = np.kron( TMP, MU[A] ) # Dipole for molecule A

                # Identites up to second position
                for C in range( A+1,B ):
                    TMP = np.kron( TMP, I_el )
                
                TMP = np.kron( TMP, MU[B] ) # Dipole for molecule B
                
                # Identites for the rest of the molecules
                for C in range( B+1, NMOL ):
                    TMP = np.kron( TMP, I_el )
            
                # Factor "2" is for double counting A != B terms
                H_PF += 2 * np.kron( TMP, I_ph ) * WC_AU * A0_SCALED ** 2

    if ( H_PF.size * H_PF.itemsize * 10 ** -9 > 0.5 ): # If H_PF larger than 0.5 GB, tell the user.
        print(f"\tWARNING!!! Large Matrix" )
        print(f"\tMemory size of numpy array in (MB, GB): ({round(H_PF.size * H_PF.itemsize * 10 ** -6,2)},{round(H_PF.size * H_PF.itemsize * 10 ** -9,2)})" )

    #print( H_PF )

    return H_PF

def get_H_JC_SubSpace_1(EAD, MU):

    #EAD -= EAD[0,0]

    MU  = np.einsum("d,mJKd->mJK", EVEC_NORM[:], MU[:,:,:,:] )
    
    NPOL = 1 + NMOL + 1 # GS + First-excited Subspace
    print (f"Dimension = {NPOL}")
    H = np.zeros( (NPOL,NPOL) )

    # Diagonal Energies
    H[0,0] = np.sum( EAD[:,0] ) # Collective Ground State Energy
    for mol in range( NMOL ):
        dE              = EAD[mol,1] - EAD[mol,0]  # Exciton Transition Energy
        H[1+mol, 1+mol] = H[0,0] + dE
    H[1+NMOL, 1+NMOL] = H[0,0] + WC_AU # Photon Mode Energy

    # Coupling Elements
    for mol in range( NMOL ):
        H[1+NMOL, 1+mol] = WC_AU * A0_SCALED * MU[mol,0,1] #* np.sqrt(1)
        H[1+mol, 1+NMOL] = H[1+NMOL, 1+mol]

    return H










@jit(nopython=True)
def build_H( H, BRAs, NMOL, EAD, MU, DSE_AA ):
    for pol1 in range( len(BRAs) ):
        for pol2 in range( pol1, len(BRAs) ):
            bra = BRAs[pol1]
            ket = BRAs[pol2]

            ### H_EL AND H_PH ###
            if ( (bra == ket).all() ):
                ### H_EL ###
                for A in range( NMOL ):
                    H[pol1,pol2] += EAD[A,bra[A]]
                ### H_PH ###
                H[pol1,pol2] += WC_AU * bra[-1]
                #print( "<", ",".join(map(str,bra)), "| H_EL + H_PH |", ",".join(map(str,ket)), ">" )
            
            ### H_EL-PH ###
            if ( abs(bra[-1] - ket[-1]) == 1 ): # Change in photon number by exactly 1
                for A in range( NMOL ):
                    if ( (bra[:A] == ket[:A]).all() and (bra[A+1:-1] == ket[A+1:-1]).all() ):
                        H[pol1,pol2] += WC_AU * A0_SCALED * MU[A,bra[A],ket[A]]
                        #if ( A == 0 ):
                            #print( "<", ",".join(map(str,bra)), "| H_EL-PH |", ",".join(map(str,ket)), ">" )
            
            ### H_DSE ###
            if ( abs(bra[-1] - ket[-1]) == 0 ): # No change in photon number
                for A in range( NMOL ):
                    for B in range( A, NMOL ):
                        if (A == B):
                            if ( (bra[:A] == ket[:A]).all() and (bra[A+1:-1] == ket[A+1:-1]).all() ):
                                H[pol1,pol2] += WC_AU * A0_SCALED**2 * DSE_AA[ A, bra[A], ket[A] ]
                        elif ( A < B ):
                            if ( (bra[:A]     == ket[:A]    ).all() and \
                                 (bra[A+1:B]  == ket[A+1:B] ).all() and \
                                 (bra[B+1:-1] == ket[B+1:-1]).all() ) :
                                    H[pol1,pol2] += 2 * WC_AU * A0_SCALED**2 * MU[ A, bra[A], ket[A] ] * MU[ B, bra[B], ket[B] ]
            H[pol2,pol1] = H[pol1,pol2]
    
    return H

def filter_BRAs( ):
    T0 = time()
    print("Default:")
    SUBS = np.zeros( (NMOL+2) )
    BRAs = []
    for n in itertools.product( [0,1], repeat=1 ): # NMOL Molecules
        for a in itertools.product( [0,1], repeat=NMOL ): # NMOL Molecules
            bra = np.append(a,n)
            if ( SUBSPACE is None or sum( bra ) <= SUBSPACE ):
                BRAs.append( bra )
                SUBS[sum(bra)] += 1
                print( bra, sum(bra) )
    print("# per Subspace:", SUBS)
    print( binomial_coeff(NMOL+1,np.arange(NMOL+2)) )
    print("Default Time: %1.2f" % (time() - T0) )
    return np.array( BRAs )

def filter_BRAs_manual( SUBSPACE, NMOL, SMAX, NUM_CONFIGS ):
    T0 = time()
    BRAs = []
    for S,config in enumerate( NUM_CONFIGS ):
        numbers   = [0 for j in range(NMOL+1)]
        for num in range( S ):
            numbers[num] = 1
        TMP = more_itertools.distinct_permutations( numbers, r=NMOL+1 )
        #TMP = distinct_permutations( numbers, r=NMOL+1 ) # No significant speed-up
        for bra in sorted(TMP):
            BRAs.append( bra )
            #print( bra )
    print("Manual Time: %1.2f s" % (time() - T0))
    return np.array( BRAs )

def get_BRAs():
    if ( SUBSPACE is None or SUBSPACE > NMOL+1 ):
        SMAX = NMOL+1
    else:
        SMAX = SUBSPACE
    
    NUM_CONFIGS = np.array( binomial_coeff(NMOL+1,np.arange(SMAX+1)) ).astype(int)
    
    MEM_SIZE = sum(NUM_CONFIGS)**2 * np.array( (1) ).itemsize * 10 ** -9 # Hamiltonian Size
    print("\tMemory Requirements (Configurations): %1.3f GB" % (sum(NUM_CONFIGS) * np.array(1).itemsize * 10 ** -9) )
    print("\tMemory Requirements (Hamiltonian): %1.3f GB" % (sum(NUM_CONFIGS)**2 * np.array(1).itemsize * 10 ** -9) )
    MEMORY = np.array(psutil.virtual_memory()[:]) * 10**-9 # Bytes --> GB
    #print( psutil.virtual_memory() )
    print( "\tAvailable System Memory %1.0f GB (free %1.0f GB) of %1.0f GB total" % (MEMORY[1], MEMORY[4], MEMORY[0]) )
    if ( MEM_SIZE > 0.95 * MEMORY[4] ): # If H_PF larger than 30.0 GB, tell the user and quit.
        print(f"\n\t!!!WARNING!!!\n\tLarge Matrix (M > 30 GB).\n\tQuitting.\n" )
        exit()


    BRAs = filter_BRAs_manual( SUBSPACE, NMOL, SMAX, NUM_CONFIGS )
    return BRAs

def get_H_PF_SubSpace_N_SuperIndex(EAD, MU):

    E0     = np.sum( EAD[:,0] ) # Collective Ground State Energy
    MU     = np.einsum("d,mJKd->mJK", EVEC_NORM[:], MU[:,:,:,:] )
    DSE_AA = np.einsum("mJK,mKL->mJL", MU[:,:,:], MU[:,:,:] ) # Matrix Multiplication for Each Molecule
                                            # Between Molecules, We Have Scalar Multiplication

    BRAs = get_BRAs()

    NSPACE = len(BRAs)
    print( "Numerical Dimension:", NSPACE )
    if ( SUBSPACE is not None and SUBSPACE > NMOL+1 ):
        print("SUBSPACE > NMOL+1. Not possible.")
        exit()

    T0 = time()
    H = np.zeros( (NSPACE,NSPACE) )
    H = build_H( H, BRAs, NMOL, EAD, MU, DSE_AA ) # Use jit compilation
    print("Time to Build H_PF: %1.2f s" % (time() - T0))
    return H





def get_ADIABATIC_DATA():

    global NMOL, A0_SCALED
    MOLECULE_DATA = open("./MOLECULE_DATA.dat","r").readlines() # Each line points to different molecule's TD-DFT folder
    NMOL = len( MOLECULE_DATA )
    if ( SCALE_A0 == True ):
        print("\n\tAttn: Rescaling coupling strength. A0 --> A0 / sqrt(NMOL)\n")
        A0_SCALED = A0 / np.sqrt( NMOL )
    else:
        A0_SCALED = A0

    EAD  = np.zeros(( NMOL,NEL ))
    MU  = np.zeros(( NMOL,NEL,NEL,3 ))

    for mol,moldir in enumerate( MOLECULE_DATA ):

        if ( RPA == True ):
            EAD[mol,:] += np.loadtxt(f"{moldir.strip()}/PLOTS_DATA/ADIABATIC_ENERGIES_RPA.dat")[:NEL] # in AU
            MU[mol,:]  += np.load(f"{moldir.strip()}/PLOTS_DATA/DIPOLE_RPA.dat.npy")[:NEL,:NEL] # in AU
        else:
            EAD[mol,:] += np.loadtxt(f"{moldir.strip()}/PLOTS_DATA/ADIABATIC_ENERGIES_TDA.dat")[:NEL] # in AU
            MU[mol,:]  += np.load(f"{moldir.strip()}/PLOTS_DATA/DIPOLE_TDA.dat.npy")[:NEL,:NEL] # in AU

    return EAD, MU

def SolvePlotandSave(H_PF,EAD,MU,name):

        sp.call(f"mkdir -p data_{name}_NMOL_{NMOL}", shell=True)

        # Diagonalize Hamiltonian
        E, U = np.linalg.eigh( H_PF ) # This is exact solution --> Should we ever try Davidson diagonalization ?
        
        # Save Data
        np.savetxt( f"data_{name}_NMOL_{NMOL}/E_{EVEC_OUT}_A0_{round(A0,6)}_WC_{round(WC_eV,6)}_NF_{NF}_NEL_{NEL}.dat", E * 27.2114 )
        np.savetxt( f"data_{name}_NMOL_{NMOL}/E_TRANS_{EVEC_OUT}_A0_{round(A0,6)}_WC_{round(WC_eV,6)}_NF_{NF}_NEL_{NEL}.dat", (E-E[0]) * 27.2114 )
        if ( WRITE_WFNS ):
            #np.savetxt( f"data_PF_NMOL_{NMOL}/U_{EVEC_OUT}_A0_{round(A0,6)}_WC_{round(WC_eV,6)}_NF_{NF}_NEL_{NEL}.dat", U ) # These can be large
            np.save( f"data_{name}_NMOL_{NMOL}/U_{EVEC_OUT}_A0_{round(A0,6)}_WC_{round(WC_eV,6)}_NF_{NF}_NEL_{NEL}.dat", U ) # Binary is smaller

        # Save original EAD and MU for convenience
        np.savetxt( f"data_{name}_NMOL_{NMOL}/EAD.dat", EAD * 27.2114 )
        np.save( f"data_{name}_NMOL_{NMOL}/MU.dat", MU )

def main():
    get_globals()
    EAD, MU = get_ADIABATIC_DATA()

    print("Coupling Coupling Strength, Electric Field Strength, and Cavity Volume:")
    print("\t# Molec. = %1.0f" % (NMOL) )
    print("\tA0 (A0') = %1.3f (%1.3f) a.u." % (A0, A0_SCALED) )
    print("\tE  (E' ) = %1.3f (%1.2f) V/nm" % (WC_AU * A0 * 514, WC_AU * A0_SCALED * 514) )
    if ( A0 > 0.0 ):
        FAC = 2*np.pi / (2 * WC_AU) / 10**3 * 0.529**3
        print("\tV  (V' ) = %1.3f (%1.3f) nm^3" % (FAC / A0**2, FAC / A0_SCALED**2) )

    if ( SUBSPACE is None or SUBSPACE == NMOL+1 ):
        if ( NMOL <= 10 ):
            H    = get_H_PF( EAD, MU )
            SolvePlotandSave( H, EAD, MU, "PF_EXACT")
            H    = get_H_PF_RWA( EAD, MU )
            SolvePlotandSave( H, EAD, MU, "PF_RWA")
            H    = get_H_PF_noDSE( EAD, MU )
            SolvePlotandSave( H, EAD, MU, "PF_noDSE")
        else:
            print( "Too many molecules for exact solution. Skipping." )

    if ( SUBSPACE is None or SUBSPACE == NMOL+1 ):
        H    = get_H_JC_SubSpace_1( EAD, MU )
        SolvePlotandSave( H, EAD, MU, "JC")
    
    H    = get_H_PF_SubSpace_N_SuperIndex( EAD, MU )
    if ( SUBSPACE is None ):
        SolvePlotandSave( H, EAD, MU, f"PF_SUBSPACE_FULL")
    else:
        SolvePlotandSave( H, EAD, MU, f"PF_SUBSPACE_{SUBSPACE}")


if __name__ == "__main__":
    main()