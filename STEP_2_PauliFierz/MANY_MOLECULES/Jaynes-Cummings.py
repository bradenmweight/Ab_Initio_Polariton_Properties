import numpy as np
from numpy import kron as kron_prod
import sys
import subprocess as sp

##### Braden M. Weight #####
#####  September 18, 2023  #####

def get_globals():
    global NEL, NF, A0, wc_eV, wc_AU
    global EVEC_INTS, EVEC_NORM, EVEC_OUT
    global RPA, WRITE_WFNS, SCALE_A0

    SCALE_A0 = True # A0 --> A0 / sqrt(NMOL)
    print("\n\tAttn: Rescaling coupling strength. A0 --> A0 / sqrt(NMOL)\n")

    NEL        = int( sys.argv[1] )  # Number of Electronic States (including ground state)
    NF         = int( sys.argv[2] )  # Number of Fock Basis States
    RPA        = True                # If True, look for TD-DFT/RPA data rather than TD-DFT/TDA data
    WRITE_WFNS = True                # If True, writes <alpha,n|\Phi_j> -- All polaritonic wfns in full adiabatic-Fock basis
                                     # (NEL=50,NF=50) --> WF ~ 48 MB
    A0         = float( sys.argv[3] ) # a.u.
    wc_eV      = float( sys.argv[4] ) # eV
    EVEC_INTS  = [ int(j) for j in sys.argv[5] ]  # Cavity Polarization Vector (input as integers without normalizing)
    
    wc_AU     = wc_eV / 27.2114
    EVEC_NORM = EVEC_INTS / np.linalg.norm(EVEC_INTS)
    EVEC_OUT = "_".join(map(str,EVEC_INTS))

def get_a_op(nf):
    a = np.zeros((nf,nf))
    for m in range(1,nf):
        a[m,m-1] = np.sqrt(m)
    return a.T

def get_H_JC(EAD, MU):
    """
    Input: HAD,   Adiabatic hamiltonian (diagonal) energies from electronic structure
    Output: H_JC, Jaynes-Cummings Hamiltonian using Adiabatic-Fock basis states
    KRON ORDER: MOL1, MOL2, MOL3, ..., MOLN, PHOTON1
    """

    print (f"Dimension = {(NEL**NMOL * NF)}")
    I_el   = np.identity(NEL)
    I_ph = np.identity(NF)
    a_op   = get_a_op(NF)
    q_op   = a_op.T + a_op
    MU     = np.einsum("d,mJKd->mJK", EVEC_NORM[:], MU[:,:,:,:] )

    MU_plus  = np.zeros_like( MU )
    MU_minus = np.zeros_like( MU )
    for j in range( NEL ):
        for k in range( NEL ):
            if ( j < k ):
                MU_minus[:,j,k] = MU[:,j,k] # Check this direction
                MU_plus[:,k,j]  = MU[:,k,j] # Check this direction

    a_plus  = a_op.T
    a_minus = a_op


    H_JC = np.zeros( (NEL**NMOL * NF, NEL**NMOL * NF) ) # Single cavity mode
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
        H_JC += np.kron( TMP, I_ph )

    #### H_PH ####
    for A in range( NMOL ):
        # FIRST TERM
        TMP = I_el
        # N-1 TERMS
        for B in range( 1, NMOL ):
            TMP = np.kron( TMP, I_el )
    H_JC += np.kron( TMP, np.diag( np.arange( NF ) * wc_AU ) )

    #### H_INT ####
    # MATTER PLUS -- PHOTON MINUS
    for A in range( NMOL ):
        # FIRST TERM
        if ( A == 0 ):
            TMP = MU[A]
        else:
            TMP = I_el
        # N-1 TERMS
        for B in range( 1, NMOL ):
            if ( A == B ):
                TMP = np.kron( TMP, MU_plus[A] )
            else:
                TMP = np.kron( TMP, I_el )
        if ( SCALE_A0 == True ):
            H_JC += kron_prod(TMP, a_minus) * wc_AU * A0 / np.sqrt( NMOL )
        else:
            H_JC += kron_prod(TMP, a_minus) * wc_AU * A0

    # MATTER PLUS -- PHOTON MINUS
    for A in range( NMOL ):
        # FIRST TERM
        if ( A == 0 ):
            TMP = MU[A]
        else:
            TMP = I_el
        # N-1 TERMS
        for B in range( 1, NMOL ):
            if ( A == B ):
                TMP = np.kron( TMP, MU_minus[A] )
            else:
                TMP = np.kron( TMP, I_el )
        if ( SCALE_A0 == True ):
            H_JC += kron_prod(TMP, a_plus) * wc_AU * A0 / np.sqrt( NMOL )
        else:
            H_JC += kron_prod(TMP, a_plus) * wc_AU * A0


    if ( H_JC.size * H_JC.itemsize * 10 ** -9 > 0.5 ): # If H_JC larger than 0.5 GB, tell the user.
        print(f"\tWARNING!!! Large Matrix" )
        print(f"\tMemory size of numpy array in (MB, GB): ({round(H_JC.size * H_JC.itemsize * 10 ** -6,2)},{round(H_JC.size * H_JC.itemsize * 10 ** -9,2)})" )

    return H_JC

def get_ADIABATIC_DATA():

    global NMOL
    MOLECULE_DATA = open("./MOLECULE_DATA.dat","r").readlines() # Each line points to different molecule's TD-DFT folder
    NMOL = len( MOLECULE_DATA )
    sp.call(f"mkdir -p data_JC_NMOL_{NMOL}", shell=True)

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

def SolvePlotandSave(H_JC,EAD,MU):

        # Diagonalize Hamiltonian
        E, U = np.linalg.eigh( H_JC ) # This is exact solution --> Should we ever try Davidson diagonalization ?
        
        # Save Data
        np.savetxt( f"data_JC_NMOL_{NMOL}/E_{EVEC_OUT}_A0_{round(A0,6)}_WC_{round(wc_eV,6)}_NF_{NF}_NEL_{NEL}.dat", E * 27.2114 )
        np.savetxt( f"data_JC_NMOL_{NMOL}/E_TRANS_{EVEC_OUT}_A0_{round(A0,6)}_WC_{round(wc_eV,6)}_NF_{NF}_NEL_{NEL}.dat", (E-E[0]) * 27.2114 )
        if ( WRITE_WFNS ):
            #np.savetxt( f"data_JC_NMOL_{NMOL}/U_{EVEC_OUT}_A0_{round(A0,6)}_WC_{round(wc_eV,6)}_NF_{NF}_NEL_{NEL}.dat", U ) # These can be large
            np.save( f"data_JC_NMOL_{NMOL}/U_{EVEC_OUT}_A0_{round(A0,6)}_WC_{round(wc_eV,6)}_NF_{NF}_NEL_{NEL}.dat", U ) # Binary is smaller

        print ( A0, wc_eV )

        # Save original EAD and MU for convenience
        np.savetxt( f"data_JC_NMOL_{NMOL}/EAD.dat", EAD * 27.2114 )
        np.save( f"data_JC_NMOL_{NMOL}/MU.dat", MU )

def main():
    get_globals()

    EAD, MU = get_ADIABATIC_DATA() 
    H_JC    = get_H_JC( EAD, MU )
    SolvePlotandSave( H_JC, EAD, MU)

    


if __name__ == "__main__":
    main()