import numpy as np
from matplotlib import pyplot as plt
import matplotlib as mpl
import subprocess as sp
import os
from scipy.signal import find_peaks

def get_globals():
    global NM, NF, A0_LIST, WC_LIST
    global EVEC_INTS, EVEC_NORM, EVEC_OUT
    global DATA_DIR, NPOL, NA0, NWC, EMIN, EMAX
    global NMOL_LIST, SIG, PFCODE

    ##### MAIN USER INPUT SECTION #####
    NMOL_LIST = np.array([10])
    NM        = 2                   # Number of Electronic States (including ground state)
    NF        = 2                   # Number of Fock Basis States
    EMIN      = 1.882800#2.8828 - 0.5#2
    EMAX      = 3.882800#2.8828 + 0.5#2
    SIG       = 0.05
    EVEC_INTS = np.array([ 1,0,0 ]) # Cavity Polarization Vector (input as integers without normalizing)
    
    #A0_LIST = np.arange( 0.0, 0.01+0.001, 0.001 ) # a.u.
    A0_LIST = np.array([0.02]) # a.u.
    #WC_LIST = np.arange( 2.6928, 3.092800, 0.01 )
    WC_LIST = np.arange( 1.882800, 3.882800, 0.01 )
    #WC_LIST = np.arange( 1.882800, 3.882800-0.01, 0.01 )
    #WC_LIST = np.array([2.8828])
    ##### END USER INPUT SECTION  #####



    ##### DO NOT MODIFY BELOW HERE #####
    NPOL = 100 # NM**NMOL_LIST[-1] * NF
    NA0  = len(A0_LIST)
    NWC  = len(WC_LIST)

    EVEC_NORM = EVEC_INTS / np.linalg.norm(EVEC_INTS)
    EVEC_OUT = "_".join(map(str,EVEC_INTS))


    DATA_DIR = "PLOTS_DATA"
    sp.call(f"mkdir -p {DATA_DIR}", shell=True)

    PFCODE = "/scratch/bweight/software/Ab_Initio_Polariton_Properties/STEP_2_PauliFierz/MANY_MOLECULES/Pauli-Fierz_DdotE.py"

def get_energies():

    EPOL = np.zeros(( len(NMOL_LIST), NPOL, NA0, NWC ))

    for moli, mol in enumerate( NMOL_LIST ):
        for A0IND, A0 in enumerate( A0_LIST ):
            for WCIND, WC in enumerate( WC_LIST ):
                A0 = round( A0, 5 )
                WC = round( WC, 5 )
                try:
                    EPOL[moli, :, A0IND, WCIND] = np.loadtxt(f"data_PF_NMOL_{mol}/E_{EVEC_OUT}_A0_{A0}_WC_{WC}_NF_{NF}_NEL_{NM}.dat")[:NPOL]
                except ValueError:
                    TMP = len(np.loadtxt(f"data_PF_NMOL_{mol}/E_{EVEC_OUT}_A0_{A0}_WC_{WC}_NF_{NF}_NEL_{NM}.dat"))
                    EPOL[moli, :TMP, A0IND, WCIND] = np.loadtxt(f"data_PF_NMOL_{mol}/E_{EVEC_OUT}_A0_{A0}_WC_{WC}_NF_{NF}_NEL_{NM}.dat")
                except OSError:
                    sp.call( f"python3 {PFCODE} {NM} {NF} {A0} {WC} {''.join(map(str,EVEC_INTS))}",shell=True)


    return EPOL

def get_average_photon_number():

    def get_a_op():
        a = np.zeros((NF,NF))
        for m in range(1,NF):
            a[m,m-1] = np.sqrt(m)
        return a.T

    PHOT = np.zeros(( len(NMOL_LIST), NPOL, NA0, NWC ))
    a_op = get_a_op()
    I_M  = np.identity( NM )
    #N_op = np.kron( I_M, a_op.T @ a_op )

    for moli, mol in enumerate( NMOL_LIST ):
        print(f"<aa>: NMOL = {mol} in {NMOL_LIST}")
        #N_op = np.zeros( (NM**mol * NF, NM**mol * NF) )
        N_op    = I_M
        counter = 1
        for mol_k in range( 1, mol ):
            counter += 1
            N_op = np.kron( N_op, I_M )
        N_op = np.kron( N_op, a_op.T @ a_op )
        print(NM**mol * NF, N_op.shape)

        for A0IND, A0 in enumerate( A0_LIST ):
            print(f"<aa>: A0 = {A0IND+1} of {len(A0_LIST)}")
            for WCIND, WC in enumerate( WC_LIST ):
                A0 = round( A0, 5 )
                WC = round( WC, 5 )
                U = np.load(f"data_PF_NMOL_{mol}/U_{EVEC_OUT}_A0_{A0}_WC_{WC}_NF_{NF}_NEL_{NM}.dat.npy")
                try:
                    PHOT[moli, :, A0IND, WCIND] = np.einsum( "Jp,JK,Kp->p", U[:,:] , N_op[:,:] , U[:,:] )[:NPOL]
                except ValueError:
                    TMP = len(np.einsum( "Jp,JK,Kp->p", U[:,:] , N_op[:,:] , U[:,:] ))
                    PHOT[moli, :TMP, A0IND, WCIND] = np.einsum( "Jp,JK,Kp->p", U[:,:] , N_op[:,:] , U[:,:] )

    #np.save(f"{DATA_DIR}/PHOT.dat.npy", PHOT)
    return PHOT

def plot_A0SCAN_WCFIXED_NMOLFIXED( EPOL, PHOT ):

    EZERO = EPOL[:,0,:,0] # Ground state for all (NMOL,A0,WC)

    cmap=mpl.colors.LinearSegmentedColormap.from_list('rg',[ "red", "darkred", "black", "darkgreen", "palegreen" ], N=256)
    for moli, mol in enumerate( NMOL_LIST ):
        for WCIND, WC in enumerate( WC_LIST ):
            WC = round( WC, 5 )
            print(f"Plotting WC = {WC} eV")
            VMAX = 1 #np.ceil(np.max(PHOT[moli,:NPOL,:,WCIND]))
            for state in range( NPOL ):
                plt.scatter( A0_LIST, EPOL[moli,state,:,WCIND] - EZERO[moli,:], s=25, cmap=cmap, c=PHOT[moli,state,:,WCIND], vmin=0.0, vmax=VMAX )
            
            plt.colorbar(pad=0.01,label="Average Photon Number")
            plt.xlim(A0_LIST[0],A0_LIST[-1])
            plt.ylim(EMIN, EMAX)
            plt.title( f"NMOL = {mol}", fontsize=15 )
            plt.xlabel( "Coupling Strength, $A_0$ (a.u.)", fontsize=15 )
            plt.ylabel( "Polariton Energy (eV)", fontsize=15 )
            plt.tight_layout()
            plt.savefig( f"{DATA_DIR}/EPOL_TRANS_A0SCAN_WC_{WC}_NMOL_{mol}.jpg", dpi=600 )
            plt.clf()


    EZERO = EPOL[:,0,0,0] # Ground state at zero coupling for all (NMOL,WC)

    cmap=mpl.colors.LinearSegmentedColormap.from_list('rg',[ "red", "darkred", "black", "darkgreen", "palegreen" ], N=256)
    for moli, mol in enumerate( NMOL_LIST ):
        for WCIND, WC in enumerate( WC_LIST ):
            WC = round( WC, 5 )
            print(f"Plotting WC = {WC} eV")
            VMAX = 1 #np.ceil(np.max(PHOT[moli,:NPOL,:,WCIND]))
            for state in range( NPOL ):
                plt.scatter( A0_LIST, EPOL[moli,state,:,WCIND] - EZERO[moli], s=25, cmap=cmap, c=PHOT[moli,state,:,WCIND], vmin=0.0, vmax=VMAX )
            
            plt.colorbar(pad=0.01,label="Average Photon Number")
            plt.xlim(A0_LIST[0],A0_LIST[-1])
            plt.ylim(EMIN, EMAX)
            plt.title( f"NMOL = {mol}", fontsize=15 )
            plt.xlabel( "Coupling Strength, $A_0$ (a.u.)", fontsize=15 )
            plt.ylabel( "Polariton Energy (eV)", fontsize=15 )
            plt.tight_layout()
            plt.savefig( f"{DATA_DIR}/EPOL_A0SCAN_WC_{WC}_NMOL_{mol}.jpg", dpi=600 )
            plt.clf()

    EZERO = EPOL[:,0,0,0] # Ground state at zero coupling for all (NMOL,WC)

    cmap=mpl.colors.LinearSegmentedColormap.from_list('rg',[ "red", "darkred", "black", "darkgreen", "palegreen" ], N=256)
    for WCIND, WC in enumerate( WC_LIST ):
        for moli, mol in enumerate( NMOL_LIST ):
            WC = round( WC, 5 )
            print(f"Plotting WC = {WC} eV")
            VMAX = 1 #np.ceil(np.max(PHOT[moli,:NPOL,:,WCIND]))
            #plt.scatter( A0_LIST, EPOL[moli,0,:,WCIND] - EZERO[moli], s=25, cmap=cmap, c=PHOT[moli,state,:,WCIND], vmin=0.0, vmax=VMAX, label="$N_{MOL}$"+f" = {mol}" )
            plt.plot( A0_LIST, EPOL[moli,0,:,WCIND] - EZERO[moli], "-o", lw=3, label="$N_{MOL}$"+f" = {mol}" )
        plt.legend()
        #plt.colorbar(pad=0.01,label="Average Photon Number")
        plt.xlim(A0_LIST[0],A0_LIST[-1])
        plt.xlabel( "Coupling Strength, $A_0$ (a.u.)", fontsize=15 )
        plt.ylabel( "Polariton Energy (eV)", fontsize=15 )
        plt.tight_layout()
        plt.savefig( f"{DATA_DIR}/EPOL_GS_A0SCAN_WC_{WC}_NMOL.jpg", dpi=600 )
        plt.clf()

def plot_WCSCAN_A0FIXED_NMOLFIXED( EPOL, PHOT ):

    EZERO = EPOL[:,0,0,0] # Ground state at zero coupling for all (NMOL,WC)
    
    cmap=mpl.colors.LinearSegmentedColormap.from_list('rg',[ "red", "darkred", "black", "darkgreen", "palegreen" ], N=256)
    for moli, mol in enumerate( NMOL_LIST ):
        for A0IND, A0 in enumerate( A0_LIST ):
            A0 = round( A0, 5 )
            VMAX = 1 #np.ceil(np.max(PHOT[moli,:NPOL,:,WCIND]))
            print(f"Plotting A0 = {A0} a.u.")

            for state in range( NPOL ):
                plt.scatter( WC_LIST, EPOL[moli,state,A0IND,:] - EZERO[moli], s=25, cmap=cmap, c=PHOT[moli,state,A0IND,:], vmin=0.0, vmax=VMAX )
            plt.plot( WC_LIST, WC_LIST*0 + EPOL[moli,2,0,0] - EZERO[moli], '--', c='black', lw=1 )
            plt.plot( WC_LIST, 100000 * (WC_LIST - (EPOL[moli,2,0,0] - EZERO[moli]) ), '--', c='black', lw=1 )    
            
            plt.colorbar(pad=0.01,label="Average Photon Number")
            plt.xlim(WC_LIST[0],WC_LIST[-1])
            plt.ylim(EMIN, EMAX)
            plt.title( "$N_{MOL}$"+f" = {mol};  "+"$A_0$"+f" = {A0} a.u.", fontsize=15 )
            plt.xlabel( "Cavity Frequency, $\omega_c$ (eV)", fontsize=15 )
            plt.ylabel( "Polariton Energy (eV)", fontsize=15 )
            plt.savefig( f"{DATA_DIR}/EPOL_WCSCAN_A0_{A0}_NMOL_{mol}.jpg", dpi=600 )
            plt.clf()

def plot_NMOLSCAN_WCFIXED_A0FIXED( EPOL, PHOT ):

    EZERO = EPOL[:,0,0,0] # Ground state at zero coupling for all (NMOL,WC)
    cmap=mpl.colors.LinearSegmentedColormap.from_list('rg',[ "red", "darkred", "black", "darkgreen", "palegreen" ], N=256)
    for A0IND, A0 in enumerate( A0_LIST ):
        for WCIND, WC in enumerate( WC_LIST ):
            A0 = round( A0, 5 )
            WC = round( WC, 5 )
            print(f"Plotting A0 = {A0} a.u.; WC = {WC} eV")

            VMAX = 1 #np.max(PHOT[:NPOL,A0IND,:])

            for state in range( NPOL ):
                plt.scatter( NMOL_LIST, EPOL[:,state,A0IND,WCIND] - EZERO[:], s=25, cmap=cmap, c=PHOT[:,state,A0IND,WCIND], vmin=0.0, vmax=VMAX )
            plt.colorbar(pad=0.01,label="Average Photon Number")
            plt.ylim(-0.01, EMAX)
            plt.xlabel( "Number of Molecules", fontsize=15 )
            plt.ylabel( "Polariton Energy (eV)", fontsize=15 )
            plt.savefig( f"{DATA_DIR}/EPOL_NMOL_SCAN_WC_{WC}_A0_{A0}.jpg", dpi=600 )
            plt.clf()

def plot_DOS( EPOL ):
    NPTS  = 1000
    EGRID = np.linspace( 0, EMAX, NPTS )
    dE    = EGRID[1] - EGRID[0]
    DOS   = np.zeros( (len(NMOL_LIST), len(A0_LIST) , len(WC_LIST) , NPTS) )
    EZERO = EPOL[:,0,0,0] # Ground state at zero coupling for all (NMOL,WC)
    for molIND, mol in enumerate( NMOL_LIST ):
        print(f"DOS {molIND+1} of {len(NMOL_LIST)}")
        for A0IND, A0 in enumerate( A0_LIST ):
            for WCIND, WC in enumerate( WC_LIST ):
                for pt in range( NPTS ):
                    DOS[molIND,A0IND,WCIND,pt] += np.sum( np.exp(  -(EGRID[pt] - (EPOL[molIND,:,A0IND,WCIND] - EZERO[molIND]))**2 / 2 / SIG**2  ) )
    
    
    for A0IND, A0 in enumerate( A0_LIST ):
        print(f"Plotting DOS {A0IND+1} of {len(A0_LIST)}")
        for WCIND, WC in enumerate( WC_LIST ):
            A0 = round( A0, 5 )
            WC = round( WC, 5 )
            for molIND, mol in enumerate( NMOL_LIST ):
                plt.plot( EGRID, DOS[molIND,A0IND,WCIND,:], label=f"NMOL = {mol}" )
            plt.legend()
            plt.xlim(EMIN,EMAX)
            plt.ylim(0)
            plt.title( f"A0 = {A0} a.u.; WC = {WC} eV", fontsize=15 )
            plt.xlabel( "Energy (eV)", fontsize=15 )
            plt.ylabel( "Density of States", fontsize=15 )
            plt.savefig( f"{DATA_DIR}/DOS_NMOL_SCAN_WC_{WC}_A0_{A0}.jpg", dpi=600 )
            plt.clf()

def plot_DOS_PHOT( EPOL, PHOT ):
    NPTS  = 500
    EGRID = np.linspace( EMIN, EMAX, NPTS )
    dE    = EGRID[1] - EGRID[0]
    DOS   = np.zeros( (len(NMOL_LIST), len(A0_LIST) , len(WC_LIST) , NPTS) )
    EZERO = EPOL[:,0,0,0] # Ground state at zero coupling for all (NMOL,WC)
 
    """
    for molIND, mol in enumerate( NMOL_LIST ):
        print(f"DOS {molIND+1} of {len(NMOL_LIST)}")
        for A0IND, A0 in enumerate( A0_LIST ):
            for WCIND, WC in enumerate( WC_LIST ):
                f = PHOT[molIND,:,A0IND,WCIND]
                for pt in range( NPTS ):
                    DOS[molIND,A0IND,WCIND,pt] += np.sum( f[:] * np.exp(  -(EGRID[pt] - (EPOL[molIND,:,A0IND,WCIND] - EZERO[molIND]))**2 / 2 / SIG**2  ) )
    
    
    for A0IND, A0 in enumerate( A0_LIST ):
        print(f"Plotting DOS {A0IND+1} of {len(A0_LIST)}")
        for WCIND, WC in enumerate( WC_LIST ):
            if ( WCIND % 10 == 0 ):
                A0 = round( A0, 5 )
                WC = round( WC, 5 )
                for molIND, mol in enumerate( NMOL_LIST ):
                    plt.plot( EGRID, DOS[molIND,A0IND,WCIND,:], label=f"NMOL = {mol}" )
                plt.legend()
                plt.xlim(EMIN,EMAX)
                plt.ylim(0)
                plt.title( f"A0 = {A0} a.u.; WC = {WC} eV", fontsize=15 )
                plt.xlabel( "Energy (eV)", fontsize=15 )
                plt.ylabel( "Density of States * $\langle\hat{a}^\dag \hat{a}\\rangle$", fontsize=15 )
                plt.savefig( f"{DATA_DIR}/DOS_PHOT_NMOL_SCAN_WC_{WC}_A0_{A0}.jpg", dpi=600 )
                plt.clf()
    """

    SIG_E  = 0.05#0.05
    SIG_WC = 0.005#dWC#0.01
    for moli, mol in enumerate( NMOL_LIST ):
        for A0IND, A0 in enumerate( A0_LIST ):
            A0 = round( A0, 5 )
            DOS_2D = np.zeros( (NPTS, len(WC_LIST)) )
            for pt1 in range( NPTS ):
                print( pt1 )
                for WCIND in range( len(WC_LIST) ):
                    DE = EGRID[pt1]     - (EPOL[moli,:,A0IND,WCIND] - EPOL[moli,0,0,0])
                    f = PHOT[moli,:,A0IND,WCIND]
                    DOS_2D[pt1,WCIND] += np.sum( f[:] * np.exp(  -DE[:]**2 / 2 / SIG_E**2  ) )


            #DOS_2D *= 2 * np.pi * SIG_E * SIG_WC
            #VMAX = 1 #np.ceil(np.max(PHOT[moli,:NPOL,:,WCIND]))
            print(f"2D DOS PHOT: NMOL = {mol} A0 = {A0} a.u.")
            X,Y = np.meshgrid(WC_LIST,EGRID)
            #plt.contourf( X,Y, DOS_2D, cmap='inferno', levels=500, origin='lower', vmin=0.0 )
            #plt.plot( WC_LIST, WC_LIST*0 + EPOL[moli,2,0,0] - EZERO[moli], '--', c='white', lw=1 )
            #plt.plot( WC_LIST, 100000 * (WC_LIST - (EPOL[moli,2,0,0] - EZERO[moli]) ), '--', c='white', lw=1 )    
            #plt.contourf( X,Y, DOS_2D, cmap='binary', levels=1000, origin='lower', vmin=0.0, vmax=1.0 )
            plt.imshow( DOS_2D, origin='lower', interpolation='gaussian', extent=(np.amin(WC_LIST), np.amax(WC_LIST), np.amin(EGRID), np.amax(EGRID)), cmap='binary', aspect='auto', norm=mpl.colors.Normalize(vmin=0, vmax=1) )
            plt.plot( WC_LIST, WC_LIST*0 + EPOL[moli,2,0,0] - EZERO[moli], '--', c='black', lw=1 )
            plt.plot( WC_LIST, 100000 * (WC_LIST - (EPOL[moli,2,0,0] - EZERO[moli]) ), '--', c='black', lw=1 )    
            
            plt.colorbar(pad=0.01,label="Average Photon Number")
            plt.xlim(WC_LIST[0],WC_LIST[-1])
            plt.ylim(EMIN, EMAX)
            plt.title( "$N_{MOL}$"+f" = {mol};  "+"$A_0$"+f" = {A0} a.u.", fontsize=15 )
            plt.xlabel( "Cavity Frequency, $\omega_c$ (eV)", fontsize=15 )
            plt.ylabel( "Polariton Energy (eV)", fontsize=15 )
            plt.savefig( f"{DATA_DIR}/DOS_PHOT_WCSCAN_2DMAP_A0_{A0}_NMOL_{mol}.jpg", dpi=600 )
            plt.clf()













def get_excitonic_dipole( NMOL, A0, WC ):

    I_M     = np.identity( NM )
    I_PH    = np.identity( NF )

    DIP_MOL = np.zeros( (NMOL,NM,NM) )
    U_POL   = np.zeros( (NMOL,NM,NM) )
    for moli in range( NMOL ):
        TMP = np.load(f"../TDDFT/MOL_1/PLOTS_DATA/DIPOLE_RPA.dat.npy")[:NM,:NM]
        DIP_MOL[moli,:,:] = np.einsum('ijd,d->ij', TMP, EVEC_INTS)

    DIP_POL = np.zeros( (NM**NMOL * NF, NM**NMOL * NF) )
    for moli in range( NMOL ):
        if ( moli == 0 ):
            TMP = DIP_MOL[moli,:,:]
        else:
            TMP = I_M
        for molk in range( 1, NMOL ):
            if ( moli == molk and moli != 0 ):
                TMP = np.kron( TMP, DIP_MOL[molk,:,:] )
            else:
                TMP = np.kron( TMP, I_M )
        DIP_POL += np.kron( TMP, I_PH )    

    return DIP_POL

def plot_excitonic_spectra( EPOL ):

    NPTS  = 1000
    EGRID = np.linspace(EMIN,EMAX,NPTS)

    ABS   = np.zeros( (len(NMOL_LIST), len(A0_LIST), len(WC_LIST), NPTS) )
    for moli, mol in enumerate( NMOL_LIST ):
        for A0IND, A0 in enumerate( A0_LIST ):
            for WCIND, WC in enumerate( WC_LIST ):
                A0 = round( A0, 6 )
                WC = round( WC, 6 )

                # Number of included polaritons
                N = NM ** mol * NF
                if ( N > 100 ): 
                    N = 100

                # Dipole matrix (excitonic part in Ad-Fock basis)
                DIP = get_excitonic_dipole( mol, A0, WC )

                # Polariton transformation
                U   = np.load(f"data_PF_NMOL_{mol}/U_{EVEC_OUT}_A0_{A0}_WC_{WC}_NF_{NF}_NEL_{NM}.dat.npy")
                DIP = U.T @ DIP @ U

                # Compute oscillator strength
                E0j     = EPOL[moli, :N, A0IND, WCIND] - EPOL[moli, 0, A0IND, WCIND]
                OSC_STR = (2/3) * E0j[:] * DIP[0,:N] ** 2

                for pt in range(NPTS):
                    ABS[moli,A0IND,WCIND,pt] = np.sum( OSC_STR[:] * np.exp( -( EGRID[pt] - E0j[:N] )**2 / 2 / SIG**2 ) )

    """
    for A0IND, A0 in enumerate( A0_LIST ):
        print(f"Plotting SPEC {A0IND+1} of {len(A0_LIST)}")
        for WCIND, WC in enumerate( WC_LIST ):
            A0 = round( A0, 5 )
            WC = round( WC, 5 )
            for molIND, mol in enumerate( NMOL_LIST ):
                plt.plot( EGRID, ABS[molIND,A0IND,WCIND,:], label=f"NMOL = {mol}" )
            plt.legend()
            plt.xlim(EMIN,EMAX)
            plt.title( f"A0 = {A0} a.u.; WC = {WC} eV", fontsize=15 )
            plt.xlabel( "Energy (eV)", fontsize=15 )
            plt.ylabel( "Absorption Spectra (Arb. Units)", fontsize=15 )
            plt.savefig( f"{DATA_DIR}/ABS_NMOL_SCAN_WC_{WC}_A0_{A0}.jpg", dpi=600 )
            plt.clf()

    MAX_RATIO = np.zeros( (len(NMOL_LIST), len(A0_LIST), len(WC_LIST)) )
    for A0IND, A0 in enumerate( A0_LIST ):
        for WCIND, WC in enumerate( WC_LIST ):
            for moli, mol in enumerate( NMOL_LIST ):
                if ( A0IND == 0 ): 
                    MAX_RATIO[moli,A0IND,WCIND] = 1.0
                    continue
                peaks = find_peaks( ABS[moli,A0IND,WCIND,:], height=1 )[0]
                if ( len(peaks) <= 1 ): 
                    MAX_RATIO[moli,A0IND,WCIND] = 1.0
                    continue
                if ( EGRID[peaks[0]] < EGRID[peaks[1]] ):
                    MAX_RATIO[moli,A0IND,WCIND] = ABS[moli,A0IND,WCIND,peaks[0]] / ABS[moli,A0IND,WCIND,peaks[1]]
                else:
                    MAX_RATIO[moli,A0IND,WCIND] = ABS[moli,A0IND,WCIND,peaks[1]] / ABS[moli,A0IND,WCIND,peaks[0]]


    for WCIND, WC in enumerate( WC_LIST ):
        for A0IND, A0 in enumerate( A0_LIST ):
            A0 = round( A0, 5 )
            WC = round( WC, 5 )
            plt.plot( NMOL_LIST, MAX_RATIO[:,A0IND,WCIND], "-o", label="$A_0$"+" = %1.3f a.u."%A0 )
        plt.legend()
        plt.title( f"WC = {WC} eV", fontsize=15 )
        plt.xlabel( "Number of Molecules", fontsize=15 )
        plt.ylabel( "Absorption Ratio, $\\frac{I_{+}}{I_{-}}$", fontsize=15 )
        plt.savefig( f"{DATA_DIR}/RATIO_ABS_NMOL_SCAN_WC_{WC}_A0.jpg", dpi=600 )
        plt.clf()

    for WCIND, WC in enumerate( WC_LIST ):
        for moli, mol in enumerate( NMOL_LIST ):
            WC = round( WC, 5 )
            plt.plot( A0_LIST, MAX_RATIO[moli,:,WCIND], "-o", label="$N_{MOL}$"+" = %1.0f"%mol )
        plt.legend()
        plt.title( f"WC = {WC} eV", fontsize=15 )
        plt.xlabel( "Coupling Strength, $A_0$, (a.u.)", fontsize=15 )
        plt.ylabel( "Absorption Ratio, $\\frac{I_{+}}{I_{-}}$", fontsize=15 )
        plt.savefig( f"{DATA_DIR}/RATIO_ABS_NMOL_WC_{WC}_A0_SCAN.jpg", dpi=600 )
        plt.clf()
    """

    for moli, mol in enumerate( NMOL_LIST ):
        for A0IND, A0 in enumerate( A0_LIST ):
            A0 = round( A0, 6 )
            VMAX = np.ceil(np.max(ABS[moli,A0IND,:,:]))
            print(f"2D ABS: NMOL = {mol} A0 = {A0} a.u.")
            X,Y = np.meshgrid(WC_LIST,EGRID)
            plt.imshow( ABS[moli,A0IND,:,:].T, origin='lower', interpolation='gaussian', extent=(np.amin(WC_LIST), np.amax(WC_LIST), np.amin(EGRID), np.amax(EGRID)), cmap='binary', aspect='auto', norm=mpl.colors.Normalize(vmin=0, vmax=VMAX) )
            plt.plot( WC_LIST, WC_LIST*0 + EPOL[moli,2,0,0] - EPOL[moli,0,0,0], '--', c='orange', lw=1 )
            plt.plot( WC_LIST, 100000 * (WC_LIST - (EPOL[moli,2,0,0] - EPOL[moli,0,0,0]) ), '--', c='orange', alpha=0.5, lw=1 )    
            
            cbar = plt.colorbar(pad=0.01)
            cbar.set_label("Osillator Strength", size=15)
            cbar.ax.tick_params(labelsize=10 ) 
            plt.xlim(WC_LIST[0],WC_LIST[-1])
            plt.ylim(EMIN, EMAX)
            plt.title( "$N_{MOL}$"+f" = {mol};  "+"$A_0$"+f" = {A0} a.u.", fontsize=15 )
            plt.xlabel( "Cavity Frequency, $\omega_c$ (eV)", fontsize=15 )
            plt.ylabel( "Energy (eV)", fontsize=15 )
            plt.savefig( f"{DATA_DIR}/ABS_WCSCAN_2DMAP_A0_{A0}_NMOL_{mol}.jpg", dpi=600 )
            plt.clf()






def main():
    get_globals()
    EPOL    = get_energies()
    PHOT    = get_average_photon_number()
    #plot_A0SCAN_WCFIXED_NMOLFIXED( EPOL, PHOT )
    #plot_NMOLSCAN_WCFIXED_A0FIXED( EPOL, PHOT )
    
    #plot_WCSCAN_A0FIXED_NMOLFIXED( EPOL, PHOT )
    #plot_excitonic_spectra( EPOL )
    #plot_DOS( EPOL )
    plot_DOS_PHOT( EPOL, PHOT )




if __name__ == "__main__":
    main()