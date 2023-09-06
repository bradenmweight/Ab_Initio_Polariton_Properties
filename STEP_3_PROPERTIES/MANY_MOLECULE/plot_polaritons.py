import numpy as np
from matplotlib import pyplot as plt
import matplotlib as mpl
import subprocess as sp
import os

def get_globals():
    global NM, NF, A0_LIST, WC_LIST
    global EVEC_INTS, EVEC_NORM, EVEC_OUT
    global DATA_DIR, NPOL, NA0, NWC, EMAX
    global NMOL_LIST

    ##### MAIN USER INPUT SECTION #####
    NMOL_LIST = np.array([1,2,3,4,5])
    NM        = 5                   # Number of Electronic States (including ground state)
    NF        = 5                   # Number of Fock Basis States
    EMAX      = 15.0
    EVEC_INTS = np.array([ 0,0,1 ]) # Cavity Polarization Vector (input as integers without normalizing)
    
    A0_LIST = np.arange( 0.0, 0.5+0.025, 0.025 ) # a.u.
    #WC_LIST = np.arange( 0.0, 20+1.0, 1.0 )
    WC_LIST = np.array([5.0])
    ##### END USER INPUT SECTION  #####



    ##### DO NOT MODIFY BELOW HERE #####
    NPOL = 100 # NM**NMOL_LIST[-1] * NF
    NA0  = len(A0_LIST)
    NWC  = len(WC_LIST)

    EVEC_NORM = EVEC_INTS / np.linalg.norm(EVEC_INTS)
    EVEC_OUT = "_".join(map(str,EVEC_INTS))

    DATA_DIR = "PLOTS_DATA"
    sp.call(f"mkdir -p {DATA_DIR}", shell=True)

def get_energies():

    EPOL = np.zeros(( NMOL_LIST[-1], NPOL, NA0, NWC ))

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

    return EPOL

def get_average_photon_number():

    def get_a_op():
        a = np.zeros((NF,NF))
        for m in range(1,NF):
            a[m,m-1] = np.sqrt(m)
        return a.T

    PHOT = np.zeros(( NPOL, NA0, NWC ))
    a_op = get_a_op() # CHECK THIS
    I_M  = np.identity( NM )
    N_op = np.kron( I_M, a_op.T @ a_op )

    if ( os.path.isfile(f"{DATA_DIR}/PHOT.dat.npy") ):
        tmp = np.load(f"{DATA_DIR}/PHOT.dat.npy")
        if ( tmp.shape == PHOT.shape ):
            return tmp

    for A0IND, A0 in enumerate( A0_LIST ):
        for WCIND, WC in enumerate( WC_LIST ):
            A0 = round( A0, 5 )
            WC = round( WC, 5 )
            U = np.load(f"data_PF/U_{EVEC_OUT}_A0_{A0}_WC_{WC}_NF_{NF}_NM_{NM}.dat.npy")
            PHOT[:, A0IND, WCIND] = np.einsum( "Jp,JK,Kp->p", U[:,:] , N_op[:,:] , U[:,:] )
    
    np.save(f"{DATA_DIR}/PHOT.dat.npy", PHOT)
    return PHOT

def plot_A0SCAN_WCFIXED( EPOL, PHOT ):

    EZERO = EPOL[0,0,0]
    cmap=mpl.colors.LinearSegmentedColormap.from_list('rg',[ "red", "darkred", "black", "darkgreen", "palegreen" ], N=256)
    for WCIND, WC in enumerate( WC_LIST ):
        WC = round( WC, 5 )
        print(f"Plotting WC = {WC} eV")
        VMAX = np.max(PHOT[:5,:,WCIND])
        for state in range( 50 ):
            plt.scatter( A0_LIST, EPOL[state,:,WCIND] - EZERO, s=25, cmap=cmap, c=PHOT[state,:,WCIND], vmin=0.0, vmax=VMAX )
        
        plt.colorbar(pad=0.01,label="Average Photon Number")
        plt.xlim(A0_LIST[0],A0_LIST[-1])
        plt.ylim(-0.01, EMAX)
        plt.xlabel( "Coupling Strength, $A_0$ (a.u.)", fontsize=15 )
        plt.ylabel( "Polariton Energy (eV)", fontsize=15 )
        plt.savefig( f"{DATA_DIR}/EPOL_A0SCAN_WC_{WC}.jpg", dpi=600 )
        plt.clf()



def plot_WCSCAN_A0FIXED( EPOL, PHOT ):

    EZERO = EPOL[0,0,0]
    cmap=mpl.colors.LinearSegmentedColormap.from_list('rg',[ "red", "darkred", "black", "darkgreen", "palegreen" ], N=256)
    for A0IND, A0 in enumerate( A0_LIST ):
        A0 = round( A0, 5 )
        VMAX = np.max(PHOT[:5,A0IND,:])
        print(f"Plotting A0 = {A0} a.u.")
        for state in range( 50 ):
            plt.scatter( WC_LIST, EPOL[state,A0IND,:] - EZERO, s=25, cmap=cmap, c=PHOT[state,A0IND,:], vmin=0.0, vmax=VMAX )
        
        plt.colorbar(pad=0.01,label="Average Photon Number")
        plt.xlim(WC_LIST[0],WC_LIST[-1])
        plt.ylim(-0.01, EMAX)
        plt.xlabel( "Cavity Frequency, $\omega_c$ (eV)", fontsize=15 )
        plt.ylabel( "Polariton Energy (eV)", fontsize=15 )
        plt.savefig( f"{DATA_DIR}/EPOL_WCSCAN_A0_{A0}.jpg", dpi=600 )
        plt.clf()

def plot_NMOLSCAN_WCFIXED_A0FIXED( EPOL):

    EZERO = EPOL[0,0,0,0]
    for A0IND, A0 in enumerate( A0_LIST ):
        for WCIND, WC in enumerate( WC_LIST ):
            A0 = round( A0, 5 )
            WC = round( WC, 5 )
            print(f"Plotting A0 = {A0} a.u.; WC = {WC} eV")
            for state in range( NPOL ):
                plt.scatter( NMOL_LIST, EPOL[:,state,A0IND,WCIND] - EZERO, s=25 )
            plt.ylim(-0.01, EMAX)
            plt.xlabel( "Number of Molecules", fontsize=15 )
            plt.ylabel( "Polariton Energy (eV)", fontsize=15 )
            plt.savefig( f"{DATA_DIR}/EPOL_WCSCAN_A0_{A0}.jpg", dpi=600 )
            plt.clf()


def main():
    get_globals()
    EPOL = get_energies()
    #PHOT = get_average_photon_number()
    #plot_A0SCAN_WCFIXED( EPOL, PHOT )
    #plot_WCSCAN_A0FIXED( EPOL, PHOT )
    plot_NMOLSCAN_WCFIXED_A0FIXED( EPOL )
    


if __name__ == "__main__":
    main()