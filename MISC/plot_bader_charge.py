import numpy as np
from matplotlib import pyplot as plt

#Q_TOTAL = 98 # Enforce total charge to be {Q_TOTAL}

#A0_LIST = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5])
A0_LIST = np.arange(0.0, 0.31, 0.01)
WC_LIST = np.array([10.0])

#PLOT_ATOMS = np.array([3,13,11,4,8,15]).astype(int) # Labels
#PLOT_ATOMS = [3] # Labels
PLOT_ATOMS = np.arange(15)+1

#ION_CHARGES = np.array([6,6,6,6,7,35]) # MAKE SURE THESE ARE CORRECT AND GREATER THAN 0
ION_CHARGES = np.array([6,6,6,6,1,1,1,7,8,8,6,1,6,1,35]) # MAKE SURE THESE ARE CORRECT AND GREATER THAN 0

NATOMS = 15
CHARGE = np.zeros(( len(A0_LIST), len(WC_LIST), NATOMS ))
for A0IND,A0 in enumerate(A0_LIST):
    for WCIND,WC in enumerate(WC_LIST):
        A0 = round(A0,6)
        WC = round(WC,6)
        #lines = open(f"difference_density_00_cavity_no-cavity_1_0_0_A0{A0}_WC{WC}_NM50_NF5.cube.ACF","r").readlines()
        lines = open(f"diagonal_density_1_0_0_A0{A0}_WC{WC}_NM50_NF5_P00.cube.ACF","r").readlines()
        for at in range( NATOMS ):
            #CHARGE[A0IND,WCIND,at] = -1 * float( lines[2+at].split()[4] ) * 1000 # |e| --> m|e|
            CHARGE[A0IND,WCIND,at] = -1 * float( lines[2+at].split()[4] ) # |e| --> m|e|

# Correct charges by uniform scaling
#for A0IND,A0 in enumerate(A0_LIST):
#    NORM = -np.sum( CHARGE[A0IND,0,:] )
#    CHARGE[A0IND,0,:] *= np.abs(Q_TOTAL / NORM)
#    print( A0, NORM, np.sum( CHARGE[A0IND,0,:] ) )
#NORM = np.einsum("AWa->AW", CHARGE[:,:,:])
#CHARGE = np.einsum("AWa,AW->AWa", CHARGE[:,:,:], Q_TOTAL / NORM[:,:] )

np.savetxt( "CHARGES_A0SCAN.dat", \
            np.c_[A0_LIST, CHARGE[:,0,PLOT_ATOMS-1]], \
            fmt="%1.6f", \
            header=" ".join(map(str,PLOT_ATOMS)) )

for count,at in enumerate(PLOT_ATOMS):
    plt.plot( A0_LIST, CHARGE[:,0,at-1]+ION_CHARGES[count], "-o", label=f"Atom {at}" )
    plt.legend()
    plt.xlim(A0_LIST[0],A0_LIST[-1])
    plt.xlabel("Coupling Strength, $A_0$ (a.u.)",fontsize=15)
    plt.ylabel("Atomic Charge, Q (|e|)",fontsize=15)
    plt.savefig(f"CHARGES_A0SCAN_ATOM_{at}.jpg",dpi=600)
    plt.clf()

for count,at in enumerate(PLOT_ATOMS):
    plt.plot( A0_LIST, CHARGE[:,0,at-1]+ION_CHARGES[count], "-o", label=f"Atom {at}" )
plt.legend()
plt.xlim(A0_LIST[0],A0_LIST[-1])
plt.xlabel("Coupling Strength, $A_0$ (a.u.)",fontsize=15)
plt.ylabel("Atomic Charge, Q (|e|)",fontsize=15)
plt.savefig(f"CHARGES_A0SCAN_ATOM_ALL.jpg",dpi=600)
plt.clf()


# PLOT CHARGE DIFFERENCE BETWEEN CONNECTED C-N ATOMS
C_LABEL = 3 # Label
C_ION   = 6 # Positive Charge
N_LABEL = 8 # Label
N_ION   = 7 # Positive Charge

Q_C = CHARGE[:,0,C_LABEL-1] + C_ION
Q_N = CHARGE[:,0,N_LABEL-1] + N_ION
dQ  = Q_C - Q_N
plt.plot( A0_LIST, dQ, "-o", label="$Q_C - Q_N$" )
plt.legend()
plt.xlim(A0_LIST[0],A0_LIST[-1])
plt.xlabel("Coupling Strength, $A_0$ (a.u.)",fontsize=15)
plt.ylabel("$Q_C - Q_N$ (|e|)",fontsize=15)
plt.savefig(f"CHARGES_A0SCAN_dQ_CN.jpg",dpi=600)
plt.clf()
