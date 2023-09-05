import numpy as np
from matplotlib import pyplot as plt

# Define sizes
NM = 50
NF = 5
NPOL = NM * NF

# Read ground state wavefunction
U0 = np.load("../data_PF/U_1_0_0_A0_0.5_WC_10.0_NF_5_NM_50.dat.npy")[:,0]

# Get full density matrix
RHO_FULL = np.outer( U0, U0 )

# Get reduced electronic density matrix
RHO_MATTER = np.zeros(( NM, NM ))
for alpha in range( NM ):
    for beta in range( NM ):
        for n in range( NF ):
            for m in range( NF ):
                POL1 = alpha * NF + n
                POL2 = beta * NF + m
                if ( n == m ):
                    RHO_MATTER[alpha,beta] += RHO_FULL[POL1,POL2]
np.savetxt("RHO_MATTER.dat", RHO_MATTER)

# Diagonalize RHO_MATTER to obtain natural wavefunctions
E,U = np.linalg.eigh( RHO_MATTER )
np.savetxt("EIGV_MATTER.dat", E)
np.savetxt("EIGU_MATTER.dat", U)

# Plot eigenvalues
plt.plot(np.arange(NM),E[:],"o-")
plt.savefig("E.jpg")
plt.clf()


# Plot eigenvectors with largest eigenvalues
#for state in range(10):
plt.plot( np.arange(NM), U[:,-1], "-o", label=f"$\lambda$" + " = %1.2f" % (E[-1]) )
plt.plot( np.arange(NM), U[:,-2], "-o", label=f"$\lambda$" + " = %1.2f" % (E[-2]) )
#plt.plot( np.arange(NM), U[:,-3], "-o", label=f"$\lambda$" + " = %1.2f" % (E[-3]) )
plt.legend()
plt.savefig("U.jpg")
plt.clf()

