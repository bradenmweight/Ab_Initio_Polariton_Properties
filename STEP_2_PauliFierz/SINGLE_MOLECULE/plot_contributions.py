import numpy as np
from matplotlib import pyplot as plt
import subprocess as sp

A0_LIST = np.arange( 0.0, 0.4+0.01, 0.01 )
WC_LIST = np.arange( 0.0, 5.1, 0.1 )

NF = 5
NM = 50

THETA = 68.8
PHI   = 34.4

DATA_DIR = "PLOTS_CONTRIBUTIONS"
sp.call(f"mkdir -p  {DATA_DIR}",shell=True)

C = np.zeros( (len(A0_LIST), len(WC_LIST), 5) ) # H_PF, H_EL, H_PH, H_INT, H_DSE
for A0i,A0 in enumerate( A0_LIST ):
    for WCi,WC in enumerate( WC_LIST ):
        A0 = round(A0,4)
        WC = round(WC,4)
        C[A0i, WCi, :] = np.loadtxt(f"data_PF/CONTRIBUTIONS_THETA_{round(THETA,2)}_PHI_{round(PHI,2)}_A0_{A0}_WC_{WC}_NF_{NF}_NM_{NM}.dat") / 27.2114 * 630

"""
for WCi,WC in enumerate( WC_LIST ):
    print(f"Saving {WCi+1} of {len(WC_LIST)}")
    np.savetxt(f"{DATA_DIR}/CONTRIBUTIONS_THETA_{round(THETA,2)}_PHI_{round(PHI,2)}_WC_{round(WC,2)}.dat", np.c_[A0_LIST, C[:,WCi,:] - (C[0,WCi,:])[None,:]], fmt="%1.6f", header="A0\tH_PF\tH_EL\tH_PH\tH_INT\tH_DSE" )
    plt.plot( A0_LIST, A0_LIST*0, "--", c="black", lw=2 )
    plt.plot( A0_LIST, C[:,WCi,0] - C[0,WCi,0], c="black", lw=4, label="$H_{PF}$" )
    plt.plot( A0_LIST, C[:,WCi,1] - C[0,WCi,1] , c="red",  lw=2,  label="$H_{EL}$" )
    plt.plot( A0_LIST, C[:,WCi,2] - C[0,WCi,2], c="green", lw=2, label="$H_{PH}$" )
    plt.plot( A0_LIST, C[:,WCi,3], c="blue", lw=2,  label="$H_{INT}$" )
    plt.plot( A0_LIST, C[:,WCi,4], c="orange", lw=2,label="$H_{DSE}$" )
    #plt.plot( A0_LIST, C[:,WCi,3] + C[:,WCi,4], c="purple",lw=2, label="$H_{INT} + H_{DSE}$" )

    plt.legend()
    plt.xlim(A0_LIST[0], A0_LIST[-1])    
    plt.xlabel("Coupling Strength, $A_0$ (a.u.)",fontsize=15)
    plt.ylabel("Energy (kcal/mol)",fontsize=15)
    plt.title("$\omega_\mathrm{c}$ = %1.2f eV" % (round(WC,2)),fontsize=15)
    plt.savefig(f"{DATA_DIR}/CONTRIBUTIONS_THETA_{round(THETA,2)}_PHI_{round(PHI,2)}_WC_{round(WC,2)}.jpg",dpi=300)
    plt.clf()
"""

for A0i,A0 in enumerate( A0_LIST ):
    print(f"Saving {A0i+1} of {len(A0_LIST)}")
    np.savetxt(f"{DATA_DIR}/CONTRIBUTIONS_THETA_{round(THETA,2)}_PHI_{round(PHI,2)}_WC_{round(WC,2)}.dat", np.c_[WC_LIST, C[A0i,:,:] - (C[A0i,0,:])[None,:]], fmt="%1.6f", header="A0\tH_PF\tH_EL\tH_PH\tH_INT\tH_DSE" )
    plt.plot( WC_LIST, WC_LIST*0, "--", c="black", lw=2 )
    plt.plot( WC_LIST, C[A0i,:,0] - C[A0i,0,0], c="black", lw=4, label="$H_{PF}$" )
    plt.plot( WC_LIST, C[A0i,:,1] - C[A0i,0,1] , c="red",  lw=2,  label="$H_{EL}$" )
    plt.plot( WC_LIST, C[A0i,:,2] - C[A0i,0,2], c="green", lw=2, label="$H_{PH}$" )
    plt.plot( WC_LIST, C[A0i,:,3], c="blue", lw=2,  label="$H_{INT}$" )
    plt.plot( WC_LIST, C[A0i,:,4], c="orange", lw=2,label="$H_{DSE}$" )
    #plt.plot( A0_LIST, C[:,WCi,3] + C[:,WCi,4], c="purple",lw=2, label="$H_{INT} + H_{DSE}$" )

    plt.legend()
    plt.xlim(WC_LIST[0], WC_LIST[-1])    
    plt.xlabel("Cavity Frequency, $\omega_\mathrm{c}$ (eV)",fontsize=15)
    plt.ylabel("Energy (kcal/mol)",fontsize=15)
    plt.title("$A_0$ = %1.2f a.u." % (round(A0,2)),fontsize=15)
    plt.savefig(f"{DATA_DIR}/CONTRIBUTIONS_THETA_{round(THETA,2)}_PHI_{round(PHI,2)}_A0_{round(A0,2)}.jpg",dpi=300)
    plt.clf()
