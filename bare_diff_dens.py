import numpy as np
import subprocess as sp

def get_globals():
    global NM, NF, DATA_DIR
    NM = 50
    NF = 5
    DATA_DIR = "data_bare_difference_density"

    sp.call(f"mkdir -p {DATA_DIR}", shell=True)

def read_TD():

    # Get size from first TD cube file from QCHEM (not Gaussian)
    global NAtoms, NGrid, Nxyz, dLxyz, Lxyz, coords
    header = np.array([ np.array(j.split(),dtype=float) for j in open(f"../QCHEM.plots/dens.0.cube","r").readlines()[2:6] ])
    NAtoms = int(header[0,0])
    NGrid  = int( header[0,1] )
    Nxyz   = (header[1:,0]).astype(int)
    dLxyz  = np.array([header[1,1],header[2,2],header[3,3] ]).astype(float)
    Lxyz   = np.array([ header[0,1], header[0,2], header[0,3] ]).astype(float)
    if ( Lxyz[0] < 0 ): Lxyz *= -1.000 # Switch Sign, Already Angstroms
    if ( Lxyz[0] > 0 ): Lxyz *= 0.529 # Convert from Bohr to Angstroms
    Vol    = Lxyz[0] * Lxyz[1] * Lxyz[2]
    
    # QCHEM EXCITED DENSITIES MAY BE WRONG... THEY FORGET to MULTIPLY BY 2 FOR RESTRICTED DFT...
    is_QCHEM = False # Let's check this and fix if necessary for correct diagonal density normalization
    GS_NORM = 0.0 # Compare excited state norms to ground state for the check. If within 1 e-, is probably fine.

    print (f'\tNAtoms   = {NAtoms}')
    print (f'\tTD Grid  = {NGrid}')
    print (f'\tNx Ny Nz = {Nxyz[0]} {Nxyz[1]} {Nxyz[2]}')
    print (f'\tLx Ly Lz = {Lxyz[0]} {Lxyz[1]} {Lxyz[2]} A')
    print (f'\tVolume   = {Vol} A^3')

    NStart = NAtoms + 6

    coords = np.array([ j for j in open(f"../QCHEM.plots/dens.0.cube","r").readlines()[6:NStart] ])


    DENS = np.zeros(( NM, Nxyz[0], Nxyz[1], Nxyz[2]  ))
    print(f"\tMemory size of transition density array in (MB, GB): ({round(DENS.size * DENS.itemsize * 10 ** -6,2)},{round(DENS.size * DENS.itemsize * 10 ** -9,2)})" )
    for state_j in range(NM):
        temp = []
        lines = open(f"../QCHEM.plots/dens.{state_j}.cube",'r').readlines()[NStart:]
        for count, line in enumerate(lines):
            t = line.split('\n')[0].split()
            for j in range(len(t)):
                temp.append( float(t[j]) )
        DENS[state_j,:,:,:] = np.array( temp ).reshape(( Nxyz[0],Nxyz[1],Nxyz[2] ))
        norm = np.sum( DENS[state_j,:,:,:],axis=(0,1,2) ) * dLxyz[0]*dLxyz[1]*dLxyz[2]
        if ( state_j == 0 ):
            GS_NORM = norm * 1.0
            GS_NORM = round(GS_NORM/2)*2 # Round to nearest even electron number
            DENS[state_j,:,:,:] *= GS_NORM/norm
            print("Renormalized ground state = ", np.sum( DENS[state_j] ) * dLxyz[0]*dLxyz[1]*dLxyz[2])
        else:
            if ( abs( norm - GS_NORM ) > 0.25 ): # Check within quarter of an electron...
                is_QCHEM = True
                print("I FOUND QCHEM ERROR ! Forcing non-ground state densities to be same as ground state density...")
                DENS[state_j,:,:,:] *= (GS_NORM/norm)
                norm                       = GS_NORM
                print("Skipping diagonal density normalization: norm =", norm)

    np.save( "DENS.dat.npy", DENS )
    return DENS

def saveDATA(DENS):
    for state in range(NM):
        print(f"\tComputing and saving {state}-0")
        Diff = DENS[state,:,:,:] - DENS[0,:,:,:]
        f = open(f'{DATA_DIR}/difference_density_S{state}-S0.cube','w')
        f.write(f"Difference Density: S{state}-S0 \n")
        f.write(f"Totally {np.product(Nxyz)} grid points\n")
        f.write(f"{NAtoms} {-Lxyz[0]/0.529} {-Lxyz[1]/0.529} {-Lxyz[2]/0.529}\n")
        f.write(f"{Nxyz[0]} {dLxyz[0]}  0.000000   0.000000\n")
        f.write(f"{Nxyz[1]} 0.000000   {dLxyz[1]} 0.000000\n")
        f.write(f"{Nxyz[2]} 0.000000   0.000000   {dLxyz[2]} \n")
        for at in range(len(coords)):
            f.write( coords[at] )
        for x in range(Nxyz[0]):
            #print(f'X = {x}')
            for y in range(Nxyz[1]):
                outArray = []
                for z in range(Nxyz[2]):
                    outArray.append( Diff[x,y,z] )
                    if ( len(outArray) % 6 == 0 or z == Nxyz[2]-1 ):
                        #outArray.append('\n')
                        f.write( " ".join(map( str, np.round(outArray,8) )) + "\n" )
                        outArray = []
        f.close()

def main():
    get_globals()
    DENS = read_TD()
    saveDATA(DENS)

if ( __name__ == "__main__" ):
    main()