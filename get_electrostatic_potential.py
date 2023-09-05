import numpy as np
#from numba import jit

ELECTRIC_CHARGE = 98 # [mBr]^+


def read_density_cube(filename):

    # Get size from first TD cube file from QCHEM (not Gaussian)
    global NAtoms, NGrid, Nxyz, dLxyz, Lxyz, coords, atom_charges, coords_array
    header = np.array([ np.array(j.split(),dtype=float) for j in open(filename,"r").readlines()[2:6] ])
    NAtoms = int(header[0,0])
    NGrid  = int( header[0,1] )
    Nxyz   = (header[1:,0]).astype(int)
    dLxyz  = np.array([header[1,1],header[2,2],header[3,3] ]).astype(float)
    Lxyz   = np.array([ header[0,1], header[0,2], header[0,3] ]).astype(float)
    if ( Lxyz[0] > 0 ): Lxyz *= -1.000 # Switch Sign, Already Angstroms
    if ( Lxyz[0] < 0 ): 
        print("Converting from Bohr to Angstroms")
        Lxyz  *= -1 #0.529 # Convert from Bohr to Angstroms
        dLxyz *= 1 #0.529 # Convert from Bohr to Angstroms
    print (f'\tNAtoms   = {NAtoms}')
    print (f'\tTD Grid  = {NGrid}')
    print (f'\tNx Ny Nz = {Nxyz[0]} {Nxyz[1]} {Nxyz[2]}')
    print (f'\tLx Ly Lz = {Lxyz[0]} {Lxyz[1]} {Lxyz[2]} A')
    print ('\tVolume   = %1.5f A^3' % (Lxyz[0] * Lxyz[1] * Lxyz[2]))

    NStart = NAtoms + 6

    coords = np.array([ j for j in open(filename,"r").readlines()[6:NStart] ])

    TD = np.zeros(( Nxyz[0], Nxyz[1], Nxyz[2]  ))
    temp = []
    lines = open(filename,'r').readlines()[NStart:]
    for count, line in enumerate(lines):
        t = line.split('\n')[0].split()
        for j in range(len(t)):
            temp.append( float(t[j]) )
    TD[:,:,:] = np.array( temp ).reshape(( Nxyz[0],Nxyz[1],Nxyz[2] ))
    norm  = np.sum( TD[:,:,:] ) * dLxyz[0]*dLxyz[1]*dLxyz[2]
    GS_NORM = norm * 1.0
    GS_NORM = round(GS_NORM/2)*2 # Round to nearest even electron number
    TD[:,:,:] *= GS_NORM/norm
    norm = np.sum( TD[:,:,:] ) * dLxyz[0]*dLxyz[1]*dLxyz[2]
    if ( int(norm) != ELECTRIC_CHARGE ):
        print( "\tCHARGE WRONG FROM USER INPUT:", int(norm), "!=", ELECTRIC_CHARGE )
        TD[:,:,:] *= ELECTRIC_CHARGE/norm
    print("\tCharge   = %1.5f" % (np.sum( TD ) * dLxyz[0]*dLxyz[1]*dLxyz[2]) )

    atom_charges = np.array([ j.split()[0] for j in coords ]).astype(int)
    coords_array = np.array([ np.array(j.split()[2:]).astype(float) for j in coords ]).astype(float)

    return TD

#@jit(nopython=True)
def compute_electrostatic_potential( CUBE_DATA ):
    """
    V(x,y,z) = \int dx' dy' dz'  1/(4 \pi \epsilon_0) * \rho(x',y',z') / |r - r' + \eta|
    \eta = 1e-10 # For numerical convergence
    """


    XGRID = np.linspace( -Lxyz[0], Lxyz[0], Nxyz[0] )
    YGRID = np.linspace( -Lxyz[1], Lxyz[1], Nxyz[1] )
    ZGRID = np.linspace( -Lxyz[2], Lxyz[2], Nxyz[2] )
    dV     = dLxyz[0] * dLxyz[1] * dLxyz[2]
    POT   = np.zeros(( Nxyz[0], Nxyz[1], Nxyz[2] ))
    for xi,x in enumerate( XGRID ):
        print("x", xi, "of", Nxyz[0])
        for yi,y in enumerate( YGRID ):
            #print("y", yi, "of", Nxyz[1])
            for zi,z in enumerate( ZGRID ):
                #print("z", zi, "of", Nxyz[2])
                #for xip,xp in enumerate( XGRID ):
                #    for yip,yp in enumerate( YGRID ):
                #        for zipp,zp in enumerate( ZGRID ):
                #            dR2 = (x - xp)**2 + (y - yp)**2 + (z - zp)**2
                #            dR  = np.sqrt( dR2 )
                #            POT[xi,yi,zi] += CUBE_DATA[xip,yip,zipp] / dR
                dx = x - XGRID
                dy = y - YGRID
                dz = z - ZGRID
                R    = np.sqrt( np.add.outer(np.add.outer(dx**2,  dy**2), dz**2 ) )
                RINV = 1/R
                RINV[ RINV == float('inf') ] = 1e8 # Choose big number
                POT[xi,yi,zi] = np.einsum( "xyz,xyz->", CUBE_DATA, RINV  ) * dV
                
                # Nuclear part
                for at in range( NAtoms ):
                    dx = x - coords_array[at,0]
                    dy = y - coords_array[at,1]
                    dz = z - coords_array[at,2]
                    R  = np.sqrt( dx**2 + dy**2 + dz**2 )
                    POT[xi,yi,zi] += -1 * atom_charges[at] / R 

    return POT

def saveCUBE( POT ):

    f = open(f'QCHEM.plots/POT_0_Braden_NUC_dV.cube','w')
    f.write(f"Electrostatic Potential \n")
    f.write(f"Totally {NGrid} grid points\n")
    f.write(f"{NAtoms} {-Lxyz[0]} {-Lxyz[1]} {-Lxyz[2]}\n")
    f.write(f"{Nxyz[0]} {dLxyz[0]}  0.000000   0.000000\n")
    f.write(f"{Nxyz[1]} 0.000000   {dLxyz[1]} 0.000000\n")
    f.write(f"{Nxyz[2]} 0.000000   0.000000   {dLxyz[2]} \n")
    for at in range(len(coords)):
        f.write( coords[at] )
    for x in range(Nxyz[0]):
        for y in range(Nxyz[1]):
            outArray = []
            for z in range(Nxyz[2]):
                outArray.append( POT[x,y,z] )
                if ( len(outArray) % 6 == 0 or z == Nxyz[2]-1 ):
                    f.write( " ".join(map( str, np.round(outArray,8) )) + "\n" )
                    outArray = []
    f.close()


def main():
    CUBE_DATA = read_density_cube("QCHEM.plots/dens.0.cube")
    POT       = compute_electrostatic_potential( CUBE_DATA )
    saveCUBE( POT )

if ( __name__ == "__main__" ):
    main()
