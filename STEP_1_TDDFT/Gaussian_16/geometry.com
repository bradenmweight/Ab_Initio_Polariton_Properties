%chk=geometry.chk
%mem=30GB
%nprocshared=12

#p B3LYP/6-31G*
#p TD=(singlets,nstates=50) IOp(6/8=3) IOp(9/40=4)

Title Card Required

0 1
C          0.00000        0.00000       -0.52896
H          0.00000        0.93788       -1.12368
H         -0.00000       -0.93788       -1.12368
O         -0.00000        0.00000        0.67764





