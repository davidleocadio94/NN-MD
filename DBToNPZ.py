from ase.io import read
import numpy as np
import os


path_data="1500K/"
atoms = read(path_data+'/trajectory.db',index=':')


# parse properties as list of dictionaries
property_list = []
E=[]
F=[]
R=[]
z=[]
CELL=[]
PBC=[]
for at in atoms:

    E.append([at.get_potential_energy()])
    F.append(at.get_forces())
    R.append(at.get_positions())
    z.append(at.numbers)
    CELL.append(at.get_cell())
    PBC.append(at.get_pbc())

E=np.array(E)
F=np.array(F)
R=np.array(R)
z=np.array(z)
CELL=np.array(CELL)
PBC=np.array(PBC)
#randomly_permuted_idx=np.random.permutation(np.arange(len(E)))
randomly_permuted_idx=np.arange(len(E))
size_test=20
#randomly_permuted_idx=np.arange(len(atoms))
idxsTrain=randomly_permuted_idx[:-size_test]
idxsTrain=idxsTrain.astype(int)
idxsTest=randomly_permuted_idx[-size_test:]
idxsTest=idxsTest.astype(int)
idxMINS = np.argpartition(E.flatten(), 5)
print(idxMINS)
idxsTest=np.append(idxsTest,idxMINS[:5])
len(idxsTest)+len(idxsTrain)


np.savez(path_data+'/HASMINtrajectory',E=E[idxsTrain],F=F[idxsTrain],R=R[idxsTrain],z=z[0],CELL=\
        CELL[0],PBC=PBC[0])
np.savez(path_data+'/HASMINtrajectoryTest',E=E[idxsTest],F=F[idxsTest],R=R[idxsTest],z=z[0],CELL=\
        CELL[0],PBC=PBC[0])
print("Train",np.shape(idxsTrain))
print("Test",np.shape(idxsTest))
