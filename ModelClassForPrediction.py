import numpy as np
import torch
import os
import logging
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import max_error
import seaborn as sns
import pandas as pd


from nequip.data import AtomicDataDict
from nequip.nn import (
    GraphModuleMixin,
    SequentialGraphNetwork,
    AtomwiseLinear,
    AtomwiseReduce,
    ForceOutput,
    PerSpeciesScaleShift,
    ConvNetLayer,
)
from nequip.nn.embedding import (
    OneHotAtomEncoding,
    RadialBasisEdgeEncoding,
    SphericalHarmonicEdgeAttrs,
)

from nequip.utils import Config
from sklearn.metrics import r2_score
import numpy as np
import matplotlib.pyplot as plt
from nequip.data import AtomicData



class Model:

    def __init__(self,PathToModel,PathToData,TestSetSize):

        self.ParentPath=os.getcwd()
        self.PathToData=os.path.join(self.ParentPath,PathToData)
        self.PathToModel=os.path.join(self.ParentPath,PathToModel)
        self.TestSetSize=TestSetSize
#         self.PathToPrediction=os.path.join(self.PathToData,"ML/")

#         if os.path.exists(self.PathToPrediction):
#             shutil.rmtree(self.PathToPrediction)
#         os.mkdir(self.PathToPrediction); print(f"Created {self.PathToPrediction}")

    def GetDataAndNumberOfAtoms(self):


        data=np.load(self.PathToData)
        return data,data['z'],data['CELL']


#         os.chdir(self.PathToData)
#         os.system("vibes u traj 2db trajectory.son")
#         atoms=read(self.PathToData+'trajectory.db',index=':')
#         self.NumberOfAtoms=len(atoms[0].numbers)
#         print(f"Number of atoms = {len(atoms[0].numbers)}")
#         return np.array(atoms[0].numbers)


    def PredictForcesAndEnergies(self):

        torch.cuda.empty_cache()
        print(self.ParentPath)
        print(self.PathToModel)
        config = Config.from_file(self.PathToModel+'tutorial.yaml')
        final_model=torch.load(self.PathToModel+'deployed.pth')
        final_model.eval()

        Data,z,CELL = self.GetDataAndNumberOfAtoms()
        
        PredictedForcesList=np.empty([len(z),3,0]); PredictedEnergiesList=np.empty([0,1])
        ActualForcesList=np.empty([len(z),3,0]); ActualEnergiesList=np.empty([0,1])

        for i in np.arange(self.TestSetSize):
            #i=0
            #CELL=Data['CELL']
            #print(CELL)
            print(f"Running point {i}")
            r = Data['R'][i] #+3
            #print(np.shape(r))
            forces = Data['F'][i]
            #print(np.shape(forces))
            energy=Data['E'][i]

            #print(np.shape(energy))
            #energies_list.append(energy)

            data = AtomicData.from_points(
        pos=r,
        r_max=config['r_max'],pbc=True,cell=CELL,\
        **{AtomicDataDict.ATOMIC_NUMBERS_KEY: torch.Tensor(torch.from_numpy(z.astype(np.float32))).to(torch.int64)}
    )
            data = data.to('cpu')

            ForcePrediction = final_model(AtomicData.to_AtomicDataDict(data))['forces'].detach().cpu().numpy()
            EnergyPrediction = final_model(AtomicData.to_AtomicDataDict(data))['total_energy'].detach().cpu().numpy()


            PredictedForcesList=np.dstack((PredictedForcesList,ForcePrediction))
            PredictedEnergiesList=np.vstack((PredictedEnergiesList,EnergyPrediction))

            ActualForcesList=np.dstack((ActualForcesList,forces))
            ActualEnergiesList=np.vstack((ActualEnergiesList,energy))



        return PredictedForcesList,PredictedEnergiesList.reshape(-1),\
    ActualForcesList,ActualEnergiesList.reshape(-1)
#         return PredictedForcesList.flatten(),\
#     PredictedEnergiesList.flatten(),ActualForcesList.flatten(),ActualEnergiesList.flatten()

