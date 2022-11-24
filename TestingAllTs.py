import ModelClassForPrediction as Model
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import r2_score


#for NNModel in ["nequip_training_NRS_1500K_pbc","nequip_training_NRS_300K_pbc","nequip_training_RS_1500K_pbc","nequip_training_RS_300K_pbc"]:
for NNModel in ["nequip_training_1500K_135Atoms_pbc_500PointsTrain","nequip_training_1500K_60Atoms_pbc_500PointsTrain",\
        "nequip_training_1500K_40Atoms_pbc_500PointsTrain"]:
 for i in ["50K/","300K/","600K/","900K/","1200K/","1500K/"]:
    testsetsize=100
    NN=Model.Model(PathToModel=NNModel+'/',\
        PathToData=i+"/trajectory.npz",\
            TestSetSize=testsetsize)

    PredictedForces,PredictedEnergies,\
    ActualForces,ActualEnergies \
    =NN.PredictForcesAndEnergies()

    np.savez(i+str(testsetsize)+"_Predictions_"+NNModel,PredictedForces=PredictedForces,PredictedEnergies=PredictedEnergies,\
        ActualForces=ActualForces,ActualEnergies=ActualEnergies )


