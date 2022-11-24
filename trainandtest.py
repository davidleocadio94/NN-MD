import torch
torch.set_default_dtype(torch.float32)
import os
torch.cuda.empty_cache()
from nequip.utils import Config
config = Config.from_file('./tutorial.yaml')

import pprint
pprint.pprint(config.as_dict())

import logging

from nequip.train.trainer import Trainer
trainer = Trainer(model=None, **dict(config))
print(trainer)




from nequip.utils import dataset_from_config
dataset = dataset_from_config(config)
logging.info(f"Successfully loaded the data set of type {dataset}...")

trainer.set_dataset(dataset)


from nequip.data import AtomicDataDict



(
    (forces_std,),
    (energies_mean, energies_std),
    (allowed_species, Z_count),
) = trainer.dataset_train.statistics(
    fields=[
        AtomicDataDict.FORCE_KEY,
        AtomicDataDict.TOTAL_ENERGY_KEY,
        AtomicDataDict.ATOMIC_NUMBERS_KEY,
    ],
    modes=["rms", "mean_std", "count"],
)
print("HEY")
print(AtomicDataDict.ATOMIC_NUMBERS_KEY)
print(allowed_species)



#data=AtomicDataDict.ATOMIC_NUMBERS_KEY
#lst = data.files
#print(lst)
#for item in lst:
#    print(type(item))
#    print(item)
#    print(np.size(data[item]))
#    print(data[item])
#    print(type(data[item]))




from nequip.models import ForceModel
config.update(dict(allowed_species=allowed_species))
force_model_ = ForceModel(**dict(config))





import logging

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

def EnergyModel(**shared_params) -> SequentialGraphNetwork:
    """Base default energy model archetecture.

    For minimal and full configuration option listings, see ``minimal.yaml`` and ``example.yaml``.
    """
    logging.debug("Start building the network model")

    num_layers = shared_params.pop("num_layers", 3)
    add_per_species_shift = shared_params.pop("PerSpeciesScaleShift_enable", False)

    layers = {
        # -- Encode --
        "one_hot": OneHotAtomEncoding,
        "spharm_edges": SphericalHarmonicEdgeAttrs,
        "radial_basis": RadialBasisEdgeEncoding,
        # -- Embed features --
        "chemical_embedding": AtomwiseLinear,
    }

    # add convnet layers
    # insertion preserves order
    for layer_i in range(num_layers):
        layers[f"layer{layer_i}_convnet"] = ConvNetLayer

    # .update also maintains insertion order
    layers.update(
        {
            # -- output block --
            "conv_to_output_hidden": AtomwiseLinear,
            "output_hidden_to_scalar": (
                AtomwiseLinear,
                dict(irreps_out="1x0e", out_field=AtomicDataDict.PER_ATOM_ENERGY_KEY),
            ),
        }
    )

    if add_per_species_shift:
        layers["per_species_scale_shift"] = (
            PerSpeciesScaleShift,
            dict(
                field=AtomicDataDict.PER_ATOM_ENERGY_KEY,
                out_field=AtomicDataDict.PER_ATOM_ENERGY_KEY,
            ),
        )

    layers["total_energy_sum"] = (
        AtomwiseReduce,
        dict(
            reduce="sum",
            field=AtomicDataDict.PER_ATOM_ENERGY_KEY,
            out_field=AtomicDataDict.TOTAL_ENERGY_KEY,
        ),
    )

    return SequentialGraphNetwork.from_parameters(
        shared_params=shared_params,
        layers=layers,
    )


def ForceModel(**shared_params) -> GraphModuleMixin:
    """Base default energy and force model archetecture.

    For minimal and full configuration option listings, see ``minimal.yaml`` and ``example.yaml``.

    A convinience method, equivalent to constructing ``EnergyModel`` and passing it to ``nequip.nn.ForceOutput``.
    """
    energy_model = EnergyModel(**shared_params)
    return ForceOutput(energy_model=energy_model)


core_model = ForceModel(**dict(config))
logging.info("Successfully built the network...")

from nequip.nn import RescaleOutput

final_model = RescaleOutput(
    model=core_model, 
    scale_keys=[AtomicDataDict.TOTAL_ENERGY_KEY, AtomicDataDict.FORCE_KEY],
    scale_by=forces_std,
    shift_keys=AtomicDataDict.TOTAL_ENERGY_KEY,
    shift_by=energies_mean,
)


trainer.model = final_model


trainer.train()


import numpy as np
import matplotlib.pyplot as plt
#benzene_data = np.load(config.dataset_file_name)
benzene_data=np.load('tutorial_data/trajectoryTest.npz')
#print(np.shape(benzene_data['R']))


results_path='tutorial-results/example-run'
# Load logged results
results = np.loadtxt(os.path.join(results_path, 'metrics_epoch.csv'), skiprows=1, delimiter=',')

# Determine time axis
time = results[:,1]-results[0,0]

# Load the validation MAEs
energy_mae = results[:,14]
forces_mae = results[:,13]
train_loss=results[:,5]
val_loss=results[:,11]
# Get final validation errors
print('Validation MAE:')
print('    energy: {:10.3f} eV/atom'.format(energy_mae[-1]))
print('    forces: {:10.3f} eV/atom/Ang'.format(forces_mae[-1]))

# Construct figure
plt.figure(figsize=(14,5))

# Plot energies
plt.subplot(1,2,1)
plt.plot(time, energy_mae)
plt.title('Energy')
plt.ylabel('MAE [eV/atom]')
plt.xlabel('Time [s]')

# Plot forces
plt.subplot(1,2,2)
plt.plot(time, forces_mae)
plt.title('Forces')
plt.ylabel('MAE [eV/atom/Ang]')
plt.xlabel('Time [s]')

plt.show()
plt.savefig('MAEs.png')
#END OF MAEs
##################################################################################################
# Construct figure
plt.figure(figsize=(14,5))
# Plot train and val loss
plt.subplot(1,2,1)
plt.plot(time, train_loss,label='train loss')
plt.plot(time, val_loss,label='val loss')
plt.title('Train and val loss')
plt.ylabel('A.u.')
plt.xlabel('Time [s]')
plt.legend()










stats=[]
predicted_forces=[]
actual_forces=[]
predicted_energies=[]
actual_energies=[]

final_model.eval()
size_test=10
for i in range(size_test):
    
    r = benzene_data['R'][-i]
    #print(r)
    forces = benzene_data['F'][-i]
    energies=benzene_data['E'][-i]
    
    
    final_model.eval(); 
    
    
    from nequip.data import AtomicData
    
    data = AtomicData.from_points(
        pos=r,
        r_max=config['r_max'], 
        **{AtomicDataDict.ATOMIC_NUMBERS_KEY: torch.Tensor(torch.from_numpy(benzene_data['z'].astype(np.float32))).to(torch.int64)}
    )
    data = data.to('cuda')
    
    prediction = final_model(AtomicData.to_AtomicDataDict(data))['forces']
    prediction=prediction.cpu()
    print("predicted forces")
    #print(prediction)
    print(np.max(prediction.ravel().tolist()))
    print(np.mean(prediction.ravel().tolist()))
    predicted_forces.append(prediction.ravel().tolist())
    prediction[abs(prediction)<1e-3]=0.0
    actual=forces
    print("actual forces")
    print(np.max(actual.ravel().tolist()))
    print(np.mean(actual.ravel().tolist()))
        #appendvalues
    actual_forces.append(actual.ravel().tolist())
    actual[abs(actual)<1e-3]=0.0
        #calculate signs with hadamard product
    hadamard_product=np.sign(np.multiply(prediction,actual)).ravel().tolist()
    hadamard_product[hadamard_product==0]=1.0
    stats.append(hadamard_product)


        #calculate similar signs and stats for energies
    prediction=final_model(AtomicData.to_AtomicDataDict(data))['total_energy']
    prediction=prediction.cpu()
        #append values
    predicted_energies.append(prediction.ravel().tolist())
    actual=energies
        #appendvalues
    actual_energies.append(actual.ravel().tolist())



############################### END OF STATS COLLECTING   
print(predicted_energies) 
from matplotlib.ticker import PercentFormatter
#calculate the similar signs
stats=sum(stats,[])
plt.subplot(1,2,2)
plt.hist(stats,weights=np.ones(len(stats)) / len(stats))
plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
plt.title('Right sign')
plt.xticks([-1,1])


plt.show()
plt.savefig('TandVLossRightSignF.png')

predicted_forces=sum(predicted_forces,[])
actual_forces=sum(actual_forces,[])
#CORRELATION PLOTS FORCES
plt.figure(figsize=(14,5))
plt.subplot(1,2,1)
m, b = np.polyfit(predicted_forces, actual_forces, 1)
from sklearn.metrics import r2_score
r2=r2_score(actual_forces,predicted_forces)
plt.plot(predicted_forces, actual_forces,'o',label='Correlation. R2 = %1.3f and slope = %1.3f' %(r2,m))
plt.plot([np.min(actual_forces), np.max(actual_forces)],[np.min(actual_forces), np.max(actual_forces)],'--')
plt.ylabel('Actual forces')
plt.xlabel('Predicted forces')
plt.legend()
#HISTOGRAM OF FORCE VALUES
plt.subplot(1,2,2)
mean_pred=np.mean(predicted_forces)
sdev_pred=np.std(predicted_forces)
mean_actual=np.mean(actual_forces)
sdev_actual=np.std(actual_forces)
plt.hist(predicted_forces,alpha=0.5, label='Pred F. mean = %1.3f, stdev = %1.3f' %(mean_pred,sdev_pred))
plt.hist(actual_forces,alpha=0.5,label='Actual F. mean = %1.3f, stdev = %1.3f' %(mean_actual,sdev_actual))
plt.title('Histograms of Forces')
plt.xlabel('Force values')
plt.legend(loc='upper right')
plt.show()
plt.savefig('CorrPlotsForceHist.png')



predicted_energies=sum(predicted_energies,[])
actual_energies=sum(actual_energies,[])

#CORRELATION PLOTS ENERGIES
plt.figure(figsize=(14,5))
plt.subplot(1,2,1)
m, b = np.polyfit(predicted_energies, actual_energies, 1)
from sklearn.metrics import r2_score
r2=r2_score(actual_energies,predicted_energies)
plt.plot(predicted_energies, actual_energies,'o',label='Correlation. R2 = %1.3f and slope = %1.3f' %(r2,m))
plt.plot([np.min(actual_energies), np.max(actual_energies)],[np.min(actual_energies), np.max(actual_energies)],'--')
plt.ylabel('Actual Energy')
plt.xlabel('Predicted Energy')
plt.legend()

#HISTOGRAM OF FORCE VALUES
plt.subplot(1,2,2)
mean_pred=np.mean(predicted_energies)
sdev_pred=np.std(predicted_energies)
mean_actual=np.mean(actual_energies)
sdev_actual=np.std(actual_energies)
plt.hist(predicted_energies,alpha=0.5, label='Pred E. mean = %1.3f, stdev = %1.3f' %(mean_pred,sdev_pred))
plt.hist(actual_energies,alpha=0.5,label='Actual E. mean = %1.3f, stdev = %1.3f' %(mean_actual,sdev_actual))
plt.title('Histograms of Energies')
plt.xlabel('Energies values')
plt.legend(loc='upper right')
plt.show()
plt.savefig('CorrPlotsEnergyHist.png')
