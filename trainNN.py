import torch
torch.set_default_dtype(torch.float32)
import os
torch.cuda.empty_cache()
from nequip.utils import Config
config = Config.from_file('./config.yaml')

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
#print("HEY")
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
#data=np.load('tutorial_data/trajectoryTest.npz')
#print(np.shape(benzene_data['R']))


results_path='results/run'
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










