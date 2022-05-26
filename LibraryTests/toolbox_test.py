"""End-to-end example for SNN Toolbox.
This script sets up a small CNN using PyTorch, trains it for one epoch on
MNIST, stores model and dataset in a temporary folder on disk, creates a
configuration file for SNN toolbox, and finally calls the main function of SNN
toolbox to convert the trained ANN to an SNN and run it using INI simulator.
"""

import os
import shutil
import inspect
import time
from functools import partial

import numpy as np
import torch
import torch.nn as nn
from tensorflow.keras import backend
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

from snntoolbox.bin.run import main
from snntoolbox.utils.utils import import_configparser
from trained import Model, compute_loss, check_gradient, train_model, save_model, f
from scipy.integrate import solve_ivp

R = 1.0
F0 = 1.0



# Pytorch to Keras parser needs image_data_format == channel_first.
backend.set_image_data_format('channels_first')

# WORKING DIRECTORY #
#####################

# Define path where model and output files will be stored.
# The user is responsible for cleaning up this temporary directory.
path_wd = os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(
    __file__)), '..', 'temp', str(time.time())))
os.makedirs(path_wd)

# GET DATASET #
###############
domain = [0.0, 1.0]
x = torch.linspace(domain[0], domain[1], steps=10, requires_grad=True)
x = x.reshape(x.shape[0], 1)

x_eval = torch.linspace(domain[0], domain[1], steps=100).reshape(-1, 1)

# numeric solution
def logistic_eq_fn(x, y):
    return R * x * (1 - x)

numeric_solution = solve_ivp(
    logistic_eq_fn, domain, [F0], t_eval=x_eval.squeeze().detach().numpy()
)

x_np = x_eval.detach().numpy()
y_np = numeric_solution.y.T

# Save dataset so SNN toolbox can find it.
np.savez_compressed(os.path.join(path_wd, 'x_test'), x_np)
np.savez_compressed(os.path.join(path_wd, 'y_test'), y_np)


# class PytorchDataset(torch.utils.data.Dataset):
#     def __init__(self, data, target, transform=None):
#         self.data = torch.from_numpy(data).float()
#         self.target = torch.from_numpy(target).long()
#         self.transform = transform
#
#     def __getitem__(self, index):
#         x = self.data[index]
#
#         if self.transform:
#             x = self.transform(x)
#
#         return x, self.target[index]
#
#     def __len__(self):
#         return len(self.data)
#
#
# trainset = torch.utils.data.DataLoader(PytorchDataset(x_train, y_train),
#                                        batch_size=64)
# testset = torch.utils.data.DataLoader(PytorchDataset(x_test, y_test),
#                                       batch_size=64)

# CREATE ANN #
##############

# This section creates a CNN using pytorch, and trains it with backpropagation.
# There are no spikes involved at this point.

# Create pytorch model from definition in separate script.
model = Model()

# Train model with backprop.
assert check_gradient(model, x)

# train the PINN
loss_fn = partial(compute_loss, x=x, verbose=True)
model, loss_evolution = train_model(
    model, loss_fn=loss_fn, learning_rate=0.1, max_epochs=20_000
)




# Store weights so SNN Toolbox can find them.
model_name = 'pytorch_cnn'
torch.save(model.state_dict(), os.path.join(path_wd, model_name + '.pkl'))

# SNN TOOLBOX CONFIGURATION #
#############################

# Create a config file with experimental setup for SNN Toolbox.
configparser = import_configparser()
config = configparser.ConfigParser()

config['paths'] = {
    'path_wd': path_wd,             # Path to model.
    'dataset_path': path_wd,        # Path to dataset.
    'filename_ann': model_name      # Name of input model.
}

config['tools'] = {
    'evaluate_ann': True,           # Test ANN on dataset before conversion.
    'normalize': False               # Normalize weights for full dynamic range.
}

config['simulation'] = {
    'simulator': 'INI',             # Chooses execution backend of SNN toolbox.
    'duration': 50,                 # Number of time steps to run each sample.
    'num_to_test': 100,             # How many test samples to run.
    'batch_size': 50,               # Batch size for simulation.
    'keras_backend': 'tensorflow'   # Which keras backend to use.
}

config['input'] = {
    'model_lib': 'pytorch'          # Input model is defined in pytorch.
}

config['output'] = {
    'plot_vars': {                  # Various plots (slows down simulation).
        'spiketrains',              # Leave section empty to turn off plots.
        'spikerates',
        'activations',
        'correlation',
        'v_mem',
        'error_t'}
}

# Store config file.
config_filepath = os.path.join(path_wd, 'config')
with open(config_filepath, 'w') as configfile:
    config.write(configfile)

# Need to copy model definition over to ``path_wd`` (needs to be in same dir as
# the weights saved above).
source_path = inspect.getfile(Model)
shutil.copyfile(source_path, os.path.join(path_wd, model_name + '.py'))

# RUN SNN TOOLBOX #
###################

main(config_filepath)