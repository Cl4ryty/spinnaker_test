import os
import numpy as np
import tensorflow as tf
import dill as pickle
import matplotlib.pyplot as plt
from snntoolbox.bin.run import main
from snntoolbox.utils.utils import import_configparser


file_path = "test_de"

# get dict entry (loss function)

f = open('serialized_custom_loss_functions.txt', 'rb')  # opened the file in write and binary mode
reconstructed_dict = pickle.load(f)  # dumping the content in the variable 'content' into the file
f.close()  # closing the file

m = tf.keras.models.load_model(file_path, custom_objects={"ann_loss_function": reconstructed_dict[file_path]})


x = tf.linspace(0., 2., num=3)
x = tf.expand_dims(x, -1)

x = tf.cast(x, tf.int32)
print("inputs", x)



# Save dataset so SNN toolbox can find it.
np.savez_compressed('x_test', x)
np.savez_compressed('y_test', x)

test = m(x)
# plot result
plt.plot(tf.squeeze(x), tf.squeeze(m(x)))
plt.show()


path_wd = os.getcwd()
model_name = file_path
input_size = 3


# SNN TOOLBOX CONFIGURATION #
#############################

# Create a config file with experimental setup for SNN Toolbox.
configparser = import_configparser()
config = configparser.ConfigParser()

config['paths'] = {
    'path_wd': path_wd,  # Path to model.
    'dataset_path': path_wd,  # Path to dataset.
    'filename_ann': model_name  # Name of input model.
}

config['tools'] = {
    'evaluate_ann': False,  # Test ANN on dataset before conversion.
    'parse': True,
    'normalize': False,
    'convert': True,
    'simulate': True
}

config['simulation'] = {
    'simulator': 'spiNNaker',  # Chooses execution backend of SNN toolbox.
    'duration': 50,  # Number of time steps to run each sample.
    'num_to_test': input_size,  # How many test samples to run.
    'batch_size': input_size,  # Batch size for simulation.
    'keras_backend': 'tensorflow'  # Which keras backend to use.
}

config['input'] = {
    'model_lib': 'keras'  # Input model is defined in pytorch.
}

config['cell'] = {
    'v_thresh': 0.01
    # Should be 0.01 for optimal correspondences between original ANN and converted SNN when simulated on PyNN
}

config['output'] = {

    'plot_vars': {  # Various plots (slows down simulation).
        'spiketrains',  # Leave section empty to turn off plots.
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
# source_path = inspect.getfile(Model)
# shutil.copyfile(source_path, os.path.join(path_wd, model_name + '.py'))

# RUN SNN TOOLBOX #
###################

main(config_filepath)