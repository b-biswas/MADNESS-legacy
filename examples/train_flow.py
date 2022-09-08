import os

import numpy as np
import tensorflow as tf

from maddeb.batch_generator import COSMOSsequence
from maddeb.FlowVAEnet import FlowVAEnet
from maddeb.train import define_callbacks
from maddeb.utils import listdir_fullpath

# define the parameters
batch_size = 200
flow_epochs = 125
latent_dim = 10
num_iter_per_epoch = None

f_net = FlowVAEnet(latent_dim=latent_dim)

print("Num GPUs Available: ", len(tf.config.list_physical_devices("GPU")))

# Keras Callbacks
path_weights = "data/cosmos + str(latent_dim) + d/"

train_path = listdir_fullpath(
    "/sps/lsst/users/bbiswas/simulations/COSMOS_btk_isolated_train/"
)
validation_path = listdir_fullpath(
    "/sps/lsst/users/bbiswas/simulations/COSMOS_btk_isolated_validation/"
)

# Define the generators
train_generator = COSMOSsequence(
    train_path,
    "isolated_gal_stamps",
    "isolated_gal_stamps",
    batch_size=batch_size,
    num_iterations_per_epoch=400,
    linear_norm_coeff=80000,
)

validation_generator = COSMOSsequence(
    validation_path,
    "isolated_gal_stamps",
    "isolated_gal_stamps",
    batch_size=batch_size,
    num_iterations_per_epoch=100,
    linear_norm_coeff=80000,
)

# load the vae weights for encoder
f_net.load_vae_weights(os.path.join(path_weights, "vae", "val_loss"))
# f_net.load_flow_weights(os.path.join(path_weights, "fvae"))

# Define all used callbacks
callbacks = define_callbacks(os.path.join(path_weights, "flow"), lr_scheduler_epochs=15)

# now train the model
hist_flow = f_net.train_flow(
    train_generator,
    validation_generator,
    callbacks=callbacks,
    epochs=flow_epochs,
)

np.save(path_weights + "/train_vae_history.npy", hist_flow.history)
