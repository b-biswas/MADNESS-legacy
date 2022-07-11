import os

import numpy as np

import tensorflow as tf
from debvader.batch_generator import COSMOSsequence
from debvader.normalize import LinearNormCosmos
from debvader.train import define_callbacks

from maddeb.FlowVAEnet import FlowVAEnet
from maddeb.utils import listdir_fullpath

# define the parameters
batch_size = 200
flow_epochs = 125
latent_dim = 10
num_iter_per_epoch = None

f_net = FlowVAEnet(latent_dim=latent_dim)

print("Num GPUs Available: ", len(tf.config.list_physical_devices("GPU")))

# Keras Callbacks
path_weights = "data/cosmos10d/"

######## List of data samples
def listdir_fullpath(d):
    return [os.path.join(d, f) for f in os.listdir(d) if not f.endswith("metadata.npy")]


datalist = listdir_fullpath("/sps/lsst/users/bbiswas/simulations/COSMOS_btk/")

train_path = datalist[:800]
validation_path = datalist[800:]

######## Define the generators
train_generator = COSMOSsequence(
    train_path,
    "isolated_gal_stamps",
    "isolated_gal_stamps",
    batch_size=batch_size,
    num_iterations_per_epoch=400,
    normalizer=LinearNormCosmos(),
)

validation_generator = COSMOSsequence(
    validation_path,
    "isolated_gal_stamps",
    "isolated_gal_stamps",
    batch_size=batch_size,
    num_iterations_per_epoch=100,
    normalizer=LinearNormCosmos(),
)

# load the vae weights for encoder
f_net.load_vae_weights(os.path.join(path_weights, "vae", "val_loss"))
# f_net.load_flow_weights(os.path.join(path_weights, "fvae"))

######## Define all used callbacks
callbacks = define_callbacks(os.path.join(path_weights, "flow"), lr_scheduler_epochs=15)

# now train the model
hist_flow = f_net.train_flow(
    train_generator,
    validation_generator,
    callbacks=callbacks,
    epochs=flow_epochs,
)

np.save(path_weights + '/train_vae_history.npy',hist_flow.history)
