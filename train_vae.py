from gc import callbacks
import os

import numpy as np

import tensorflow as tf
from debvader.batch_generator import COSMOSsequence
from debvader.normalize import LinearNormCosmos

from scripts.FlowVAEnet import FlowVAEnet
from scripts.utils import listdir_fullpath

from debvader.train import define_callbacks

# define the parameters
batch_size = 200
generative_epochs = 100
vae_epochs = 150
deblender_epochs = 150
latent_dim = 10

f_net = FlowVAEnet(latent_dim=latent_dim)

######## List of data samples
def listdir_fullpath(d):
    return [os.path.join(d, f) for f in os.listdir(d) if not f.endswith("metadata.npy")]


datalist = listdir_fullpath("/sps/lsst/users/bbiswas/simulations/COSMOS_btk/")

train_path = datalist[:700]
validation_path = datalist[700:]

# Keras Callbacks
path_weights = "data/" + "cosmos10d_largesig_largekl/"

######## Define the generators

normalizer = LinearNormCosmos()

train_generator_vae = COSMOSsequence(train_path, 'isolated_gal_stamps', 'isolated_gal_stamps', 
                                 batch_size=batch_size, num_iterations_per_epoch=400,
                                 normalizer=normalizer)

validation_generator_vae = COSMOSsequence(validation_path, 'isolated_gal_stamps', 'isolated_gal_stamps', 
                                 batch_size=batch_size, num_iterations_per_epoch=120, 
                                 normalizer=normalizer)

######## Define all used callbacks
callbacks = define_callbacks(os.path.join(path_weights, "vae"), lr_scheduler_epochs=15)

hist_vae = f_net.train_vae(
    train_generator_vae,
    validation_generator_vae,
    callbacks=callbacks,
    epochs=vae_epochs,
    train_encoder=True,
    train_decoder=True,
    optimizer=tf.keras.optimizers.Adam(5e-4)
)


#f_net.load_vae_weights(os.path.join(path_weights, "vae" , "val_loss"))
#f_net.randomize_encoder()

train_generator_deblender = COSMOSsequence(train_path, 'blended_gal_stamps', 'isolated_gal_stamps', 
                                 batch_size=200, num_iterations_per_epoch=400,
                                 normalizer=normalizer)

validation_generator_deblender = COSMOSsequence(validation_path, 'blended_gal_stamps', 'isolated_gal_stamps', 
                                 batch_size=200, num_iterations_per_epoch=120, 
                                 normalizer=normalizer)

######## Define all used callbacks
callbacks = define_callbacks(os.path.join(path_weights, "deblender"), lr_scheduler_epochs=15)

hist_deblender = f_net.train_vae(
    train_generator_deblender,
    validation_generator_deblender,
    callbacks=callbacks,
    epochs=deblender_epochs,
    train_encoder=True,
    train_decoder=False,
    optimizer=tf.keras.optimizers.Adam(1e-4),
)

#np.save(path_weights + '/train_vae_history.npy',hist_vae.history)
np.save(path_weights + '/train_deblender_history.npy', hist_deblender.history)
