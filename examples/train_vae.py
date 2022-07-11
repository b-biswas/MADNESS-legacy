from gc import callbacks
import os

import numpy as np

import tensorflow as tf
import tensorflow_probability as tfp

from debvader.batch_generator import COSMOSsequence
from debvader.normalize import LinearNormCosmos

from maddeb.FlowVAEnet import FlowVAEnet
from maddeb.utils import listdir_fullpath
from maddeb.FlowVAEnet import vae_loss_fn_wrapper
from debvader.train import define_callbacks

tfd = tfp.distributions

# define the parameters
batch_size = 100
vae_epochs = 120
flow_epochs = 125
deblender_epochs = 120
latent_dim = 8

prior = tfd.Independent(tfd.Normal(loc=tf.zeros(latent_dim), scale=.5), reinterpreted_batch_ndims=1)
f_net = FlowVAEnet(latent_dim=latent_dim, kl_prior=prior, kl_weight=.01)

######## List of data samples
def listdir_fullpath(d):
    return [os.path.join(d, f) for f in os.listdir(d) if not f.endswith("metadata.npy")]


datalist_isolated = listdir_fullpath("/sps/lsst/users/bbiswas/simulations/COSMOS_btk_isolated/")

train_path_isolated_gal = datalist_isolated[:700]
validation_path_isolated_gal = datalist_isolated[700:]

# Keras Callbacks
path_weights = "data/" + "cosmos8d/"

######## Define the generators

normalizer = LinearNormCosmos()

train_generator_vae = COSMOSsequence(train_path_isolated_gal, 'blended_gal_stamps', 'blended_gal_stamps', 
                                 batch_size=batch_size, num_iterations_per_epoch=400,
                                 normalizer=normalizer)

validation_generator_vae = COSMOSsequence(validation_path_isolated_gal, 'blended_gal_stamps', 'blended_gal_stamps', 
                                 batch_size=batch_size, num_iterations_per_epoch=100, 
                                 normalizer=normalizer)

######## Define all used callbacks
callbacks = define_callbacks(os.path.join(path_weights, "vae"), lr_scheduler_epochs=25)

hist_vae = f_net.train_vae(
    train_generator_vae,
    validation_generator_vae,
    callbacks=callbacks,
    epochs=vae_epochs,
    train_encoder=True,
    train_decoder=True,
    track_kl = True,
    optimizer=tf.keras.optimizers.Adam(2e-4),
)

np.save(path_weights + '/train_vae_history.npy',hist_vae.history)

f_net = FlowVAEnet(latent_dim=latent_dim, kl_prior=None, kl_weight=None)
f_net.load_vae_weights(os.path.join(path_weights, "vae" , "val_loss"))

######## Define all used callbacks
callbacks = define_callbacks(os.path.join(path_weights, "flow"), lr_scheduler_epochs=25)

# now train the model
hist_flow = f_net.train_flow(
    train_generator_vae,
    validation_generator_vae,
    callbacks=callbacks,
    epochs=flow_epochs,
)

np.save(os.path.join(path_weights, 'train_vae_history.npy'), hist_flow.history)

f_net.flow.trainable = False
deblend_prior = f_net.td
deblend_prior.trainable=False
print(f_net.flow.trainable_variables)


f_net = FlowVAEnet(latent_dim=latent_dim, kl_prior=None, kl_weight=None)
f_net.load_vae_weights(os.path.join(path_weights, "vae" , "val_loss"))
#f_net.randomize_encoder()

datalist_blended = listdir_fullpath("/sps/lsst/users/bbiswas/simulations/COSMOS_btk/")

train_path_blended_gal = datalist_blended[:700]
validation_path_blended_gal = datalist_blended[700:]

train_generator_deblender = COSMOSsequence(train_path_blended_gal, 'blended_gal_stamps', 'isolated_gal_stamps', 
                                 batch_size=batch_size, num_iterations_per_epoch=400,
                                 normalizer=normalizer)

validation_generator_deblender = COSMOSsequence(validation_path_blended_gal, 'blended_gal_stamps', 'isolated_gal_stamps', 
                                 batch_size=batch_size, num_iterations_per_epoch=100, 
                                 normalizer=normalizer)
######## Define all used callbacks
callbacks = define_callbacks(os.path.join(path_weights, "deblender"), lr_scheduler_epochs=25)

#f_net.vae_model.get_layer("latent_space").activity_regularizer=None

hist_deblender = f_net.train_vae(
    train_generator_deblender,
    validation_generator_deblender,
    callbacks=callbacks,
    epochs=deblender_epochs,
    train_encoder=True,
    train_decoder=False,
    track_kl=True,
    optimizer=tf.keras.optimizers.Adam(1e-4),
    loss_function=vae_loss_fn_wrapper(sigma=None, linear_norm_coeff=80000),
)

np.save(path_weights + '/train_deblender_history.npy', hist_deblender.history)
