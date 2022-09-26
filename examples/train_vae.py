import os

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from maddeb.batch_generator import COSMOSsequence
from maddeb.FlowVAEnet import FlowVAEnet, vae_loss_fn_wrapper
from maddeb.train import define_callbacks
from maddeb.utils import get_data_dir_path, listdir_fullpath

tfd = tfp.distributions

# define the parameters
batch_size = 100
vae_epochs = 250
flow_epochs = 150
deblender_epochs = 250
lr_scheduler_epochs = 40
latent_dim = 8
linear_norm_coeff = 80000

kl_prior = tfd.Independent(
    tfd.Normal(loc=tf.zeros(latent_dim), scale=1), reinterpreted_batch_ndims=1
)
kl_weight = 1

f_net = FlowVAEnet(latent_dim=latent_dim, kl_prior=kl_prior, kl_weight=kl_weight)

train_path_isolated_gal = listdir_fullpath(
    "/sps/lsst/users/bbiswas/simulations/COSMOS_btk_isolated_train/"
)
validation_path_isolated_gal = listdir_fullpath(
    "/sps/lsst/users/bbiswas/simulations/COSMOS_btk_isolated_validation/"
)

# Keras Callbacks
data_path = get_data_dir_path()

path_weights = os.path.join(data_path, "cosmos" + str(latent_dim) + "d")

# Define the generators

train_generator_vae = COSMOSsequence(
    train_path_isolated_gal,
    "blended_gal_stamps",
    "blended_gal_stamps",
    batch_size=batch_size,
    num_iterations_per_epoch=400,
    linear_norm_coeff=linear_norm_coeff,
)

validation_generator_vae = COSMOSsequence(
    validation_path_isolated_gal,
    "blended_gal_stamps",
    "blended_gal_stamps",
    batch_size=batch_size,
    num_iterations_per_epoch=100,
    linear_norm_coeff=linear_norm_coeff,
)

# Define all used callbacks
callbacks = define_callbacks(os.path.join(path_weights, "vae"), lr_scheduler_epochs=lr_scheduler_epochs)

hist_vae = f_net.train_vae(
    train_generator_vae,
    validation_generator_vae,
    callbacks=callbacks,
    epochs=vae_epochs,
    train_encoder=True,
    train_decoder=True,
    track_kl=True,
    optimizer=tf.keras.optimizers.Adam(2e-4),
)

np.save(path_weights + "/train_vae_history.npy", hist_vae.history)

f_net = FlowVAEnet(latent_dim=latent_dim, kl_prior=None, kl_weight=None)
f_net.load_vae_weights(os.path.join(path_weights, "vae", "val_loss"))

# Define all used callbacks
callbacks = define_callbacks(os.path.join(path_weights, "flow"), lr_scheduler_epochs=lr_scheduler_epochs)

# now train the model
hist_flow = f_net.train_flow(
    train_generator_vae,
    validation_generator_vae,
    callbacks=callbacks,
    epochs=flow_epochs,
)

np.save(os.path.join(path_weights, "train_vae_history.npy"), hist_flow.history)

f_net.flow.trainable = False
# deblend_prior = f_net.td
# deblend_prior.trainable = False
# print(f_net.flow.trainable_variables)


f_net = FlowVAEnet(latent_dim=latent_dim, kl_prior=kl_prior, kl_weight=kl_weight)
f_net.load_vae_weights(os.path.join(path_weights, "vae", "val_loss"))
# f_net.randomize_encoder()

train_path_blended_gal = listdir_fullpath(
    "/sps/lsst/users/bbiswas/simulations/COSMOS_btk_blended_train/"
)
validation_path_blended_gal = listdir_fullpath(
    "/sps/lsst/users/bbiswas/simulations/COSMOS_btk_blended_validation/"
)

train_generator_deblender = COSMOSsequence(
    train_path_blended_gal,
    "blended_gal_stamps",
    "isolated_gal_stamps",
    batch_size=batch_size,
    num_iterations_per_epoch=400,
    linear_norm_coeff=linear_norm_coeff,
)

validation_generator_deblender = COSMOSsequence(
    validation_path_blended_gal,
    "blended_gal_stamps",
    "isolated_gal_stamps",
    batch_size=batch_size,
    num_iterations_per_epoch=100,
    linear_norm_coeff=linear_norm_coeff,
)
# Define all used callbacks
callbacks = define_callbacks(
    os.path.join(path_weights, "deblender"), lr_scheduler_epochs=lr_scheduler_epochs,
)

# f_net.vae_model.get_layer("latent_space").activity_regularizer=None

hist_deblender = f_net.train_vae(
    train_generator_deblender,
    validation_generator_deblender,
    callbacks=callbacks,
    epochs=deblender_epochs,
    train_encoder=True,
    train_decoder=False,
    track_kl=True,
    optimizer=tf.keras.optimizers.Adam(2e-4),
    loss_function=vae_loss_fn_wrapper(sigma=None, linear_norm_coeff=linear_norm_coeff),
)

np.save(path_weights + "/train_deblender_history.npy", hist_deblender.history)
