import numpy as np

import tensorflow as tf
import tensorflow_probability as tfp

import matplotlib as mpl
import warnings
import os

from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Conv2D, Input, Dense, Dropout, MaxPool2D, Flatten,  Reshape, UpSampling2D, Cropping2D, Conv2DTranspose, PReLU, Concatenate, Lambda, BatchNormalization, concatenate, LeakyReLU

tfd = tfp.distributions
tfb = tfp.bijectors

def build_encoder(latent_dim=64, hidden_dim=256, filters=[32, 64, 128, 256], kernels=[3,3,3,3],nb_of_bands=6, conv_activation=None, dense_activation=None):#'sofplus'
    """
    Return encoder as model
    latent_dim : dimension of the latent variable
    hidden_dim : dimension of the dense hidden layer
    filters: list of the sizes of the filters used for this model
    list of the size of the kernels used for each filter of this model
    conv_activation: type of activation layer used after the convolutional layers
    dense_activation: type of activation layer used after the dense layers
    nb_of bands : nb of band-pass filters needed in the model
    """
    input_layer = Input(shape=(64,64,nb_of_bands))

    h = Reshape((64,64,nb_of_bands))(input_layer)
    h = BatchNormalization()(h)
    for i in range(len(filters)):
        h = Conv2D(filters[i], (kernels[i],kernels[i]), activation=conv_activation, padding='same')(h)
        h = PReLU()(h)
        h = Conv2D(filters[i], (kernels[i],kernels[i]), activation=conv_activation, padding='same', strides=(2,2))(h)
        h = PReLU()(h)
    h = Flatten()(h)
    h = Dense(hidden_dim, activation=dense_activation)(h)
    h = PReLU()(h)
    latent_space = Dense(latent_dim)(h)
    return Model(input_layer, latent_space)

def flow(latent_dim=64, num_nf_layers=5):
    
    my_bijects = []
    zdist = tfd.MultivariateNormalDiag(loc=[0.0] * latent_dim)
    # loop over desired bijectors and put into list
    for i in range(num_nf_layers):
        # Syntax to make a MAF
        anet = tfb.AutoregressiveNetwork(
            params=2, hidden_units=[16, 16], activation="relu"
        )
        ab = tfb.MaskedAutoregressiveFlow(anet)
        # Add bijector to list
        my_bijects.append(ab)
        # Now permuate (!important!)
        permutation=np.random.permutation(latent_dim)
        permute = tfb.Permute(permutation)
        my_bijects.append(permute)
    # put all bijectors into one "chain bijector"
    # that looks like one
    big_bijector = tfb.Chain(my_bijects)
    # make transformed dist
    td = tfd.TransformedDistribution(zdist, bijector=big_bijector)
    return td
