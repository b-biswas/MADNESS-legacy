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

def build_encoder(latent_dim=32, hidden_dim=256, filters=[32, 64, 128, 256], kernels=[3,3,3,3],nb_of_bands=6, conv_activation=None, dense_activation=None):#'sofplus'
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
    mu = Dense(latent_dim)(h)
    sig = Dense(latent_dim, activation='softplus')(h)
    return Model(input_layer, [mu, sig], name ='encoder')

#### Create encooder
def build_decoder(input_shape, latent_dim=32, hidden_dim=256, filters=[32, 64, 128, 256], kernels=[3,3,3,3], conv_activation=None, dense_activation=None, linear_norm=False):
    """
    Return decoder as model
    input_shape: shape of the input data
    latent_dim : dimension of the latent variable
    hidden_dim : dimension of the dense hidden layer
    filters: list of the sizes of the filters used for this model
    list of the size of the kernels used for each filter of this model
    conv_activation: type of activation layer used after the convolutional layers
    dense_activation: type of activation layer used after the dense layers
    """
    input_layer = Input(shape=(latent_dim,))
    h = Dense(hidden_dim, activation=dense_activation)(input_layer)
    h = PReLU()(h)
    w = int(np.ceil(input_shape[0]/2**(len(filters))))
    h = Dense(w*w*filters[-1], activation=dense_activation)(h)
    h = PReLU()(h)
    h = Reshape((w,w,filters[-1]))(h)
    for i in range(len(filters)-1,-1,-1):
        h = Conv2DTranspose(filters[i], (kernels[i],kernels[i]), activation=conv_activation, padding='same', strides=(2,2))(h)
        h = PReLU()(h)
        h = Conv2DTranspose(filters[i], (kernels[i],kernels[i]), activation=conv_activation, padding='same')(h)
        h = PReLU()(h)
    if linear_norm:
        h = Conv2D(input_shape[-1], (3,3), activation='relu', padding='same')(h)
    else:
        h = Conv2D(input_shape[-1], (3,3), activation='sigmoid', padding='same')(h)
    cropping = int(h.get_shape()[1]-input_shape[0])
    if cropping>0:
        print('in cropping')
        if cropping % 2 == 0:
            h = Cropping2D(cropping/2)(h)
        else:
            h = Cropping2D(((cropping//2,cropping//2+1),(cropping//2,cropping//2+1)))(h)

    return Model(input_layer, h, name='decoder')


# Function to define model

def vae_model(latent_dim, hidden_dim, filters, kernels, nb_of_bands, conv_activation=None, dense_activation=None, linear_norm = True):
    """
    Function to create VAE model
    nb_of bands : nb of band-pass filters needed in the model
    """

    #### Parameters to fix
    # input_shape: shape of the input data
    # latent_dim : dimension of the latent variable
    # hidden_dim : dimension of the dense hidden layer
    # filters: list of the sizes of the filters used for this model
    # kernels: list of the size of the kernels used for each filter of this model
    
    input_shape = (64, 64, nb_of_bands)

    # Build the encoder
    encoder = build_encoder(latent_dim, hidden_dim, filters, kernels, nb_of_bands,conv_activation='relu')
    # Build the decoder
    decoder = build_decoder(input_shape, latent_dim, hidden_dim, filters, kernels, linear_norm=linear_norm, conv_activation='relu', dense_activation=None)
    
    return encoder, decoder

def init_permutation_once(x, name):
    return tf.Variable(name=name, initial_value=x, trainable=False)

def flow(latent_dim=32, num_nf_layers=5):
    
    my_bijects = []
    zdist = tfd.MultivariateNormalDiag(loc=[0.0] * latent_dim)
    # loop over desired bijectors and put into list

    np.random.seed(43)

    for i in range(num_nf_layers):
        # Syntax to make a MAF
        anet = tfb.AutoregressiveNetwork(
            params=2, hidden_units=[256, 256], activation="tanh"
        )
        ab = tfb.MaskedAutoregressiveFlow(anet)
        # Add bijector to list
        my_bijects.append(ab)
        # Now permuate (!important!)
        permute = tfb.Permute(permutation=init_permutation_once(np.random.permutation(latent_dim).astype('int32'), name='permutation'+str(i)))
        my_bijects.append(permute)
        #TODO: include batchnorm?
    # put all bijectors into one "chain bijector"
    # that looks like one
    big_bijector = tfb.Chain(my_bijects)
    # make transformed dist
    td = tfd.TransformedDistribution(zdist, bijector=big_bijector)

    input_layer = Input(shape=(latent_dim))
    return Model(input_layer, td.log_prob(input_layer), name='flow')
