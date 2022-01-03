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

def create_encoder(
    input_shape,
    latent_dim,
    filters,
    kernels,
    conv_activation=None,
    dense_activation=None,
):
    # Define the prior for the latent space
    prior = tfd.Independent(
        tfd.Normal(loc=tf.zeros(latent_dim), scale=1), reinterpreted_batch_ndims=1
    )

    # Input layer
    input_layer = Input(shape=(input_shape))

    # Define the model
    h = BatchNormalization()(input_layer)
    for i in range(len(filters)):
        h = Conv2D(
            filters[i], (kernels[i], kernels[i]), activation=None, padding="same"
        )(h)
        h = PReLU()(h)
        h = Conv2D(
            filters[i],
            (kernels[i], kernels[i]),
            activation=None,
            padding="same",
            strides=(2, 2),
        )(h)
        h = PReLU()(h)

    h = Flatten()(h)
    h = PReLU()(h)
    h = Dense(
        tfp.layers.MultivariateNormalTriL.params_size(latent_dim), activation=None
    )(h)

    return Model(input_layer, h, name='encoder')


def create_decoder(
    input_shape,
    latent_dim,
    filters,
    kernels,
    conv_activation=None,
    dense_activation=None,
):

    input_layer = Input(shape=(latent_dim,))
    h = PReLU()(input_layer)
    h = Dense(tfp.layers.MultivariateNormalTriL.params_size(32))(h)
    h = PReLU()(h)
    w = int(np.ceil(input_shape[0] / 2 ** (len(filters))))
    h = Dense(w * w * filters[-1], activation=dense_activation)(tf.cast(h, tf.float32))
    h = PReLU()(h)
    h = Reshape((w, w, filters[-1]))(h)
    for i in range(len(filters) - 1, -1, -1):
        h = Conv2DTranspose(
            filters[i],
            (kernels[i], kernels[i]),
            activation=conv_activation,
            padding="same",
            strides=(2, 2),
        )(h)
        h = PReLU()(h)
        h = Conv2DTranspose(
            filters[i],
            (kernels[i], kernels[i]),
            activation=conv_activation,
            padding="same",
        )(h)
        h = PReLU()(h)
        
    h = Conv2D(input_shape[-1]*2, (3, 3), activation="relu", padding="same")(h)

    # In case the last convolutional layer does not provide an image of the size of the input image, cropp it.
    cropping = int(h.get_shape()[1] - input_shape[0])
    if cropping > 0:
        print("in cropping")
        if cropping % 2 == 0:
            h = Cropping2D(cropping / 2)(h)
        else:
            h = Cropping2D(
                ((cropping // 2, cropping // 2 + 1), (cropping // 2, cropping // 2 + 1))
            )(h)

    # Build the encoder only
    h = tfp.layers.DistributionLambda(make_distribution_fn=lambda t: tfd.Normal(loc=t[...,:input_shape[-1]], scale=1e-4 +t[...,input_shape[-1]:])
                                          ,convert_to_tensor_fn=tfp.distributions.Distribution.sample)(h)
    
    
    decoder = Model(input_layer,h, name='decoder')
    return Model(input_layer,h, name='decoder')

def init_permutation_once(x, name):
    return tf.Variable(name=name, initial_value=x, trainable=False)

def create_flow(latent_dim=32, num_nf_layers=5):

    my_bijects = []
    zdist = tfd.MultivariateNormalDiag(tf.zeros(latent_dim))
    # zdist = tfd.Independent(tfd.Normal(loc=tf.zeros(latent_dim), scale=1), reinterpreted_batch_ndims=1)

    # loop over desired bijectors and put into list

    permute_arr = np.arange(latent_dim)[::-1]

    for i in range(num_nf_layers):
        # Syntax to make a MAF
        anet = tfb.AutoregressiveNetwork(
            params=2, hidden_units=[256, 256], activation="relu"
        )
        ab = tfb.MaskedAutoregressiveFlow(anet)
        # Add bijector to list
        my_bijects.append(ab)
        # Now permuate (!important!)
        permute = tfb.Permute(permute_arr)
        my_bijects.append(permute)
        #TODO: include batchnorm?
    # put all bijectors into one "chain bijector"
    # that looks like one
    bijector_chain = tfb.Chain(my_bijects)
    # make transformed dist
    td = tfd.TransformedDistribution(zdist, bijector=bijector_chain)

    input_layer = Input(shape=(latent_dim,))
    return Model(input_layer, td.log_prob(input_layer), name='flow')

# Function to define model

def create_model_fvae(
    input_shape,
    latent_dim,
    filters,
    kernels,
    conv_activation=None,
    dense_activation=None,
    linear_norm=False,
):
    """
    Create the VAE model
    parameters:
        input_shape: shape of input tensor
        latent_dim: size of the latent space
        filters: filters used for the convolutional layers
        kernels: kernels used for the convolutional layers
    """
    
    encoder = create_encoder(
    input_shape,
    latent_dim,
    filters,
    kernels,
    conv_activation=None,
    dense_activation=None,
)
    
    decoder = create_decoder(
    input_shape,
    latent_dim,
    filters,
    kernels,
    conv_activation=None,
    dense_activation=None,
)

    flow = create_flow(latent_dim=latent_dim, num_nf_layers=5)

    # Define the prior for the latent space
    prior = tfd.Independent(
        tfd.Normal(loc=tf.zeros(latent_dim), scale=1), reinterpreted_batch_ndims=1
    )

    # Build the model
    x_input = Input(shape=(input_shape))
    z = tfp.layers.MultivariateNormalTriL(
        latent_dim,
        activity_regularizer=tfp.layers.KLDivergenceRegularizer(prior, weight=0.01),
    )(encoder(x_input))
    
    net = Model(inputs = x_input, outputs=[decoder(z), flow(z.sample()), z])

    return net, encoder, decoder, flow
