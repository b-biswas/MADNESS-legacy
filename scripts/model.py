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
    """
    function to create the encoder

    Parameters
    ----------
    input_shape: list
        shape of input tensor
    latent_dim: int
        size of the latent space
    filters: list
        filters used for the convolutional layers
    kernels: list
        kernels used for the convolutional layers
    conv_activation: str
        activation for conv layers
    dense_activation: str
        activation for dense layers

    Returns
    -------
    encoder: tf.keras.Model
       model that takes as input the image of a galaxy and projects it to the latent space
    """

    # Input layer
    input_layer = Input(shape=(input_shape))

    # Define the model
    h = BatchNormalization(name='batchnorm1')(input_layer)
    for i in range(len(filters)):
        h = Conv2D(
            filters[i],
            (kernels[i], kernels[i]),
            activation=None,
            padding="same",
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
    h = Dense(1024)(h)
    h = PReLU()(h)
    h = Dense(
        tfp.layers.MultivariateNormalTriL.params_size(latent_dim),
        activation=None,
    )(h)

    return Model(input_layer, h, name="encoder")


def create_decoder(
    input_shape,
    latent_dim,
    filters,
    kernels,
    conv_activation=None,
    dense_activation=None,
):
    """
    function to create the decoder 

    Parameters
    ----------
    input_shape: list
        shape of input tensor
    latent_dim: int
        size of the latent space
    filters: list
        filters used for the convolutional layers
    kernels: list
        kernels used for the convolutional layers
    conv_activation: str
        activation for conv layers
    dense_activation: str
        activation for dense layers

    Returns
    -------
    decoder: tf.keras.Model
        model that takes as input a point in the latent space and decodes it to reconstruct a noiseless galaxy.
    """

    input_layer = Input(shape=(latent_dim,))
    h = Dense(256)(input_layer)
    h = PReLU()(h)
    w = int(np.ceil(input_shape[0] / 2 ** (len(filters))))
    h = Dense(w * w * filters[-1], activation=None)(tf.cast(h, tf.float32))
    h = PReLU()(h)
    h = Reshape((w, w, filters[-1]))(h)
    for i in range(len(filters) - 1, -1, -1):
        h = Conv2DTranspose(
            filters[i],
            (kernels[i], kernels[i]),
            activation=None,
            padding="same",
            strides=(2, 2),
        )(h)
        h = PReLU()(h)
        h = Conv2DTranspose(
            filters[i],
            (kernels[i], kernels[i]),
            activation=None,
            padding="same",
        )(h)
        h = PReLU()(h)

    # keep the output of the last layer as relu as we want only positive flux values.
    h = Conv2D(input_shape[-1] * 2, (3, 3), activation="relu", padding="same")(h)

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
    h = tfp.layers.DistributionLambda(
        make_distribution_fn=lambda t: tfd.Normal(
            loc=t[..., : input_shape[-1]], scale=1e-6 + t[..., input_shape[-1] :]
        ),
        convert_to_tensor_fn=tfp.distributions.Distribution.sample,
    )(h)

    return Model(input_layer, h, name="decoder")

def make_MAF(hidden_units=[16, 16], activation='relu'):
    MADE = tfb.AutoregressiveNetwork(
        params=2,
        hidden_units=hidden_units,
        activation=activation)
    return tfb.MaskedAutoregressiveFlow(shift_and_log_scale_fn=MADE)

class Flow(tf.keras.layers.Layer):
    def __init__(self, latent_dim, num_nf_layers):
        super(Flow, self).__init__(name="flow")
        self.latent_dim = latent_dim
        self.num_nf_layers = num_nf_layers

    def build(self, input_shape):
        bijectors = []
        permute_arr = np.arange(self.latent_dim)[::-1]
        for i in range(self.num_nf_layers):
            masked_auto_i = make_MAF(hidden_units=[64, 64],
                                    activation='sigmoid')
            bijectors.append(masked_auto_i)
            bijectors.append(tfb.Permute(permutation=permute_arr))
            #bijectors.append(tfb.BatchNormalization())
        
        flow_bijectors = tfb.Chain(list(reversed(bijectors[:-1])))
        base_dist = tfd.Sample(tfd.Normal(loc=0, scale=1), self.latent_dim)
        self.trainable_dist = tfd.TransformedDistribution(
            distribution=base_dist,
            bijector=flow_bijectors)

    def call(self, inputs):
        return self.trainable_dist.log_prob(inputs)

# Function to define model
def create_model_fvae(
    input_shape,
    latent_dim,
    filters,
    kernels,
    conv_activation=None,
    dense_activation=None,
    num_nf_layers=6,
):
    """
    Create the sinmultaneously create the VAE and the flow model.

    Parameters
    ----------
    input_shape: list
        shape of input tensor
    latent_dim: int
        size of the latent space
    filters: list
        filters used for the convolutional layers
    kernels: list
        kernels used for the convolutional layers
    conv_activation: str
        activation for conv layers
    dense_activation: str
        activation for dense layers
    num_nf_layers: int
        number of layers in the flow network

    Returns
    -------
    vae_model: tf.keras.Model
        vae model which consists of the encoder and decoder. 
    flow_model: tf.keras.Model
        flow model which consists of the encoder and flow transormation layers
    encoder: tf.keras.Model
        encoder which is common to both the vae_model and the flow_model
        model that takes as input the image of a galaxy and projects it to the latent space
    decoder: tf.keras.Model
        decoder which is present in the vae_model
        model that takes as input a point in the latent space and decodes it to reconstruct a noiseless galaxy.
    flow: tf.keras.Model
        flow network which is present in the flow_model
        model that takes as input a point in the latent sapce and returns the log_prob wrt the base distribution
    """

    # create the encoder
    encoder = create_encoder(
    input_shape,
    latent_dim,
    filters,
    kernels,
    conv_activation=None,
    dense_activation=None,
)
    
    # create the decoder
    decoder = create_decoder(
    input_shape,
    latent_dim,
    filters,
    kernels,
    conv_activation=None,
    dense_activation=None,
)

    # create the flow transformation
    flow_layer = Flow(latent_dim=latent_dim, num_nf_layers=num_nf_layers)
    _ = flow_layer(np.zeros((1, latent_dim)))
    # Define the prior for the latent space
    prior = tfd.Independent(
        tfd.Normal(loc=tf.zeros(latent_dim), scale=1), reinterpreted_batch_ndims=1)

    # Build the model
    x_input = Input(shape=(input_shape))
    z = tfp.layers.MultivariateNormalTriL(
        latent_dim,
        activity_regularizer=tfp.layers.KLDivergenceRegularizer(prior, weight=0.01),
    )(encoder(x_input))

    vae_model = Model(inputs=x_input, outputs=[decoder(z), z])
    flow_model = Model(inputs=x_input, outputs=flow_layer(z)) # without sample I get the following error: AttributeError: 'MultivariateNormalTriL' object has no attribute 'graph'

    return vae_model, flow_model, encoder, decoder, flow_layer
