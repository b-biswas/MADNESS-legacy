import tensorflow as tf
import tensorflow.keras.backend as K
import tensorflow_probability as tfp

from scripts.model import create_model_fvae
from scripts.model import create_encoder

tfd = tfp.distributions
tfb = tfp.bijectors

import logging

logging.basicConfig(format="%(message)s", level=logging.INFO)

LOG = logging.getLogger(__name__)

@tf.function(autograph=False)
def vae_loss_fn(x, predicted_distribution):
    log_prob = predicted_distribution.log_prob(x)

    weight = tf.add(tf.math.sqrt(x), .1)
    loss = tf.math.multiply(log_prob, weight)
    #loss = log_prob

    objective = -tf.math.reduce_mean(tf.math.reduce_sum(loss, axis=[1, 2, 3]))

    return objective

@tf.function(autograph=False)
def vae_loss_fn(x, predicted_distribution):
    mean = predicted_distribution.mean()

    diff = tf.subtract(mean, x)
    diff_square = tf.square(diff)
    mse = tf.math.sqrt(tf.math.reduce_mean(diff_square, axis=[1, 2, 3]))

    objective = tf.math.reduce_mean(mse)

    return objective

def deblender_loss_fn(x, predicted_distribution):
    loss = predicted_distribution.log_prob(x)
    objective = -tf.math.reduce_mean(tf.math.reduce_sum(loss, axis=[1, 2, 3]))
    return objective

#@tf.function(autograph=False)
def flow_loss_fn(x, output):
    return -tf.math.reduce_mean(output)


class FlowVAEnet:
    def __init__(
        self,
        input_shape=[45, 45, 6],
        latent_dim=10,
        filters_encoder=[50, 100, 200, 400],
        filters_decoder=[50, 100, 200, 300],
        kernels_encoder=[3, 3, 3, 3],
        kernels_decoder=[3, 3, 3, 3],
        num_nf_layers=6,
    ):
        """
        Creates the required models according to the specifications.

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
        num_nf_layers: int
            number of layers in the flow network
        """

        self.input_shape = input_shape
        self.latent_dim = latent_dim

        self.filters_encoder = filters_encoder
        self.kernels_encoder = kernels_encoder

        self.filters_decoder = filters_decoder
        self.kernels_decoder = kernels_decoder

        self.nb_of_bands = input_shape[2]
        self.num_nf_layers = num_nf_layers

        (
            self.vae_model,
            self.flow_model,
            self.encoder,
            self.decoder,
            self.flow,
            self.td,
        ) = create_model_fvae(
            input_shape=self.input_shape,
            latent_dim=self.latent_dim,
            filters_encoder=self.filters_encoder,
            kernels_encoder=self.kernels_encoder,
            filters_decoder=self.filters_decoder,
            kernels_decoder=self.kernels_decoder,
            num_nf_layers=self.num_nf_layers,
        )

        self.optimizer = None
        self.callbacks = None

    def train_vae(
        self,
        train_generator,
        validation_generator,
        callbacks,
        train_encoder=True,
        train_decoder=True,
        optimizer=tf.keras.optimizers.Adam(1e-4),
        epochs=35,
        verbose=1,
    ):
        """
        trains only the vae model. (both the encoder and the decoder)

        Parameters
        ----------
        train_generator:
            generator to be used for training the network.
            keras.utils.Sequence returning (inputs, targets) or (inputs, targets, sample_weights)
        validation_generator:
            generator to be used for validation
            keras.utils.Sequence returning (inputs, targets) or (inputs, targets, sample_weights)
        callbacks: list
            List of keras.callbacks.Callback instances.
            List of callbacks to apply during training.
            See tf.keras.callbacks
        optimizer: str or tf.keras.optimizers
            String (name of optimizer) or optimizer instance. See tf.keras.optimizers.
        epochs: int
            number of epochs for which the model is going to be trained
        verbose: int
            verbose option for training.
            'auto', 0, 1, or 2. Verbosity mode. 0 = silent, 1 = progress bar, 2 = one line per epoch.
            'auto' defaults to 1 for most cases, but 2 when used with ParameterServerStrategy.
            Note that the progress bar is not particularly useful when logged to a file, so verbose=2 is recommended when not running interactively (eg, in a production environment).
        """
        if (not train_decoder) and (not train_encoder):
            raise ValueError(
                "Training failed because both encoder and decoder are not trainable"
            )

        self.encoder.trainable = train_encoder
        self.decoder.trainable = train_decoder

        if train_encoder is False:
            self.encoder.get_layer("batchnorm1").trainable = False

        self.vae_model.summary()
        LOG.info("\n--- Training only VAE network ---")
        LOG.info("Encoder status: " + str(train_encoder))
        LOG.info("Decoder status: " + str(train_decoder))
        # LOG.info("Initial learning rate: " + str(lr))

        # Custom metric to display the KL divergence during training
        def kl_metric(y_true, y_pred):
            return K.sum(self.vae_model.losses)

        LOG.info("Number of epochs: " + str(epochs))
        self.vae_model.compile(
            optimizer=optimizer,
            loss={"decoder": vae_loss_fn},
            experimental_run_tf_function=True,
            metrics=["mse", kl_metric],
        )
        hist = self.vae_model.fit(
            x=train_generator,
            epochs=epochs,
            verbose=verbose,
            shuffle=True,
            validation_data=validation_generator,
            callbacks=callbacks,
            workers=8,
            use_multiprocessing=True,
        )
        return hist

    def train_flow(
        self,
        train_generator,
        validation_generator,
        callbacks,
        optimizer=tf.keras.optimizers.Adam(1e-4),
        epochs=35,
        verbose=1,
    ):
        """
        Trains only the flow part of the flow_model while keeping the encoder constant.

        Parameters
        ----------
        train_generator:
            generator to be used for training the network.
            keras.utils.Sequence returning (inputs, targets) or (inputs, targets, sample_weights)
        validation_generator:
            generator to be used for validation
            keras.utils.Sequence returning (inputs, targets) or (inputs, targets, sample_weights)
        callbacks: list
            List of keras.callbacks.Callback instances.
            List of callbacks to apply during training.
            See tf.keras.callbacks
        optimizer: str or tf.keras.optimizers
            String (name of optimizer) or optimizer instance. See tf.keras.optimizers.
        epochs: int
            number of epochs for which the model is going to be trained
        num_scheduler_epochs: int
            number of epochs after which learning rate is reduced by a factor of `e`
        verbose: int
            verbose option for training.
            'auto', 0, 1, or 2. Verbosity mode. 0 = silent, 1 = progress bar, 2 = one line per epoch.
            'auto' defaults to 1 for most cases, but 2 when used with ParameterServerStrategy.
            Note that the progress bar is not particularly useful when logged to a file, so verbose=2 is recommended when not running interactively (eg, in a production environment).
        """
        self.flow.trainable = True
        self.encoder.trainable = False
        # TODO: find a better way to fix all batchnorm layers
        self.encoder.get_layer("batchnorm1").trainable = False
        self.flow_model.compile(
            optimizer=optimizer,
            loss={"flow": flow_loss_fn},
            experimental_run_tf_function=True,
        )
        self.flow_model.summary()

        LOG.info("\n--- Training only FLOW network ---")
        # LOG.info("Initial learning rate: " + str(lr))
        LOG.info("Number of epochs: " + str(epochs))

        hist = self.flow_model.fit(
            x=train_generator,
            epochs=epochs,
            verbose=verbose,
            shuffle=True,
            validation_data=validation_generator,
            callbacks=callbacks,
            workers=8,
            use_multiprocessing=True,
        )

        return hist

    def load_vae_weights(self, weights_path, is_folder=True):
        """
        Parameters
        ----------
        weights_path: str
            path to the weights of the vae_model.
            The path must point to either a folder with the saved checkpoints or directly to the checkpoint.
        is_folder: bool
            specifies if the weights_path points to a folder.
            If True, then the latest checkpoint is loaded.
            else, the checkpoint specified in the path is loaded
        """
        if is_folder:
            weights_path = tf.train.latest_checkpoint(weights_path)
        self.vae_model.load_weights(weights_path).expect_partial()

    def load_flow_weights(self, weights_path, is_folder=True):
        """
        Parameters
        ----------
        weights_path: str
            path to the weights of the vae_model.
            The path must point to either a folder with the saved checkpoints or directly to the checkpoint.
        is_folder: bool
            specifies if the weights_path points to a folder.
            If True, then the latest checkpoint is loaded.
            else, the checkpoint specified in the path is loaded
        """
        if is_folder:
            weights_path = tf.train.latest_checkpoint(weights_path)
        self.flow_model.load_weights(weights_path).expect_partial()

    def randomize_encoder(self):
        new_encoder = create_encoder(
        input_shape=self.input_shape, latent_dim=self.latent_dim, filters=self.filters_encoder, kernels=self.kernels_encoder
        )
        self.encoder.set_weights(new_encoder.get_weights())
