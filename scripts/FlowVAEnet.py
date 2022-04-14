import tensorflow as tf
import tensorflow.keras.backend as K
import tensorflow_probability as tfp

from scripts.model import create_model_fvae

tfd = tfp.distributions
tfb = tfp.bijectors

import logging

logging.basicConfig(format="%(message)s", level=logging.INFO)

LOG = logging.getLogger(__name__)


def vae_loss_fn(x, x_decoded_mean):
    return -tf.math.reduce_mean(
        tf.math.reduce_sum(x_decoded_mean.log_prob(x), axis=[1, 2, 3])
    )


def flow_loss_fn(x, output):
    return -tf.math.reduce_mean(output)


class FlowVAEnet:
    def __init__(
        self,
        input_shape=[59, 59, 6],
        latent_dim=32,
        filters=[32, 64, 128, 256],
        kernels=[3, 3, 3, 3],
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

        self.filters = filters
        self.kernels = kernels
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
            filters=self.filters,
            kernels=self.kernels,
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
        self.vae_model.summary()
        LOG.info("\n--- Training only VAE network ---")
        LOG.info("Encoder status: " + str(train_encoder))
        LOG.info("Decoder status: " + str(train_decoder))
        # LOG.info("Initial learning rate: " + str(lr))

        # Custom metric to display the KL divergence during training
        def kl_metric(y_true, y_pred):
            return K.sum(self.vae_model.losses)

        LOG.info("Number of epochs: " + str(epochs))
        terminate_on_nan = [tf.keras.callbacks.TerminateOnNaN()]
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
        optimizer=tf.keras.optimizers.Adam(1e-3),
        epochs=35,
        num_scheduler_epochs=30,
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
            experimental_run_tf_function=False,
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
