"""Defines models and training functions for the Deblender."""

import logging

import tensorflow as tf
import tensorflow.keras.backend as K
import tensorflow_probability as tfp

from maddeb.model import create_encoder, create_model_fvae

tfd = tfp.distributions
tfb = tfp.bijectors
logging.basicConfig(format="%(message)s", level=logging.INFO)

LOG = logging.getLogger(__name__)


def vae_loss_fn_wrapper(sigma, linear_norm_coeff):
    """Return function to compute loss with Gaussian approx to Poisson noise.

    Parameters
    ----------
    sigma: list/array
        Contains information of the noise in each band.
    linear_norm_coeff: int/list/array
        Scaling factor for normalization.
        If single number is passed, all bands are normalized with same constant.

    Returns
    -------
    vae_loss_func:
        loss function to compute the VAE loss

    """

    @tf.function(experimental_compile=True)
    def vae_loss_fn(ground_truth, predicted_galaxy):
        """Compute the gaussian approximation to poisson noise.

        Parameters
        ----------
        ground_truth: array/tensor
            ground truth of the field.
        predicted_galaxy:
            galaxy predicted my the model.

        Returns
        -------
        objective: float
            objective to be minimized by the minimizer.

        """
        # predicted_galaxies = predicted_distribution.mean()

        weight = tf.add(tf.divide(ground_truth, linear_norm_coeff), tf.square(sigma))
        mse = tf.square(tf.subtract(predicted_galaxy, ground_truth))
        loss = tf.math.divide(mse, weight)
        # loss = log_prob

        objective = tf.math.reduce_mean(tf.math.reduce_sum(loss, axis=[1, 2, 3]))

        return objective

    return vae_loss_fn


@tf.function(autograph=False)
def vae_loss_fn_mse(x, predicted_distribution):
    """Compute the MSE loss function.

    Parameters
    ----------
    x: array/tensor
        Galaxy ground truth.
    predicted_distribution: tf distribution
        pixel wise distribution of the flux.
        The mean is used to compute the MSE.

    Returns
    -------
    objective: float
        objective to be minimized by the minimizer.

    """
    mean = predicted_distribution.sample()

    weight = tf.sqrt(tf.reduce_max(x, axis=[1, 2, 3]))
    diff = tf.subtract(mean, x)
    pixel_mse = tf.square(diff)
    mse = tf.math.reduce_sum(pixel_mse, axis=[1, 2, 3])

    objective = tf.math.reduce_mean(tf.divide(mse, weight))

    return objective


@tf.function(experimental_compile=True)
def deblender_ssim_loss_fn(x, predicted_distribution):
    """Compute the loss under predicted distribution, weighted by the SSIM.

    Parameters
    ----------
    x: array/tensor
        Galaxy ground truth.
    predicted_distribution: tf distribution
        pixel wise distribution of the flux.
        The mean is used to compute the MSE.

    Returns
    -------
    objective: float
        objective to be minimized by the minimizer.

    """
    loss = predicted_distribution.log_prob(x)

    band_normalizer = tf.reduce_max(x, axis=[1, 2], keepdims=True)
    ssim = tf.image.ssim(
        x / band_normalizer, predicted_distribution.mean() / band_normalizer, max_val=1
    )
    objective = -tf.reduce_mean(tf.reduce_sum(loss, axis=[1, 2, 3]) * ssim)

    # weight = tf.math.reduce_max(x, axis= [1, 2])
    # objective = tf.math.reduce_sum(loss, axis=[1, 2])
    # weighted_objective = -tf.math.reduce_mean(tf.divide(objective, weight))
    return objective


@tf.function(experimental_compile=True)
def deblender_loss_fn(x, predicted_distribution):
    """Compute the loss under predicted distribution.

    Parameters
    ----------
    x: array/tensor
        Galaxy ground truth.
    predicted_distribution: tf distribution
        pixel wise distribution of the flux.
        The mean is used to compute the MSE.

    Returns
    -------
    objective: float
        objective to be minimized by the minimizer.

    """
    loss = predicted_distribution.log_prob(x)
    objective = -tf.math.reduce_mean(tf.reduce_sum(loss, axis=[1, 2, 3]))
    # weight = tf.math.reduce_max(x, axis= [1, 2])
    # objective = tf.math.reduce_sum(loss, axis=[1, 2])
    # weighted_objective = -tf.math.reduce_mean(tf.divide(objective, weight))
    return objective


@tf.function(experimental_compile=True)
def flow_loss_fn(x, output):
    """Compute the loss under predicted distribution.

    Parameters
    ----------
    x: array/tensor
        Galaxy ground truth.
    output: tensor
        log probability output over a batch from Normalizing Flow

    Returns
    -------
    objective: float
        objective to be minimized by the minimizer.

    """
    return -tf.math.reduce_mean(output)


class FlowVAEnet:
    """Initialize and train the Neural Network model."""

    def __init__(
        self,
        input_shape=[45, 45, 6],
        latent_dim=8,
        filters_encoder=[64, 128, 256, 512],
        filters_decoder=[64, 128, 256],
        kernels_encoder=[5, 3, 5, 3],
        kernels_decoder=[5, 3, 5, 3],
        num_nf_layers=8,
        kl_prior=None,
        kl_weight=None,
        decoder_sigma_cutoff=None,
    ):
        """Create the required models according to the specifications.

        Parameters
        ----------
        input_shape: list
            shape of input tensor
        latent_dim: int
            size of the latent space
        filters_encoder: list
            filters used for the convolutional layers in encoder
        filters_decoder: list
            filters used for the convolutional layers in decoder
        kernels_encoder: list
            kernels used for the convolutional layers in encoder
        kernels_decoder: list
            kernels used for the convolutional layers in decoder
        num_nf_layers: int
            number of layers in the flow network
        kl_prior: tf distribution
            KL prior to be applied on the latent space.
        kl_weight: float
            Weight to be multiplied tot he kl_prior
        decoder_sigma_cutoff: list of float
            backgound noise-level in each band

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
            kl_prior=kl_prior,
            kl_weight=kl_weight,
            decoder_sigma_cutoff=decoder_sigma_cutoff,
        )

        self.optimizer = None
        self.callbacks = None

    def train_vae(
        self,
        train_generator,
        validation_generator,
        callbacks,
        loss_function,
        train_encoder=True,
        train_decoder=True,
        optimizer=tf.keras.optimizers.Adam(1e-4),
        track_kl=False,
        epochs=35,
        verbose=1,
    ):
        """Train only the the components of VAE model (encoder and/or decoder).

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
        loss_function: python function
            function that can compute the loss.
        train_encoder: bool
            Flag to decide if encoder is to be trained.
        train_decoder: bool
            Flag to decide if decoder is to be trained.
        optimizer: str or tf.keras.optimizers
            String (name of optimizer) or optimizer instance. See tf.keras.optimizers.
        track_kl:bool
            To decide if the kl loss is to be tracked through the iterations.
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

        metrics = ["mse"]

        # Custom metric to display the KL divergence during training
        def kl_metric(y_true, y_pred):
            return K.sum(self.vae_model.losses)

        if track_kl:
            metrics += [kl_metric]

        LOG.info("Number of epochs: " + str(epochs))

        if loss_function is None:
            print("pass valid loss function")
        self.vae_model.compile(
            optimizer=optimizer,
            loss={"decoder": loss_function},
            experimental_run_tf_function=False,
            metrics=metrics,
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
        """Train only the flow part of the flow_model while keeping the encoder constant.

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
        """Load the trained weights of VAE model (encoder and decoder).

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
        """Load the trained weights encoder and NF (encoder and Flow).

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
        """Randomize the weights of encoder to restart training."""
        new_encoder = create_encoder(
            input_shape=self.input_shape,
            latent_dim=self.latent_dim,
            filters=self.filters_encoder,
            kernels=self.kernels_encoder,
        )
        self.encoder.set_weights(new_encoder.get_weights())
