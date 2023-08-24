"""Perform Deblending."""

import logging
import os
import time

import numpy as np
import sep
import tensorflow as tf
import tensorflow_probability as tfp

from maddeb.extraction import extract_cutouts
from maddeb.FlowVAEnet import FlowVAEnet
from maddeb.utils import get_data_dir_path

tfd = tfp.distributions

# logging level set to INFO
logging.basicConfig(format="%(message)s", level=logging.INFO)

LOG = logging.getLogger(__name__)


class Deblend:
    """Run the deblender."""

    def __init__(
        self,
        latent_dim=16,
        weights_path=None,

    ):
        """Initialize class variables.

        Parameters
        ----------
        latent_dim: int
            size of latent space.
        """

        self.latent_dim = latent_dim
        self.flow_vae_net = FlowVAEnet(latent_dim=latent_dim)
        
        if weights_path is None:
            data_dir_path = get_data_dir_path()
            weights_path = os.path.join(data_dir_path, "catsim_kl416d")
        self.flow_vae_net.load_flow_weights(
            weights_path=os.path.join(
                weights_path, "flow8/val_loss"
            )
        )
        self.flow_vae_net.load_vae_weights(
            weights_path=os.path.join(
                weights_path, "deblender/val_loss"
            )
        )

        # self.flow_vae_net.vae_model.trainable = False
        # self.flow_vae_net.flow_model.trainable = False

        # self.flow_vae_net.vae_model.summary()

        self.postage_stamp = None
        self.detected_positions = None
        self.cutout_size = None
        self.num_components = None
        self.channel_last = None
        self.noise_sigma = None
        self.num_bands = None
        self.field_size = None
        self.use_log_prob = None
        self.linear_norm_coeff = None

        self.optimizer = None
        self.max_iter = None
        self.z = None

    def __call__(
        self,
        postage_stamp,
        detected_positions,
        noise_sigma=None,
        cutout_size=45,
        num_components=1,
        max_iter=60,
        use_log_prob=True,
        channel_last=False,
        linear_norm_coeff=80000,
        convergence_criterion=None,
        use_debvader=False,
        optimizer=None,
        lr=0.075,
        compute_sig_dynamically=False,
    ):
        """Run the Deblending operation.

        Parameters
        ----------
        postage_stamp: np.ndarray
            input stamp/field that is to be deblended
        detected_positions: list
            List of detected positions.
            as in array and not image
        noise_sigma: list of float
            backgound noise-level in each band
        cutout_size: int
            size of cutouts (in the VAE input/output)
        num_components: int
            number of galaxies present in the image.
        max_iter: int
            number of iterations in the deblending step
        use_log_prob: bool
            decides whether or not to use the log_prob output of the flow deblender in the optimization.
        channel_last: bool
            if channel is the last column of the postage_stamp
        linear_norm_coeff: int/list
            list stores the bandwise linear normalizing/scaling factor.
            if int is passed, same scaling factor is used for all.
        convergence_criterion: tfp.optimizer.convergence_criteria
            For termination of the optimization loop
        use_debvader: bool
            Use encoder as a deblender to set initial position for deblending.
        optimizer: tf.keras.optimizers
            Optimizer ot use used for gradient descent.
        lr: float
            learning rate for optimization.
        compute_sig_dynamically: bool
            to estimate noise level in image. (can be slow)
            Otherwise it uses sep to compute the background noise.

        """
        # tf.config.run_functions_eagerly(False)
        self.linear_norm_coeff = linear_norm_coeff
        self.max_iter = max_iter
        self.num_components = num_components
        self.use_log_prob = use_log_prob
        self.components = None
        self.channel_last = channel_last

        self.noise_sigma = noise_sigma

        if self.channel_last:
            self.postage_stamp = postage_stamp / linear_norm_coeff
        else:
            self.postage_stamp = (
                np.transpose(postage_stamp, axes=[1, 2, 0]) / linear_norm_coeff
            )
        self.detected_positions = detected_positions
        self.cutout_size = cutout_size

        if channel_last:
            self.num_bands = np.shape(postage_stamp)[-1]
            self.field_size = np.shape(postage_stamp)[1]
        else:
            self.num_bands = np.shape(postage_stamp)[0]
            self.field_size = np.shape(postage_stamp)[1]

        self.results = self.gradient_decent(
            convergence_criterion=convergence_criterion,
            use_debvader=use_debvader,
            optimizer=optimizer,
            lr=lr,
            compute_sig_dynamically=compute_sig_dynamically,
        )

    def get_components(self):
        """Return the predicted components.

        The final returned image has same value of channel_last as input image.
        """
        if self.channel_last:
            return self.components.copy()
        return np.transpose(self.components, axes=(0, 3, 1, 2)).copy()

    def compute_residual(
        self,
        postage_stamp,
        reconstructions=None,
        use_scatter_and_sub=False,
        index_pos_to_sub=None,
        padding_infos=None,
    ):
        """Compute residual in a field.

        Parameters
        ----------
        postage_stamp: tf tensor
            field with all the galaxies
        reconstructions: tf tensor
            reconstructions to be subtracted
        use_scatter_and_sub: bool
            to use tf.tensor_scatter_nd_sub or no
        index_pos_to_sub:
            index position for substraction is `use_scatter_and_sub` is True
        padding_infos:
            padding parameters for reconstructions to that they can be subtracted from the field.
            Used when `use_scatter_and_sub` is False.

        Returns
        -------
        residual_field: tf tensor
            residual of the field after subtracting the reconsturctions.

        """
        if reconstructions is None:
            reconstructions = tf.convert_to_tensor(self.components, dtype=tf.float32)
        residual_field = tf.convert_to_tensor(postage_stamp, dtype=tf.float32)

        # residual_field = tf.cast(residual_field, tf.float32)

        if use_scatter_and_sub:

            if index_pos_to_sub is not None:

                def one_step(i, residual_field):
                    indices = index_pos_to_sub[i]
                    reconstruction = reconstructions[i]

                    residual_field = tf.tensor_scatter_nd_sub(
                        residual_field,
                        indices,
                        tf.reshape(
                            reconstruction, [tf.math.reduce_prod(reconstruction.shape)]
                        ),
                    )

                    return i + 1, residual_field

                c = lambda i, *_: i < self.num_components

                _, residual_field = tf.while_loop(
                    c,
                    one_step,
                    (0, residual_field),
                    maximum_iterations=self.num_components,
                )

            else:
                for i in range(self.num_components):
                    detected_position = self.detected_positions[i]

                    starting_pos_x = round(detected_position[0]) - int(
                        (self.cutout_size - 1) / 2
                    )
                    starting_pos_y = round(detected_position[1]) - int(
                        (self.cutout_size - 1) / 2
                    )

                    indices = (
                        np.indices((self.cutout_size, self.cutout_size, self.num_bands))
                        .reshape(3, -1)
                        .T
                    )
                    indices[:, 0] += int(starting_pos_x)
                    indices[:, 1] += int(starting_pos_y)

                    reconstruction = reconstructions[i]

                    residual_field = tf.tensor_scatter_nd_sub(
                        residual_field,
                        indices,
                        tf.reshape(
                            reconstruction, [tf.math.reduce_prod(reconstruction.shape)]
                        ),
                    )

        else:

            def one_step(i, residual_field):
                # padding = tf.cast(padding_infos[i], dtype=tf.int32)
                padding = padding_infos[i]
                reconstruction = tf.pad(
                    tf.gather(reconstructions, i), padding, "CONSTANT", name="padding"
                )
                # tf.where(mask, tf.zeros_like(tensor), tensor)
                residual_field = residual_field - reconstruction
                return i + 1, residual_field

            c = lambda i, _: i < self.num_components

            _, residual_field = tf.while_loop(
                c,
                one_step,
                (tf.constant(0, dtype=tf.int32), residual_field),
                maximum_iterations=self.num_components,
            )
        return residual_field

    def compute_loss(
        self,
        z,
        postage_stamp,
        compute_sig_dynamically,
        sig_sq,
        use_scatter_and_sub,
        index_pos_to_sub,
        padding_infos,
    ):
        """Compute loss at each epoch of Deblending optimization.

        Parameters
        ----------
        z: tf tensor
            latent space representations of the reconstructions.
        postage_stamp: tf tensor
            field with all the galaxies
        compute_sig_dynamically: bool
            to estimate noise level in image. (can be slow)
            Otherwise it uses sep to compute the background noise.
        sig_sq: tf tensor
            Factor for division to convert the MSE to Gaussian approx to poisson noise.
        reconstructions: tf tensor
            reconstructions to be subtracted
        use_scatter_and_sub: bool
            to use tf.tensor_scatter_nd_sub or no
        index_pos_to_sub:
            index position for substraction is `use_scatter_and_sub` is True
        padding_infos:
            padding parameters for reconstructions to that they can be subtracted from the field.
            Used when `use_scatter_and_sub` is False.

        Returns
        -------
        final_loss: tf float
            Final loss to be minimized
        reconstruction_loss: tf float
            Loss form residuals in the field
        log_prob: tf float
            log prob evaluated by normalizing flow
        residual_field: tf tensor
            residual field after deblending.

        """
        reconstructions = self.flow_vae_net.decoder(z)

        residual_field = self.compute_residual(
            postage_stamp,
            reconstructions,
            use_scatter_and_sub=use_scatter_and_sub,
            index_pos_to_sub=index_pos_to_sub,
            padding_infos=padding_infos,
        )

        if compute_sig_dynamically:
            sig_sq = tf.stop_gradient(
                tf.math.reduce_std(residual_field, axis=[0, 1]) ** 2
            )

        reconstruction_loss = residual_field**2 / sig_sq
        # tf.print(sig_sq, output_stream=sys.stdout)

        reconstruction_loss = tf.math.reduce_sum(reconstruction_loss)

        reconstruction_loss = reconstruction_loss / 2

        log_prob = tf.math.reduce_sum(
            self.flow_vae_net.flow(
                tf.reshape(z, (self.num_components, self.latent_dim))
            )
        )

        # tf.print(reconstruction_loss, output_stream=sys.stdout)
        # tf.print(log_likelihood, output_stream=sys.stdout)

        # tf.print(reconstruction_loss)
        # tf.print(log_likelihood)

        final_loss = reconstruction_loss

        if self.use_log_prob:
            final_loss = reconstruction_loss - log_prob

        return final_loss, reconstruction_loss, log_prob, residual_field

    def get_index_pos_to_sub(self):
        """Get index position to run tf.tensor_scatter_nd_sub."""
        index_list = []
        for i in range(self.num_components):
            detected_position = self.detected_positions[i]

            starting_pos_x = round(detected_position[0]) - int(
                (self.cutout_size - 1) / 2
            )
            starting_pos_y = round(detected_position[1]) - int(
                (self.cutout_size - 1) / 2
            )

            indices = (
                np.indices((self.cutout_size, self.cutout_size, self.num_bands))
                .reshape(3, -1)
                .T
            )
            indices[:, 0] += int(starting_pos_x)
            indices[:, 1] += int(starting_pos_y)
            index_list.append(indices)

        return np.array(index_list)

    def get_padding_infos(self):
        """Compute padding info to convert galaxy cutout into field."""
        padding_infos_list = []
        for detected_position in self.detected_positions:

            starting_pos_x = round(detected_position[0]) - int(
                (self.cutout_size - 1) / 2
            )
            starting_pos_y = round(detected_position[1]) - int(
                (self.cutout_size - 1) / 2
            )

            padding = [
                [
                    starting_pos_x,
                    self.field_size - (starting_pos_x + int(self.cutout_size)),
                ],
                [
                    starting_pos_y,
                    self.field_size - (starting_pos_y + int(self.cutout_size)),
                ],
                [0, 0],
            ]

            padding_infos_list.append(padding)
        return np.array(padding_infos_list)

    def run_debvader(self):
        """Feed forward to use encoder as Deblender."""
        m, n, b = np.shape(self.postage_stamp)

        cutouts = extract_cutouts(
            self.postage_stamp,
            pos=self.detected_positions,
            cutout_size=self.cutout_size,
            nb_of_bands=b,
            channel_last=True,
        )
        z = tfp.layers.MultivariateNormalTriL(self.latent_dim)(
            self.flow_vae_net.encoder(cutouts)
        )
        self.components = self.flow_vae_net.decoder(z) * self.linear_norm_coeff

    def compute_noise_sigma(self):
        """Compute noise level with sep."""
        sig = []
        for i in range(6):

            sig.append(sep.Background(self.postage_stamp[:, :, i]).globalrms)
        print(sig)

        return np.array(sig)

    def gradient_decent(
        self,
        initZ=None,
        convergence_criterion=None,
        use_debvader=False,
        optimizer=None,
        lr=0.075,
        compute_sig_dynamically=False,
    ):
        """Perform the gradient descent step to separate components (galaxies).

        Parameters
        ----------
        initZ: np.ndarray
            initial value of the latent space.
        convergence_criterion: tfp.optimizer.convergence_criteria
            For termination of the optimization loop
        use_debvader: bool
            Use encoder as a deblender to set initial position for deblending.
        optimizer: tf.keras.optimizers
            Optimizer ot use used for gradient descent.
        lr: float
            learning rate for optimization.
        compute_sig_dynamically: bool
            to estimate noise level in image. (can be slow)
            Otherwise it uses sep to compute the background noise.

        Returns
        -------
        results: list
            variation of the loss over deblending iterations.

        """
        # X = self.postage_stamp
        # if not self.channel_last:
        #     X = np.transpose(X, axes=(1, 2, 0))

        m, n, b = np.shape(self.postage_stamp)

        if not use_debvader:
            # check constraint parameter over here
            z = tf.Variable(self.flow_vae_net.td.sample(self.num_components))

        else:
            # z = tf.Variable(name="z", initial_value=tf.random_normal_initializer(mean=0, stddev=1)(shape=[self.num_components, self.latent_dim], dtype=tf.float32))
            # use the encoder to find a good starting point.
            cutouts = extract_cutouts(
                self.postage_stamp,
                pos=self.detected_positions,
                cutout_size=self.cutout_size,
                nb_of_bands=b,
                channel_last=True,
            )
            initZ = tfp.layers.MultivariateNormalTriL(self.latent_dim)(
                self.flow_vae_net.encoder(cutouts)
            )
            LOG.info("\n\nUsing encoder for initial point")
            z = tf.Variable(initZ.mean())

        # self.optimizer = tf.keras.optimizers.Adam(lr=lr)

        if optimizer is None:
            if isinstance(lr, tf.keras.optimizers.schedules):
                lr_scheduler = tf.keras.optimizers.schedules.ExponentialDecay(
                    initial_learning_rate=.2, decay_steps=30, decay_rate=0.9, staircase=False,
                )
            optimizer = tf.keras.optimizers.Adam(learning_rate=lr_scheduler)

        LOG.info("\n--- Starting gradient descent in the latent space ---")
        LOG.info("Maximum number of iterations: " + str(self.max_iter))
        # LOG.info("Learning rate: " + str(optimizer.lr.numpy()))
        LOG.info("Number of Galaxies: " + str(self.num_components))
        LOG.info("Dimensions of latent space: " + str(self.latent_dim))

        t0 = time.time()

        index_pos_to_sub = self.get_index_pos_to_sub()
        # index_pos_to_sub = tf.TensorArray(
        #   tf.int32,
        #   size=np.shape(index_pos_to_sub)[0],
        #   clear_after_read=False).unstack(index_pos_to_sub)

        padding_infos = self.get_padding_infos()
        # padding_infos = tf.TensorArray(
        #   tf.int32,
        #   size=np.shape(padding_infos)[0],
        #   clear_after_read=False).unstack(padding_infos)

        def trace_fn(traceable_quantities):
            return {"loss": traceable_quantities.loss}

        if self.noise_sigma is None:
            noise_level = self.compute_noise_sigma()

        # Calculate sigma^2 with gaussian approximation to poisson noise.
        # Note here that self.postage stamp is normalized but it must be divided again
        # to ensure that the loglikelihood does not change due to scaling/normalizing

        sig_sq = self.postage_stamp / self.linear_norm_coeff
        # sig_sq[sig_sq <= (5 * noise_level)] = 0
        sig_sq = tf.convert_to_tensor(
            np.add(sig_sq, np.square(noise_level)),
            dtype=tf.float32,
        )

        # sig_sq = tf.convert_to_tensor(np.square(noise_level), dtype=tf.float32)

        results = tfp.math.minimize(
            loss_fn=self.generate_grad_step_loss(
                z=z,
                postage_stamp=tf.convert_to_tensor(
                    self.postage_stamp, dtype=tf.float32
                ),
                compute_sig_dynamically=tf.convert_to_tensor(compute_sig_dynamically),
                sig_sq=sig_sq,
                use_scatter_and_sub=tf.convert_to_tensor(True),
                index_pos_to_sub=tf.convert_to_tensor(index_pos_to_sub, dtype=tf.int32),
                padding_infos=tf.convert_to_tensor(padding_infos, dtype=tf.float32),
            ),
            trainable_variables=[z],
            num_steps=self.max_iter,
            optimizer=optimizer,
            convergence_criterion=convergence_criterion,
        )

        # for i in range(self.max_iter):
        # print("log prob flow:" + str(log_likelihood.numpy()))
        # print("reconstruction loss"+str(reconstruction_loss.numpy()))
        #    self.gradient_descent_step(z, self.postage_stamp, use_scatter_and_sub=True, index_pos_to_sub=index_pos_to_sub, padding_infos=padding_infos)

        """ LOG.info(f"Final loss {output.objective_value.numpy()}")
        LOG.info("converged "+ str(output.converged.numpy()))
        LOG.info("converged "+ str(output.num_iterations.numpy()))

        z_flatten = output.position
        z = tf.reshape(z_flatten, shape=[self.num_components, self.latent_dim]) """
        # for i in range(self.max_iter):
        # print("log prob flow:" + str(log_likelihood.numpy()))
        # print("reconstruction loss"+str(reconstruction_loss.numpy()))
        #    self.gradient_descent_step(z, self.postage_stamp, use_scatter_and_sub=True, index_pos_to_sub=index_pos_to_sub, padding_infos=padding_infos)

        LOG.info("--- Gradient descent complete ---")
        LOG.info("\nTime taken for gradient descent: " + str(time.time() - t0))

        self.components = self.flow_vae_net.decoder(z) * self.linear_norm_coeff
        self.z = z
        # print(self.components)

        return results

    def generate_grad_step_loss(
        self,
        z,
        postage_stamp,
        compute_sig_dynamically,
        sig_sq,
        use_scatter_and_sub,
        index_pos_to_sub,
        padding_infos,
    ):
        """Return function compute training loss that has no arguments.

        Parameters
        ----------
        z: tf tensor
            latent space representations of the reconstructions.
        postage_stamp: tf tensor
            field with all the galaxies
        compute_sig_dynamically: bool
            to estimate noise level in image. (can be slow)
            Otherwise it uses sep to compute the background noise.
        sig_sq: tf tensor
            Factor for division to convert the MSE to Gaussian approx to poisson noise.
        reconstructions: tf tensor
            reconstructions to be subtracted
        use_scatter_and_sub: bool
            to use tf.tensor_scatter_nd_sub or no
        index_pos_to_sub:
            index position for substraction is `use_scatter_and_sub` is True
        padding_infos:
            padding parameters for reconstructions to that they can be subtracted from the field.
            Used when `use_scatter_and_sub` is False.

        Returns
        -------
        training_loss: python function
            computes loss without taking any arguments.

        """

        @tf.function(autograph=False)
        def training_loss():
            """Compute training loss."""
            loss, *_ = self.compute_loss(
                z=z,
                postage_stamp=postage_stamp,
                compute_sig_dynamically=compute_sig_dynamically,
                sig_sq=sig_sq,
                use_scatter_and_sub=use_scatter_and_sub,
                index_pos_to_sub=index_pos_to_sub,
                padding_infos=padding_infos,
            )
            return loss

        return training_loss
