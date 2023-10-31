"""Perform Deblending."""

import logging
import os
import time

import galcheat
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


def vectorized_compute_residual(args):
    (
        postage_stamp, #todo: rename this to field
        reconstructions,
        use_scatter_and_sub,
        index_pos_to_sub,
        padding_infos,
        num_components,
    ) = args

    residual_field = compute_residual(
        postage_stamp, #todo: rename this to field
        reconstructions,
        use_scatter_and_sub=use_scatter_and_sub,
        index_pos_to_sub=index_pos_to_sub,
        padding_infos=padding_infos,
        num_components=num_components,
    )

    return residual_field

@tf.function
def compute_residual(
        postage_stamp,
        reconstructions=None,
        use_scatter_and_sub=False,
        index_pos_to_sub=None,
        padding_infos=None,
        num_components=1,
    ):
        """Compute residual in a field.

        Parameters
        ----------
        postage_stamp: tf tensor
            field with all the galaxies
        reconstructions: tf tensor
            reconstructions to be subtracted
        use_scatter_and_sub: bool
            uses tf.scatter_and_sub for substraction instead of padding.
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
        residual_field = tf.convert_to_tensor(postage_stamp, dtype=tf.float32)

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

        c = lambda i, *_: i < num_components

        _, residual_field = tf.while_loop(
            c,
            one_step,
            (0, residual_field),
            maximum_iterations=num_components,
        )

        return residual_field

class Deblend:
    """Run the deblender."""

    def __init__(
        self,
        stamp_shape=45,
        latent_dim=16,
        filters_encoder=[32, 128, 256, 512],
        filters_decoder=[64, 96, 128],
        kernels_encoder=[5, 5, 5, 5],
        kernels_decoder=[5, 5, 5],
        dense_layer_units=512,
        num_nf_layers=6,
        weights_path=None,
        load_weights=True,
        survey=galcheat.get_survey("LSST"),
    ):
        """Initialize class variables.

        Parameters
        ----------
        stamp_shape: int
            size of input postage stamp
        latent_dim: int
            size of latent space.
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
        dense_layer_units: int
            number of units in the dense layer
        weights_path: string
            base path to load weights.
            flow weights are loaded from weights_path/flow6/val_loss
            vae weights are loaded from weights_path/deblender/val_loss
        survey: galcheat.survey object
            galcheat survey object to fetch survey details
        load_weights: bool
            Should be used as True to load pre-trained weights.
            if False, random weights are used(used for testing purposes).

        """
        self.latent_dim = latent_dim
        self.survey = survey
        self.flow_vae_net = FlowVAEnet(
            stamp_shape=stamp_shape,
            latent_dim=latent_dim,
            filters_encoder=filters_encoder,
            kernels_encoder=kernels_encoder,
            filters_decoder=filters_decoder,
            kernels_decoder=kernels_decoder,
            dense_layer_units=dense_layer_units,
            num_nf_layers=num_nf_layers,
        )

        if load_weights:
            if weights_path is None:
                data_dir_path = get_data_dir_path()
                weights_path = os.path.join(data_dir_path, survey.name)
            self.flow_vae_net.load_flow_weights(
                weights_path=os.path.join(weights_path, "flow/val_loss")
            )
            self.flow_vae_net.flow_model.trainable = False

            self.flow_vae_net.load_vae_weights(
                weights_path=os.path.join(weights_path, "vae/val_loss")
            )
            self.flow_vae_net.load_encoder_weights(
                weights_path=os.path.join(weights_path, "deblender/val_loss")
            )
        self.flow_vae_net.vae_model.trainable = False

        # self.flow_vae_net.vae_model.trainable = False
        # self.flow_vae_net.flow_model.trainable = False

        # self.flow_vae_net.vae_model.summary()

        self.postage_stamp = None
        self.detected_positions = None
        self.cutout_size = stamp_shape
        self.num_components = None
        self.channel_last = None
        self.noise_sigma = None
        self.num_bands = len(survey.available_filters)
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
        num_components=1,
        max_iter=60,
        use_log_prob=True,
        channel_last=False,
        linear_norm_coeff=10000,
        convergence_criterion=None,
        use_debvader=True,
        optimizer=None,
        use_scatter_and_sub=True,
        compute_sig_dynamically=False,
        map_solution=True,
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
        use_scatter_and_sub: bool
            uses tf.scatter_and_sub for substraction instead of padding.
        compute_sig_dynamically: bool
            to estimate noise level in image. (can be slow)
            Otherwise it uses sep to compute the background noise.
        map_solution: bool
            To obtain the map solution (MADNESS) or debvader solution.
            Both `map_solution` and `use_debvader` cannot be False at the same time.

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
                np.transpose(postage_stamp, axes=[0, 2, 3, 1]) / linear_norm_coeff
            )

        self.detected_positions = detected_positions
        self.max_number = detected_positions.shape[1]
        self.num_fields = detected_positions.shape[0]

        self.field_size = np.shape(postage_stamp)[1]

        self.results = self.gradient_decent(
            convergence_criterion=convergence_criterion,
            use_debvader=use_debvader,
            optimizer=optimizer,
            use_scatter_and_sub=use_scatter_and_sub,
            compute_sig_dynamically=compute_sig_dynamically,
            map_solution=map_solution,
        )

    def get_components(self):
        """Return the predicted components.

        The final returned image has same value of channel_last as input image.
        """
        if self.channel_last:
            return self.components.copy()
        return np.transpose(self.components, axes=(0, 3, 1, 2)).copy()

    # @tf.function
    

    def compute_loss(
        self,
        z,
        postage_stamp,
        compute_sig_dynamically,
        sig_sq,
        use_scatter_and_sub,
        index_pos_to_sub,
        padding_infos,
        num_components,
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
            uses tf.scatter_and_sub for substraction instead of padding.
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
        
        reconstructions = tf.reshape(reconstructions,[self.num_fields, self.max_number, self.cutout_size, self.cutout_size, self.num_bands])

        residual_field = tf.vectorized_map(
            vectorized_compute_residual,
            elems = (
                postage_stamp, #todo: rename this to field
                reconstructions,
                use_scatter_and_sub,
                index_pos_to_sub,
                padding_infos,
                num_components,
            ),
            # parallel_iterations=5,
            # fn_output_signature=tf.TensorSpec(
            #         postage_stamp.shape[1:], 
            #         dtype=tf.float32, 
            #     ),
        )

        reconstruction_loss = residual_field**2 / sig_sq
        # tf.print(sig_sq, output_stream=sys.stdout)

        reconstruction_loss = tf.math.reduce_sum(reconstruction_loss)

        reconstruction_loss = reconstruction_loss / 2 

        log_prob = tf.math.reduce_sum(
            self.flow_vae_net.flow(tf.reshape(z, (self.num_fields*self.max_number, self.latent_dim)))
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
        for field_num in range(self.num_fields):
            inner_list = []
            for i in range(self.max_number):
                indices = (
                    np.indices((self.cutout_size, self.cutout_size, self.num_bands))
                    .reshape(3, -1)
                    .T
                )
                detected_position = self.detected_positions[field_num][i]

                starting_pos_x = round(detected_position[0]) - int(
                    (self.cutout_size - 1) / 2
                )
                starting_pos_y = round(detected_position[1]) - int(
                    (self.cutout_size - 1) / 2
                )
                indices[:, 0] += int(starting_pos_x)
                indices[:, 1] += int(starting_pos_y)
                inner_list.append(indices)
            index_list.append(inner_list)

        return np.array(index_list)

    def get_padding_infos(self):
        """Compute padding info to convert galaxy cutout into field."""
        padding_infos_list = []
        for field_num in range(self.num_fields):
            inner_list = []
            for detected_position in self.detected_positions[field_num]:
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

                inner_list.append(padding)
            padding_infos_list.append(inner_list)
        return np.array(padding_infos_list)

    def compute_noise_sigma(self):
        """Compute noise level with sep."""
        sig = []
        print(self.postage_stamp.shape)
        for i in range(len(self.survey.available_filters)):
            sig.append(
                sep.Background(
                    np.ascontiguousarray(self.postage_stamp[0][:, :, i])
                ).globalrms
            )

        return np.array(sig)

    def gradient_decent(
        self,
        initZ=None,
        convergence_criterion=None,
        use_debvader=True,
        optimizer=None,
        use_scatter_and_sub=True,
        compute_sig_dynamically=False,
        map_solution=True,
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
        use_scatter_and_sub: bool
            uses tf.scatter_and_sub for substraction instead of padding.
        compute_sig_dynamically: bool
            to estimate noise level in image. (can be slow)
            Otherwise it uses sep to compute the background noise.
        map_solution: bool
            To obtain the map solution or debvader solution.
            Both `map_solution` and `use_debvader` cannot be False at the same time.

        Returns
        -------
        results: list
            variation of the loss over deblending iterations.

        """
        # X = self.postage_stamp
        # if not self.channel_last:
        #     X = np.transpose(X, axes=(1, 2, 0))
        LOG.info("use debvader: " + str(use_debvader))
        LOG.info("MAP solution: " + str(map_solution))
        if not map_solution and not use_debvader:
            raise ValueError(
                "Both use_debvader and map_solution cannot be False at the same time"
            )

        if not use_debvader:
            # check constraint parameter over here
            z = tf.Variable(self.flow_vae_net.td.sample((self.num_fields * self.max_number)))

        else:
            # z = tf.Variable(name="z", initial_value=tf.random_normal_initializer(mean=0, stddev=1)(shape=[self.num_components, self.latent_dim], dtype=tf.float32))
            # use the encoder to find a good starting point.
            cutouts = np.zeros(
                (
                    self.num_fields*self.max_number, 
                    self.cutout_size, 
                    self.cutout_size, 
                    self.num_bands,
                )
            )
            for field_num in range(self.num_fields):
                cutouts[field_num*self.max_number : field_num*self.max_number + self.num_components[field_num]] = extract_cutouts(
                    self.postage_stamp[field_num],
                    pos=self.detected_positions[field_num][:self.num_components[field_num]],
                    cutout_size=self.cutout_size,
                    nb_of_bands=self.num_bands,
                    channel_last=True,
                )[0]
            initZ = tfp.layers.MultivariateNormalTriL(self.latent_dim)(
                self.flow_vae_net.encoder(cutouts)
            )
            LOG.info("\n\nUsing encoder for initial point")
            z = tf.Variable(tf.reshape(initZ.mean(), (self.num_fields * self.max_number, 16)))

        # tf.print(z.shape)

        # self.optimizer = tf.keras.optimizers.Adam(lr=lr)
        if map_solution:
            if optimizer is None:
                lr_scheduler = tf.keras.optimizers.schedules.ExponentialDecay(
                    initial_learning_rate=0.25,
                    decay_steps=20,
                    decay_rate=0.9,
                    staircase=False,
                )
                optimizer = tf.keras.optimizers.Adam(learning_rate=lr_scheduler)

            LOG.info("\n--- Starting gradient descent in the latent space ---")
            LOG.info("Maximum number of iterations: " + str(self.max_iter))
            # LOG.info("Learning rate: " + str(optimizer.lr.numpy()))
            LOG.info("Number of Galaxies: " + str(self.num_components))
            LOG.info("Dimensions of latent space: " + str(self.latent_dim))

            t0 = time.time()

            index_pos_to_sub = self.get_index_pos_to_sub()

            padding_infos = self.get_padding_infos()

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
                sig_sq + noise_level**2,
                dtype=tf.float32,
            )
            # tf.print(sig_sq.shape)
            # sig_sq = tf.convert_to_tensor(np.square(noise_level), dtype=tf.float32)

            results = tfp.math.minimize(
                loss_fn=self.generate_grad_step_loss(
                    z=z,
                    postage_stamp=tf.convert_to_tensor(
                        self.postage_stamp, dtype=tf.float32
                    ),
                    compute_sig_dynamically=tf.convert_to_tensor(
                        [compute_sig_dynamically] * self.num_fields,
                    ),
                    sig_sq=sig_sq,
                    use_scatter_and_sub=tf.convert_to_tensor(
                        [use_scatter_and_sub] * self.num_fields,
                    ),
                    index_pos_to_sub=tf.convert_to_tensor(
                        index_pos_to_sub, dtype=tf.int32
                    ),
                    padding_infos=tf.convert_to_tensor(padding_infos, dtype=tf.int32),
                    num_components=tf.convert_to_tensor(self.num_components, dtype=tf.int32),
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
        else:
            results = None
        self.components = tf.reshape(self.flow_vae_net.decoder(z) * self.linear_norm_coeff, [self.num_fields, self.max_number, self.cutout_size, self.cutout_size, self.num_bands])
        self.z = tf.reshape(z, (self.num_fields, self.max_number, 16))

        # print(self.components)

        return results

    def compute_loss_vectorized(self, arg):

        (
            z,
            postage_stamp,
            compute_sig_dynamically,
            sig_sq,
            use_scatter_and_sub,
            index_pos_to_sub,
            padding_infos,
            num_components,
        ) = arg

        loss, *_ = self.compute_loss(
            z=z,
            postage_stamp=postage_stamp,
            compute_sig_dynamically=compute_sig_dynamically,
            sig_sq=sig_sq,
            use_scatter_and_sub=use_scatter_and_sub,
            index_pos_to_sub=index_pos_to_sub,
            padding_infos=padding_infos,
            num_components=num_components,
        )
        return loss

    def generate_grad_step_loss(
        self,
        z,
        postage_stamp,
        compute_sig_dynamically,
        sig_sq,
        use_scatter_and_sub,
        index_pos_to_sub,
        padding_infos,
        num_components,
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
            uses tf.scatter_and_sub for substraction instead of padding.
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
        
        @tf.function
        def training_loss():
            """Compute training loss."""
            # loss = tf.vectorized_map(
            #     self.compute_loss_vectorized,
            #     (
            #         z,
            #         postage_stamp,
            #         compute_sig_dynamically,
            #         sig_sq,
            #         use_scatter_and_sub,
            #         index_pos_to_sub,
            #         padding_infos,
            #         num_components,
            #     ),
            # )
            loss, *_ = self.compute_loss(
                z=z,
                postage_stamp=postage_stamp,
                compute_sig_dynamically=compute_sig_dynamically,
                sig_sq=sig_sq,
                use_scatter_and_sub=use_scatter_and_sub,
                index_pos_to_sub=index_pos_to_sub,
                padding_infos=padding_infos,
                num_components=num_components,
            )
            # tf.print(loss.shape)
            mean_loss = tf.reduce_mean(loss)

            return mean_loss

        return training_loss
