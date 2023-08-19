"""Losses to train the ML models."""

import logging

import tensorflow as tf
import tensorflow_probability as tfp

tfd = tfp.distributions
tfb = tfp.bijectors
logging.basicConfig(format="%(message)s", level=logging.INFO)

LOG = logging.getLogger(__name__)


# def vae_loss_fn_wrapper(sigma, linear_norm_coeff):
#     """Return function to compute loss with Gaussian approx to Poisson noise.

#     Parameters
#     ----------
#     sigma: list/array
#         Contains information of the noise in each band.
#     linear_norm_coeff: int/list/array
#         Scaling factor for normalization.
#         If single number is passed, all bands are normalized with same constant.

#     Returns
#     -------
#     vae_loss_func:
#         loss function to compute the VAE loss

#     """

#     @tf.function(experimental_compile=True)
#     def vae_loss_fn(ground_truth, predicted_galaxy):
#         """Compute the gaussian approximation to poisson noise.

#         Parameters
#         ----------
#         ground_truth: array/tensor
#             ground truth of the field.
#         predicted_galaxy:
#             galaxy predicted my the model.

#         Returns
#         -------
#         objective: float
#             objective to be minimized by the minimizer.

#         """
#         # predicted_galaxies = predicted_distribution.mean()

#         weight = tf.add(tf.divide(ground_truth, linear_norm_coeff), tf.square(sigma))
#         mse = tf.square(tf.subtract(predicted_galaxy, ground_truth))
#         loss = tf.math.divide(mse, weight)
#         # loss = log_prob

#         objective = tf.math.reduce_mean(tf.math.reduce_sum(loss, axis=[1, 2, 3]))

#         return objective

#     return vae_loss_fn


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


def deblender_ssim_loss_fn_wrapper(sigma_cutoff, ch_alpha):
    """Input field sigma into ssim loss function.

    Parameters
    ----------
    sigma_cutoff: list
        list of sigma levels (normalized) in the bands.

    Returns
    -------
    deblender_ssim_loss_fn:
        function to compute the loss using SSIM weight.

    """

    @tf.function(experimental_compile=True)
    def deblender_ssim_loss_fn(x, predicted_galaxy):
        """Compute the loss under predicted distribution, weighted by the SSIM.

        Parameters
        ----------
        x: array/tensor
            Galaxy ground truth.
        predicted_galaxy: tf tensor
            pixel wise prediction of the flux.

        Returns
        -------
        objective: float
            objective to be minimized by the minimizer.

        """
        loss = tf.math.reduce_mean(
            tf.reduce_sum(
                (x - predicted_galaxy) ** 2 / sigma_cutoff**2, axis=[1, 2, 3]
            )
        )

        band_normalizer = tf.reduce_max(x, axis=[1, 2], keepdims=True)
        ssim = tf.image.ssim(
            x / band_normalizer,
            predicted_galaxy / band_normalizer,
            max_val=1,
        )
        tf.stop_gradient(ch_alpha.alpha)
        objective = tf.reduce_mean(loss*(1 + ch_alpha.alpha * ssim))

        # weight = tf.math.reduce_max(x, axis= [1, 2])
        # objective = tf.math.reduce_sum(loss, axis=[1, 2])
        # weighted_objective = -tf.math.reduce_mean(tf.divide(objective, weight))

        return objective

    return deblender_ssim_loss_fn


def deblender_loss_fn_wrapper(sigma_cutoff):
    """Input field sigma into deblender loss function.

    Parameters
    ----------
    sigma_cutoff: list
        list of sigma levels (normalized) in the bands.

    Returns
    -------
    deblender_loss_fn:
        function to compute the deblender loss.

    """

    @tf.function(experimental_compile=True)
    def deblender_loss_fn(x, predicted_galaxy):
        """Compute the loss under predicted distribution.

        Parameters
        ----------
        x: array/tensor
            Galaxy ground truth.
        predicted_galaxy: tf tensor
            pixel wise prediction of the flux.

        Returns
        -------
        objective: float
            objective to be minimized by the minimizer.

        """
        # predicted_distribution = tfd.Normal(loc=predicted_galaxy, scale=sigma_cutoff)
        # loss = predicted_distribution.log_prob(x)
        # objective = -tf.math.reduce_mean(tf.reduce_sum(loss, axis=[1, 2, 3]))

        objective = tf.math.reduce_mean(
            tf.reduce_sum(
                (x - predicted_galaxy) ** 2 / sigma_cutoff**2, axis=[1, 2, 3]
            )
        )

        # weight = tf.math.reduce_max(x, axis= [1, 2])
        # objective = tf.math.reduce_sum(loss, axis=[1, 2])
        # weighted_objective = -tf.math.reduce_mean(tf.divide(objective, weight))
        return objective

    return deblender_loss_fn


def vae_loss_fn_wrapper(sigma_cutoff):
    """Input field sigma into deblender loss function.

    Parameters
    ----------
    sigma_cutoff: list
        list of sigma levels (normalized) in the bands.

    Returns
    -------
    deblender_loss_fn:
        function to compute the deblender loss.

    """

    @tf.function(experimental_compile=True)
    def deblender_loss_fn(x, predicted_galaxy):
        """Compute the loss under predicted distribution.

        Parameters
        ----------
        x: array/tensor
            Galaxy ground truth.
        predicted_galaxy: tf tensor
            pixel wise prediction of the flux.

        Returns
        -------
        objective: float
            objective to be minimized by the minimizer.

        """
        predicted_distribution = tfd.Normal(
            loc=predicted_galaxy, scale=sigma_cutoff + tf.math.sqrt(predicted_galaxy)
        )
        loss = predicted_distribution.log_prob(x)
        objective = -tf.math.reduce_mean(tf.reduce_sum(loss, axis=[1, 2, 3]))
        # weight = tf.math.reduce_max(x, axis= [1, 2])
        # objective = tf.math.reduce_sum(loss, axis=[1, 2])
        # weighted_objective = -tf.math.reduce_mean(tf.divide(objective, weight))
        return objective

    return deblender_loss_fn


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

