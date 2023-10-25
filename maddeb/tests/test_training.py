"""Test training."""
import os

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from galcheat.utilities import mean_sky_level

from maddeb.callbacks import changeAlpha, define_callbacks
from maddeb.dataset_generator import batched_CATSIMDataset
from maddeb.FlowVAEnet import FlowVAEnet
from maddeb.losses import deblender_encoder_loss_wrapper, deblender_loss_fn_wrapper
from maddeb.utils import get_data_dir_path

tfd = tfp.distributions

def test_training():
    """Test training."""
    vae_epochs =2 

    kl_prior = tfd.Independent(
        tfd.Normal(loc=tf.zeros(1), scale=1), reinterpreted_batch_ndims=1
    )

    f_net = FlowVAEnet(
        stamp_shape=11,
        latent_dim=4,
        filters_encoder=[1, 1, 1, 1],
        filters_decoder=[1, 1, 1],
        kernels_encoder=[1, 1, 1, 1],
        kernels_decoder=[1, 1, 1],
        dense_layer_units=1,
        num_nf_layers=1,
        kl_prior=kl_prior,
        kl_weight=1,
    )

    data = np.random.rand(20, 11, 11, 6)

    # Keras Callbacks
    data_path = get_data_dir_path()
    ch_alpha = changeAlpha(max_epochs=int(.5 * vae_epochs))

    path_weights = os.path.join(data_path)
    callbacks = define_callbacks(
        os.path.join(path_weights, "test_temp"),
        lr_scheduler_epochs=1,
        patience=1,
    )

    _ = f_net.train_vae(
        (data[:5],data[5:10]),
        (data[10:15],data[15:]),
        callbacks=callbacks + [ch_alpha],
        epochs=int(.5 * vae_epochs),
        train_encoder=True,
        train_decoder=True,
        track_kl=True,
        optimizer=tf.keras.optimizers.Adam(1e-5, clipvalue=0.1),
        loss_function=deblender_loss_fn_wrapper(
            sigma_cutoff=np.array([1]*6),
            use_ssim=True,
            ch_alpha=ch_alpha,
            linear_norm_coeff=1,
        ),
        verbose=2,
        # loss_function=vae_loss_fn_wrapper(sigma=noise_sigma, linear_norm_coeff=linear_norm_coeff),
    )
