"""Test Deblending."""
import numpy as np

from maddeb.Deblender import Deblend

def test_deblending():
    deb = Deblend(
        stamp_shape=5,
        latent_dim=4,
        filters_encoder=[1, 1, 1, 1],
        filters_decoder=[1, 1, 1],
        kernels_encoder=[1, 1, 1, 1],
        kernels_decoder=[1, 1, 1],
        dense_layer_units=1,
        num_nf_layers=1,
        load_weights=False,
    )

    data = np.random.rand(15, 15, 6)

    detected_pos = [[9, 10], [11, 11]]

    deb(
        data,
        detected_pos,
        num_components=len(detected_pos),  # redundant parameter
        use_log_prob=True,
        linear_norm_coeff=1,
        max_iter=3,
        use_debvader=True,
        compute_sig_dynamically=False,
        map_solution=True,
        channel_last=True,
    )