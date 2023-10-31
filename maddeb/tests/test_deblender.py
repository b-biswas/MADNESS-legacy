"""Test Deblending."""
import numpy as np

from maddeb.Deblender import Deblend


def test_deblending():
    """Test deblending."""
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
        data.copy(),
        detected_pos,
        num_components=len(detected_pos),  # redundant parameter
        use_log_prob=True,
        linear_norm_coeff=1,
        max_iter=2,
        use_debvader=True,
        compute_sig_dynamically=False,
        map_solution=True,
        channel_last=True,
    )

    deb(
        np.transpose(data.copy(), axes=[2, 0, 1]),
        detected_pos,
        num_components=len(detected_pos),  # redundant parameter
        use_log_prob=True,
        linear_norm_coeff=1,
        max_iter=2,
        use_debvader=True,
        compute_sig_dynamically=False,
        map_solution=True,
        channel_last=False,
        use_scatter_and_sub=False,
    )

    residual1 = deb.compute_residual(data, use_scatter_and_sub=True).numpy()

    padding_infos = deb.get_padding_infos()
    residual2 = deb.compute_residual(
        data, use_scatter_and_sub=False, padding_infos=padding_infos
    ).numpy()

    np.testing.assert_array_equal(residual1, residual2)

    def test_scatter_and_sub():
        """Test scatter and sub."""
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

        detected_pos = [[10, 10]]

        deb(
            data.copy(),
            detected_pos,
            num_components=len(detected_pos),  # redundant parameter
            use_log_prob=True,
            linear_norm_coeff=1,
            max_iter=2,
            use_debvader=True,
            compute_sig_dynamically=False,
            map_solution=True,
            channel_last=True,
        )

        residual1 = deb.compute_residual(data, use_scatter_and_sub=True).numpy()

        padding_infos = deb.get_padding_infos()
        residual2 = deb.compute_residual(
            data, use_scatter_and_sub=False, padding_infos=padding_infos
        ).numpy()

        np.testing.assert_array_equal(residual1, residual2)
