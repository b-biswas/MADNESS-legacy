import logging
import time

import btk
import galsim
import matplotlib.pyplot as plt
import numpy as np
import scarlet
import scarlet.psf
import seaborn as sns
import sep
import tensorflow as tf
import tensorflow_probability as tfp

from maddeb.Deblender import Deblend
from maddeb.extraction import extract_cutouts

# logging level set to INFO
logging.basicConfig(format="%(message)s", level=logging.INFO)

LOG = logging.getLogger(__name__)


COSMOS_CATALOG_PATHS = [
    "/sps/lsst/users/bbiswas/COSMOS_catalog/COSMOS_25.2_training_sample/real_galaxy_catalog_25.2.fits",
    "/sps/lsst/users/bbiswas/COSMOS_catalog/COSMOS_25.2_training_sample/real_galaxy_catalog_25.2_fits.fits",
]


stamp_size = 100.2
max_number = 150
batch_size = 1
max_shift = 44
catalog = btk.catalog.CosmosCatalog.from_file(COSMOS_CATALOG_PATHS)
survey = btk.survey.get_surveys("LSST")
seed = 3

galsim_catalog = galsim.COSMOSCatalog(
    COSMOS_CATALOG_PATHS[0], exclusion_level="marginal"
)

sampling_function = btk.sampling_functions.DefaultSampling(
    max_number=max_number, maxshift=max_shift, stamp_size=stamp_size, seed=seed
)

draw_generator = btk.draw_blends.CosmosGenerator(
    catalog,
    sampling_function,
    survey,
    batch_size=batch_size,
    stamp_size=stamp_size,
    cpus=1,
    add_noise="all",
    verbose=False,
    gal_type="parametric",
    seed=seed,
)

blend = next(draw_generator)

field_images = blend["blend_images"]
isolated_images = blend["isolated_images"]


def compute_pixel_covariance_and_flux(predicted_galaxy, simulated_galaxy, field_image):
    ground_truth_pixels = []
    predicted_pixels = []
    sig = []

    actual_flux = []
    predicted_flux = []

    for band_number in range(len(bands)):
        sig.append(sep.Background(field_image[band_number]).globalrms)
        # print(sig)
        # print(sig[band_number])
        mask1 = simulated_galaxy[band_number] > 0 * sig[band_number]
        mask2 = predicted_galaxy[band_number] > 0 * sig[band_number]
        mask = np.logical_or(mask1, mask2)
        #             fig, ax = plt.subplots(1, 2)
        #             plt.subplot(1,2,1)
        #             plt.imshow(cutout_galaxy[:, :, band_number])
        #             plt.subplot(1, 2, 2)
        #             plt.imshow(madness_predictions[blend_number][galaxy_number][band_number])
        #             plt.show()
        ground_truth_pixels.extend(simulated_galaxy[band_number][mask])
        predicted_pixels.extend(predicted_galaxy[band_number][mask])

        actual_flux.append(np.sum(simulated_galaxy[band_number][mask1]))
        predicted_flux.append(np.sum(predicted_galaxy[band_number][mask2]))

    pixel_covariance = np.sum(np.multiply(predicted_pixels, ground_truth_pixels)) / (
        np.sqrt(np.sum(np.square(predicted_pixels)))
        * np.sqrt(np.sum(np.square(ground_truth_pixels)))
    )

    return pixel_covariance, actual_flux, predicted_flux


psf = np.array(
    [
        p.drawImage(
            galsim.Image(field_images[0].shape[1], field_images[0].shape[2]),
            scale=survey.pixel_scale.to_value("arcsec"),
        ).array
        for p in blend["psf"]
    ]
)
bands = [f for f in survey._filters]
wcs = blend["wcs"]


def predict_with_scarlet(image, x_pos, y_pos, show_scene, show_sources, filters):
    sig = []
    weights = np.ones_like(image)
    for i in range(6):
        sig.append(sep.Background(image[i]).globalrms)
        weights[i] = weights[i] / (sig[i] ** 2)
    observation = scarlet.Observation(
        image, psf=scarlet.psf.ImagePSF(psf), weights=weights, channels=bands, wcs=wcs
    )

    model_psf = scarlet.GaussianPSF(sigma=(0.8,) * len(filters))
    model_frame = scarlet.Frame(image.shape, psf=model_psf, channels=filters, wcs=wcs)

    observation = observation.match(model_frame)
    sources = []
    for i in range(len(x_pos)):
        result = scarlet.ExtendedSource(
            model_frame,
            model_frame.get_sky_coord((x_pos[i], y_pos[i])),
            observation,
            thresh=1,
            shifting=True,
        )
        sources.append(result)

    scarlet.initialization.set_spectra_to_match(sources, observation)

    t0 = time.time()
    scarlet_blend = scarlet.Blend(sources, observation)

    scarlet_blend.fit(200, e_rel=1e-5)
    t1 = time.time()

    LOG.info("SCARLET TIME: " + str(t1 - t0))
    # print(f"scarlet ran for {it} iterations to logL = {logL}")
    # scarlet.display.show_likelihood(scarlet_blend)
    # plt.show()

    if show_scene:
        scarlet.display.show_scene(
            sources,
            norm=None,
            observation=observation,
            show_rendered=True,
            show_observed=True,
            show_residual=True,
        )

        plt.show()

    if show_sources:
        scarlet.display.show_sources(
            sources,
            norm=None,
            observation=observation,
            show_rendered=True,
            show_observed=True,
            add_boxes=True,
        )
        plt.show()

    predicted_sources = []
    for src in sources:
        predicted_sources.append(observation.render(src.get_model(frame=model_frame)))
    # print(np.shape(src.get_model(frame=model_frame)))
    return predicted_sources


x_pos = blend["blend_list"][0]["y_peak"]
y_pos = blend["blend_list"][0]["x_peak"]

scarlet_predictions = []
for i, image in enumerate(field_images):
    image = field_images[i]
    x_pos = blend["blend_list"][i]["y_peak"]
    y_pos = blend["blend_list"][i]["x_peak"]
    predicted_sources = predict_with_scarlet(
        image,
        x_pos=x_pos,
        y_pos=y_pos,
        show_scene=False,
        show_sources=False,
        filters=bands,
    )
    scarlet_predictions.append(predicted_sources)


madness_predictions = []
for i in range(len(blend["blend_list"])):
    blends = blend["blend_list"][i]
    # print(blends)
    detected_positions = []
    for j in range(len(blends)):
        detected_positions.append([blends["y_peak"][j], blends["x_peak"][j]])

    deb = Deblend(
        field_images[i],
        detected_positions,
        latent_dim=8,
        num_components=len(blends),
        use_likelihood=True,
        linear_norm_coeff=80000,
        max_iter=300,
    )
    # tf.config.run_functions_eagerly(False)
    convergence_criterion = tfp.optimizer.convergence_criteria.LossNotDecreasing(
        atol=0.00001 * 45 * 45 * len(blends) * 3, min_num_steps=50, window_size=20
    )
    # convergence_criterion = None
    lr_scheduler = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=0.08, decay_steps=15, decay_rate=0.8, staircase=True
    )
    optimizer = tf.keras.optimizers.RMSprop(learning_rate=lr_scheduler)

    deb(
        convergence_criterion,
        optimizer=optimizer,
        use_debvader=True,
        compute_sig_dynamically=False,
    )
    madness_predictions.append(deb.get_components())

    cov_madness = []
madness_actual_flux = []
madness_predicted_flux = []
for blend_number in range(len(field_images)):
    blends_meta_data = blend["blend_list"][blend_number]
    # print(blends)

    for galaxy_number in range(len(blends_meta_data)):
        detected_position = [
            [
                blends_meta_data["y_peak"][galaxy_number],
                blends_meta_data["x_peak"][galaxy_number],
            ]
        ]
        distances_to_center = list(
            np.array(detected_position) - int((np.shape(field_images[0])[1] - 1) / 2)
        )
        cutout_galaxy, idx = extract_cutouts(
            isolated_images[blend_number][galaxy_number],
            distances_to_center,
            cutout_size=45,
        )
        # print(idx)
        if idx == []:
            continue
        cutout_galaxy = cutout_galaxy[0]
        cutout_galaxy = np.transpose(cutout_galaxy, axes=(2, 0, 1))
        # print(np.shape(cutout_galaxy))
        ground_truth_pixels = []
        predicted_pixels = []
        sig = []
        #        fig, ax = plt.subplots(1, 2)
        #         plt.subplot(1,2,1)
        #         plt.imshow(cutout_galaxy[2])
        #         plt.subplot(1, 2, 2)
        #         plt.imshow(madness_predictions[blend_number][galaxy_number][2])
        #         plt.show()
        cov, actual, predicted = compute_pixel_covariance_and_flux(
            madness_predictions[blend_number][galaxy_number],
            cutout_galaxy,
            field_images[0],
        )

        cov_madness.append(cov)
        madness_actual_flux.append(actual)
        madness_predicted_flux.append(predicted)

scarlet_cov = []
scarlet_actual_flux = []
scarlet_predicted_flux = []

for blend_number in range(len(field_images)):

    for galaxy_number in range(len(blend["blend_list"][blend_number])):

        ground_truth_pixels = []
        predicted_pixels = []
        sig = []

        current_galaxy = isolated_images[blend_number][galaxy_number]
        cov, actual, predicted = compute_pixel_covariance_and_flux(
            scarlet_predictions[blend_number][galaxy_number],
            current_galaxy,
            field_images[0],
        )

        scarlet_cov.append(cov)
        scarlet_actual_flux.append(actual)
        scarlet_predicted_flux.append(predicted)

scarlet_actual_flux = np.array(scarlet_actual_flux)
scarlet_predicted_flux = np.array(scarlet_predicted_flux)

scarlet_relative_difference = np.abs(
    np.divide(scarlet_predicted_flux - scarlet_actual_flux, scarlet_actual_flux)
)

madness_actual_flux = np.array(madness_actual_flux)
madness_predicted_flux = np.array(madness_predicted_flux)

madness_relative_difference = np.abs(
    np.divide(madness_predicted_flux - madness_actual_flux, madness_actual_flux)
)

# print(madness_relative_difference[np.logical_not(np.isinf(madness_relative_difference))].reshape(-1))
sns.set_theme(
    style={
        "axes.grid": True,
        "axes.labelcolor": "white",
        "figure.facecolor": ".15",
        "xtick.color": "white",
        "ytick.color": "white",
        "text.color": "black",
        "image.cmap": "viridis",
    }
)
plt.figure(figsize=(10, 7))
bins = plt.hist(
    madness_relative_difference[
        np.logical_not(np.isnan(madness_relative_difference))
    ].reshape(-1),
    bins=50,
    alpha=0.7,
    label="MADNESS",
)
plt.hist(
    scarlet_relative_difference[
        np.logical_not(np.isnan(scarlet_relative_difference))
    ].reshape(-1),
    bins=bins[1],
    alpha=0.5,
    label="scarlet",
)
plt.legend(fontsize=20)
ax = plt.gca()
plt.xlabel("relative flux reconstruction error", fontsize=20)
ax.tick_params(labelsize=15)
plt.ylabel("number of galaxies", fontsize=20)

plt.savefig("flux_err")


bkgrms = sep.Background(blend["blend_images"][0][2]).globalrms

np.shape(scarlet_predictions[0])

actual_gal_fluxes = []
actual_gal_fluxerrs = []
actual_gal_flags = []

scarlet_gal_fluxes = []
scarlet_gal_fluxerrs = []
scarlet_gal_flags = []

madness_gal_fluxes = []
madness_gal_fluxerrs = []
madness_gal_flags = []


actual_residual_field = blend["blend_images"][0]
scarlet_residual_field = blend["blend_images"][0]

for i in range(len(scarlet_predictions[0])):
    actual_residual_field = actual_residual_field - blend["isolated_images"][0][i]
    scarlet_residual_field = scarlet_residual_field - scarlet_predictions[0][i]

padding_infos = deb.get_padding_infos()
madness_residual_field = deb.compute_residual(
    blend["blend_images"][0], use_scatter_and_sub=True
).numpy()

for i in range(len(scarlet_predictions[0])):

    # actual galaxy
    for band in range(6):

        bkgrms = sep.Background(blend["blend_images"][0][band]).globalrms

        actual_galaxy = actual_residual_field + blend["isolated_images"][0][i]
        actual_galaxy = actual_galaxy[band].copy(order="C")
        flux, fluxerr, flag = sep.sum_circle(
            actual_galaxy,
            [blend["blend_list"][0]["x_peak"][i]],
            [blend["blend_list"][0]["y_peak"][i]],
            3.0,
            err=bkgrms,
        )
        print(flux)
        actual_gal_fluxes.extend(flux)
        actual_gal_fluxerrs.extend(fluxerr)
        actual_gal_flags.extend(flag)

        # scarlet galaxy
        scarlet_galaxy = scarlet_residual_field + scarlet_predictions[0][i]
        scarlet_galaxy = scarlet_galaxy[band].copy(order="C")
        # plt.imshow(scarlet_galaxy)
        flux, fluxerr, flag = sep.sum_circle(
            scarlet_galaxy,
            [blend["blend_list"][0]["x_peak"][i]],
            [blend["blend_list"][0]["y_peak"][i]],
            3.0,
            err=bkgrms,
        )
        print(flux)
        scarlet_gal_fluxes.extend(flux)
        scarlet_gal_fluxerrs.extend(fluxerr)
        scarlet_gal_flags.extend(flag)

        # madness galaxy
        madness_galaxy = madness_residual_field + np.pad(
            deb.components[i], padding_infos[i]
        )
        madness_galaxy = madness_galaxy[:, :, band].copy(order="C")
        flux, fluxerr, flag = sep.sum_circle(
            madness_galaxy,
            [blend["blend_list"][0]["x_peak"][i]],
            [blend["blend_list"][0]["y_peak"][i]],
            3.0,
            err=bkgrms,
        )
        madness_gal_fluxes.extend(flux)
        madness_gal_fluxerrs.extend(fluxerr)
        madness_gal_flags.extend(flag)

madness_gal_fluxes = np.asarray(madness_gal_fluxes)
scarlet_gal_fluxes = np.asarray(scarlet_gal_fluxes)
actual_gal_fluxes = np.asarray(actual_gal_fluxes)

plt.figure(figsize=(10, 7))
bins = np.arange(0, 0.1, 0.001)
plt.hist(
    np.abs((madness_gal_fluxes - actual_gal_fluxes) / actual_gal_fluxes),
    bins=bins,
    alpha=0.5,
    label="MADNESS",
)
plt.hist(
    np.abs((scarlet_gal_fluxes - actual_gal_fluxes) / actual_gal_fluxes),
    bins=bins,
    alpha=0.5,
    label="scarlet",
)

plt.legend(fontsize=20)
print("yes its done")
plt.savefig("aperturephoto")
