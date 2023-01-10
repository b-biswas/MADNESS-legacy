from copyreg import pickle
import os
import logging
import time
import pandas as pd

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
from scipy.optimize import curve_fit
from scipy.stats import norm

from astropy.table import vstack, hstack

from maddeb.Deblender import Deblend
from maddeb.metrics import (
    compute_apperture_photometry,
    compute_pixel_covariance_and_fluxes,
)
from maddeb.utils import get_data_dir_path
import hickle

# logging level set to INFO
logging.basicConfig(format="%(message)s", level=logging.INFO)

LOG = logging.getLogger(__name__)

COSMOS_CATALOG_PATHS = "/sps/lsst/users/bbiswas/OneDegSq_snr_10.fits"

stamp_size = 41
min_number = 12
max_number = 15
batch_size = 20
maxshift = 15
num_repetations = 20
catalog = btk.catalog.CatsimCatalog.from_file(COSMOS_CATALOG_PATHS)
survey = btk.survey.get_surveys("LSST")
seed = 13
run_name = "test_run_catsim"

sampling_function = btk.sampling_functions.DefaultSampling(
    max_number=max_number, min_number=min_number, max_shift=maxshift, stamp_size=stamp_size, seed=seed
)

draw_generator = btk.draw_blends.CatsimGenerator(
    catalog,
    sampling_function,
    survey,
    batch_size=batch_size,
    stamp_size=stamp_size,
    cpus=1,
    add_noise="all",
    verbose=False,
    seed=seed,
)

# Define function to make predictions iwth scarlet
def predict_with_scarlet(image, x_pos, y_pos, show_scene, show_sources, filters):
    sig = []
    weights = np.ones_like(image)
    for i in range(6):
        sig.append(sep.Background(image[i]).globalrms)
        weights[i] = weights[i] / (sig[i] ** 2)
    observation = scarlet.Observation(
        image, psf=scarlet.psf.ImagePSF(psf), weights=weights, channels=bands, wcs=wcs
    )

    model_psf = scarlet.GaussianPSF(sigma=(0.382, .365, .344, .335, .327, .323)) # These numbers are derived from the FWHM given for LSST filters in the galcheat v1.0 repo https://github.com/aboucaud/galcheat/blob/main/galcheat/data/LSST.yaml
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

    scarlet_blend = scarlet.Blend(sources, observation)

    t0 = time.time()
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

for file_num in range(num_repetations):
    print("Processing file " + str(file_num))
    blend = next(draw_generator)

    field_images = blend["blend_images"]
    isolated_images = blend["isolated_images"]

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

    x_pos = blend["blend_list"][0]["y_peak"]
    y_pos = blend["blend_list"][0]["x_peak"]

    # Get Scarlet Predictions
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

    blend["scarlet_predictions"] = scarlet_predictions

    # get MADNESS predictions
    madness_predictions = []
    linear_norm_coeff = [1000, 5000, 10000, 10000, 10000, 10000]

    for field_num in range(len(blend["blend_list"])):

        current_field_predictions = []
        current_blend = blend["blend_list"][field_num]
        # print(blends)
        detected_positions = []
        for j in range(len(current_blend)):
            detected_positions.append([current_blend["y_peak"][j], current_blend["x_peak"][j]])

        # tf.config.run_functions_eagerly(False)
        # convergence_criterion = tfp.optimizer.convergence_criteria.LossNotDecreasing(
        #     atol=0.00001 * 45 * 45 * len(blend) * 3, min_num_steps=100, window_size=20
        # )
        convergence_criterion = tfp.optimizer.convergence_criteria.SuccessiveGradientsAreUncorrelated(min_num_steps=120, window_size=30)
        # convergence_criterion = None
        lr_scheduler = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=1., decay_steps=25, decay_rate=0.9, staircase=True
        )
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr_scheduler)

        deb = Deblend(
            field_images[field_num],
            detected_positions,
            num_components=len(current_blend), #redundant parameter
            latent_dim=16,
            use_likelihood=True,
            linear_norm_coeff=linear_norm_coeff,
            max_iter=500,
        )
        deb(
            convergence_criterion=convergence_criterion,
            optimizer=optimizer,
            use_debvader=True,
            compute_sig_dynamically=False,
        )
        padding_infos = deb.get_padding_infos()
        for component_num in range(deb.num_components):
            prediction = np.pad(deb.components[component_num], padding_infos[component_num])
            prediction = np.transpose(prediction, axes=(2, 0, 1))
            current_field_predictions.append(prediction)
        madness_predictions.append(current_field_predictions)

    blend["madness_predictions"] = madness_predictions

    blend_list = []
    madness_results = []
    scarlet_results = []

    actual_photometry = []
    madness_photometry = []
    scarlet_photometry = []
    blended_photometry = []

    for field_num in range(len(blend["blend_list"])):


        num_galaxies = len(blend["blend_list"][field_num])

        isolated_images = blend["isolated_images"][field_num][0:num_galaxies]

        madness_current_res = compute_pixel_covariance_and_fluxes(
            blend["madness_predictions"][field_num], isolated_images, blend["blend_images"][field_num]
        )
        scarlet_current_res = compute_pixel_covariance_and_fluxes(
            blend["scarlet_predictions"][field_num], isolated_images, blend["blend_images"][field_num]
        )

        size = blend['blend_list'][field_num]['btk_size']

        madness_current_res['size'] = size
        scarlet_current_res['size'] = size

        madness_current_res['field_num'] = [field_num]*num_galaxies
        scarlet_current_res['field_num'] = [field_num]*num_galaxies

        madness_current_res['file_num'] = [file_num]*num_galaxies
        scarlet_current_res['file_num'] = [file_num]*num_galaxies
        #make this a table

        madness_results.append(madness_current_res)
        scarlet_results.append(scarlet_current_res)

        bkg_rms = {}
        for band in range(6):
            bkg_rms[band] = sep.Background(blend["blend_images"][field_num][band]).globalrms


        actual_results_current = compute_apperture_photometry(
            field_image = blend["blend_images"][field_num],
            predictions=blend["isolated_images"][field_num],
            xpos=blend["blend_list"][field_num]["x_peak"],
            ypos=blend["blend_list"][field_num]["y_peak"],
            bkg_rms=bkg_rms,
        )
        actual_results_current["field_num"] = field_num
        actual_results_current["file_num"] = file_num
        actual_photometry.append(actual_results_current)


        madness_results_current = compute_apperture_photometry(
            field_image=blend["blend_images"][field_num],
            predictions=blend["madness_predictions"][field_num],
            xpos=blend["blend_list"][field_num]["x_peak"],
            ypos=blend["blend_list"][field_num]["y_peak"],
            bkg_rms=bkg_rms,
        )
        madness_results_current["field_num"] = field_num
        madness_results_current["file_num"] = file_num
        madness_photometry.append(madness_results_current)

        scarlet_results_current = compute_apperture_photometry(
            field_image=blend["blend_images"][field_num],
            predictions=blend["scarlet_predictions"][field_num],
            xpos=blend["blend_list"][field_num]["x_peak"],
            ypos=blend["blend_list"][field_num]["y_peak"],
            bkg_rms=bkg_rms,
        )
        scarlet_results_current["field_num"] = field_num
        scarlet_results_current["file_num"] = file_num
        scarlet_photometry.append(scarlet_results_current)

        blended_results_current = compute_apperture_photometry(
            field_image=blend["blend_images"][field_num],
            predictions=None,
            xpos=blend["blend_list"][field_num]["x_peak"],
            ypos=blend["blend_list"][field_num]["y_peak"],
            bkg_rms=bkg_rms,
        )
        blended_results_current["field_num"] = field_num
        blended_results_current["file_num"] = file_num
        blended_photometry.append(blended_results_current)

    madness_results = vstack(madness_results)
    scarlet_results = vstack(scarlet_results)

    madness_results = hstack([madness_results, vstack(blend["blend_list"])])
    scarlet_results = hstack([scarlet_results, vstack(blend["blend_list"])])

    actual_photometry = vstack(actual_photometry)
    madness_photometry = vstack(madness_photometry)
    scarlet_photometry = vstack(scarlet_photometry)
    blended_photometry = vstack(blended_photometry)

    # reconstruction_file = open("debblend_results" + str(rep_num)+ ".pkl", "wb")
    save_file_name = os.path.join("/sps/lsst/users/bbiswas", "scarlet_comparison", "debblend_results" + str(file_num) + ".hkl")    
    hickle.dump(blend, save_file_name, mode="w")
    # reconstruction_file.close()

    save_file_name = os.path.join(get_data_dir_path(), "results", run_name,  "scarlet_reconstruction", str(file_num) + ".hkl")
    hickle.dump(scarlet_results, save_file_name, mode="w")

    save_file_name = os.path.join(get_data_dir_path(), "results", run_name, "madness_reconstruction", str(file_num) + ".hkl")
    hickle.dump(madness_results, save_file_name, mode="w")


    save_file_name = os.path.join(get_data_dir_path(), "results", run_name, "scarlet_photometry", str(file_num) + ".hkl")
    hickle.dump(scarlet_photometry, save_file_name, mode="w")

    save_file_name = os.path.join(get_data_dir_path(), "results", run_name, "madness_photometry", str(file_num) +  ".hkl")
    hickle.dump(madness_photometry, save_file_name, mode="w")

    save_file_name = os.path.join(get_data_dir_path(), "results", run_name, "actual_photometry", str(file_num) + ".hkl")
    hickle.dump(actual_photometry, save_file_name, mode="w")

    save_file_name = os.path.join(get_data_dir_path(), "results", run_name, "blended_photometry", str(file_num) + ".hkl")
    hickle.dump(blended_photometry, save_file_name, mode="w")

# # Compute covariance, actual and predicted fluxes
# madness_cov = []
# madness_actual_flux = []
# madness_predicted_flux = []

# scarlet_cov = []
# scarlet_actual_flux = []
# scarlet_predicted_flux = []

# for blend_number in range(len(field_images)):

#     current_galaxies = isolated_images[blend_number]

#     madness_res = compute_pixel_covariance_and_fluxes(
#         madness_predictions[blend_number], current_galaxies, field_images[0]
#     )
#     scarlet_res = compute_pixel_covariance_and_fluxes(
#         scarlet_predictions[blend_number], current_galaxies, field_images[0]
#     )

#     madness_cov.append(madness_res[0])
#     madness_actual_flux.append(madness_res[1])
#     madness_predicted_flux.append(madness_res[2])

#     scarlet_cov.append(scarlet_res[0])
#     scarlet_actual_flux.append(scarlet_res[1])
#     scarlet_predicted_flux.append(scarlet_res[2])

# bins = np.arange(0.95, 1, 0.001)
# plt.hist(scarlet_cov, bins=bins, alpha=0.5, label="scarlet")
# plt.hist(madness_cov, bins=bins, alpha=0.7, label="MADNESS")
# plt.legend()
# plt.xlim([0.95, 1])
# plt.savefig("cov_results")

# madness_actual_flux = np.array(madness_actual_flux)
# madness_predicted_flux = np.array(madness_predicted_flux)

# scarlet_actual_flux = np.array(scarlet_actual_flux)
# scarlet_predicted_flux = np.array(scarlet_predicted_flux)

# scarlet_relative_difference = np.divide(
#     scarlet_predicted_flux - scarlet_actual_flux, scarlet_actual_flux
# )
# madness_relative_difference = np.divide(
#     madness_predicted_flux - madness_actual_flux, madness_actual_flux
# )


# # print(madness_relative_difference[np.logical_not(np.isinf(madness_relative_difference))].reshape(-1))

# # Fit Gaussians
# def gauss(x, sig, mu):
#     return 1 / np.sqrt(2.0 * np.pi) / sig * np.exp(-0.5 * (x - mu) ** 2 / sig**2)


# n_bins = 100
# hist, bin_tmp = np.histogram(madness_relative_difference, n_bins, density=True)
# bins = np.mean((bin_tmp[:-1], bin_tmp[1:]), 0)
# madness_fit = curve_fit(gauss, bins, hist, p0=[np.std(bins), np.mean(bins)])

# hist, bin_tmp = np.histogram(scarlet_relative_difference, n_bins, density=True)
# bins = np.mean((bin_tmp[:-1], bin_tmp[1:]), 0)
# scarlet_fit = curve_fit(gauss, bins, hist, p0=[np.std(bins), np.mean(bins)])


# # Plot relative flux error
# sns.set_theme(
#     style={
#         "axes.grid": True,
#         "axes.labelcolor": "black",
#         "figure.facecolor": "1",
#         "xtick.color": "black",
#         "ytick.color": "black",
#         "text.color": "black",
#         "image.cmap": "viridis",
#     }
# )
# plt.figure(figsize=(10, 7))
# bins = np.arange(-0.5, 0.5, 0.01)
# plt.hist(
#     madness_relative_difference[
#         np.logical_not(np.isnan(madness_relative_difference))
#     ].reshape(-1),
#     bins=bins,
#     density=True,
#     alpha=0.7,
#     color="coral",
#     label="MADNESS",
# )
# plt.plot(bins, norm.pdf(bins, madness_fit[0][1], madness_fit[0][0]), color="coral")
# LOG.info("Madness mu: " + str(madness_fit[0][1]))
# LOG.info("Madness sig: " + str(madness_fit[0][0]))
# plt.hist(
#     scarlet_relative_difference[
#         np.logical_not(np.isnan(scarlet_relative_difference))
#     ].reshape(-1),
#     bins=bins,
#     density=True,
#     alpha=0.5,
#     color="cornflowerblue",
#     label="scarlet",
# )
# plt.plot(
#     bins, norm.pdf(bins, scarlet_fit[0][1], scarlet_fit[0][0]), color="cornflowerblue"
# )
# LOG.info("Scarlet mu: " + str(scarlet_fit[0][1]))
# LOG.info("Scarlet sig: " + str(scarlet_fit[0][0]))
# plt.legend(fontsize=20)
# ax = plt.gca()
# plt.xlabel("relative flux reconstruction error", fontsize=20)
# ax.tick_params(labelsize=15)
# plt.ylabel("number of galaxies", fontsize=20)
# plt.xlim([-0.5, 0.5])
# plt.savefig("flux_err", transparent=True)

# # Compare apperture photometry

# # Compute the residual fields


# # ------------- TODO: check if x and y pos are correct

# actual_residual_field = blend["blend_images"][0]
# scarlet_residual_field = blend["blend_images"][0]

# for i in range(len(scarlet_predictions[0])):
#     actual_residual_field = actual_residual_field - blend["isolated_images"][0][i]
#     scarlet_residual_field = scarlet_residual_field - scarlet_predictions[0][i]

# padding_infos = deb.get_padding_infos()
# madness_residual_field = deb.compute_residual(
#     blend["blend_images"][0], use_scatter_and_sub=True
# ).numpy()

# bkg_rms = {}
# for band in range(6):
#     bkg_rms[band] = sep.Background(blend["blend_images"][0][band]).globalrms

# actual_gal_fluxes, actual_gal_fluxerrs, actual_gal_flags = compute_apperture_photometry(
#     residual_field=actual_residual_field,
#     predictions=blend["isolated_images"][0],
#     xpos=blend["blend_list"][0]["x_peak"],
#     ypos=blend["blend_list"][0]["y_peak"],
#     bkg_rms=bkg_rms,
# )

# (
#     madness_gal_fluxes,
#     madness_gal_fluxerrs,
#     madness_gal_flags,
# ) = compute_apperture_photometry(
#     residual_field=np.transpose(madness_residual_field, axes=(2, 0, 1)),
#     predictions=madness_predictions[0],
#     xpos=blend["blend_list"][0]["x_peak"],
#     ypos=blend["blend_list"][0]["y_peak"],
#     bkg_rms=bkg_rms,
# )

# (
#     scarlet_gal_fluxes,
#     scarlet_gal_fluxerrs,
#     scarlet_gal_flags,
# ) = compute_apperture_photometry(
#     residual_field=scarlet_residual_field,
#     predictions=scarlet_predictions[0],
#     xpos=blend["blend_list"][0]["x_peak"],
#     ypos=blend["blend_list"][0]["y_peak"],
#     bkg_rms=bkg_rms,
# )

# (
#     blended_gal_fluxes,
#     blended_gal_fluxerrs,
#     blended_gal_flags,
# ) = compute_apperture_photometry(
#     residual_field=blend["blend_images"][0],
#     predictions=None,
#     xpos=blend["blend_list"][0]["x_peak"],
#     ypos=blend["blend_list"][0]["y_peak"],
#     bkg_rms=bkg_rms,
# )


# plt.figure(figsize=(10, 7))
# bins = np.arange(0, 1, 0.001)
# plt.hist(
#     (madness_gal_fluxes - actual_gal_fluxes) / actual_gal_fluxes,
#     bins=bins,
#     alpha=0.5,
#     label="MADNESS",
# )
# print(np.shape(scarlet_gal_fluxes))
# plt.hist(
#     (scarlet_gal_fluxes - actual_gal_fluxes) / actual_gal_fluxes,
#     bins=bins,
#     alpha=0.5,
#     label="scarlet",
# )


# plt.hist(
#     (blended_gal_fluxes - actual_gal_fluxes) / blended_gal_fluxes,
#     bins=bins,
#     alpha=0.5,
#     label="blended",
# )
# plt.xlim([0, .1])

# plt.legend(fontsize=20)

# plt.savefig("aperturephoto")
