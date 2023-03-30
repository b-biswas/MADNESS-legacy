"""Run MADNESS on test dataset."""

import logging
import os
import sys

import btk
import hickle
import numpy as np
import pandas as pd
import sep
import tensorflow as tf
import tensorflow_probability as tfp

from maddeb.Deblender import Deblend
from maddeb.metrics import (
    compute_apperture_photometry,
    compute_pixel_covariance_and_fluxes,
)

# logging level set to INFO
logging.basicConfig(format="%(message)s", level=logging.INFO)

LOG = logging.getLogger(__name__)

survey = btk.survey.get_surveys("LSST")

num_repetations = 200

density = sys.argv[1]

if density not in ["high", "low"]:
    raise ValueError("The second arguemnt should be either isolated or blended")

simulation_path = "/sps/lsst/users/bbiswas/simulations/test_data/"
results_path = "/sps/lsst/users/bbiswas/MADNESS_results/"
run_name = "catsim_" + density + "_density_ssim_20"

for file_num in range(num_repetations):
    LOG.info("Processing file " + str(file_num))
    blend = hickle.load(
        os.path.join(
            simulation_path,
            str(file_num) + ".hkl",
        )
    )

    field_images = blend["blend_images"]
    isolated_images = blend["isolated_images"]

    # get MADNESS predictions
    madness_predictions = []
    madness_models = []
    linear_norm_coeff = [1000, 5000, 10000, 10000, 10000, 10000]

    blend_list = []
    madness_results = []

    actual_photometry = []
    madness_photometry = []
    blended_photometry = []

    for field_num in range(len(blend["blend_list"])):

        current_field_predictions = []
        current_madness_models = {"images": [], "field_num": [], "galaxy_num": []}

        current_blend = blend["blend_list"][field_num]
        # print(blends)
        detected_positions = []
        for j in range(len(current_blend)):
            detected_positions.append(
                [current_blend["y_peak"][j], current_blend["x_peak"][j]]
            )

        # tf.config.run_functions_eagerly(False)
        # convergence_criterion = tfp.optimizer.convergence_criteria.LossNotDecreasing(
        #     atol=0.00001 * 45 * 45 * len(blend) * 3, min_num_steps=100, window_size=20
        # )
        convergence_criterion = (
            tfp.optimizer.convergence_criteria.SuccessiveGradientsAreUncorrelated(
                min_num_steps=120, window_size=30
            )
        )
        # convergence_criterion = None
        lr_scheduler = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=1.0, decay_steps=25, decay_rate=0.9, staircase=True
        )
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr_scheduler)

        deb = Deblend(
            field_images[field_num],
            detected_positions,
            num_components=len(current_blend),  # redundant parameter
            latent_dim=16,
            use_log_prob=True,
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
            prediction = np.pad(
                deb.components[component_num], padding_infos[component_num]
            )
            prediction = np.transpose(prediction, axes=(2, 0, 1))
            current_field_predictions.append(prediction)

        # madness_predictions.append(current_field_predictions)

        num_galaxies = len(blend["blend_list"][field_num])

        isolated_images = blend["isolated_images"][field_num][0:num_galaxies]

        madness_current_res = compute_pixel_covariance_and_fluxes(
            current_field_predictions,
            isolated_images,
            blend["blend_images"][field_num],
        )

        # madness_current_res["images"] = current_field_predictions
        madness_current_res["size"] = blend["blend_list"][field_num]["btk_size"]
        madness_current_res["field_num"] = [field_num] * num_galaxies
        madness_current_res["file_num"] = [file_num] * num_galaxies
        madness_current_res["r_band_snr"] = blend["blend_list"][field_num]["r_band_snr"]
        madness_current_res["ref_mag"] = blend["blend_list"][field_num]["ref_mag"]
        # make this a table

        # madness_results.append(madness_current_res)

        bkg_rms = {}
        for band in range(6):
            bkg_rms[band] = sep.Background(
                blend["blend_images"][field_num][band]
            ).globalrms

        madness_photometry_current = compute_apperture_photometry(
            field_image=blend["blend_images"][field_num],
            predictions=current_field_predictions,
            xpos=blend["blend_list"][field_num]["x_peak"],
            ypos=blend["blend_list"][field_num]["y_peak"],
            bkg_rms=bkg_rms,
        )
        madness_current_res.update(madness_photometry_current)
        madness_current_res = pd.DataFrame.from_dict(madness_current_res)
        madness_results.append(madness_current_res)

        actual_results_current = compute_apperture_photometry(
            field_image=blend["blend_images"][field_num],
            predictions=blend["isolated_images"][field_num],
            xpos=blend["blend_list"][field_num]["x_peak"],
            ypos=blend["blend_list"][field_num]["y_peak"],
            bkg_rms=bkg_rms,
        )
        actual_results_current["field_num"] = field_num * num_galaxies
        actual_results_current["file_num"] = file_num * num_galaxies
        actual_results_current["galaxy_num"] = np.arange(num_galaxies)
        actual_results_current = pd.DataFrame.from_dict(actual_results_current)
        actual_photometry.append(actual_results_current)

        blended_results_current = compute_apperture_photometry(
            field_image=blend["blend_images"][field_num],
            predictions=None,
            xpos=blend["blend_list"][field_num]["x_peak"],
            ypos=blend["blend_list"][field_num]["y_peak"],
            bkg_rms=bkg_rms,
        )
        blended_results_current["field_num"] = field_num * num_galaxies
        blended_results_current["file_num"] = file_num * num_galaxies
        blended_results_current["galaxy_num"] = np.arange(num_galaxies)
        blended_results_current = pd.DataFrame.from_dict(blended_results_current)
        blended_photometry.append(blended_results_current)

    madness_results = pd.concat(madness_results, ignore_index=True)
    # madness_results = hstack([madness_results, vstack(blend["blend_list"])])
    # madness_results = hstack([madness_results,vstack(madness_photometry)])

    # madness_results = hstack([madness_results, vstack(blend["blend_list"])])

    actual_photometry = pd.concat(actual_photometry, ignore_index=True)
    blended_photometry = pd.concat(blended_photometry, ignore_index=True)

    # reconstruction_file = open("debblend_results" + str(rep_num)+ ".pkl", "wb")

    save_file_name = os.path.join(
        results_path,
        run_name,
        "madness_results",
        str(file_num) + ".pkl",
    )
    # np.save(save_file_name,
    #     madness_results.to_records())
    madness_results.to_pickle(save_file_name)
    # hickle.dump(madness_results, save_file_name, mode="w")
    # ascii.write(madness_results, save_file_name, overwrite=True)

    save_file_name = os.path.join(
        results_path,
        run_name,
        "actual_photometry",
        str(file_num) + ".pkl",
    )
    # np.save(save_file_name,
    #     actual_photometry.to_records())
    actual_photometry.to_pickle(save_file_name)
    # hickle.dump(actual_photometry, save_file_name, mode="w")
    # ascii.write(actual_photometry, save_file_name, overwrite=True)

    save_file_name = os.path.join(
        results_path,
        run_name,
        "blended_photometry",
        str(file_num) + ".pkl",
    )
    # np.save(save_file_name,
    #     blended_photometry.to_records())
    blended_photometry.to_pickle(save_file_name)
    # hickle.dump(blended_photometry, save_file_name, mode="w")
    # ascii.write(blended_photometry, save_file_name, overwrite=True)
