"""Run MADNESS on test dataset."""

import logging
import math
import os
import sys

import galcheat
import hickle
import numpy as np
import pandas as pd
import sep
import tensorflow as tf
import tensorflow_probability as tfp

from maddeb.Deblender import Deblend
from maddeb.metrics import compute_aperture_photometry, compute_pixel_cosdist
from maddeb.utils import get_data_dir_path

# logging level set to INFO
logging.basicConfig(format="%(message)s", level=logging.INFO)

LOG = logging.getLogger(__name__)

survey_name = sys.argv[1]

if survey_name not in ["LSST"]:
    raise ValueError("survey should be one of: LSST")  # other surveys to be added soon!

survey = galcheat.get_survey(survey_name)

density = sys.argv[2]
run_name = sys.argv[3]
map_solution = sys.argv[4].lower() == "true"
max_number = 20
#LOG.info(map_solution)

if density not in ["high", "low"]:
    raise ValueError("The second arguemnt should be either isolated or blended")

simulation_path = os.path.join(
    "/sps/lsst/users/bbiswas/simulations/test_data/", density
)
results_path = "/sps/lsst/users/bbiswas/MADNESS_results/"
density_level = density + "_density"


weights_path = os.path.join(get_data_dir_path(), f"LSST")
deb = Deblend(latent_dim=16, weights_path=weights_path, survey=survey)

psf_fwhm = []
for band in survey.available_filters:
    filt = survey.get_filter(band)
    psf_fwhm.append(filt.psf_fwhm.value)

num_repetations = 300

for file_num in range(num_repetations):
    LOG.info(f"\n\n######### Processing file: {file_num} #########")
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
    linear_norm_coeff = 10000

    blend_list = []
    madness_results = []

    actual_photometry = []
    madness_photometry = []
    blended_photometry = []

    detected_positions = np.zeros((len(blend['blend_list']), max_number, 2)) 
    num_components = []
    for field_num in range(len(blend['blend_list'])):
        for gal_num in range(len(blend['blend_list'][field_num])):
            detected_positions[field_num][gal_num][0] = blend['blend_list'][field_num]['y_peak'][gal_num]
            detected_positions[field_num][gal_num][1] = blend['blend_list'][field_num]['x_peak'][gal_num]
        num_components.append(len(blend['blend_list'][field_num]))

    convergence_criterion = tfp.optimizer.convergence_criteria.LossNotDecreasing(
        rtol=0.05,
        min_num_steps=40,
        window_size=15,
    )
    # convergence_criterion = None
    lr_scheduler = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=0.075,
        decay_steps=30,
        decay_rate=0.8,
        staircase=True,
    )
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_scheduler)

    deb(
        field_images,
        detected_positions,
        num_components=num_components,  # redundant parameter
        use_log_prob=True,
        linear_norm_coeff=linear_norm_coeff,
        max_iter=200,
        convergence_criterion=convergence_criterion,
        optimizer=optimizer,
        use_debvader=True,
        map_solution=map_solution,
    )
    padding_infos_all_fields = deb.get_padding_infos()

    for field_num in range(len(blend["blend_list"])):
        #LOG.info(field_num)

        current_field_predictions = []
        current_madness_models = {"images": [], "field_num": [], "galaxy_num": []}

        current_blend = blend["blend_list"][field_num]

        padding_infos = padding_infos_all_fields[field_num]
        for component_num in range(deb.num_components[field_num]):
            #LOG.info(component_num)
            #LOG.info(padding_infos)
            prediction = np.pad(
                deb.components[field_num][component_num], 
                padding_infos[component_num],
            )
            prediction = np.transpose(prediction, axes=(2, 0, 1))
            current_field_predictions.append(prediction)

        # madness_predictions.append(current_field_predictions)

        num_galaxies = len(blend["blend_list"][field_num])

        isolated_images = blend["isolated_images"][field_num][0:num_galaxies]

        madness_current_res = compute_pixel_cosdist(
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
        for band_name in ["u_ab", "g_ab", "r_ab", "i_ab", "z_ab", "y_ab"]:
            madness_current_res[band_name] = blend["blend_list"][field_num][band_name]
        # make this a table

        # madness_results.append(madness_current_res)

        bkg_rms = {}
        for band in range(len(survey.available_filters)):
            bkg_rms[band] = sep.Background(
                blend["blend_images"][field_num][band]
            ).globalrms

        a = blend["blend_list"][field_num]["a_d"].value
        b = blend["blend_list"][field_num]["b_d"].value
        theta = blend["blend_list"][field_num]["pa_disk"].value

        cond = (
            blend["blend_list"][field_num]["a_d"]
            < blend["blend_list"][field_num]["a_b"]
        )
        a = np.where(cond, blend["blend_list"][field_num]["a_b"].value, a)
        b = np.where(cond, blend["blend_list"][field_num]["b_b"].value, b)
        theta = np.where(cond, blend["blend_list"][field_num]["pa_bulge"].value, theta)

        theta = theta % 180
        theta = theta * math.pi / 180

        theta = np.where(theta > math.pi / 2, theta - math.pi, theta)

        madness_photometry_current = compute_aperture_photometry(
            field_image=blend["blend_images"][field_num],
            predictions=current_field_predictions,
            xpos=blend["blend_list"][field_num]["x_peak"],
            ypos=blend["blend_list"][field_num]["y_peak"],
            a=a / survey.pixel_scale.value,
            b=b / survey.pixel_scale.value,
            theta=theta,
            psf_fwhm=np.array(psf_fwhm) / survey.pixel_scale.value,
            bkg_rms=bkg_rms,
        )
        madness_current_res.update(madness_photometry_current)
        madness_current_res = pd.DataFrame.from_dict(madness_current_res)
        madness_results.append(madness_current_res)

        actual_results_current = compute_aperture_photometry(
            field_image=blend["blend_images"][field_num],
            predictions=blend["isolated_images"][field_num],
            xpos=blend["blend_list"][field_num]["x_peak"],
            ypos=blend["blend_list"][field_num]["y_peak"],
            a=a / survey.pixel_scale.value,
            b=b / survey.pixel_scale.value,
            theta=theta,
            psf_fwhm=np.array(psf_fwhm) / survey.pixel_scale.value,
            bkg_rms=bkg_rms,
        )
        actual_results_current["field_num"] = field_num * num_galaxies
        actual_results_current["file_num"] = file_num * num_galaxies
        actual_results_current["galaxy_num"] = np.arange(num_galaxies)
        actual_results_current = pd.DataFrame.from_dict(actual_results_current)
        actual_photometry.append(actual_results_current)

        blended_results_current = compute_aperture_photometry(
            field_image=blend["blend_images"][field_num],
            predictions=None,
            xpos=blend["blend_list"][field_num]["x_peak"],
            ypos=blend["blend_list"][field_num]["y_peak"],
            a=a / survey.pixel_scale.value,
            b=b / survey.pixel_scale.value,
            theta=theta,
            psf_fwhm=np.array(psf_fwhm) / survey.pixel_scale.value,
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

    madness_results["r_contamination"] = (
        blended_photometry["r_phot_flux"].values
        - actual_photometry["r_phot_flux"].values
    ) / actual_photometry["r_phot_flux"].values

    # reconstruction_file = open("debblend_results" + str(rep_num)+ ".pkl", "wb")

    save_file_name = os.path.join(
        results_path,
        density_level,
        run_name,
        "madness_results" if map_solution else "debvader_results",
        str(file_num) + ".pkl",
    )
    # np.save(save_file_name,
    #     madness_results.to_records())
    madness_results.to_pickle(save_file_name)
    # hickle.dump(madness_results, save_file_name, mode="w")
    # ascii.write(madness_results, save_file_name, overwrite=True)

    save_file_name = os.path.join(
        results_path,
        density_level,
        "actual_photometry",
        str(file_num) + ".pkl",
    )

    actual_photometry.to_pickle(save_file_name)

    save_file_name = os.path.join(
        results_path,
        density_level,
        "blended_photometry",
        str(file_num) + ".pkl",
    )

    blended_photometry.to_pickle(save_file_name)
