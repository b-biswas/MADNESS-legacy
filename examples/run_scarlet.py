"""Run Scarlet on test data."""

import logging
import math
import os
import sys
import time

import btk
import galsim
import hickle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scarlet
import scarlet.psf
import sep

from maddeb.metrics import (
    compute_apperture_photometry,
    compute_pixel_cosdist,
)

# logging level set to INFO
logging.basicConfig(format="%(message)s", level=logging.INFO)

LOG = logging.getLogger(__name__)

survey = btk.survey.get_surveys("LSST")

num_repetations = 300
density = sys.argv[1]

if density not in ["high", "low"]:
    raise ValueError("The second arguemnt should be either isolated or blended")


simulation_path = os.path.join(
    "/sps/lsst/users/bbiswas/simulations/test_data/", density
)
results_path = "/sps/lsst/users/bbiswas/MADNESS_results/"
density_level = density + "_density"

psf_fwhm = []
for band in ["u", "g", "r", "i", "z", "y"]:
    filt = survey.get_filter(band)
    psf_fwhm.append(filt.psf_fwhm.value * 5)


# Define function to make predictions with scarlet
def predict_with_scarlet(image, x_pos, y_pos, show_scene, show_sources, filters):
    """Deblend using the SCARLET deblender.

    Parameters
    ----------
    image: array
        field to be deblended.
    x_pos: array
        x positions of detections.
    y_pos: array
        y position of detetions.
    show_scene: bool
        To run scarlet.display.show_scene or not.
    show_sources: bool
        To run scarlet.display.show_sources or not.
    filters: list of hashable elements
        Names/identifiers of spectral channels

    Returns
    -------
    predicted_sources:
        array with reconsturctions predicted by SCARLET

    """
    sig = []
    weights = np.ones_like(image)
    for i in range(6):
        sig.append(sep.Background(image[i]).globalrms)
        weights[i] = weights[i] / (sig[i] ** 2)
    observation = scarlet.Observation(
        image, psf=scarlet.psf.ImagePSF(psf), weights=weights, channels=bands, wcs=wcs
    )

    model_psf = scarlet.GaussianPSF(
        sigma=(0.382, 0.365, 0.344, 0.335, 0.327, 0.323)
    )  # These numbers are derived from the FWHM given for LSST filters in the galcheat v1.0 repo https://github.com/aboucaud/galcheat/blob/main/galcheat/data/LSST.yaml
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
    LOG.info("Processing file " + str(file_num))
    blend = hickle.load(
        os.path.join(
            simulation_path,
            str(file_num) + ".hkl",
        )
    )

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

    scarlet_results = []
    scarlet_photometry = []

    # Get Scarlet Predictions

    for field_num in range(len(blend["blend_list"])):
        scarlet_current_predictions = []
        image = field_images[field_num]
        x_pos = blend["blend_list"][field_num]["y_peak"]
        y_pos = blend["blend_list"][field_num]["x_peak"]
        scarlet_current_predictions = predict_with_scarlet(
            image,
            x_pos=x_pos,
            y_pos=y_pos,
            show_scene=False,
            show_sources=False,
            filters=bands,
        )

        num_galaxies = len(blend["blend_list"][field_num])

        isolated_images = blend["isolated_images"][field_num][0:num_galaxies]

        scarlet_current_res = compute_pixel_cosdist(
            scarlet_current_predictions,
            isolated_images,
            blend["blend_images"][field_num],
        )

        # scarlet_current_res["images"] = scarlet_current_predictions

        size = blend["blend_list"][field_num]["btk_size"]

        scarlet_current_res["size"] = size
        scarlet_current_res["field_num"] = [field_num] * num_galaxies
        scarlet_current_res["file_num"] = [file_num] * num_galaxies
        scarlet_current_res["r_band_snr"] = blend["blend_list"][field_num]["r_band_snr"]
        # make this a table

        # scarlet_results.append(scarlet_current_res)

        bkg_rms = {}
        for band in range(6):
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

        scarlet_photometry_current = compute_apperture_photometry(
            field_image=blend["blend_images"][field_num],
            predictions=scarlet_current_predictions,
            xpos=blend["blend_list"][field_num]["x_peak"],
            ypos=blend["blend_list"][field_num]["y_peak"],
            a=5 * a,
            b=5 * b,
            theta=theta,
            psf_fwhm=psf_fwhm,
            bkg_rms=None,
        )
        scarlet_current_res.update(scarlet_photometry_current)
        scarlet_current_res = pd.DataFrame.from_dict(scarlet_current_res)
        print(scarlet_current_res)
        scarlet_results.append(scarlet_current_res)
    # scarlet_results = vstack(scarlet_results)
    # scarlet_results = hstack([scarlet_results, vstack(blend["blend_list"])])
    # scarlet_results = hstack(scarlet_results,vstack(scarlet_photometry))

    scarlet_results = pd.concat(scarlet_results)

    save_file_name = os.path.join(
        results_path,
        density_level,
        "scarlet_results",
        str(file_num) + ".pkl",
    )

    scarlet_results.to_pickle(save_file_name)
    # np.save(
    #     save_file_name,
    #     scarlet_results.to_records(),
    # )

    # hickle.dump(scarlet_results, save_file_name, mode="w", compression='gzip')
