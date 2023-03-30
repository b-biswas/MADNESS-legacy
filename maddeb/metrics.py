"""Metrics for comparison."""

import numpy as np
import sep
from numba import jit

# def compute_reconstruction_metrics(predicted_images, ground_truth):
#     """Calculate mean squared error, peak signal to noise ratio, ssim.

#     Parameters
#     ----------
#     predicted_images: np array
#         galaxies predicted by the model
#     ground_truth: np array
#         simulated ground truth of the model.

#     Returns
#     -------
#     results_dict: dict
#         evaluated metrics

#     """
#     msr_results = []
#     psnr_results = []
#     ssim_results = []

#     for i in range(len(predicted_images)):

#         msr_results.append(
#             skimage.metrics.mean_squared_error(predicted_images[i], ground_truth[i])
#         )

#         psnr_results.append(
#             skimage.metrics.peak_signal_noise_ratio(
#                 predicted_images[i],
#                 ground_truth[i],
#                 data_range=np.max(ground_truth[i]),
#             )
#         )

#         ssim_results.append(
#             skimage.metrics.structural_similarity(
#                 ground_truth[i],
#                 predicted_images[i],
#                 channel_axis=-1,
#                 multichannel=True,
#             )
#         )

#     results_dict = {"mse": msr_results, "psnr": psnr_results, "ssim": ssim_results}

#     return results_dict


def compute_pixel_covariance_and_fluxes(
    predicted_galaxies,
    simulated_galaxies,
    field_image,
    get_blendedness=True,
):
    """Calculate pixel covariances and fluxes.

    Parameters
    ----------
    predicted_galaxies: np array
        galaxies predicted by the model
    simulated_galaxies: np array
        simulated ground truth of the model.
    field_image: np array
        the entire simulated field.
    get_blendedness: bool
        to return the blendedness metric

    Returns
    -------
    results_dict: astropy.Table
        Table with the evaluated metrics

    """
    noiseless_galaxy_field = np.zeros_like(field_image)
    for simulated_galaxy in simulated_galaxies:
        noiseless_galaxy_field += simulated_galaxy

    results = {}

    columns = ["_covariance", "_actual_flux", "_predicted_flux"]
    if get_blendedness:
        columns = columns + ["_blendedness"]

    for band in ["u", "g", "r", "i", "z", "y"]:
        for col_name in columns:
            results[band + col_name] = []
    results["galaxy_num"] = []

    for gal_num in range(len(predicted_galaxies)):
        # (
        #     pixel_covriance,
        #     actual_flux,
        #     predicted_flux,
        # ) = compute_pixel_covariance_and_flux(
        #     predicted_galaxy=predicted_galaxies[gal_num],
        #     simulated_galaxy=simulated_galaxies[gal_num],
        #     field_image=field_image,
        # )

        predicted_galaxy = predicted_galaxies[gal_num]
        simulated_galaxy = simulated_galaxies[gal_num]

        for band_number, band in enumerate(["u", "g", "r", "i", "z", "y"]):

            sig = sep.Background(field_image[band_number]).globalrms
            # print(sig)
            # print(sig[band_number])

            # mask1 = simulated_galaxy[band_number] > 0 * sig
            # mask2 = predicted_galaxy[band_number] > 0 * sig
            # mask = np.logical_or(mask1, mask2)

            #             fig, ax = plt.subplots(1, 2)
            #             plt.subplot(1,2,1)
            #             plt.imshow(cutout_galaxy[:, :, band_number])
            #             plt.subplot(1, 2, 2)
            #             plt.imshow(madness_predictions[blend_number][galaxy_number][band_number])
            #             plt.show()
            (
                band_actual_flux,
                band_predicted_flux,
                pixel_covariance,
            ) = convariance_and_flux_helper(
                predicted_galaxy[band_number], simulated_galaxy[band_number], sig
            )

            results[band + "_actual_flux"].append(band_actual_flux)
            results[band + "_predicted_flux"].append(band_predicted_flux)
            results[band + "_covariance"].append(pixel_covariance)
            if get_blendedness:
                blendedness = compute_blendedness(
                    isolated_galaxy_band=simulated_galaxy[band_number],
                    field_band=noiseless_galaxy_field[band_number],
                )
                results[band + "_blendedness"].append(blendedness)

        results["galaxy_num"].append(gal_num)
    return results


def convariance_and_flux_helper(predicted_band_galaxy, simulated_band_galaxy, sig):
    """Calculate pixel fluxes and covariances in a band.

    Parameters
    ----------
    predicted_band_galaxy: np array
        A specific band a predicted galaxy.
    simulated_band_galaxy: np array
        simulated ground truth of the same band.
    sig: float
        noise level

    Returns
    -------
    band_actual_flux: float
        Actual flux in the band
    band_predicted_flux: float
        Flux predicted by the model
    pixel_covariance: float
        pixel-wise covariance in the band

    """
    mask1 = simulated_band_galaxy > 0 * sig
    mask2 = predicted_band_galaxy > 0 * sig
    mask = np.logical_or(mask1, mask2)
    ground_truth_pixels = simulated_band_galaxy[mask].flatten()
    predicted_pixels = predicted_band_galaxy[mask].flatten()

    band_actual_flux = np.sum(simulated_band_galaxy[mask1])
    band_predicted_flux = np.sum(predicted_band_galaxy[mask2])

    pixel_covariance = np.sum(np.multiply(predicted_pixels, ground_truth_pixels)) / (
        np.sqrt(np.sum(np.square(predicted_pixels)))
        * np.sqrt(np.sum(np.square(ground_truth_pixels)))
    )

    return band_actual_flux, band_predicted_flux, pixel_covariance


@jit
def compute_blendedness(isolated_galaxy_band, field_band):
    """Calculate pixel fluxes and covariances in a band.

    Parameters
    ----------
    isolated_galaxy_band: np array
        simulated ground truth of a specific band an isolated galaxy.
    field_band: np array
        simulated ground truth of the same region of band.

    Returns
    -------
    blendedness: float
        blendedness in the band

    """
    isolated_pixels = isolated_galaxy_band.flatten()
    field_pixels = field_band.flatten()

    blendedness = 1 - np.sum(np.multiply(isolated_pixels, isolated_pixels)) / np.sum(
        np.multiply(field_pixels, isolated_pixels)
    )

    return blendedness


def compute_apperture_photometry(field_image, predictions, xpos, ypos, bkg_rms):
    """Calculate apperture photometry.

    Parameters
    ----------
    field_image: np array
        image the field of galaxies.
    predictions: np array
        predictions of the model/ ground truth
    xpos: np array/list
        x positions of the detections
    ypos: np array/list
        y positions of the detections
    bkg_rms: list
        list with the rms background in each band.

    Returns
    -------
    results: astropy.Table
        Table with flux and flux_errs.

    """
    results = {}
    for band in ["u", "g", "r", "i", "z", "y"]:
        for column in ["_phot_flux", "_phot_fluxerrs", "_phot_flags"]:
            results[band + column] = []

    results["galaxy_num"] = []

    residual_field = field_image
    if predictions is not None:
        for prediction in predictions:
            residual_field = residual_field - prediction

    for galaxy_num in range(len(xpos)):

        # actual galaxy
        if predictions is None:
            galaxy = residual_field
        else:
            galaxy = residual_field + predictions[galaxy_num]

        galaxy = galaxy.copy(order="C")

        for band_num, band in enumerate(["u", "g", "r", "i", "z", "y"]):

            flux, fluxerr, flag = sep.sum_circle(
                galaxy[band_num],
                [xpos[galaxy_num]],
                [ypos[galaxy_num]],
                5.0,
                err=bkg_rms[band_num],
            )

            results[band + "_phot_flux"].extend(flux)
            results[band + "_phot_fluxerrs"].extend(fluxerr)
            results[band + "_phot_flags"].extend(flag)

        results["galaxy_num"].append(galaxy_num)

    return results
