import numpy as np
import sep
import skimage
from numba import jit
import pandas as pd

def compute_reconstruction_metrics(predicted_images, ground_truth, channel_last=True):
    """
    calculates reconsurction metrics such as:
    mean squared error, peak signal to noise ratio, ssim,

    args:
    predicted_images:
    ground_truth:

    """

    msr_results = []
    psnr_results = []
    ssim_results = []

    for i in range(len(predicted_images)):

        msr_results.append(
            skimage.metrics.mean_squared_error(predicted_images[i], ground_truth[i])
        )

        psnr_results.append(
            skimage.metrics.peak_signal_noise_ratio(
                predicted_images[i],
                ground_truth[i],
                data_range=np.max(ground_truth[i]),
            )
        )

        ssim_results.append(
            skimage.metrics.structural_similarity(
                ground_truth[i],
                predicted_images[i],
                channel_axis=-1,
                multichannel=True,
            )
        )

    results_dict = {"mse": msr_results, "psnr": psnr_results, "ssim": ssim_results}

    return results_dict


def compute_pixel_covariance_and_fluxes(
    predicted_galaxies, simulated_galaxies, field_image, get_blendedness=True,
):
    noiseless_galaxy_field = np.zeros_like(field_image)
    for simulated_galaxy in simulated_galaxies:
        noiseless_galaxy_field += simulated_galaxy

    results = {}

    columns = ["_covariance", "_actual_flux", "_predicted_flux"]
    if get_blendedness:
        columns = columns + ["_blendedness"]

    for band in ['u', 'g', 'r', 'i', 'z', 'y']:
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

        predicted_galaxy=predicted_galaxies[gal_num]
        simulated_galaxy=simulated_galaxies[gal_num]

        for band_number, band in enumerate(['u', 'g', 'r', 'i', 'z', 'y']):

            sig = sep.Background(field_image[band_number]).globalrms
            # print(sig)
            # print(sig[band_number])
            mask1 = simulated_galaxy[band_number] > 0 * sig
            mask2 = predicted_galaxy[band_number] > 0 * sig
            mask = np.logical_or(mask1, mask2)
            #             fig, ax = plt.subplots(1, 2)
            #             plt.subplot(1,2,1)
            #             plt.imshow(cutout_galaxy[:, :, band_number])
            #             plt.subplot(1, 2, 2)
            #             plt.imshow(madness_predictions[blend_number][galaxy_number][band_number])
            #             plt.show()
            band_actual_flux, band_predicted_flux, pixel_covariance = convariance_and_flux_helper(predicted_galaxy[band_number], simulated_galaxy[band_number], sig)

            
            results[band + "_actual_flux"].append(band_actual_flux)
            results[band + "_predicted_flux"].append(band_predicted_flux)
            results[band + "_covariance"].append(pixel_covariance)
            if get_blendedness:
                blendedness = compute_blendedness(isolated_galaxy_band=simulated_galaxy[band_number], field_band=noiseless_galaxy_field[band_number])
                results[band + "_blendedness"].append(blendedness)

        results["galaxy_num"].append(gal_num)
    return pd.DataFrame(results)

@jit
def convariance_and_flux_helper(predicted_band_galaxy, simulated_band_galaxy, sig):
    """
    only one band
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

    isolated_pixels = isolated_galaxy_band.flatten()
    field_pixels = field_band.flatten()

    blendedness = 1 - np.sum(np.multiply(isolated_pixels, isolated_pixels))/np.sum(np.multiply(field_pixels, isolated_pixels))

    return blendedness


def compute_apperture_photometry(field_image, predictions, xpos, ypos, bkg_rms):
    """
    field_image:
    prediction: predictions of model or None
    """
    results = {}
    for band in ['u', 'g', 'r', 'i', 'z', 'y']:
        for column in ["_flux", "_fluxerrs", "_flags"]:
            results[band + column] = []

    results["galaxy_num"] = []

    residual_field=field_image
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

        for band_num, band in enumerate(['u', 'g', 'r', 'i', 'z', 'y']):

            flux, fluxerr, flag = sep.sum_circle(
                galaxy[band_num],
                [xpos[galaxy_num]],
                [ypos[galaxy_num]],
                3.0,
                err=bkg_rms[band_num],
            )

            results[band + "_flux"].extend(flux)
            results[band + "_fluxerrs"].extend(fluxerr)
            results[band + "_flags"].extend(flag)

        results["galaxy_num"].append(galaxy_num)

    return pd.DataFrame(results)
