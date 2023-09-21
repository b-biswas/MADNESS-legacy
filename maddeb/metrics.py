"""Metrics for comparison."""

import numpy as np
import sep
from numba import jit


def compute_pixel_cosdist(
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
    for simulated_galaxy, predicted_galaxy in zip(
        simulated_galaxies, predicted_galaxies
    ):
        noiseless_galaxy_field += simulated_galaxy

    results = {}

    columns = ["_covariance"]
    if get_blendedness:
        columns = columns + ["_blendedness"]

    for band in ["u", "g", "r", "i", "z", "y"]:
        for col_name in columns:
            results[band + col_name] = []
    results["galaxy_num"] = []

    for gal_num in range(len(predicted_galaxies)):

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
            pixel_covariance = cosdist_helper(
                predicted_galaxy[band_number],
                simulated_galaxy[band_number],
                sig,
            )

            results[band + "_covariance"].append(pixel_covariance)
            if get_blendedness:
                blendedness = compute_blendedness(
                    isolated_galaxy_band=simulated_galaxy[band_number],
                    field_band=noiseless_galaxy_field[band_number],
                )
                results[band + "_blendedness"].append(blendedness)

        results["galaxy_num"].append(gal_num)
    return results

@jit
def cosdist_helper(predicted_band_galaxy, simulated_band_galaxy, sig):
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
    pixel_covariance: float
        pixel-wise covariance in the band

    """
    # mask1 = simulated_band_galaxy > 0 * sig
    # mask2 = predicted_band_galaxy > 0 * sig
    # mask = np.logical_or(mask1, mask2)
    ground_truth_pixels = simulated_band_galaxy.flatten()
    predicted_pixels = predicted_band_galaxy.flatten()

    pixel_covariance = np.sum(np.multiply(predicted_pixels, ground_truth_pixels)) / (
        np.sqrt(np.sum(np.square(predicted_pixels)))
        * np.sqrt(np.sum(np.square(ground_truth_pixels)))
    )

    return pixel_covariance


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


def compute_apperture_photometry(
    field_image,
    predictions,
    xpos,
    ypos,
    bkg_rms,
    a,
    b,
    theta,
    psf_fwhm=None,
    r=2,
):
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
    a: float
        hlr semi major axis of galaxy in pixels
    b: float
        hlr semi minor axis of galaxy in pixels
    theta: float
        orientation of the galaxy in degrees
    psf_fwhm: float
        fwhm of PSF in pixels
    r: int
        factor by which the major-minor-axis is multiplied

    Returns
    -------
    results: astropy.Table
        Table with flux and flux_errs.

    """
    results = {}
    for band in ["u", "g", "r", "i", "z", "y"]:
        for column in ["_phot_flux", "_phot_fluxerrs", "_phot_flags"]:
            results[band + column] = []

    if psf_fwhm is None:
        psf_fwhm = [0] * 6

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

            flux, fluxerr, flag = sep.sum_ellipse(
                data=galaxy[band_num],
                x=[xpos[galaxy_num]],
                y=[ypos[galaxy_num]],
                a=(a[galaxy_num] ** 2 + (psf_fwhm[band_num] / 2) ** 2) ** 0.5,
                b=(b[galaxy_num] ** 2 + (psf_fwhm[band_num] / 2) ** 2) ** 0.5,
                theta=theta[galaxy_num],
                r=r,
                err=bkg_rms[band_num],
            )

            results[band + "_phot_flux"].extend(flux)
            results[band + "_phot_fluxerrs"].extend(fluxerr)
            results[band + "_phot_flags"].extend(flag)

        results["galaxy_num"].append(galaxy_num)

    return results
