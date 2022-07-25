import numpy as np
import sep
import skimage


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
    predicted_galaxies, simulated_galaxies, field_image
):
    pixel_covriances = []
    actual_fluxes = []
    predicted_fluxes = []

    for gal_num in range(len(predicted_galaxies)):
        (
            pixel_covriance,
            actual_flux,
            predicted_flux,
        ) = compute_pixel_covariance_and_flux(
            predicted_galaxy=predicted_galaxies[gal_num],
            simulated_galaxy=simulated_galaxies[gal_num],
            field_image=field_image,
        )

        pixel_covriances.append(pixel_covriance)
        actual_fluxes.append(actual_flux)
        predicted_fluxes.append(predicted_flux)

    return (
        np.array(pixel_covriances),
        np.array(actual_fluxes),
        np.array(predicted_fluxes),
    )


def compute_pixel_covariance_and_flux(predicted_galaxy, simulated_galaxy, field_image):
    """
    accepts only channel first"""
    ground_truth_pixels = []
    predicted_pixels = []
    sig = []

    actual_flux = []
    predicted_flux = []

    for band_number in range(6):
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


def compute_apperture_photomery(residual_field, predictions, xpos, ypos, bkg_rms):

    gal_fluxes = []
    gal_fluxerrs = []
    gal_flags = []

    for i in range(len(xpos)):

        # actual galaxy

        galaxy = residual_field + predictions[i]
        galaxy = galaxy.copy(order="C")

        for band in range(6):

            flux, fluxerr, flag = sep.sum_circle(
                galaxy[band],
                [xpos[i]],
                [ypos[i]],
                3.0,
                err=bkg_rms[band],
            )

            gal_fluxes.extend(flux)
            gal_fluxerrs.extend(fluxerr)
            gal_flags.extend(flag)

    return np.array(gal_fluxes), np.array(gal_fluxerrs), np.array(gal_flags)
