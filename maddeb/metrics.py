import numpy as np
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
