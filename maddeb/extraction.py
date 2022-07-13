import logging

import numpy as np

logging.basicConfig(format="%(message)s", level=logging.INFO)

LOG = logging.getLogger(__name__)


def extract_cutouts(
    field_image,
    galaxy_distances_to_center,
    cutout_size=59,
    nb_of_bands=6,
    channel_last=False,
):
    """
    Extract the cutouts around particular galaxies in the field
    parameters:
        field_image: image of the field to deblend
        field_size: size of the field
        galaxy_distances_to_center: distances of the galaxies to deblend from the center of the field. In pixels.
        cutout_size: size of the stamps
    """
    cutout_images = np.zeros(
        (len(galaxy_distances_to_center), cutout_size, cutout_size, nb_of_bands)
    )
    list_idx = []
    flag = False

    field_size = np.shape(field_image)[1]

    for i in range(len(galaxy_distances_to_center)):
        try:
            x_shift = galaxy_distances_to_center[i][0]
            y_shift = galaxy_distances_to_center[i][1]

            x_start = -int(cutout_size / 2) + round(x_shift) + int(field_size / 2)
            x_end = int(cutout_size / 2) + round(x_shift) + int(field_size / 2) + 1

            y_start = -int(cutout_size / 2) + round(y_shift) + int(field_size / 2)
            y_end = int(cutout_size / 2) + round(y_shift) + int(field_size / 2) + 1
            if channel_last:
                cutout_images[i] = field_image[x_start:x_end, y_start:y_end]
            else:
                cutout_regions = field_image[:, x_start:x_end, y_start:y_end]
                cutout_images[i] = np.transpose(cutout_regions, axes=(1, 2, 0))

            list_idx.append(i)

        except ValueError:
            flag = True

    if flag:

        LOG.warning(
            "Some galaxies are too close from the border of the field to be considered here."
        )

    return cutout_images, list_idx
