"""Simulations for training models."""

import logging
import os
import sys

import btk
import btk.catalog
import btk.draw_blends
import btk.plot_utils
import btk.sampling_functions
import btk.survey
import numpy as np
import pandas as pd

from maddeb.extraction import extract_cutouts
from maddeb.utils import CustomSampling

logging.basicConfig(format="%(message)s", level=logging.INFO)

LOG = logging.getLogger(__name__)

LOG.info(sys.argv)
dataset = sys.argv[1]  # should be either training or validation
if dataset not in ["training", "validation"]:
    raise ValueError(
        "The first arguement (dataset) should be either training or validation"
    )


blend_type = sys.argv[2]  # set to 4 to generate blended scenes
if blend_type not in ["isolated", "blended"]:
    raise ValueError("The second arguemnt should be either isolated or blended")

if blend_type == "isolated":
    max_number = 1
    unique_galaxies = True
    if dataset == "training":
        batch_size = 100
    if dataset == "validation":
        batch_size = 100
else:
    unique_galaxies = False
    max_number = 3
    batch_size = 100

seed = 993

CATSIM_CATALOG_PATH = "/sps/lsst/users/bbiswas/OneDegSq_snr_10.fits"
SAVE_PATH = "/sps/lsst/users/bbiswas/simulations/CATSIM_tfDataset"
print("saving data at " + SAVE_PATH)

stamp_size = 15
maxshift = 2

if dataset == "training":
    index_range = [0, 150000]
    num_batches = 1500
elif dataset == "validation":
    index_range = [150000, 200000]
    num_batches = 500

catalog = btk.catalog.CatsimCatalog.from_file(CATSIM_CATALOG_PATH)
survey = btk.survey.get_surveys("LSST")

sampling_function = CustomSampling(
    index_range=index_range,
    max_number=max_number,
    maxshift=maxshift,
    stamp_size=stamp_size,
    seed=seed,
    unique=unique_galaxies,
)

draw_generator = btk.draw_blends.CatsimGenerator(
    catalog,
    sampling_function,
    survey,
    batch_size=batch_size,
    stamp_size=stamp_size,
    cpus=4,
    add_noise="all",
    verbose=False,
    seed=seed,
    augment_data=False,
)

total_galaxy_stamps = num_batches * batch_size
stamp_counter = 0
shift_rng = np.random.default_rng(12345)
for batch_num in range(num_batches):

    print("simulating file number:" + str(batch_num))

    batch = next(draw_generator)

    for blended_image_num in range(len(batch["blend_images"])):

        blended_image = batch["blend_images"][blended_image_num]

        for galaxy_num in range(len(batch["blend_list"][blended_image_num]["x_peak"])):

            postage_stamps = {}
            # print("Image number "+ str(blended_image_num))
            # print("Galaxy number " + str(galaxy_num))
            isolated_image = batch["isolated_images"][blended_image_num][galaxy_num]
            x_pos = batch["blend_list"][blended_image_num]["y_peak"][galaxy_num]
            y_pos = batch["blend_list"][blended_image_num]["x_peak"][galaxy_num]

            # x_shift = shift_rng.random() - 0.5
            # y_shift = shift_rng.random() - 0.5
            x_shift = 0
            y_shift = 0

            pos = (x_pos + x_shift, y_pos + y_shift)
            gal_blended = extract_cutouts(
                blended_image,
                [pos],
                distances_to_center=False,
                channel_last=False,
                cutout_size=45,
            )[0][0]
            postage_stamps["blended_gal_stamps"] = [gal_blended]
            gal_isolated = extract_cutouts(
                isolated_image,
                [pos],
                distances_to_center=False,
                channel_last=False,
                cutout_size=45,
            )[0][0]
            postage_stamps["isolated_gal_stamps"] = [gal_isolated]
            postage_stamps["gal_locations_y_peak"] = [
                batch["blend_list"][blended_image_num]["y_peak"] - pos[0]
            ]
            postage_stamps["gal_locations_x_peak"] = [
                batch["blend_list"][blended_image_num]["x_peak"] - pos[1]
            ]
            postage_stamps["r_band_snr"] = [
                batch["blend_list"][blended_image_num]["r_band_snr"]
            ]

            postage_stamps = pd.DataFrame(postage_stamps)

            np.save(
                os.path.join(
                    SAVE_PATH,
                    blend_type + "_" + dataset,
                    f"gal_{batch_num}_{blended_image_num}_{galaxy_num}.npy",
                ),
                postage_stamps.to_records(),
            )

            stamp_counter += 1
            if stamp_counter == total_galaxy_stamps:
                LOG.info(f"simulated {stamp_counter} stamps")
                sys.exit()
