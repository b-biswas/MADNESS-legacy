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
import yaml

from btksims.sampling import CustomSampling
from btksims.utils import get_btksims_config_path
from maddeb.extraction import extract_cutouts

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

with open(get_btksims_config_path()) as f:
    btksims_config = yaml.safe_load(f)

if blend_type == "isolated":
    sim_config = btksims_config["ISOLATED_SIM_PARAMS"]
else:
    sim_config = btksims_config["BLENDED_SIM_PARAMS"]

CATSIM_CATALOG_PATH = btksims_config.CAT_PATH
SAVE_PATH = btksims_config["TRAIN_DATA_SAVE_PATH"]
print("saving data at " + SAVE_PATH)

if dataset == "training":
    index_range = [sim_config["train_index_start"], sim_config["train_index_end"]]
    num_batches = sim_config["train_num_batches"]
elif dataset == "validation":
    index_range = [sim_config["val_index_start"], sim_config["val_index_end"]]
    num_batches = sim_config["val_num_batches"]

catalog = btk.catalog.CatsimCatalog.from_file(CATSIM_CATALOG_PATH)
survey = btk.survey.get_surveys(sim_config["survey_name"])

sampling_function = CustomSampling(
    index_range=index_range,
    max_number=sim_config["max_number"],
    maxshift=sim_config["maxshift"],
    stamp_size=sim_config["stamp_size"],
    seed=sim_config["btk_seed"],
    unique=sim_config["unique_galaxies"].lower() == "true",
)

draw_generator = btk.draw_blends.CatsimGenerator(
    catalog,
    sampling_function,
    survey,
    batch_size=sim_config["btk_batch_size"],
    stamp_size=sim_config["stamp_size"],
    cpus=4,
    add_noise="all",
    verbose=False,
    seed=sim_config["btk_seed"],
    augment_data=False,
)

total_galaxy_stamps = num_batches * sim_config["btk_batch_size"]
stamp_counter = 0
# shift_rng = np.random.default_rng(12345)
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
