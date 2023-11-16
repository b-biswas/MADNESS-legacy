"""Simulate test dataset."""

import logging
import os
import sys

import btk
import hickle
import yaml

from btksims.sampling import CustomSampling
from btksims.utils import get_btksims_config_path

# logging level set to INFO
logging.basicConfig(format="%(message)s", level=logging.INFO)

LOG = logging.getLogger(__name__)

density = sys.argv[1]

if density not in ["high", "low"]:
    raise ValueError("The first arguemnt should be either high or low")

with open(get_btksims_config_path()) as f:
    btksims_config = yaml.safe_load(f)

survey = btk.survey.get_surveys(btksims_config["survey_name"])
simulation_path = btksims_config["TEST_DATA_SAVE_PATH"][btksims_config["survey_name"]]
CATALOG_PATH = btksims_config["CAT_PATH"][btksims_config["survey_name"]]

if density == "high":
    sim_config = btksims_config["TEST_PARAMS"]

catalog = btk.catalog.CatsimCatalog.from_file(CATALOG_PATH)

index_range = [sim_config["index_start"], len(catalog.table)]
sampling_function = CustomSampling(
    index_range=index_range,
    min_number=sim_config[density + "_density"]["min_number"],
    max_number=sim_config[density + "_density"]["max_number"],
    maxshift=sim_config["maxshift"],
    stamp_size=sim_config["stamp_size"],
    seed=sim_config["btk_seed"],
    unique=sim_config["unique_galaxies"],
)

draw_generator = btk.draw_blends.CatsimGenerator(
    catalog,
    sampling_function,
    survey,
    batch_size=sim_config["batch_size"],
    stamp_size=sim_config["stamp_size"],
    cpus=1,
    add_noise="all",
    verbose=False,
    seed=sim_config["btk_seed"],
    augment_data=False,
)

for file_num in range(sim_config["num_repetations"]):
    print("Processing file " + str(file_num))
    blend = next(draw_generator)

    save_file_name = os.path.join(
        simulation_path,
        density,
        str(file_num) + ".hkl",
    )
    print(save_file_name)
    hickle.dump(blend, save_file_name, mode="w")
