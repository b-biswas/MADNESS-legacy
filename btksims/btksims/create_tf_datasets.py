"""First call to generate tf datasets."""
import os

import yaml

from btksims.utils import get_btksims_config_path
from maddeb.dataset_generator import loadCATSIMDataset
from maddeb.utils import get_maddeb_config_path

with open(get_btksims_config_path()) as f:
    btksims_config = yaml.safe_load(f)

with open(get_maddeb_config_path()) as f:
    maddeb_config = yaml.safe_load(f)

if maddeb_config["survey_name"] != btksims_config["survey_name"]:
    raise ValueError("survey_name in config files of maddeb and btksims must match")

survey_name = maddeb_config["survey_name"]

loadCATSIMDataset(
    train_data_dir=os.path.join(
        btksims_config["TRAIN_DATA_SAVE_PATH"][survey_name],
        "blended_training",
    ),
    val_data_dir=os.path.join(
        btksims_config["TRAIN_DATA_SAVE_PATH"][survey_name],
        "blended_validation",
    ),
    output_dir=os.path.join(
        maddeb_config["TF_DATASET_PATH"][survey_name], "blended_tfDataset"
    ),
)

loadCATSIMDataset(
    train_data_dir=os.path.join(
        btksims_config["TRAIN_DATA_SAVE_PATH"][survey_name],
        "isolated_training",
    ),
    val_data_dir=os.path.join(
        btksims_config["TRAIN_DATA_SAVE_PATH"][survey_name],
        "isolated_validation",
    ),
    output_dir=os.path.join(
        maddeb_config["TF_DATASET_PATH"][survey_name],
        "isolated_tfDataset",
    ),
)
