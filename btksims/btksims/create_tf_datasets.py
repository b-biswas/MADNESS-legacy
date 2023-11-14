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

loadCATSIMDataset(
    train_data_dir=os.path.join(
        btksims_config["TRAIN_DATA_SAVE_PATH"],
        "blended_training",
    ),
    val_data_dir=os.path.join(
        btksims_config["TRAIN_DATA_SAVE_PATH"],
        "blended_validation",
    ),
    output_dir=os.path.join(maddeb_config["TF_DATASET_PATH"], "blended_tfDataset"),
)

loadCATSIMDataset(
    train_data_dir=os.path.join(
        btksims_config["TRAIN_DATA_SAVE_PATH"],
        "isolated_training",
    ),
    val_data_dir=os.path.join(
        btksims_config["TRAIN_DATA_SAVE_PATH"],
        "isolated_validation",
    ),
    output_dir=os.path.join(
        maddeb_config["TF_DATASET_PATH"],
        "isolated_tfDataset",
    ),
)
