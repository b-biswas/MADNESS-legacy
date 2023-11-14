"""Define utils."""
import os


def get_btksims_config_path():
    """Fetch path to maddeb config yaml file.

    Returns
    -------
    data_dir: str
        path to data folder

    """
    curdir = os.path.dirname(os.path.abspath(__file__))
    get_btksims_config_path = os.path.join(curdir, "btksims_config.yaml")

    return get_btksims_config_path
