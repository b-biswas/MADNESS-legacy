import os


def listdir_fullpath(d):
    return [os.path.join(d, f) for f in os.listdir(d)]

def get_data_dir_path():
    """Function to return path to the data folder of kndetect
    Returns
    -------
    data_dir: str
        path to data folder
    """
    curdir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(curdir, "data")

    return data_dir
