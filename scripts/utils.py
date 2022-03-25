import os


def listdir_fullpath(d):
    return [os.path.join(d, f) for f in os.listdir(d)]
