# Import packages
import numpy as np
import pathlib
from pathlib import Path
import os



def listdir_fullpath(d):
    return [os.path.join(d, f) for f in os.listdir(d)]

############## NORMALIZATION OF IMAGES
# Value used for normalization
beta = 2.5

def norm(x, bands, path, channel_last=False, inplace=True, linear_norm=False):
    '''
    Return image x normalized

    Parameters:
    -----------
    x: image to normalize
    bands: filter number
    path: path to the normalization constants
    channel_last: is the channels (filters) in last in the array shape
    inplace: boolean: change the value of array itself
    linear_norm: boolean: if normalization is to be linear
    '''
    if not inplace:
        y = np.copy(x)
    else:
        y = x
    if linear_norm:
        return y/80000

    while bands[0]>9:
        bands = np.array(bands)-10
    full_path = pathlib.PurePath(path)
    isolated_or_blended = full_path.parts[6][0:len(full_path.parts[6])-9]

    test_dir = str(Path(path).parents[0])+'/test/'
    I = np.load(test_dir+'galaxies_'+isolated_or_blended+'_20191024_0_I_norm.npy', mmap_mode = 'c')

    if channel_last:
        assert y.shape[-1] == len(bands)
        for i in range (len(y)):
            for ib, b in enumerate(bands):
                y[i,:,:,ib] = np.tanh(np.arcsinh(y[i,:,:,ib]/(I[b]/beta)))
    else:
        assert y.shape[1] == len(bands)
        for i in range (len(y)):
            for ib, b in enumerate(bands):
                y[i,ib] = np.tanh(np.arcsinh(y[i,ib]/(I[b]/beta)))
    return y

def denorm(x, bands, path, channel_last=False, inplace=True, linear_norm=False):
    '''
    Return image x denormalized

    Parameters:
    ----------
    x: image to denormalize
    bands: filter number
    path: path to the normalization constants
    channel_last: is the channels (filters) in last in the array shape
    inplace: boolean: change the value of array itself
    linear_norm: boolean: if normalization is to be linear
    '''
    if not inplace:
        y = np.copy(x)
    else:
        y = x

    if linear_norm:
        return y*80000

    while bands[0]>9:
        bands = np.array(bands)-10
    full_path = pathlib.PurePath(path)
    isolated_or_blended = full_path.parts[6][0:len(full_path.parts[6])-9]
    #print(isolated_or_blended)
    test_dir = str(Path(path).parents[0])+'/test/'
    I = np.load(test_dir+'galaxies_'+isolated_or_blended+'_20191024_0_I_norm.npy', mmap_mode = 'c')#I = np.concatenate([I_euclid,n_years*I_lsst])
    
    if channel_last:
        print(y.shape)
        assert y.shape[-1] == len(bands)
        for i in range (len(y)):
            for ib, b in enumerate(bands):
                y[i,:,:,ib] = np.sinh(np.arctanh(y[i,:,:,ib]))*(I[b]/beta)
    else:
        assert y.shape[1] == len(bands)
        for i in range (len(y)):
            for ib, b in enumerate(bands):
                y[i,ib] = np.sinh(np.arctanh(x[i,ib]))*(I[b]/beta)
    return y
    