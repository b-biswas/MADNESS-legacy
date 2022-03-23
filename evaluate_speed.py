import sys 
sys.path.insert(0,'../')

from scripts.Deblender import Deblend
from scripts.utils import listdir_fullpath
from scripts.utils import norm, denorm

import numpy as np
import os

import matplotlib.pyplot as plt

bands = [4,5,6,7,8,9]

import btk
import btk.plot_utils
import btk.survey
import btk.draw_blends
import btk.catalog
import btk.sampling_functions
import astropy.table

COSMOS_CATALOG_PATHS = [
    "/sps/lsst/users/bbiswas/COSMOS_catalog/COSMOS_25.2_training_sample/real_galaxy_catalog_25.2.fits",
    "/sps/lsst/users/bbiswas/COSMOS_catalog/COSMOS_25.2_training_sample/real_galaxy_catalog_25.2_fits.fits",
]


stamp_size = 70
max_number = 25
batch_size = 1
max_shift = 25
catalog = btk.catalog.CosmosCatalog.from_file(COSMOS_CATALOG_PATHS)
survey = btk.survey.get_surveys("Rubin")
seed=17

import galsim 
galsim_catalog = galsim.COSMOSCatalog(COSMOS_CATALOG_PATHS[0], exclusion_level="marginal")

sampling_function = btk.sampling_functions.DefaultSampling(max_number=max_number, maxshift=max_shift, stamp_size=stamp_size, seed=seed)

draw_generator = btk.draw_blends.CosmosGenerator(
    catalog,
    sampling_function,
    survey,
    batch_size=batch_size,
    stamp_size=stamp_size,
    cpus=1,
    add_noise="all",
    verbose=False,
    gal_type="parametric",
    seed=seed,
)


import time
nb_gal = []
time_per_itr = []
for i in range(20):
    detected_positions = []
    blend = next(draw_generator)

    blend['blend_list'][0]['x_peak']

    np.shape(blend['blend_images'][0])

    for i in range(len(blend['blend_list'][0])):
        detected_positions.append([blend['blend_list'][0]['y_peak'][i], blend['blend_list'][0]['x_peak'][i]])
    print(detected_positions)

    t0 = time.time()
    deb = Deblend(blend['blend_images'][0]/80000, detected_positions, latent_dim=10, num_components=len(blend['blend_list'][0]), use_likelihood=True)
    time_per_itr.append(time.time()-t0)
    nb_gal.append(len(detected_positions))
    print(time.time()-t0)
    print(len(detected_positions))
    del(blend)

print(time_per_itr)
print(nb_gal)

