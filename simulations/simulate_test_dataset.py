"""Simulate test dataset."""

import logging
import os
import sys

import btk
import hickle

# logging level set to INFO
logging.basicConfig(format="%(message)s", level=logging.INFO)

LOG = logging.getLogger(__name__)

COSMOS_CATALOG_PATHS = "/sps/lsst/users/bbiswas/OneDegSq_snr_10.fits"

simulation_path = "/sps/lsst/users/bbiswas/simulations/test_data/"

stamp_size = 41
density = sys.argv[1]

if density not in ["high", "low"]:
    raise ValueError("The second arguemnt should be either isolated or blended")

if density == "high":
    min_number = 12
    max_number = 20
else:
    min_number = 8
    max_number = 15

batch_size = 20
maxshift = 15
num_repetations = 300
catalog = btk.catalog.CatsimCatalog.from_file(COSMOS_CATALOG_PATHS)
survey = btk.survey.get_surveys("LSST")
seed = 13

sampling_function = btk.sampling_functions.DefaultSampling(
    max_number=max_number,
    min_number=min_number,
    max_shift=maxshift,
    stamp_size=stamp_size,
    seed=seed,
)

draw_generator = btk.draw_blends.CatsimGenerator(
    catalog,
    sampling_function,
    survey,
    batch_size=batch_size,
    stamp_size=stamp_size,
    cpus=1,
    add_noise="all",
    verbose=False,
    seed=seed,
    augment_data=False,
)

for file_num in range(num_repetations):
    print("Processing file " + str(file_num))
    blend = next(draw_generator)

    save_file_name = os.path.join(
        simulation_path,
        density,
        str(file_num) + ".hkl",
    )
    print(save_file_name)
    hickle.dump(blend, save_file_name, mode="w")
