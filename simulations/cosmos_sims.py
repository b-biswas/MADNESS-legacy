import sys
import os 

import btk
import btk.catalog
import btk.draw_blends
import btk.plot_utils
import btk.sampling_functions
import btk.survey
import numpy as np
import pandas as pd

from maddeb.extraction import extract_cutouts
from maddeb.utils import CustomSampling, CustomUniformSampling

print(sys.argv)
dataset = sys.argv[1] # should be either training or validation
if dataset not in ["training", "validation"]:
    raise ValueError("The first arguement (dataset) should be either training or validation")


blend_type = sys.argv[2]  # set to 4 to generate blended scenes
if blend_type not in ["isolated", "blended"]:
    raise ValueError("The second arguemnt should be either isolated or blended")

if blend_type == "isolated":
    max_number = 1
    batch_size = 200
else:
    max_number = 4
    batch_size = 100

seed = 993

CATSIM_CATALOG_PATH = "/sps/lsst/users/bbiswas/OneDegSq_snr_10.fits"

stamp_size = 15
maxshift = 1.5

sky_level_factor = 1.0

if dataset == "training":
    index_range = [0, 150000]
    num_files = 1000
elif dataset == "validation":
    index_range = [150000, 200000]
    num_files = 400

catalog = btk.catalog.CatsimCatalog.from_file(CATSIM_CATALOG_PATH)
survey = btk.survey.get_surveys("LSST")

sampling_function = CustomSampling(
    index_range=index_range,
    max_number=max_number,
    maxshift=maxshift,
    stamp_size=stamp_size,
    seed=seed,
)

draw_generator = btk.draw_blends.CatsimGenerator(
    catalog,
    sampling_function,
    survey,
    batch_size=batch_size,
    stamp_size=stamp_size,
    cpus=4,
    add_noise="all",
    verbose=False,
    seed=seed,
    augment_data=True,
    sky_level_factor=sky_level_factor,
)

for file_num in range(num_files):

    print("simulating file number:" + str(file_num))

    # postage_stamps = {
    #     "blended_gal_stamps": [],
    #     "isolated_gal_stamps": [],
    #     "btk_index": [],
    # }

    postage_stamps = {
        "blended_gal_stamps": [],
        "isolated_gal_stamps": [],
    }

    # meta_data = []

    batch = next(draw_generator)

    for blended_image_num in range(len(batch["blend_images"])):

        blended_image = batch["blend_images"][blended_image_num]

        for galaxy_num in range(len(batch["blend_list"][blended_image_num]["x_peak"])):

            # print("Image number "+ str(blended_image_num))
            # print("Galaxy number " + str(galaxy_num))
            isolated_image = batch["isolated_images"][blended_image_num][galaxy_num]
            x_pos = batch["blend_list"][blended_image_num]["y_peak"][galaxy_num]
            y_pos = batch["blend_list"][blended_image_num]["x_peak"][galaxy_num]
            shift_rng = np.random.default_rng(12345)
            x_shift = shift_rng.random() - .5
            y_shift = shift_rng.random() - .5
            pos = (x_pos+x_shift, y_pos+y_shift)
            gal_blended = extract_cutouts(
                blended_image,
                [pos],
                distances_to_center=False,
                channel_last=False,
                cutout_size=45,
            )[0][0]
            postage_stamps["blended_gal_stamps"].append(gal_blended)
            gal_isolated = extract_cutouts(
                isolated_image,
                [pos],
                distances_to_center=False,
                channel_last=False,
                cutout_size=45,
            )[0][0]
            postage_stamps["isolated_gal_stamps"].append(gal_isolated)
            # postage_stamps["btk_index"].append(
            #     [batch["blend_list"][blended_image_num]["btk_index"][galaxy_num]]
            # )
            # plt.subplot(121)
            # plt.imshow(gal_blended)

            # plt.subplot(122)
            # plt.imshow(gal_isolated)

            # plt.show()

        # meta_data.append(batch['blend_list'][blended_image_num])


    np.save(
        os.path.join(
            "/sps/lsst/users/bbiswas/simulations/CATSIM_10_btk_shifted_" + blend_type + "_" + dataset,
            "batch" + str(file_num + 1) + ".npy",
        ),
        pd.DataFrame(postage_stamps).to_records(),
    )

    print("saved to /sps/lsst/users/bbiswas/simulations/CATSIM_10_btk_shifted" + blend_type + "_" + dataset)
