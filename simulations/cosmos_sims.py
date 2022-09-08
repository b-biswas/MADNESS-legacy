import btk
import btk.catalog
import btk.draw_blends
import btk.plot_utils
import btk.sampling_functions
import btk.survey
import numpy as np
import pandas as pd

from maddeb.extraction import extract_cutouts
from maddeb.utils import CustomSampling

seed = 993

COSMOS_CATALOG_PATHS = [
    "/sps/lsst/users/bbiswas/COSMOS_catalog/COSMOS_25.2_training_sample/real_galaxy_catalog_25.2.fits",
    "/sps/lsst/users/bbiswas/COSMOS_catalog/COSMOS_25.2_training_sample/real_galaxy_catalog_25.2_fits.fits",
]

stamp_size = 15
max_number = 1  # set to 4 to generate blended scenes
batch_size = 200
maxshift = 1.5

dataset = "validation"  # either training or validation
if dataset == "training":
    index_range = [0, 50000]
    num_files = 400
elif dataset == "validation":
    index_range = [50000, 60000]
    num_files = 100

catalog = btk.catalog.CosmosCatalog.from_file(COSMOS_CATALOG_PATHS)
survey = btk.survey.get_surveys("LSST")

sampling_function = CustomSampling(
    index_range=index_range,
    max_number=max_number,
    maxshift=maxshift,
    stamp_size=stamp_size,
    seed=seed,
)

draw_generator = btk.draw_blends.CosmosGenerator(
    catalog,
    sampling_function,
    survey,
    batch_size=batch_size,
    stamp_size=stamp_size,
    cpus=8,
    add_noise="all",
    verbose=False,
    seed=seed,
    gal_type="parametric",
)


for file_num in range(num_files):

    print("simulating file number:" + str(file_num))

    postage_stamps = {
        "blended_gal_stamps": [],
        "isolated_gal_stamps": [],
        "btk_index": [],
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
            pos = (x_pos, y_pos)
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
            postage_stamps["btk_index"].append(
                [batch["blend_list"][blended_image_num]["btk_index"][galaxy_num]]
            )
            # plt.subplot(121)
            # plt.imshow(gal_blended)

            # plt.subplot(122)
            # plt.imshow(gal_isolated)

            # plt.show()

        # meta_data.append(batch['blend_list'][blended_image_num])

    if max_number == 1:
        np.save(
            "/sps/lsst/users/bbiswas/simulations/COSMOS_btk_isolated_"
            + dataset
            + "/batch"
            + str(file_num + 1)
            + ".npy",
            pd.DataFrame(postage_stamps).to_records(),
        )
    else:
        np.save(
            "/sps/lsst/users/bbiswas/simulations/COSMOS_btk_blended_"
            + dataset
            + "/batch"
            + str(file_num + 1)
            + ".npy",
            pd.DataFrame(postage_stamps).to_records(),
        )
    del batch
    del postage_stamps
    del blended_image
