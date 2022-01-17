from scripts.Deblender import Deblend
from scripts.utils import listdir_fullpath
from scripts.utils import norm, denorm, convert_to_linear_norm

import numpy as np
import os

import matplotlib.pyplot as plt

bands = [4,5,6,7,8,9]

######## List of data samples
images_dir = '/sps/lsst/users/barcelin/data/isolated_galaxies/' + '27.5/centered/'
list_of_samples = [x for x in listdir_fullpath(os.path.join(images_dir,'training')) if x.endswith('.npy')]
list_of_samples_val = [x for x in listdir_fullpath(os.path.join(images_dir,'validation')) if x.endswith('.npy')]

images = np.load(list_of_samples_val[0])

images = images[:200]

weights_path = '/sps/lsst/users/barcelin/data/isolated_galaxies/' + '27.5/centered/test' 
print(np.shape(images[:, 1, 4:]))
 
images_noisy = convert_to_linear_norm(images[:, 1, 4:])
images_noisy = np.transpose(images_noisy, axes=(0, 2, 3, 1))

images_isolated = convert_to_linear_norm(images[:, 0, 4:])
images_isolated = np.transpose(images_isolated, axes=(0, 2, 3, 1))

#Fix norm over here
num_images_to_denoise = 10
shuffled_indices = np.arange(200)
np.random.shuffle(shuffled_indices)

print(shuffled_indices)
fig, ax = plt.subplots(num_images_to_denoise, 5, figsize=(20,5*num_images_to_denoise))

for i in range(num_images_to_denoise):
    print("image number: " + str(i))
    image_index = shuffled_indices[i]

    deblender = Deblend(images_noisy[image_index], channel_last=True)
    residual = images_noisy[image_index] - deblender.components

    im1 = ax[i, 0].imshow(images_noisy[image_index, :, :, 2])
    fig.colorbar(im1, ax=ax[i, 0])
    ax[i, 0].set_title("noisy image")

    im2 = ax[i, 1].imshow(images_isolated[image_index, :, :, 2])
    fig.colorbar(im2, ax=ax[i, 1])
    ax[i, 1].set_title("original isolated")

    im3 = ax[i, 2].imshow(deblender.components[: ,:, 2])
    fig.colorbar(im3, ax=ax[i, 2])
    ax[i, 2].set_title("prediction")

    im4 = ax[i, 3].imshow(residual[: ,:, 2])
    fig.colorbar(im4, ax=ax[i, 3])
    ax[i, 3].set_title("predicted residual")

    im5 = ax[i, 4].imshow(images_noisy[image_index, : ,:, 2] - images_isolated[image_index, :, :, 2])
    fig.colorbar(im5, ax=ax[i, 4])
    ax[i, 4].set_title("actual residual")

plt.savefig("result")
