import os
import tensorflow as tf
from random import choice
import h5py as h5py
import tensorflow_datasets as tfds
import numpy as np

_DESCRIPTION = """
#Galaxies from CATSIM WL Deblending catalogue
"""

_CITATION = ""
_URL = "https://github.com/b-biswas/MADNESS"

def Logger(str,verbosity=0):
    """

      Simple Logger
      By default logs .. increase verbosity in case of well .. more verbose logs

      use 
      os.environ["DIFF_TRACE"]="1"
      in your notebook to change verbosity

    """
    if int(os.environ.get("DIFF_TRACE",0)) > verbosity:
        print(str)

class CatsimDataset(tfds.core.GeneratorBasedBuilder):
    """Catsim galaxy dataset"""  

    VERSION = tfds.core.Version("1.0.0")
    RELEASE_NOTES = {'1.0.0': 'Initial release.',}
    MANUAL_DOWNLOAD_INSTRUCTIONS = "Nothing to download. Dataset was generated at first call."

    def __init__(self,
            train_data_dir,
            val_data_dir,
            **kwargs):
        super(CatsimDataset,self).__init__(**kwargs)
        self.train_data_dir = train_data_dir
        self.val_data_dir = val_data_dir


    def PopulateFileList(self, fit_path):
        

        list_of_images=[]
        for root, dirs, files in os.walk(fit_path, topdown=False):
            for name in files:
                file_path = os.path.join(root, name)
                if os.path.splitext(file_path)[1] != ".npy": 
                    continue
                list_of_images.append(file_path)
        # Make sure that some data exists
        assert len(list_of_images) > 0

        Logger("File list is populated .. there is {0} files".format(len(list_of_images)))

        return list_of_images

    def _info(self) -> tfds.core.DatasetInfo:
        """Returns the dataset metadata."""
        N_TIMESTEPS = 280
        return tfds.core.DatasetInfo(
            builder=self,
            description=_DESCRIPTION,
            homepage=_URL,
            citation=_CITATION,
            features=tfds.features.FeaturesDict({
                "isolated_gal_stamps": tfds.features.Tensor(shape=(45, 45, 6), dtype=tf.float32),
                "blended_gal_stamps": tfds.features.Tensor(shape=(45, 45, 6), dtype=tf.dtypes.float32),
            }),
        )

    def _split_generators(self, dl):
        """Returns generators according to split"""
        return {tfds.Split.TRAIN: self._generate_examples(self.train_data_dir),
                tfds.Split.VALIDATION: self._generate_examples(self.val_data_dir)}

    def _generate_examples(self, data_folder):
        """Yields examples."""

        # Only populated the first time
        list_of_images = self.PopulateFileList(data_folder)

        for gal_file in list_of_images:
            # very verbose
            Logger("File : {0} is being treated".format(gal_file), 2)
            current_sample = np.load(gal_file, allow_pickle=True)
            key = os.path.splitext(gal_file)[0]
            example = {}
            example["isolated_gal_stamps"] = current_sample["isolated_gal_stamps"][0].astype('float32')
            example["blended_gal_stamps"] = current_sample["blended_gal_stamps"][0].astype('float32')
            yield key , example

def loadCATSIMDataset(
            train_data_dir, 
            val_data_dir,
            output_dir,
        ):

    arg_dict = {
        'train_data_dir': train_data_dir,
        'val_data_dir': val_data_dir,
    }
    #subsets = [tfds.Split.TRAIN,tfds.Split.VALIDATION]
    ds = tfds.load('CatsimDataset', data_dir=output_dir, builder_kwargs=arg_dict)

    return ds

def batched_CATSIMDataset(
            tf_dataset_dir,
            linear_norm_coeff,
            batch_size,
            x_col_name="blended_gal_stamps", 
            y_col_name="isolated_gal_stamps"
        ):

    # normalized train and val dataset generator
    def preprocess_batch(ds, linear_norm_coeff, x_col_name, y_col_name):

        def pre_process(galaxy):
            """ Pre-processing function preparing data for denoising task
            """
            # Cutout a portion of the map
            x = galaxy[x_col_name]/linear_norm_coeff
            y = galaxy[y_col_name]/linear_norm_coeff

            do_flip_lr = tf.random.uniform([]) > 0.5
            if do_flip_lr:
                x = tf.image.flip_left_right(x)
                y = tf.image.flip_left_right(y)

            do_flip_ud = tf.random.uniform([]) > 0.5
            if do_flip_ud:
                x = tf.image.flip_up_down(x)
                y = tf.image.flip_up_down(y)

            return (x, y)

        # ds = ds.repeat(2)
        ds = ds.shuffle(buffer_size=15*batch_size)
        ds = ds.batch(batch_size)
        ds = ds.map(pre_process)
        ds = ds.prefetch(tf.data.experimental.AUTOTUNE)
        return ds
    
    ds = loadCATSIMDataset(
        train_data_dir=None,
        val_data_dir=None,
        output_dir=tf_dataset_dir,
    )

    ds_train = preprocess_batch(ds=ds[tfds.Split.TRAIN], 
                                linear_norm_coeff=linear_norm_coeff,
                                x_col_name=x_col_name, 
                                y_col_name=y_col_name,
                                )
    ds_val = preprocess_batch(ds=ds[tfds.Split.VALIDATION], 
                              linear_norm_coeff=linear_norm_coeff,
                              x_col_name=x_col_name,y_col_name=y_col_name)

    return ds_train, ds_val