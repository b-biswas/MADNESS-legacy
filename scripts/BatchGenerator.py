
import numpy as np
import tensorflow as tf
import scripts.utils as utils

class BatchGenerator(tf.keras.utils.Sequence):
    """
    Class to create batch generator for VAEs.
    """
    def __init__(self, bands, list_of_samples, total_sample_size, batch_size, trainval_or_test, do_norm, denorm, path, blended_and_isolated, num_iter_per_epoch=None, list_of_weights_e=None, linear_norm = False):
        """
        Initialization function
        bands: filters to use for the input and target images
        list_of_samples: list of the numpy arrays which correspond to the whole training (or validation) sample
        total_sample_size: size of the whole training (or validation) sample
        batch_size: size of the batches to provide
        trainval_or_test: choice between training, validation or test generator
        denorm: boolean to denormalize images
        path: path to the normalization values
        list_of_weights_e: numpy array of the weights to use to weight inputs
        linear_norm: option to do linear nromalization of input data
        """
        self.bands = bands
        self.nbands = len(bands)
        self.total_sample_size = total_sample_size
        self.batch_size = batch_size
        self.list_of_samples = list_of_samples
        self.trainval_or_test = trainval_or_test
        self.path = path

        self.epoch = 0
        self.do_norm = do_norm
        self.denorm = denorm
        self.linear_norm = linear_norm

        self.num_iter_per_epoch = num_iter_per_epoch
        self.blended_and_isolated = blended_and_isolated

        # Weights computed from the lengths of lists
        self.p = []
        for sample in self.list_of_samples:
            temp = np.load(sample, mmap_mode = 'c')
            self.p.append(float(len(temp)))
        self.p = np.array(self.p)
        self.total_sample_size = int(np.sum(self.p))
        print("[BatchGenerator] total_sample_size = ", self.total_sample_size)
        print("[BatchGenerator] len(list_of_samples) = ", len(self.list_of_samples))

        self.p /= np.sum(self.p)

        self.produced_samples = 0
        self.list_of_weights_e = list_of_weights_e

    def __len__(self):
        """
        Function to define the length of an epoch
        """
        #return int(float(self.total_sample_size) / float(self.batch_size))      
        if self.num_iter_per_epoch is None:
            self.num_iter_per_epoch = int(float(self.total_sample_size) / float(self.batch_size))
        return self.num_iter_per_epoch

    def on_epoch_end(self):
        """
        Function executed at the end of each epoch
        """
        print("Produced samples", self.produced_samples)
        self.produced_samples = 0
        
    def __getitem__(self, idx):
        """
        Function which returns the input and target batches for the network
        """
        # If the generator is a training generator, the whole sample is displayed
        index = np.random.choice(list(range(len(self.p))), p=self.p)
        sample_filename = self.list_of_samples[index]
        sample = np.load(sample_filename, mmap_mode = 'c')

        # If needed, possibility to apply weights to inputs for training.
        if self.list_of_weights_e == None:
            indices = np.random.choice(len(sample), size=self.batch_size, replace=False)
        else:
            self.weights_e = np.load(self.list_of_weights_e[index])
            indices = np.random.choice(len(sample), size=self.batch_size, replace=False, p = self.weights_e/np.sum(self.weights_e))

        self.produced_samples += len(indices)

        x = sample[indices,1][:,self.bands]
        y = sample[indices,0][:,self.bands]
        
        # Preprocessing of the data to be easier for the network to learn
        if self.do_norm:
            x = utils.norm(x, self.bands, self.path)
            y = utils.norm(y, self.bands, self.path)

        if self.denorm or self.linear_norm:
            x = utils.denorm(x, self.bands, self.path)
            y = utils.denorm(y, self.bands, self.path)

            if self.linear_norm:
                x = utils.norm(x, [], None, linear_norm=True)
                y = utils.norm(y, [], None, linear_norm=True)


        #  flip : flipping the image array
        rand = np.random.randint(4)
        if rand == 1: 
            x = np.flip(x, axis=-1)
            y = np.flip(y, axis=-1)
        elif rand == 2 : 
            x = np.swapaxes(x, -1, -2)
            y = np.swapaxes(y, -1, -2)
        elif rand == 3:
            x = np.swapaxes(np.flip(x, axis=-1), -1, -2)
            y = np.swapaxes(np.flip(y, axis=-1), -1, -2)
        
        # Change the shape of inputs and targets to feed the network
        x = np.transpose(x, axes = (0,2,3,1))
        y = np.transpose(y, axes = (0,2,3,1))
        
        if self.trainval_or_test == 'training' or self.trainval_or_test == 'validation':
            if self.blended_and_isolated:
                print("the shape is ")
                print(np.concatenate((x,y)).shape)
                return np.concatenate((x,y)), np.concatenate((x,y))
            else:
                return y,y

            