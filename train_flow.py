import tensorflow as tf
from scripts.BatchGenerator import BatchGenerator
from scripts.FlowVAEnet import FlowVAEnet
from scripts.utils import listdir_fullpath
import os

# define the parameters
bands = [4,5,6,7,8,9] # corresponds to LSST bands in the simulation
batch_size = 200 
linear_norm = True
flow_epochs = 20
latent_dim = 32
num_iter_per_epoch = None

f_net = FlowVAEnet(latent_dim=latent_dim, linear_norm=linear_norm)

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

# Keras Callbacks
path_weights = '/sps/lsst/users/bbiswas/weights/LSST/FlowDeblender/' + 'separated_architecture/'

######## List of data samples
images_dir = '/sps/lsst/users/barcelin/data/blended_galaxies/' + '27.5/centered/'
list_of_samples = [x for x in listdir_fullpath(os.path.join(images_dir,'training')) if x.endswith('.npy')]
list_of_samples_val = [x for x in listdir_fullpath(os.path.join(images_dir,'validation')) if x.endswith('.npy')]

######## Define the generators
training_generator_flow = BatchGenerator(bands, list_of_samples, total_sample_size=None,
                                    batch_size=batch_size,
                                    trainval_or_test='training',
                                    do_norm=False,
                                    denorm=False,
                                    linear_norm=linear_norm,
                                    path=os.path.join(images_dir, "test/"),
                                    list_of_weights_e=None,
                                    num_iter_per_epoch=num_iter_per_epoch,
                                    blended_and_isolated=False)

validation_generator_flow = BatchGenerator(bands, list_of_samples_val, total_sample_size=None,
                                    batch_size=batch_size,
                                    trainval_or_test='validation',
                                    do_norm=False,
                                    denorm=False,
                                    linear_norm=linear_norm,
                                    path=os.path.join(images_dir, "test/"),
                                    list_of_weights_e=None,
                                    num_iter_per_epoch=num_iter_per_epoch,
                                    blended_and_isolated=False)

# load the vae weights for encoder 
f_net.load_vae_weights(str(path_weights + "vae/"))
#f_net.load_flow_weights(str(path_weights + "fvae/"))

######## Define all used callbacks
checkpointer_flow_loss = tf.keras.callbacks.ModelCheckpoint(filepath=path_weights + "fvae/" + 'weights_isolated.{epoch:02d}-{val_loss:.2f}.ckpt', monitor='val_loss', verbose=1, save_best_only=True,save_weights_only=True, mode='min', period=1)

# now train the model
f_net.train_flow(training_generator_flow, validation_generator_flow, callbacks= [checkpointer_flow_loss], epochs=flow_epochs, path_weights=path_weights)
