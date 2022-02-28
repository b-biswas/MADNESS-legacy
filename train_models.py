import tensorflow as tf
from debvader.batch_generator import COSMOSsequence
from debvader.normalize import linear_normalization_cosmos
from scripts.FlowVAEnet import FlowVAEnet
from scripts.utils import listdir_fullpath
import os

# define the parameters
batch_size = 200 
linear_norm = True
generative_epochs = 20
flow_epochs = 20
deblender_epochs =20
latent_dim = 32
num_iter_per_epoch = None

f_net = FlowVAEnet(latent_dim=latent_dim, linear_norm=linear_norm)

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

# Keras Callbacks
path_weights = '/sps/lsst/users/bbiswas/weights/LSST/FlowDeblender/' + 'trail/'

######## List of data samples
def listdir_fullpath(d):
    return [os.path.join(d, f) for f in os.listdir(d) if not f.endswith("metadata.npy")]

datalist = listdir_fullpath("/sps/lsst/users/bbiswas/simulations/COSMOS_btk/")

train_path = datalist[:800]
validation_path = datalist[800:]


######## Define the generators
train_generator = COSMOSsequence(train_path, 'blended_gal_stamps', 'isolated_gal_stamps', 
                                 batch_size=200, num_iterations_per_epoch=400,
                                 normalization_func=linear_normalization_cosmos, use_only_isolated=True)

validation_generator = COSMOSsequence(validation_path, 'blended_gal_stamps', 'isolated_gal_stamps', 
                                 batch_size=200, num_iterations_per_epoch=100, 
                                 normalization_func=linear_normalization_cosmos, use_only_isolated=True)


# Step 1: learn latent space representation of the Galaxies.

######## Define all used callbacks
checkpointer_vae_loss = tf.keras.callbacks.ModelCheckpoint(filepath=path_weights + "vae/" + 'weights_isolated.{epoch:02d}-{val_loss:.2f}.ckpt', monitor='val_loss', verbose=1, save_best_only=True,save_weights_only=True, mode='min', period=1)
# now train the model
f_net.train_vae(train_generator, validation_generator, callbacks=[checkpointer_vae_loss], epochs=generative_epochs, path_weights=path_weights)


# Step 2: optimize the log prob using the flow network

######## Define all used callbacks
checkpointer_flow_loss = tf.keras.callbacks.ModelCheckpoint(filepath=path_weights + "fvae/" + 'weights_isolated.{epoch:02d}-{val_loss:.2f}.ckpt', monitor='val_loss', verbose=1, save_best_only=True,save_weights_only=True, mode='min', period=1)

# now train the model
f_net.train_flow(train_generator, validation_generator, callbacks= [checkpointer_flow_loss], epochs=flow_epochs, path_weights=path_weights)


# Step 3: to find  a good initial guess for the deblender, train the encoder as a deblender.

######## Define the generators
# To deblend, feed blended galaies as input to the model
train_generator = COSMOSsequence(train_path, 'blended_gal_stamps', 'isolated_gal_stamps', 
                                 batch_size=200, num_iterations_per_epoch=400,
                                 normalization_func=linear_normalization_cosmos, use_only_isolated=False)

validation_generator = COSMOSsequence(validation_path, 'blended_gal_stamps', 'isolated_gal_stamps', 
                                 batch_size=200, num_iterations_per_epoch=100, 
                                 normalization_func=linear_normalization_cosmos, use_only_isolated=False)


######## Define all used callbacks
checkpointer_deblender_loss = tf.keras.callbacks.ModelCheckpoint(filepath=path_weights + "deblender/" + 'weights_isolated.{epoch:02d}-{val_loss:.2f}.ckpt', monitor='val_loss', verbose=1, save_best_only=True,save_weights_only=True, mode='min', period=1)
# now train the model
f_net.train_vae(train_generator, validation_generator, train_decoder = False, callbacks=[checkpointer_deblender_loss], epochs=deblender_epochs, path_weights=path_weights)
