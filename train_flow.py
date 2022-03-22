import tensorflow as tf
from debvader.batch_generator import COSMOSsequence
from debvader.normalize import LinearNormCosmos
from scripts.FlowVAEnet import FlowVAEnet
from scripts.utils import listdir_fullpath
import os

# define the parameters
batch_size = 200
linear_norm = True
flow_epochs = 200
latent_dim = 10
num_iter_per_epoch = None

f_net = FlowVAEnet(latent_dim=latent_dim, linear_norm=linear_norm)

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

# Keras Callbacks
path_weights = '/pbs/throng/lsst/users/bbiswas/train_debvader/cosmos/updated_cosmos10dim_small_sig/'

######## List of data samples
def listdir_fullpath(d):
    return [os.path.join(d, f) for f in os.listdir(d) if not f.endswith("metadata.npy")]

datalist = listdir_fullpath("/sps/lsst/users/bbiswas/simulations/COSMOS_btk/")

train_path = datalist[:800]
validation_path = datalist[800:]

######## Define the generators
train_generator = COSMOSsequence(train_path, 'isolated_gal_stamps', 'isolated_gal_stamps', 
                                 batch_size=batch_size, num_iterations_per_epoch=400,
                                 normalizer=LinearNormCosmos())

validation_generator = COSMOSsequence(validation_path, 'isolated_gal_stamps', 'isolated_gal_stamps', 
                                 batch_size=batch_size, num_iterations_per_epoch=100, 
                                 normalizer=LinearNormCosmos())

# load the vae weights for encoder 
f_net.load_vae_weights(os.path.join(path_weights, "vae/val_loss"))
#f_net.load_flow_weights(os.path.join(path_weights, "fvae"))

######## Define all used callbacks
checkpointer_flow_loss = tf.keras.callbacks.ModelCheckpoint(filepath=os.path.join(path_weights, "fvae", "weights_isolated.ckpt"), monitor='val_loss', verbose=1, save_best_only=True,save_weights_only=True, mode='min', period=1)

# now train the model
f_net.train_flow(train_generator, validation_generator, callbacks=[checkpointer_flow_loss], epochs=flow_epochs)
