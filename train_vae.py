import tensorflow as tf
from debvader.batch_generator import COSMOSsequence
from debvader.normalize import LinearNormCosmos
from scripts.FlowVAEnet import FlowVAEnet
from scripts.utils import listdir_fullpath
import os

# define the parameters
batch_size = 200
linear_norm = True
generative_epochs = 100
vae_epochs = 100
deblender_epochs = 100
latent_dim = 32
num_iter_per_epoch = None

f_net = FlowVAEnet(latent_dim=latent_dim, linear_norm=linear_norm)

# Keras Callbacks
path_weights = '/sps/lsst/users/bbiswas/weights/LSST/FlowDeblender/' + 'separated_architecture/'

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

f_net.load_vae_weights(weights_path='/pbs/throng/lsst/users/bbiswas/train_debvader/cosmos/2step_scheduled_lr/deblender/val_loss')

######## Define all used callbacks
checkpointer_vae_loss = tf.keras.callbacks.ModelCheckpoint(filepath=path_weights + "vae/" + 'weights_isolated.{epoch:02d}-{val_loss:.2f}.ckpt', monitor='val_loss', verbose=1, save_best_only=True,save_weights_only=True, mode='min', period=1)
f_net.train_vae(train_generator, validation_generator, callbacks=[checkpointer_vae_loss], epochs=vae_epochs)
