import tensorflow as tf
from scripts.BatchGenerator import BatchGenerator
from scripts.FlowVAEnet import FlowVAEnet
from scripts.utils import listdir_fullpath
import os

bands = [4,5,6,7,8,9]
batch_size = 200
linear_norm = True
epochs = 5 

f_net = FlowVAEnet(linear_norm=linear_norm)

# Keras Callbacks
path_weights = '/sps/lsst/users/bbiswas/weights/LSST/FlowDeblender/' + 'trial_run/'

#checkpointer_mse = tf.keras.callbacks.ModelCheckpoint(filepath=path_weights+'mse/weights_noisy_v4.{epoch:02d}-{val_mean_squared_error:.2f}.ckpt', monitor='val_mean_squared_error', verbose=1, save_best_only=True,save_weights_only=True, mode='min', period=1)
checkpointer_vae_loss = tf.keras.callbacks.ModelCheckpoint(filepath=path_weights+'weights_noisy_v4.{epoch:02d}-{val_functional_3_loss:.2f}.ckpt', monitor='val_functional_3_loss', verbose=1, save_best_only=True,save_weights_only=True, mode='min', period=1)
checkpointer_flow_loss = tf.keras.callbacks.ModelCheckpoint(filepath=path_weights+'weights_noisy_v4.{epoch:02d}-{val_functional_5_loss:.2f}.ckpt', monitor='val_functional_5_loss', verbose=1, save_best_only=True,save_weights_only=True, mode='min', period=1)

terminate_on_nan = tf.keras.callbacks.TerminateOnNaN()

######## Define all used callbacks


######## List of data samples
images_dir = '/sps/lsst/users/barcelin/data/blended_galaxies/' + '27.5/centered/'
list_of_samples = [x for x in listdir_fullpath(os.path.join(images_dir,'training')) if x.endswith('.npy')]
list_of_samples_val = [x for x in listdir_fullpath(os.path.join(images_dir,'validation')) if x.endswith('.npy')]


######## Define the generators
training_generator = BatchGenerator(bands, list_of_samples, total_sample_size=None,
                                    batch_size=batch_size,
                                    trainval_or_test='training',
                                    do_norm=False,
                                    denorm = False,
                                    linear_norm = linear_norm,
                                    path = os.path.join(images_dir, "test/"),
                                    list_of_weights_e = None)

validation_generator = BatchGenerator(bands, list_of_samples_val, total_sample_size=None,
                                    batch_size=batch_size,
                                    trainval_or_test='validation',
                                    do_norm=False,
                                    denorm = False,
                                    linear_norm = linear_norm,
                                    path = os.path.join(images_dir, "test/"),
                                    list_of_weights_e = None)

f_net.train_vae(training_generator, validation_generator, callbacks=[checkpointer_vae_loss, terminate_on_nan], epochs=epochs)
f_net.train_flow_model(training_generator, validation_generator, callbacks=[checkpointer_flow_loss, terminate_on_nan], epochs=epochs)
