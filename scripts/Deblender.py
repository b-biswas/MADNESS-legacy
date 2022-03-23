import numpy as np
import tensorflow as tf
from scripts.FlowVAEnet import FlowVAEnet
from scripts.extraction import extract_cutouts
import tensorflow_probability as tfp
import logging
import time

tfd = tfp.distributions

# logging level set to INFO
logging.basicConfig(format='%(message)s',
                    level=logging.INFO)

LOG = logging.getLogger(__name__)

class Deblend:

    def __init__(self, postage_stamp, detected_positions, cutout_size=59, num_components=1, max_iter=60, lr= .3, latent_dim=10, initZ=None, use_likelihood=True, channel_last=False):
        """
        Parameters
        __________
        postage_stamp: np.ndarray
            input stamp/field that is to be deblended
        detected_positions: as in array and not image
        cutout_size:
        num_components: int
            number of galaxies present in the image.
        max_iter: int
            number of iterations in the deblending step
        lr: float
            learning rate for the gradient descent in the latent space.
        initZ: np.ndarry
            initial value for the latent space
        use_likelihood: bool
            decides whether or not to use the log_prob output of the flow deblender in the optimization. 
        channel_last: bool
            if channel is the last column of the postage_stamp
        """

        self.postage_stamp = postage_stamp
        self.max_iter = max_iter 
        self.lr = lr 
        self.num_components = num_components 
        self.use_likelihood = use_likelihood
        self.components = None
        self.channel_last = channel_last
        self.detected_positions = detected_positions
        self.cutout_size = cutout_size

        self.latent_dim = latent_dim
        self.flow_vae_net = FlowVAEnet(latent_dim=latent_dim)

        self.flow_vae_net.load_flow_weights(weights_path='/pbs/throng/lsst/users/bbiswas/train_debvader/cosmos/updated_cosmos10dim_small_sig/fvae/')
        self.flow_vae_net.load_vae_weights(weights_path='/pbs/throng/lsst/users/bbiswas/train_debvader/cosmos/updated_cosmos10dim_small_sig/deblender/val_loss')

        #self.flow_vae_net.vae_model.trainable = False
        #self.flow_vae_net.flow_model.trainable = False

        #self.flow_vae_net.vae_model.summary()
        self.gradient_decent(initZ)

    def compute_residual(self, reconstructions=None):
        if reconstructions is None:
            reconstructions = self.components
        if self.channel_last:
            residual_field = self.postage_stamp.copy()
        else:
            residual_field = np.transpose(self.postage_stamp, axes = (1,2,0)).copy()

        residual_field = tf.Variable(residual_field, dtype=tf.float32)

        for i in range(self.num_components):
            detected_position = self.detected_positions[i]

            #TODO: make this optional

            #cutout prediction 
            reconstruction = reconstructions[i]
            
            starting_pos_x = round(detected_position[0] - (self.cutout_size-1)/2)
            starting_pos_y = round(detected_position[1] - (self.cutout_size-1)/2)

            indices = np.indices((self.cutout_size, self.cutout_size, tf.shape(reconstruction)[2])).reshape(3, -1).T
            indices[:, 0] += int(starting_pos_x)
            indices[:, 1] += int(starting_pos_y)

            residual_field = tf.tensor_scatter_nd_sub(residual_field, indices, tf.reshape(reconstruction, -1))

        return residual_field

    def gradient_decent(self, optimizer=None, initZ=None):
        """
        perform the gradient descent step to separate components (galaxies)

        Parameters
        ----------
        optimizer: tf.keras.optimizers
            optimizer to be used for hte gradient descent
        initZ: np.ndarray
            initial value of the latent space.
        """
        X = self.postage_stamp
        if not self.channel_last:
            X = np.transpose(X, axes = (1,2,0))

        m, n, b = np.shape(X) 
        
        if initZ is not None: 
            # check constraint parameter over here
            z = tf.Variable(initial_value = initZ, name ='z')

        else:
            # z = tf.Variable(name="z", initial_value=tf.random_normal_initializer(mean=0, stddev=1)(shape=[self.num_components, self.latent_dim], dtype=tf.float32))
            # use the encoder to find a good starting point.
            distances_to_center = list(np.array(self.detected_positions) - int((m-1)/2))
            cutouts = extract_cutouts(X, m, distances_to_center, cutout_size=self.cutout_size, nb_of_bands=b)
            initZ = tfp.layers.MultivariateNormalTriL(self.latent_dim)(self.flow_vae_net.encoder(cutouts))
            LOG.info("Using encoder for initial point")
            z = tf.Variable(initZ.mean())

        optimizer = tf.keras.optimizers.Adam(lr = self.lr)

        sig = tf.math.reduce_std(X)

        LOG.info("\n--- Starting gradient descent in the latent space ---")
        LOG.info("Number of iterations: " + str(self.max_iter))
        LOG.info("Learning rate: " + str(self.lr))
        LOG.info("Number of components: " + str(self.num_components))
        LOG.info("Dimensions of latent space: " + str(self.latent_dim))

        t0 = time.time()
        for i in range(self.max_iter):

            with tf.GradientTape() as tape:
                
                reconstructions = self.flow_vae_net.decoder(z).mean()

                residual_field = self.compute_residual(reconstructions)

                reconstruction_loss = tf.cast(tf.math.reduce_sum(tf.square(residual_field)), tf.float32) / tf.cast(tf.square(sig), tf.float32)

                log_likelihood = tf.cast(tf.math.reduce_sum(self.flow_vae_net.flow(tf.reshape(z,(self.num_components, self.latent_dim)))), tf.float32)
                if self.use_likelihood:
                    loss = tf.math.subtract(reconstruction_loss, log_likelihood)
                else:
                    loss = reconstruction_loss

                sig = tf.math.reduce_std(residual_field)

            #print("log prob flow:" + str(log_likelihood.numpy()))
            #print("reconstruction loss"+str(reconstruction_loss.numpy()))
            #print(loss)
            grad = tape.gradient(loss, [z])
            grads_and_vars=[(grad, [z])]
            optimizer.apply_gradients(zip(grad, [z]))

        LOG.info("--- Gradient descent complete ---")
        LOG.info("\nTime taken for gradient descent: " + str(time.time()-t0))
        self.components = reconstructions.numpy()
