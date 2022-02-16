import numpy as np
import tensorflow as tf
from scripts.FlowVAEnet import FlowVAEnet
import tensorflow_probability as tfp

tfd = tfp.distributions

class Deblend:

    def __init__(self, postage_stamp, detected_positions, cutout_size=64, num_components=1, max_iter=150, lr= .05, initZ=None, use_likelihood=True, channel_last=False):
        """
        Parameters
        __________
        postage_stamp: np.ndarray
            input stamp/field that is to be deblended
        detected_positions:
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

        self.flow_vae_net = FlowVAEnet()

        self.flow_vae_net.load_vae_weights('/sps/lsst/users/bbiswas/weights/LSST/FlowDeblender/separated_architecture/vae/')
        self.flow_vae_net.load_flow_weights('/sps/lsst/users/bbiswas/weights/LSST/FlowDeblender/separated_architecture/fvae/')

        #self.flow_vae_net.vae_model.trainable = False
        #self.flow_vae_net.flow_model.trainable = False

        #self.flow_vae_net.vae_model.summary()
        self.gradient_decent(initZ)

    def compute_residual(self, reconstructions):
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
            
            starting_pos_y = detected_position[0] - (self.cutout_size-1)/2
            ending_pos_y = detected_position[0] + (self.cutout_size-1)/2

            starting_pos_x = detected_position[1] - (self.cutout_size-1)/2
            ending_pos_x = detected_position[1] + (self.cutout_size-1)/2

            residual_field[int(starting_pos_x) : int(ending_pos_x) + 1, int(starting_pos_y) : int(ending_pos_y) + 1] = tf.subtract(residual_field[int(starting_pos_x) : int(ending_pos_x) + 1, int(starting_pos_y) : int(ending_pos_y) + 1], reconstruction)

        return residual_field

    def predicted_field(self, reconstructions=None):
        if reconstructions is None:
            reconstructions = self.num_components
        
        prediction = np.zeros_like(self.postage_stamp)

        if not self.channel_last:
            prediction = np.transpose(prediction, axes=(1,2,0))
        
        for i in range(self.num_components):
            detected_position = self.detected_positions[i]

            #TODO: make this optional

            #cutout prediction 
            reconstruction = reconstructions[i]
            max_loc = tf.math.argmax(reconstruction).numpy()
            
            starting_pos_x = detected_position[0] - (self.cutout_size-1)/2
            ending_pos_x = detected_position[0] + (self.cutout_size-1)/2

            starting_pos_y = detected_position[1] - (self.cutout_size-1)/2
            ending_pos_y = detected_position[1] + (self.cutout_size-1)/2

            prediction[int(starting_pos_x) : int(ending_pos_x) + 1, int(starting_pos_y) : int(ending_pos_y) + 1] += prediction

        return prediction

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

        print(np.shape(X))
        m, n, b = np.shape(X) 

        initializer = tf.random_uniform_initializer(0,1)
        
        if initZ is not None: 
            # check constraint parameter over here
            z = tf.Variable(initial_value = initZ, name ='z')

        else:

            z = tf.Variable(name="z", initial_value=tf.random_normal_initializer(mean=0, stddev=1)(shape=[self.num_components, 32], dtype=tf.float32))
            # TODO: re-train the encoder to find a good starting point.
            #_, initZ = self.flow_vae_net.vae_model(np.expand_dims(X, 0))
            # z = tf.Variable(initial_value = initZ, shape=tf.TensorShape((1,32)), name ='z')

        zdist = tfd.MultivariateNormalDiag(loc=[0.0] * 32)

        optimizer = tf.keras.optimizers.Adam(lr = self.lr)

        sig = tf.math.reduce_std(X)

        for i in range(self.max_iter):
            #print(i)
            with tf.GradientTape() as tape:
                
                reconstructions = self.flow_vae_net.decoder(z).mean()
                #reconstruction = tf.math.reduce_sum(reconstruction, axis=0)

                residual_field = self.compute_residual(reconstructions)

                reconstruction_loss = tf.math.reduce_sum(tf.square(residual_field)) / tf.cast(tf.square(sig), tf.float64)

                log_likelihood = tf.cast(tf.math.reduce_sum(self.flow_vae_net.flow(tf.reshape(z,(self.num_components, 32)))), tf.float64)
                if self.use_likelihood:
                    loss = tf.math.subtract(reconstruction_loss, log_likelihood)
                else:
                    loss = reconstruction_loss

                sig = tf.math.reduce_std(residual_field)
            #print(tf.shape(tf.math.reduce_sum(W, axis=0)))
            print("sigma :" + str(sig.numpy()))
            print("log prob flow:" + str(log_likelihood.numpy()))
            print("reconstruction loss"+str(reconstruction_loss.numpy()))
            print(loss)
            grad = tape.gradient(loss, [z])
            grads_and_vars=[(grad, [z])]
            optimizer.apply_gradients(zip(grad, [z]))

        self.components = reconstructions.numpy()
        #print(self.components)
