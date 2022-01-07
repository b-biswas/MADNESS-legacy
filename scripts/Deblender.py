import numpy as np
import tensorflow as tf
from scripts.FlowVAEnet import FlowVAEnet
import tensorflow_probability as tfp

tfd = tfp.distributions

class Deblend:

    def __init__(self, postage_stamp, num_components = 1, max_iter = 100, lr = .1, initW = None, use_likelihood=True, channel_last=False):

        self.postage_stamp = postage_stamp
        print(np.shape(postage_stamp))
        self.max_iter = max_iter 
        self.lr = lr 
        self.num_components = num_components 
        self.use_likelihood = use_likelihood
        self.components = None
        self.channel_last = channel_last

        self.flow_vae_net = FlowVAEnet()

        self.flow_vae_net.load_vae_weights('/sps/lsst/users/bbiswas/weights/LSST/FlowDeblender/separated_architecture/vae/')
        self.flow_vae_net.load_flow_weights('/sps/lsst/users/bbiswas/weights/LSST/FlowDeblender/separated_architecture/fvae/')

        self.flow_vae_net.vae_model.trainable = False
        self.flow_vae_net.flow_model.trainable = False

        self.flow_vae_net.vae_model.summary()
        self.gradient_decent()


    def gradient_decent(self, optimizer=None, initZ=None):


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

            z = tf.Variable(name="z", initial_value=tf.random_normal_initializer(mean=0, stddev=.01)(shape=[32], dtype=tf.float32))
            #_, initZ = self.flow_vae_net.vae_model(np.expand_dims(X, 0))
            #z = tf.Variable(initial_value = initZ, shape=tf.TensorShape((1,32)), name ='z')

        zdist = tfd.MultivariateNormalDiag(loc=[0.0] * 32)

        optimizer = tf.keras.optimizers.Adam(lr = self.lr)

        sig = tf.math.reduce_std(X)

        for i in range(self.max_iter):
            #print(i)
            with tf.GradientTape() as tape:

                reconstruction = self.flow_vae_net.decoder(tf.reshape(z,(1, 32))).mean()
                reconstruction = tf.math.reduce_sum(reconstruction, axis=0)
                reconstruction_loss = tf.math.reduce_sum(tf.square(X - reconstruction)) / tf.cast(tf.square(sig), tf.float32)

                log_likelihood = self.flow_vae_net.td(tf.reshape(z,(1, 32)))
                if self.use_likelihood:
                    loss = reconstruction_loss - log_likelihood
                else:
                    loss = reconstruction_loss

            #print(tf.shape(tf.math.reduce_sum(W, axis=0)))
            #print(log_likelihood)
            #print(loss)
            grad = tape.gradient(loss, [z])
            
            grads_and_vars=[(grad, [z])]

            #print(W)

            optimizer.apply_gradients(zip(grad, [z]))

        self.components = reconstruction.numpy()
        print(self.components)
