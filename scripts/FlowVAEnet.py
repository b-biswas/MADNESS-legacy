from scripts.model import flow, vae_model
import tensorflow as tf
import tensorflow_probability as tfp

from tensorflow.keras.layers import Input

tfd = tfp.distributions
tfb = tfp.bijectors

def vae_loss_fn(x, output):
    x_decoded_mean = output[1]
    tf.print(x_decoded_mean.shape)
    mse_loss = tf.reduce_sum(tf.keras.metrics.mean_squared_error(x, x_decoded_mean), axis=[1,2])
    return mse_loss 

def flow_loss_fn(x, output):
    return -output[0]

class FlowVAEnet:
    def __init__(self, latent_dim=32, hidden_dim=256, filters=[32, 64, 128, 256], kernels=[3,3,3,3],nb_of_bands=6, conv_activation=None, dense_activation=None, num_nf_layers=5, linear_norm=True):

        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.filters = filters
        self.kernels = kernels
        self.nb_of_bands = nb_of_bands
        self.conv_activation = conv_activation
        self.dense_activation = dense_activation
        self.num_nf_layers = num_nf_layers
        self.linear_norm = linear_norm

        self.encoder, self.decoder, self.td, self.model = self.build_vae_model()
        self.model.summary()
        self.optimizer = None
        self.callbacks = None


    def build_vae_model(self):

        encoder, decoder = vae_model(latent_dim=self.latent_dim, hidden_dim=self.hidden_dim, filters=self.filters, kernels=self.kernels, nb_of_bands=self.nb_of_bands, conv_activation=self.conv_activation, dense_activation=self.dense_activation, linear_norm=self.linear_norm)
        
        input_vae = Input(shape=encoder.input.shape[1:])

        mu, sig = encoder(input_vae)
        
        latent_dist = tfd.MultivariateNormalDiag(mu, sig)
        x = latent_dist.sample()

        reconstruction = decoder(x)
        td = flow(latent_dim=self.latent_dim, num_nf_layers=self.num_nf_layers)
        log_prob = td(x)
        tf.print(log_prob.shape)
        
        model = tf.keras.Model(input_vae, outputs=[log_prob, decoder(x)])
        
        return encoder, decoder, td, model
       
    def train_vae(self, train_generator, validation_generator, callbacks, optimizer=tf.keras.optimizers.Adam(1e-2, clipvalue=1.0), epochs = 35, verbose=1):
        self.td.trainable=False
        self.model.summary()
        self.model.compile(optimizer=optimizer, loss={'decoder': vae_loss_fn})
        self.model.fit_generator(generator=train_generator, epochs=epochs,
                  verbose=verbose,
                  shuffle=True,
                  validation_data=validation_generator,
                  callbacks=callbacks,
                  workers=0, 
                  use_multiprocessing = True)

    def train_flow_model(self, train_generator, validation_generator, callbacks, optimizer=tf.keras.optimizers.Adam(1e-2), epochs = 35, verbose=1):

        self.encoder.trainable = False
        self.decoder.trainable = False
        self.td.trainable = True
        self.model.summary()
        self.td.summary()
        self.model.compile(optimizer=optimizer, loss = {'flow': flow_loss_fn})
        self.model.fit_generator(generator=train_generator, epochs=epochs,
                  verbose=verbose,
                  shuffle=True,
                  validation_data=validation_generator,
                  callbacks=callbacks,
                  workers=0, 
                  use_multiprocessing = True)

    def load_weights(self, weights_path, Folder=True):
        if Folder:
            weights_path = tf.train.latest_checkpoint(weights_path)
        self.model.load_weights(weights_path)
    
        self.model.trainable=False
