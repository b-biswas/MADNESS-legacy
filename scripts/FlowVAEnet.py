from scripts.model import flow, vae_model
import tensorflow as tf
import tensorflow_probability as tfp

from tensorflow.keras.layers import Input

tfd = tfp.distributions
tfb = tfp.bijectors

def vae_loss_fn(x, output):
    x_decoded_mean = output
    mse_loss = tf.reduce_sum(tf.keras.metrics.mean_squared_error(x, x_decoded_mean), axis=[1,2]) # TODO: prevent avg over the last dimension multiply suitably
    tf.print(mse_loss.shape)
    return mse_loss*6

def flow_loss_fn(x, output):
    tf.print(tf.shape(output))
    return -output

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
        
        model = tf.keras.Model(input_vae, outputs=[log_prob, decoder(x), mu, sig])
        
        return encoder, decoder, td, model
       
    def train_vae(self, train_generator, validation_generator, path_weights, callbacks, optimizer=tf.keras.optimizers.Adam(1e-6, clipvalue=1), epochs = 35, verbose=1):
        self.td.trainable=False
        self.encoder.trainable=True
        self.decoder.trainable=True
        self.model.summary()
        print("Training only VAE network")
        terminate_on_nan = [tf.keras.callbacks.TerminateOnNaN()]
        self.model.compile(optimizer=optimizer, loss={'decoder': vae_loss_fn})
        self.model.fit_generator(generator=train_generator, epochs=epochs,
                  verbose=verbose,
                  shuffle=True,
                  validation_data=validation_generator,
                  callbacks= callbacks + terminate_on_nan,
                  workers=0, 
                  use_multiprocessing = True)

    def train_flow(self, train_generator, validation_generator, path_weights, callbacks, optimizer=tf.keras.optimizers.Adam(1e-6, clipvalue=1), epochs = 35, verbose=1):
        print("Training only Flow net")
        self.td.trainable = True
        self.encoder.trainable=True
        self.decoder.trainable=True
        self.model.summary()
        self.model.compile(optimizer=optimizer, loss={'decoder': vae_loss_fn, 'flow': flow_loss_fn})
        #self.model.compile(optimizer=optimizer, loss={'flow': flow_loss_fn})
        terminate_on_nan = [tf.keras.callbacks.TerminateOnNaN()]
        self.model.fit_generator(generator=train_generator, epochs=epochs,
                  verbose=verbose,
                  shuffle=True,
                  validation_data=validation_generator,
                  callbacks=callbacks + terminate_on_nan,
                  workers=0, 
                  use_multiprocessing = True)

    def load_weights(self, weights_path, Folder=True):
        if Folder:
            weights_path = tf.train.latest_checkpoint(weights_path)
        self.model.load_weights(weights_path)

        self.model.trainable=False
