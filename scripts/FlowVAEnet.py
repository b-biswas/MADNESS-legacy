from scripts.model import build_encoder, flow
import tensorflow as tf
import tensorflow_probability as tfp

from tensorflow.keras.layers import Input

tfd = tfp.distributions
tfb = tfp.bijectors

class FlowVAEnet:
    def __init__(self, latent_dim=32, hidden_dim=256, filters=[32, 64, 128, 256], kernels=[3,3,3,3],nb_of_bands=6, conv_activation=None, dense_activation=None, num_nf_layers=5):

        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.filters = filters
        self.kernels = kernels
        self.nb_of_bands = nb_of_bands
        self.conv_activation = conv_activation
        self.dense_activation = dense_activation
        self.num_nf_layers = num_nf_layers
        self.model = self.build_flow_vae()
        self.optimizer = None
        self.callbacks = None


    def build_flow_vae(self):

        encoder = build_encoder(latent_dim=self.latent_dim, hidden_dim=self.hidden_dim, filters=self.filters, kernels=self.kernels, nb_of_bands=self.nb_of_bands, conv_activation=self.conv_activation, dense_activation=self.dense_activation)
        td = flow(latent_dim=self.latent_dim, num_nf_layers=self.num_nf_layers)

        input_vae = Input(shape=encoder.input.shape[1:])

        mu, sig = encoder(input_vae)
        
        latent_dist = tfd.MultivariateNormalDiag(mu, sig)
        x = latent_dist.sample()

        log_prob = td.log_prob(x)
        
        model = tf.keras.Model(input_vae, log_prob)
        
        return model

    def loss_fn(self, y, log_prob):
        return -log_prob
    

    def train_model(self, train_generator, validation_generator, callbacks, optimizer=tf.keras.optimizers.Adam(1e-3), epochs = 100, verbose=1):

        self.model.compile(optimizer=optimizer, loss = self.loss_fn)
        self.model.fit_generator(generator=train_generator, epochs=epochs,
                  verbose=verbose,
                  shuffle=True,
                  validation_data=validation_generator,
                  callbacks=callbacks,
                  workers=0, 
                  use_multiprocessing = True)