from scripts.model import create_model_fvae
import tensorflow as tf
import tensorflow_probability as tfp

from tensorflow.keras.layers import Input
import tensorflow.keras.backend as K

tfd = tfp.distributions
tfb = tfp.bijectors

def vae_loss_fn(x, x_decoded_mean):
    # tf.print(tf.shape(x_decoded_mean.log_prob(x)))
    return -tf.math.reduce_mean(tf.math.reduce_sum(x_decoded_mean.log_prob(x), axis=[1, 2, 3]))

def flow_loss_fn(x, output):
    return -tf.math.reduce_mean(output)

class FlowVAEnet:

    def __init__(self,
        input_shape=[59, 59, 6],
        latent_dim=32,
        filters=[32,64,128,256],
        kernels=[3,3,3,3],
        conv_activation=None,
        dense_activation=None,
        linear_norm=True,
        num_nf_layers=5,
    ):
        """
        Creates the required models according to the specifications.

        Parameters
        ----------
        input_shape: list
            shape of input tensor
        latent_dim: int
            size of the latent space
        filters: list
            filters used for the convolutional layers
        kernels: list
            kernels used for the convolutional layers
        conv_activation: str
            activation for conv layers
        dense_activation: str
            activation for dense layers
        linear_norm: bool
            to specify by normalization is linear or not
        num_nf_layers: int
            number of layers in the flow network
        """

        self.input_shape = input_shape
        self.latent_dim = latent_dim

        self.filters = filters
        self.kernels = kernels
        self.nb_of_bands = input_shape[2]
        self.conv_activation = conv_activation
        self.dense_activation = dense_activation
        self.num_nf_layers = num_nf_layers
        self.linear_norm = linear_norm

        self.vae_model, self.flow_model, self.encoder, self.decoder, self.flow = create_model_fvae(input_shape=self.input_shape, 
                                                                                latent_dim=self.latent_dim, 
                                                                                filters=self.filters, 
                                                                                kernels=self.kernels, 
                                                                                conv_activation=self.conv_activation, 
                                                                                dense_activation=self.dense_activation,
                                                                                num_nf_layers=self.num_nf_layers)

        self.optimizer = None
        self.callbacks = None

    def train_vae(self, 
                    train_generator, 
                    validation_generator, 
                    callbacks, 
                    train_encoder=True,
                    train_decoder=True,
                    optimizer=tf.keras.optimizers.Adam(1e-4), 
                    epochs=35, 
                    verbose=1):
        """
        trains only the vae model. (both the encoder and the decoder)

        Parameters
        ----------
        train_generator:
            generator to be used for training the network.
            keras.utils.Sequence returning (inputs, targets) or (inputs, targets, sample_weights)
        validation_generator:
            generator to be used for validation
            keras.utils.Sequence returning (inputs, targets) or (inputs, targets, sample_weights)
        callbacks: list
            List of keras.callbacks.Callback instances. 
            List of callbacks to apply during training. 
            See tf.keras.callbacks
        optimizer: str or tf.keras.optimizers
            String (name of optimizer) or optimizer instance. See tf.keras.optimizers.
        epochs: int
            number of epochs for which the model is going to be trained
        verbose: int
            verbose option for training.
            'auto', 0, 1, or 2. Verbosity mode. 0 = silent, 1 = progress bar, 2 = one line per epoch. 
            'auto' defaults to 1 for most cases, but 2 when used with ParameterServerStrategy. 
            Note that the progress bar is not particularly useful when logged to a file, so verbose=2 is recommended when not running interactively (eg, in a production environment). 
        """
        if (not train_decoder) and (not train_encoder):
            raise ValueError("Training failed because both encoder and decoder are not trainable")
        
        self.encoder.trainable=train_encoder
        self.decoder.trainable=train_decoder
        self.vae_model.summary()
        print("Training only VAE network")
        terminate_on_nan = [tf.keras.callbacks.TerminateOnNaN()]
        self.vae_model.compile(optimizer=optimizer, loss={"decoder": vae_loss_fn}, experimental_run_tf_function=False)
        self.vae_model.fit_generator(generator=train_generator, 
                                    epochs=epochs,
                                    verbose=verbose,
                                    shuffle=True,
                                    validation_data=validation_generator,
                                    callbacks= callbacks + terminate_on_nan,
                                    workers=0, 
                                    use_multiprocessing = True)

    def train_flow(self, 
                    train_generator, 
                    validation_generator, 
                    callbacks, 
                    optimizer=tf.keras.optimizers.Adam(1e-4), 
                    epochs=35, 
                    verbose=1):
        """
        Trains only the flow part of the flow_model while keeping the encoder constant.

        Parameters
        ----------
        train_generator:
            generator to be used for training the network.
            keras.utils.Sequence returning (inputs, targets) or (inputs, targets, sample_weights)
        validation_generator:
            generator to be used for validation
            keras.utils.Sequence returning (inputs, targets) or (inputs, targets, sample_weights)
        callbacks: list
            List of keras.callbacks.Callback instances. 
            List of callbacks to apply during training. 
            See tf.keras.callbacks
        optimizer: str or tf.keras.optimizers
            String (name of optimizer) or optimizer instance. See tf.keras.optimizers.
        epochs: int
            number of epochs for which the model is going to be trained
        verbose: int
            verbose option for training.
            'auto', 0, 1, or 2. Verbosity mode. 0 = silent, 1 = progress bar, 2 = one line per epoch. 
            'auto' defaults to 1 for most cases, but 2 when used with ParameterServerStrategy. 
            Note that the progress bar is not particularly useful when logged to a file, so verbose=2 is recommended when not running interactively (eg, in a production environment). 
        """
        print("Training only Flow net")
        
        self.flow.trainable = True
        self.encoder.trainable = False
        self.flow_model.compile(optimizer=optimizer, loss=lambda _, log_prob: -log_prob)
        self.flow_model.summary()
        #self.model.compile(optimizer=optimizer, loss={'flow': flow_loss_fn})
        terminate_on_nan = [tf.keras.callbacks.TerminateOnNaN()]

        def scheduler(epoch, lr):
            if (epoch + 1) % 30 != 0:
                return lr
            else:
                return lr * tf.math.exp(-1.0)

        lr_scheduler = tf.keras.callbacks.LearningRateScheduler(scheduler)

        callbacks += [lr_scheduler]
        self.flow_model.fit_generator(generator=train_generator,
                                    epochs=epochs,
                                    verbose=verbose,
                                    shuffle=True,
                                    validation_data=validation_generator,
                                    callbacks=callbacks + terminate_on_nan,
                                    workers=4, 
                                    use_multiprocessing=True)

    def load_vae_weights(self, weights_path, is_folder=True):
        """
        Parameters
        ----------
        weights_path: str
            path to the weights of the vae_model.
            The path must point to either a folder with the saved checkpoints or directly to the checkpoint.
        is_folder: bool
            specifies if the weights_path points to a folder.
            If True, then the latest checkpoint is loaded.
            else, the checkpoint specified in the path is loaded
        """
        if is_folder:
            weights_path = tf.train.latest_checkpoint(weights_path)
        self.vae_model.load_weights(weights_path).expect_partial()

    def load_flow_weights(self, weights_path, is_folder=True):
        """
        Parameters
        ----------
        weights_path: str
            path to the weights of the vae_model.
            The path must point to either a folder with the saved checkpoints or directly to the checkpoint.
        is_folder: bool
            specifies if the weights_path points to a folder.
            If True, then the latest checkpoint is loaded.
            else, the checkpoint specified in the path is loaded
        """
        if is_folder:
            weights_path = tf.train.latest_checkpoint(weights_path)
        self.flow_model.load_weights(weights_path).expect_partial()
