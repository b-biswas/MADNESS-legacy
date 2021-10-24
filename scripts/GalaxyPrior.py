from scripts.model import build_encoder, flow, build_flow
import tensorflow as tf

class GalaxyPrior:

    def __init__(self, encoder=build_encoder(), td=flow()):
        self.encoder = build_encoder()
        self.td = flow()
        self.flow_model = build_flow(encoder, td)
        self.optimizer = None
        self.callbacks = None

    def loss_fn(self, y, log_prob):
        return -log_prob
    

    def train_model(self, train_generator, validation_generator, callbacks, optimizer=tf.keras.optimizers.Adam(1e-4), epochs = 100, verbose=1):

        self.flow_model.compile(optimizer=optimizer, loss = self.loss_fn)
        self.flow_model.fit_generator(generator=train_generator, epochs=epochs,
                  verbose=verbose,
                  shuffle=True,
                  validation_data=validation_generator,
                  callbacks=callbacks,
                  workers=0, 
                  use_multiprocessing = True)

