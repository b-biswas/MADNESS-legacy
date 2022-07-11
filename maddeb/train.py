import os

import tensorflow as tf


def define_callbacks(weights_save_path, lr_scheduler_epochs=None):
    """
    Define callbacks for a network to train

    parameters:
        weights_save_path: path at which weights are to be saved.path at which weights are to be saved. By default, it saves weights in the data folder.
        lr_scheduler_epochs: number of iterations after which the learning rate is decreased by a factor of $e$.
            Default is None, and a constant learning rate is used
    """

    checkpointer_val_mse = tf.keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(weights_save_path, "val_mse", "weights.ckpt"),
        monitor="val_mse",
        verbose=1,
        save_best_only=True,
        save_weights_only=True,
        mode="min",
        save_freq="epoch",
    )
    checkpointer_val_loss = tf.keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(weights_save_path, "val_loss", "weights.ckpt"),
        monitor="val_loss",
        verbose=1,
        save_best_only=True,
        save_weights_only=True,
        mode="min",
        save_freq="epoch",
    )

    early_stopping = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=15)

    callbacks = [checkpointer_val_mse, checkpointer_val_loss, early_stopping]

    if lr_scheduler_epochs is not None:

        def scheduler(epoch, lr):
            if (epoch + 1) % lr_scheduler_epochs != 0:
                return lr
            else:
                return lr * tf.math.exp(-1.0)

        lr_scheduler = tf.keras.callbacks.LearningRateScheduler(scheduler)

        callbacks += [lr_scheduler]

    callbacks += [tf.keras.callbacks.TerminateOnNaN()]

    return callbacks
