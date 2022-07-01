import tensorflow as tf
import tensorflow_addons as tfa


class EchoStateNetwork(tf.keras.Sequential):
    def __init__(self, num_units):
        super().__init__([
            tfa.layers.ESN(
                units=num_units,
                connectivity=0.1,
                leaky=1,
                spectral_radius=0.9,
                use_norm2=False,
                use_bias=True,
                activation="tanh",
                kernel_initializer="glorot_uniform",
                recurrent_initializer="glorot_uniform",
                bias_initializer="zeros",
                return_sequences=True,
                go_backwards=False,
                unroll=False,
                name="echo_state_network"),
            tf.keras.layers.Dense(3, activation="relu", name="consumptions")])

    def compile_and_fit(self, window, patience=2, epochs=2):
        self.compile(
            optimizer=tfa.optimizers.RectifiedAdam(),
            loss=tf.losses.MeanSquaredError(),
            metrics=[tf.metrics.MeanAbsoluteError()])

        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=patience, mode='min')

        history = self.fit(
            window.train, epochs=epochs,
            validation_data=window.val,
            callbacks=[early_stopping])

        return history

