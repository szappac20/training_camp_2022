import tensorflow as tf


class LongShortTimeMemory(tf.keras.Sequential):
    def __init__(self, num_units):
        super().__init__([
            # Shape [batch, time, features] => [batch, time, lstm_units]
            tf.keras.layers.LSTM(32, return_sequences=True),
            # Shape => [batch, time, features]
            tf.keras.layers.Dense(
                units=num_units, name="consumptions")])

    def compile_and_fit(self, window, patience=2, epochs=2):
        self.compile(
            optimizer=tf.keras.optimizers.Adam(),
            loss=tf.losses.MeanSquaredError(),
            metrics=[tf.metrics.MeanAbsoluteError()])

        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=patience, mode='min')

        history = self.fit(
            window.train, epochs=epochs,
            validation_data=window.val,
            callbacks=[early_stopping])

        return history
