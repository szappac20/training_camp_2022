import tensorflow as tf


class Linear(tf.keras.Sequential):
    def __init__(self, num_units, name="linear"):
        super().__init__(
            [tf.keras.layers.Dense(units=num_units)],
            name=name)

    def compile_and_fit(self, window, patience=2, epochs=10):
        self.compile(
            loss=tf.keras.losses.MeanSquaredError(),
            optimizer=tf.keras.optimizers.Adam(),
            metrics=[tf.keras.metrics.MeanAbsoluteError()])

        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=patience, mode="min")

        history = self.fit(
            window.train, epochs=epochs, validation_data=window.val,
            callbacks=[early_stopping])

        return history
