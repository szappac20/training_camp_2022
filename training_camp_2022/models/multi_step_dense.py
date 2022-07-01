import tensorflow as tf


class MultiStepDense(tf.keras.Sequential):
    def __init__(self, num_units, name="multi_step_dense"):
        # Shape: (time, features) => (time*features)

        # Add back the time dimension
        # Shape: (outputs) => (1, outputs)

        super(MultiStepDense, self).__init__([
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(units=32, activation="relu"),
            tf.keras.layers.Dense(units=32, activation="relu"),
            tf.keras.layers.Dense(units=num_units),
            tf.keras.layers.Reshape([-1, num_units])], name=name)

    def compile_and_fit(self, window, patience=2, epochs=2):
        self.compile(
            loss=tf.keras.losses.MeanSquaredError(),
            optimizer=tf.keras.optimizers.Adam(),
            metrics=[tf.keras.metrics.MeanAbsoluteError()])

        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=patience, mode='min')

        history = self.fit(
            window.train, epochs=epochs,
            validation_data=window.val,
            callbacks=[early_stopping])

        return history
