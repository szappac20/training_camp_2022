import tensorflow as tf


class Feedback(tf.keras.Model):
    def __init__(self, units, out_steps, num_features, name="ar_lstm"):
        super().__init__(name=name)
        self.out_steps = out_steps
        self.units = units
        self.lstm_cell = tf.keras.layers.LSTMCell(units)
        # Also wrap the LSTMCell in an RNN to simplify the `warmup` method.
        self.lstm_rnn = tf.keras.layers.RNN(
            self.lstm_cell, return_state=True)
        self.dense = tf.keras.layers.Dense(num_features)

    def warmup(self, inputs):
        # inputs.shape => (batch, time, features)
        # x.shape => (batch, lstm_units)
        x, *state = self.lstm_rnn(inputs)

        # predictions.shape => (batch, features)
        prediction = self.dense(x)
        return prediction, state

    def call(self, inputs, training=None):
        # Use a TensorArray to capture dynamically unrolled outputs
        predictions = []
        # Initialize the LSTM state
        prediction, state = self.warmup(inputs)

        # Insert the first prediction
        predictions.append(prediction)

        # Run the rest of the prediction steps
        for n in range(1, self.out_steps):
            # Use the last prediction as input
            x = prediction
            # Execute one lstm step
            x, state = self.lstm_cell(
                x, states=state, training=training)
            # Convert the lstm output to a prediction
            prediction = self.dense(x)
            # Add the prediction to the output
            predictions.append(prediction)

        # predictions.shape => (time, batch, features)
        predictions = tf.stack(predictions)
        # predictions.shape => (batch, time, features)
        predictions = tf.transpose(predictions, [1, 0, 2])
        return predictions

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

