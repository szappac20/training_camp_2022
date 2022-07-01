import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

import training_camp_2022.config
import training_camp_2022.data.window_generator
import training_camp_2022.experiments.energy_consumptions_prediction


def test_window_generator_shapes():

    task_1_path = os.path.join(training_camp_2022.config.data_path, "task_1")
    task_2_path = os.path.join(training_camp_2022.config.data_path, "task_2")

    training_features_1 = pd.read_csv(
        os.path.join(task_1_path, "training_features.csv"),
        sep=";", encoding="latin1")
    training_features_2 = pd.read_csv(
        os.path.join(task_2_path, "training_features.csv"),
        sep=";", encoding="latin1")

    validation_features_1 = pd.read_csv(
        os.path.join(task_1_path, "validation_features.csv"),
        sep=";", encoding="latin1")
    validation_features_2 = pd.read_csv(
        os.path.join(task_2_path, "validation_features.csv"),
        sep=";", encoding="latin1")

    test_features_1 = pd.read_csv(
        os.path.join(task_1_path, "test_features.csv"),
        sep=";", encoding="latin1")
    test_features_2 = pd.read_csv(
        os.path.join(task_2_path, "test_features.csv"),
        sep=";", encoding="latin1")


def test_tutorial():
    zip_path = os.path.join(
        "C:\\Users", "zappac20", ".keras", "datasets",
        "jena_climate_2009_2016.csv.zip")

    if os.path.exists(zip_path) is False:
        url = "https://storage.googleapis.com/tensorflow/tf-keras-datasets"
        zip_path = tf.keras.utils.get_file(
            origin=url + "/jena_climate_2009_2016.csv.zip",
            fname="jena_climate_2009_2016.csv.zip",
            extract=True)

    csv_path, _ = os.path.splitext(zip_path)

    df = pd.read_csv(csv_path)
    # Slice [start:stop:step], starting from index 5 take every 6th record.
    df = df[5::6]

    wv = df["wv (m/s)"]
    bad_wv = wv == -9999.0
    wv[bad_wv] = 0.0

    max_wv = df["max. wv (m/s)"]
    bad_max_wv = max_wv == -9999.0
    max_wv[bad_max_wv] = 0.0

    # The above inplace edits are reflected in the DataFrame.
    df["wv (m/s)"].min()

    date_time = pd.to_datetime(df.pop("Date Time"), format="%d.%m.%Y %H:%M:%S")

    timestamp_s = date_time.map(pd.Timestamp.timestamp)
    day = 24 * 60 * 60
    year = (365.2425) * day

    df["Day sin"] = np.sin(timestamp_s * (2 * np.pi / day))
    df["Day cos"] = np.cos(timestamp_s * (2 * np.pi / day))
    df["Year sin"] = np.sin(timestamp_s * (2 * np.pi / year))
    df["Year cos"] = np.cos(timestamp_s * (2 * np.pi / year))

    column_indices = {name: i for i, name in enumerate(df.columns)}

    n = len(df)
    train_df = df[0:int(n * 0.7)]
    val_df = df[int(n * 0.7):int(n * 0.9)]
    test_df = df[int(n * 0.9):]

    num_features = df.shape[1]

    train_mean = train_df.mean()
    train_std = train_df.std()

    train_df = (train_df - train_mean) / train_std
    val_df = (val_df - train_mean) / train_std
    test_df = (test_df - train_mean) / train_std

    w1 = training_camp_2022.data.window_generator.WindowGenerator(
        input_width=24, label_width=1, shift=24,
        train_df=train_df, val_df=val_df, test_df=test_df,
        label_columns=["T (degC)"])
    print(w1)

    print()

    w2 = training_camp_2022.data.window_generator.WindowGenerator(
        input_width=6, label_width=1, shift=1,
        train_df=train_df, val_df=val_df, test_df=test_df,
        label_columns=["T (degC)"])
    print(w2)

    # Stack three slices, the length of the total window.
    example_window = tf.stack([
        np.array(train_df[:w2.total_window_size]),
        np.array(train_df[100:100+w2.total_window_size]),
        np.array(train_df[200:200+w2.total_window_size])])
    print(example_window)

    example_inputs, example_labels = w2.split_window(example_window)

    print("All shapes are: (batch, time, features)")
    print(f"Window shape: {example_window.shape}")
    print(f"Inputs shape: {example_inputs.shape}")
    print(f"Labels shape: {example_labels.shape}")

    for example_inputs, example_labels in w2.train.take(1):
        print(f"Inputs shape (batch, time, features): {example_inputs.shape}")
        print(f"Labels shape (batch, time, features): {example_labels.shape}")

    single_step_window = training_camp_2022.data.window_generator.WindowGenerator(
        input_width=1, label_width=1, shift=1,
        train_df=train_df, val_df=val_df, test_df=test_df,
        label_columns=["T (degC)"])

    class Baseline(tf.keras.Model):
        def __init__(self, label_index=None):
            super().__init__()
            self.label_index = label_index

        def call(self, inputs):
            if self.label_index is None:
                return inputs
            result = inputs[:, :, self.label_index]
            return result[:, :, tf.newaxis]

    baseline = Baseline(label_index=column_indices["T (degC)"])

    baseline.compile(
        loss=tf.keras.losses.MeanSquaredError(),
        metrics=[tf.keras.metrics.MeanAbsoluteError()])

    val_performance = {}
    performance = {}
    val_performance["Baseline"] = baseline.evaluate(
        single_step_window.val)
    performance["Baseline"] = baseline.evaluate(
        single_step_window.test, verbose=0)

    wide_window = training_camp_2022.data.window_generator.WindowGenerator(
        input_width=24, label_width=24, shift=1,
        train_df=train_df, val_df=val_df, test_df=test_df,
        label_columns=["T (degC)"])
    print("Input shape:", wide_window.example[0].shape)
    print("Output shape:", baseline(wide_window.example[0]).shape)

    print("Final plot")
    wide_window.plot(baseline, plot_cols=["T (degC)"])

    linear = tf.keras.Sequential([tf.keras.layers.Dense(units=1)])

    def compile_and_fit(model, window, patience=2):
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=patience, mode="min")

        model.compile(
            loss=tf.keras.losses.MeanSquaredError(),
            optimizer=tf.keras.optimizers.Adam(),
            metrics=[tf.keras.metrics.MeanAbsoluteError()])

        history = model.fit(
            window.train, epochs=20,
            validation_data=window.val,
            callbacks=[early_stopping])
        return history

    history = compile_and_fit(linear, single_step_window)

    val_performance["Linear"] = linear.evaluate(single_step_window.val)
    performance["Linear"] = linear.evaluate(single_step_window.test, verbose=0)

    wide_window.plot(linear, plot_cols=["T (degC)"])

    conv_window = training_camp_2022.data.window_generator.WindowGenerator(
        input_width=3, label_width=1, shift=1,
        train_df=train_df, val_df=val_df, test_df=test_df,
        label_columns=["T (degC)"])

    multi_step_dense = tf.keras.Sequential([
        # Shape: (time, features) => (time*features)
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(units=32, activation="relu"),
        tf.keras.layers.Dense(units=32, activation="relu"),
        tf.keras.layers.Dense(units=1),
        # Add back the time dimension.
        # Shape: (outputs) => (1, outputs)
        tf.keras.layers.Reshape([1, -1]),
    ], name="multi_step_dense")

    print("Input shape:", conv_window.example[0].shape)
    print("Output shape:", multi_step_dense(conv_window.example[0]).shape)

    conv_window.plot(multi_step_dense, plot_cols=["T (degC)"])

    model_plot_path = os.path.join(
        training_camp_2022.config.plot_path,
        "multi_step_dense_model_tutorial.png")
    print(multi_step_dense.summary())
    print(model_plot_path)
    tf.keras.utils.plot_model(
        multi_step_dense, model_plot_path, show_shapes=True)


def test_imputation():

    train_features_ds_pd, train_labels_ds_pd, \
    validation_features_ds_pd, validation_labels_ds_pd, \
    test_features_ds_pd, test_labels_ds_pd = \
        training_camp_2022.experiments.energy_consumptions_prediction.\
            prepare_data()

    train_df = \
        training_camp_2022.experiments.energy_consumptions_prediction.\
            impute_data(train_features_ds_pd)
    val_df = \
        training_camp_2022.experiments.energy_consumptions_prediction.\
            impute_data(validation_features_ds_pd)
    test_df = \
        training_camp_2022.experiments.energy_consumptions_prediction.\
            impute_data(test_features_ds_pd)

    win_gen = training_camp_2022.data.window_generator.WindowGenerator(
        input_width=6, label_width=1, shift=1,
        train_df=train_df, val_df=val_df, test_df=test_df,
        label_columns=["f1"])

    print(win_gen)
    print(train_df.head(60))

    for example_inputs, example_labels in win_gen.train.take(3):
        print(f"Inputs shape (batch, time, features): {example_inputs.shape}")
        print(f"Labels shape (batch, time, features): {example_labels.shape}")

        print("Inputs")
        print(*example_inputs)
        print("Labels")
        print(*example_labels)


if __name__ == "__main__":
    # test_window_generator_shapes()
    test_tutorial()
    # test_imputation()

    plt.show()

