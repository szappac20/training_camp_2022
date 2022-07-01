import os
import datetime
import numpy as np
import pandas as pd
import tensorflow as tf

import training_camp_2022.config
import training_camp_2022.data.window_generator
import training_camp_2022.models.echo_state_network
import training_camp_2022.models.baseline
import training_camp_2022.models.linear_neural_network
import training_camp_2022.models.dense
import training_camp_2022.models.multi_step_dense
import training_camp_2022.models.long_short_time_memory
import training_camp_2022.models.feedback

import training_camp_2022.view.time_series


def prepare_data():
    task_1_path = os.path.join(
        training_camp_2022.config.data_path, "task_1")

    train_features_ds_pd = pd.read_csv(
        os.path.join(task_1_path, "training_features.csv"),
        sep=";", encoding="latin1")
    train_labels_ds_pd = pd.read_csv(
        os.path.join(task_1_path, "training_labels.csv"),
        sep=";", encoding="latin1")

    validation_features_ds_pd = pd.read_csv(
        os.path.join(task_1_path, "validation_features.csv"),
        sep=";", encoding="latin1")
    validation_labels_ds_pd = pd.read_csv(
        os.path.join(task_1_path, "validation_labels.csv"),
        sep=";", encoding="latin1")

    test_features_ds_pd = pd.read_csv(
        os.path.join(task_1_path, "test_features.csv"),
        sep=";", encoding="latin1")
    test_labels_ds_pd = pd.read_csv(
        os.path.join(task_1_path, "test_labels.csv"), sep=";",
        encoding="latin1")

    # train_features_ds = tfdf.keras.pd_dataframe_to_tf_dataset(train_ds_pd)
    # test_features_ds = tfdf.keras.pd_dataframe_to_tf_dataset(test_ds_pd)

    return train_features_ds_pd, train_labels_ds_pd, \
        validation_features_ds_pd, validation_labels_ds_pd, \
        test_features_ds_pd, test_labels_ds_pd


def impute_data(df):
    train_range = pd.date_range(
        start=datetime.datetime(year=2021, month=7, day=1),
        end=datetime.datetime(year=2022, month=6, day=1), freq="MS")

    df = df.drop(["city_density", "gender"], axis=1)

    imputed = pd.DataFrame()
    for user in list(set(df["user"])):
        # print("Elaborating user", user)
        cond = (df["user"] == user)
        if sum(np.array(cond)) > 3:
            tmp_df = df[cond].copy()

            tmp_df["date"] = tmp_df.apply(
                lambda row: datetime.datetime(
                    year=int(row.year), month=int(row.month), day=1), axis=1)
            tmp_df = tmp_df.set_index("date")
            # tmp_df = tmp_df.drop(["year", "month"], axis=1)
            tmp_df = tmp_df.drop(["year"], axis=1)

            tmp_df = tmp_df.reindex(train_range, fill_value=np.nan)
            tmp_df = tmp_df.interpolate(method="cubic").interpolate(
                fill_value="extrapolate", limit_direction="both")

            imputed = pd.concat([imputed, tmp_df])
            imputed = imputed.drop(["user"], axis=1)

    return imputed


def mask_data(df, fill_value=-1000):
    train_range = pd.date_range(
        start=datetime.datetime(year=2021, month=7, day=1),
        end=datetime.datetime(year=2022, month=6, day=1), freq="MS")

    df = df.drop(["city_density", "gender"], axis=1)

    masked = pd.DataFrame()
    for user in list(set(df["user"])):
        # print("Elaborating user", user)
        cond = (df["user"] == user)
        if sum(np.array(cond)) > 3:
            tmp_df = df[cond].copy()

            tmp_df["date"] = tmp_df.apply(
                lambda row: datetime.datetime(
                    year=int(row.year), month=int(row.month), day=1), axis=1)
            tmp_df = tmp_df.set_index("date")
            # tmp_df = tmp_df.drop(["year", "month"], axis=1)
            tmp_df = tmp_df.drop(["year"], axis=1)

            tmp_df = tmp_df.reindex(train_range, fill_value=fill_value)

            masked = pd.concat([masked, tmp_df])
            masked = masked.drop(["user"], axis=1)

    return masked


def train_evaluate_model(
        model, monthly_window, quadrimestral_window,
        val_performance, test_performance):

    history = model.compile_and_fit(monthly_window)

    val_performance[model.name] = model.evaluate(
        monthly_window.val)
    test_performance[model.name] = model.evaluate(
        monthly_window.test, verbose=False)

    print(f"Final plot {model.name}")
    quadrimestral_window.plot(model)

    return history


def launch_train():
    train_features_ds_pd, train_labels_ds_pd, \
        validation_features_ds_pd, validation_labels_ds_pd, \
        test_features_ds_pd, test_labels_ds_pd = prepare_data()

    train_df = impute_data(train_features_ds_pd)
    val_df = impute_data(validation_features_ds_pd)
    test_df = impute_data(test_features_ds_pd)

    column_indices = {name: i for i, name in enumerate(train_df.columns)}

    monthly_window = \
        training_camp_2022.data.window_generator.WindowGenerator(
            input_width=1, label_width=1, shift=1,
            train_df=train_df, val_df=val_df, test_df=test_df,
            label_columns=["f1", "f2", "f3"])

    quadrimestral_window = \
        training_camp_2022.data.window_generator.WindowGenerator(
            input_width=6, label_width=6, shift=1,
            train_df=train_df, val_df=val_df, test_df=test_df,
            label_columns=["f1", "f2", "f3"])

    conv_window = training_camp_2022.data.window_generator.WindowGenerator(
        input_width=3, label_width=1, shift=1,
        train_df=train_df, val_df=val_df, test_df=test_df,
        label_columns=["f1", "f2", "f3"])

    multi_window = training_camp_2022.data.window_generator.WindowGenerator(
        input_width=12, label_width=12, shift=12,
        train_df=train_df, val_df=val_df, test_df=test_df,
        label_columns=["f1", "f2", "f3"])

    val_performance = {}
    test_performance = {}

    print("Baseline model")
    baseline = training_camp_2022.models.baseline.Baseline(
        label_indices=[column_indices[f] for f in ["f1", "f2", "f3"]])
    baseline_history = train_evaluate_model(
        baseline, monthly_window, quadrimestral_window,
        val_performance, test_performance)

    print("Linear model")
    linear = training_camp_2022.models.linear_neural_network.Linear(
        num_units=3)
    linear_history = train_evaluate_model(
        linear, monthly_window, quadrimestral_window,
        val_performance, test_performance)

    print("Echo State Network")
    esn = training_camp_2022.models.echo_state_network.EchoStateNetwork(
        num_units=100)
    esn_history = train_evaluate_model(
        esn, monthly_window, quadrimestral_window,
        val_performance, test_performance)

    print("Dense")
    dense = training_camp_2022.models.dense.Dense(num_units=3)
    dense_history = train_evaluate_model(
        dense, monthly_window, quadrimestral_window,
        val_performance, test_performance)

    print("Multi-Step Dense")
    multi_step_dense = \
        training_camp_2022.models.multi_step_dense.MultiStepDense(num_units=3)
    multi_step_dense_history = train_evaluate_model(
        multi_step_dense, conv_window, conv_window,
        val_performance, test_performance)

    print("Long-Short Time Memory")
    lstm = \
        training_camp_2022.models.long_short_time_memory.LongShortTimeMemory(
            num_units=3)
    lstm_history = train_evaluate_model(
        lstm, monthly_window, quadrimestral_window,
        val_performance, test_performance)

    """
    feedback = training_camp_2022.models.feedback.Feedback(
        units=32, out_steps=12, num_features=8)
    feedback_history = train_evaluate_model(
        feedback, multi_window, multi_window,
        val_performance, test_performance)
    """

    model_list = [linear, esn, dense, multi_step_dense, lstm]
    plot_models(model_list)

    metric_name = "mean_absolute_error"
    metric_index = lstm.metrics_names.index(metric_name)

    training_camp_2022.view.time_series.comparing_performance(
        val_performance, test_performance, metric_index)
    training_camp_2022.view.time_series.show()


def launch_eval():
    pass


def plot_models(model_list):

    for model in model_list:
        model_plot_path = os.path.join(
            training_camp_2022.config.plot_path,
            f"{model.name}_model.png")
        model.summary()
        tf.keras.utils.plot_model(
            model, model_plot_path, show_shapes=True)


if __name__ == "__main__":
    launch_train()
    # launch_eval()
