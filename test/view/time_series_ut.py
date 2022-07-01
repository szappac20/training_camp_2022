import os
import pandas as pd

import training_camp_2022.config
import training_camp_2022.view.time_series


def test_generated_consumptions():
    task_1_path = os.path.join(training_camp_2022.config.data_path, "task_1")
    task_2_path = os.path.join(training_camp_2022.config.data_path, "task_2")

    features_1_df = pd.read_csv(
        os.path.join(task_1_path, "training_features.csv"),
        sep=";", encoding="latin1")

    training_camp_2022.view.time_series.plot_consumptions(features_1_df)


def test_generated_consumptions_users():
    task_1_path = os.path.join(training_camp_2022.config.data_path, "task_1")
    task_2_path = os.path.join(training_camp_2022.config.data_path, "task_2")

    features_2_df = pd.read_csv(
        os.path.join(task_2_path, "training_features.csv"),
        sep=";", encoding="latin1")

    training_camp_2022.view.time_series.plot_consumptions_users(
        features_2_df)


if __name__ == "__main__":
    # test_generated_consumptions()
    test_generated_consumptions_users()
