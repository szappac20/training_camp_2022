import os
import json
import pandas as pd

import training_camp_2022.config
import training_camp_2022.data.energy_bill_generator
import training_camp_2022.experiments.dataset_generator
import training_camp_2022.experiments.energy_consumption_profiler


def test_dataset_generation_shapes():
    """
    Test the shape of the generated datasets

    """
    task_1_path = os.path.join(training_camp_2022.config.data_path, "task_1")
    task_2_path = os.path.join(training_camp_2022.config.data_path, "task_2")

    training_features_1 = pd.read_csv(
        os.path.join(task_1_path, "training_features.csv"),
        sep=";", encoding="latin1")
    training_features_2 = pd.read_csv(
        os.path.join(task_2_path, "training_features.csv"),
        sep=";", encoding="latin1")

    training_users_1 = set(training_features_1["user"])
    training_users_2 = set(training_features_2["user"])
    err_msg = \
        f"Users in training set for task 1 and task 2 are different:\n" + \
        f"training set for task 1 has {training_users_1} users,\n" + \
        f"training set for task 2 has {training_users_2} users,"
    assert training_users_1 == training_users_2, err_msg

    validation_features_1 = pd.read_csv(
        os.path.join(task_1_path, "validation_features.csv"),
        sep=";", encoding="latin1")
    validation_features_2 = pd.read_csv(
        os.path.join(task_2_path, "validation_features.csv"),
        sep=";", encoding="latin1")

    validation_users_1 = set(validation_features_1["user"])
    validation_users_2 = set(validation_features_2["user"])
    err_msg = "Users in validation set for task 1 and task 2 are different"
    assert validation_users_1 == validation_users_2, err_msg

    test_features_1 = pd.read_csv(
        os.path.join(task_1_path, "test_features.csv"),
        sep=";", encoding="latin1")
    test_features_2 = pd.read_csv(
        os.path.join(task_2_path, "test_features.csv"),
        sep=";", encoding="latin1")

    test_users_1 = set(test_features_1["user"])
    test_users_2 = set(test_features_2["user"])
    err_msg = "Users in test set for task 1 and task 2 are different"
    assert test_users_1 == test_users_2, err_msg


    training_features_1 = training_features_1.drop(
        ["city_elevation", "is_resident", "upper_bound"], axis=1)

    commercial_offers = training_camp_2022.experiments.\
        energy_consumption_profiler.prepare_commercial_offers()
    training_year_projection_1 = training_camp_2022.experiments.\
        energy_consumption_profiler.project_over_year(
            training_features_1, commercial_offers)

    training_labels_1 = pd.read_csv(
        os.path.join(task_1_path, "training_labels.csv"),
        sep=";", encoding="latin1")
    training_labels_1["cluster"] = training_labels_1.apply(
        training_camp_2022.experiments.energy_consumption_profiler.best_offer,
        axis=1)
    training_labels_2 = pd.read_csv(
        os.path.join(task_2_path, "training_labels.csv"),
        sep=";", encoding="latin1")

    print(training_features_1.shape)
    # print(training_labels_1.columns)
    print(training_labels_1.shape)
    print(training_year_projection_1.shape)

    test_features_1 = test_features_1.drop(
        ["city_elevation", "is_resident", "upper_bound"], axis=1)

    commercial_offers = training_camp_2022.experiments.\
        energy_consumption_profiler.prepare_commercial_offers()
    test_year_projection_1 = training_camp_2022.experiments.\
        energy_consumption_profiler.project_over_year(
            test_features_1, commercial_offers)

    test_labels_1 = pd.read_csv(
        os.path.join(task_1_path, "test_labels.csv"),
        sep=";", encoding="latin1")
    test_labels_1["cluster"] = test_labels_1.apply(
        training_camp_2022.experiments.energy_consumption_profiler.best_offer,
        axis=1)
    test_labels_2 = pd.read_csv(
        os.path.join(task_2_path, "test_labels.csv"),
        sep=";", encoding="latin1")

    print(test_features_1.shape)
    # print(test_labels_1.columns)
    print(test_labels_1.shape)
    print(test_year_projection_1.shape)


def test_json_structure():
    def json_compare(json1, json2):
        # Compare all keys
        for key in json1.keys():
            # print("Elaborating key", key)

            if key == "64936980-6136-4bdc-94e3-d30f68599649":
                # In order to check the key corresponding to
                # the item of the fake json and the real one
                # the two different keys must be the same
                oldkey = list(json2.keys())[0]
                json2[key] = json2[oldkey]
                del json2[oldkey]

            if key in json2.keys():
                # print("Key", key, "is also in the fake json")
                # If subjson
                if isinstance(json1[key], dict):
                    json_compare(json1[key], json2[key])
                # else:
                #     if json1[key] != json2[key]:
                #         print("These entries are different:")
                #         print(json1[key])
                #         print(json2[key])
            else:
                print("Key", key, "is missing in the second json")

        return True

    energy_bill_json_generator = \
        training_camp_2022.data.energy_bill_generator.EnergyBillJsonGenerator()
    fake_energy_bills = energy_bill_json_generator.run(1)

    energy_bill_path = os.path.join(
        training_camp_2022.config.data_path, "test_energy_bill.json")
    with open(energy_bill_path, "rb") as fp:
        energy_bill_dict = json.loads(fp.read())

    json_compare(energy_bill_dict, fake_energy_bills[0])


if __name__ == "__main__":
    test_dataset_generation_shapes()
    # test_json_structure()
