import os

import training_camp_2022.config
import training_camp_2022.data.energy_bill_generator
import training_camp_2022.data.commercial_offer


def generate_datasets(num_users=100000):
    """
    Generate 4 datasets:
      - 2 features' datasets
      - 2 labels' datasets

    The two features' datasets are collections of monthly bills.
    They contain the same num_users different users
    with the following rationale:
      - Dataset 1: bills from the past months
                  (one year before with a variable number of bills)
      - Dataset 2: bills form the incoming months
                   (one year further with a fixed number of 12 bills)

    The two labels' datasets are collections of the best commercial offers.
    Each offer correspond to a single user, therefore the two datasets
    have the same size of num_users.
    They are defined by the following rationale:
      - Dataset 1: the best commercial offers for past year
      - Dataset 2: the best commercial offers for incoming year

    Each dataset will be created with the following split
      - 70% training
      - 20% validation
      - 10% test

    Args:
        num_users (int): total number of users

    Returns:
    """
    mono_oraria = training_camp_2022.data.commercial_offer.CommercialOffer(
        price_dict={"f1": 0.28, "f2": 0.28, "f3": 0.28}, name="mono_oraria")

    serale = training_camp_2022.data.commercial_offer.CommercialOffer(
        price_dict={"f1": 0.31, "f2": 0.26, "f3": 0.26}, name="serale")

    notturna = training_camp_2022.data.commercial_offer.CommercialOffer(
        price_dict={"f1": 0.56, "f2": 0.56, "f3": 0.}, name="notturna")

    bill_generator = \
        training_camp_2022.data.energy_bill_generator.EnergyBillGenerator(
            commercial_offers=[mono_oraria, serale, notturna])
    features_1_df, features_2_df, labels_1_df, labels_2_df = \
        bill_generator.run(num_users=num_users)

    training_index = num_users * 70 // 100
    validation_index = num_users * 20 // 100

    task_1_path = os.path.join(training_camp_2022.config.data_path, "task_1")
    task_2_path = os.path.join(training_camp_2022.config.data_path, "task_2")
    os.makedirs(task_1_path, exist_ok=True)
    os.makedirs(task_2_path, exist_ok=True)

    cond = features_1_df["user"] < training_index
    features_1_df[cond].to_csv(
        os.path.join(task_1_path, "training_features.csv"),
        sep=";", index=False)
    cond = features_2_df["user"] < training_index
    features_2_df[cond].to_csv(
        os.path.join(task_2_path, "training_features.csv"),
        sep=";", index=False)

    cond = labels_1_df["user"] < training_index
    labels_1_df[cond].to_csv(
        os.path.join(task_1_path, "training_labels.csv"),
        sep=";", index=False)
    cond = labels_2_df["user"] < training_index
    labels_2_df[cond].to_csv(
        os.path.join(task_2_path, "training_labels.csv"),
        sep=";", index=False)

    cond = (
        (features_1_df["user"] < training_index + validation_index) &
        (features_1_df["user"] >= training_index))
    features_1_df[cond].to_csv(
        os.path.join(task_1_path, "validation_features.csv"),
        sep=";", index=False)
    cond = (
        (features_2_df["user"] < training_index + validation_index) &
        (features_2_df["user"] >= training_index))
    features_2_df[cond].to_csv(
        os.path.join(task_2_path, "validation_features.csv"),
        sep=";", index=False)

    cond = (
        (labels_1_df["user"] < training_index + validation_index) &
        (labels_1_df["user"] >= training_index))
    labels_1_df[cond].to_csv(
        os.path.join(task_1_path, "validation_labels.csv"),
        sep=";", index=False)
    cond = (
        (labels_2_df["user"] < training_index + validation_index) &
        (labels_2_df["user"] >= training_index))
    labels_2_df[cond].to_csv(
        os.path.join(task_2_path, "validation_labels.csv"),
        sep=";", index=False)

    cond = features_1_df["user"] >= training_index + validation_index
    features_1_df[cond].to_csv(
        os.path.join(task_1_path, "test_features.csv"),
        sep=";", index=False)
    cond = features_2_df["user"] >= training_index + validation_index
    features_2_df[cond].to_csv(
        os.path.join(task_2_path, "test_features.csv"),
        sep=";", index=False)

    cond = labels_1_df["user"] >= training_index + validation_index
    labels_1_df[cond].to_csv(
        os.path.join(task_1_path, "test_labels.csv"),
        sep=";", index=False)
    cond = labels_2_df["user"] >= training_index + validation_index
    labels_2_df[cond].to_csv(
        os.path.join(task_2_path, "test_labels.csv"),
        sep=";", index=False)

    # print(dataset_1)
    # print(dataset_2)
    # print(dataset_3)
    # print(dataset_4)


if __name__ == "__main__":
    generate_datasets(num_users=1000)
    import matplotlib.pyplot as plt
    plt.show()
