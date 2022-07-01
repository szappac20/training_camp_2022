# import tensorflow_decision_forests as tfdf

import os
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

import training_camp_2022.config
import training_camp_2022.data.energy_bill_generator
import training_camp_2022.data.commercial_offer
import training_camp_2022.models
import training_camp_2022.experiments
import training_camp_2022.view.clustering


"""
References:
    https://ec.europa.eu/eurostat/statistics-explained/
    index.php?title=Energy_consumption_in_households
    https://ieeexplore.ieee.org/document/7966398
    Monthly energy consumption forecast: A deep learning approach

    Front. Energy Res., 22 October 2021 |
    https://doi.org/10.3389/fenrg.2021.779587
    Power Consumption Predicting and Anomaly Detection
    Based on Transformer and K-Means

    Italian Household Load Profiles: A Monitoring Campaign
    https://www.mdpi.com/2075-5309/10/12/217


    https://www.sciencedirect.com/science/article/pii/S0169023X17303282
    Learning process modeling phases from
    modeling interactions and eye tracking data

    https://www.techedgegroup.com/blog/data-science-process-problem-statement-definition


Luce
https://energy.poste.it/ui/index?accepted=L&L=211118LFIXPOSTE&commodity=L

Gas
https://energy.poste.it/ui/index?accepted=G&G=211118GFIXPOSTE&commodity=G


Terna
https://www.terna.it/en/electric-system/statistical-data-forecast/statistical-forecasts



Predictive model
    Trainable system fed by the fields parsed by the OCR (json file)
    The output is the collection of forecasts for the next n months (n=6)

Prescriptive model
    Deterministic (non-trainable) model chosing the best commercial offer
    on the base of the energy consumptions estimated by the predictive model
    It is supervised by the predictive model
    The commercial offer must fit the energy consumptions estimates
    and the exceeding of the energy threshold forecasted by
    the predictive model.

Steps

    1. Generate a dataset with at least 100k records
       The data range must cover at least 6 months
       The dataset must be compliant with respect to the
       fields of the json file produced by the OCR
       (obtained by scanning energy bills from customers
       previous energy suppliers or prospect)

    2. Define a baseline model for the predictive model

    3. Define a baseline model for the prescriptive model

    4. Compare the baseline models with other simple models

"""

"""

Training Camp 2022
Title: Driving Business Decisions through AI

Abstract
Algorithmic Business Thinking is a paradigm defining algorithms bases
on a symbiotic cooperation of humans and machines working side-by-side
in mitigating the risk of unconscious biases
that could cause effectiveness loss on the final decision.

The partnership of humans and machines derives AI driven business decisions
in a faster and more effective way, supporting better scaling
when heterogeneous business use-cases demanding increases.

AI algorithms will improve marketing strategies and
drive product evolutionary transformation,
embedding the AI in the product itself or using AI to design for innovation.
On the other hand, in some cases this paradigm may raise the need
for an ethical consilience among decisions provided by machine vs human
("Is this the right thing to do; what are the unintended consequences?")
This camp is aimed at the accomplishment of the following targets:
   Collect the data provided during the on-boarding of a customer
   purchasing energy supply offer in order
   to predict the energy consumption in the next period of time,
   Drive the identification of a business decision (i.e. commercial offer),
   and ensure consilience of sustainability and consumption demand.

As a prerequisite for this training, it's supposed that the students start
from a solid knowledge of Python and machine learning
most frequently adopted libraries.

"""

# # Check the version of TensorFlow Decision Forests
# print("Found TensorFlow Decision Forests v" + tfdf.__version__


def prepare_commercial_offers():

    mono_oraria = training_camp_2022.data.commercial_offer.CommercialOffer(
        price_dict={"f1": 0.28, "f2": 0.28, "f3": 0.28}, name="mono_oraria")

    serale = training_camp_2022.data.commercial_offer.CommercialOffer(
        price_dict={"f1": 0.31, "f2": 0.26, "f3": 0.26}, name="serale")

    notturna = training_camp_2022.data.commercial_offer.CommercialOffer(
        price_dict={"f1": 0.56, "f2": 0.56, "f3": 0.}, name="notturna")

    commercial_offers = [mono_oraria, serale, notturna]
    return commercial_offers


def best_offer(row):
    commercial_offers = prepare_commercial_offers()
    columns = [f"prob_{co.name}" for co in commercial_offers]

    cluster = np.argmax(row[columns])
    return cluster


def best_projected_offer(row):
    commercial_offers = prepare_commercial_offers()
    columns = [co.name for co in commercial_offers]
    cluster = np.argmin(row[columns])
    return cluster


def prepare_data():
    task_1_path = os.path.join(training_camp_2022.config.data_path, "task_1")

    train_features_ds_pd = pd.read_csv(
        os.path.join(task_1_path, "training_features.csv"),
        sep=";", encoding="latin1")
    train_labels_ds_pd = pd.read_csv(
        os.path.join(task_1_path, "training_labels.csv"),
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
        test_features_ds_pd, test_labels_ds_pd


def project_over_year(train_features_ds_pd, commercial_offers):
    year_projection = \
        12 * train_features_ds_pd[["user", "f1", "f2", "f3"]].groupby(
            by=["user"]).mean()

    for offer in commercial_offers:
        year_projection[offer.name] = year_projection.apply(
            offer.compute_yearly_cost, axis=1)

    year_projection["cluster"] = year_projection.apply(
        best_projected_offer, axis=1)
    return year_projection


def launch_train(model_name):
    commercial_offers = prepare_commercial_offers()
    train_features_ds_pd, train_labels_ds_pd, _, _ = prepare_data()

    if model_name == "baseline":
        train_features_ds_pd = train_features_ds_pd.drop(
            ["city_elevation", "is_resident", "upper_bound"], axis=1)
        train_labels_ds_pd["cluster"] = train_labels_ds_pd.apply(
            best_offer, axis=1)

        train_year_projection = project_over_year(
            train_features_ds_pd, commercial_offers)

    elif model_name == "random_forest":
        rf_train_df = train_features_ds_pd.groupby(by=["user"]).first()
        rf_train_df["len"] = train_features_ds_pd[["user", "age"]].groupby(
            by=["user"]).count()

        train_year_projection = project_over_year(
            train_features_ds_pd, commercial_offers)

        # rf_train_df = pd.concat([
        #     rf_train_df[[
        #         "city_density", "city_elevation", "is_resident",
        #         "gender", "age", "upper_bound"]],
        #     year_projection[["f1", "f2", "f3", "cluster"]]], axis=1)
        rf_train_df = pd.concat([
            rf_train_df[["gender", "age"]],
            train_year_projection[["f1", "f2", "f3", "cluster"]]], axis=1)

        train_ds = tfdf.keras.pd_dataframe_to_tf_dataset(
            rf_train_df, label="cluster")

        # Specify the model
        random_forest_model = tfdf.keras.RandomForestModel()

        # Optionally, add evaluation metrics
        random_forest_model.compile(metrics=["accuracy"])

        # Train the model
        random_forest_model.fit(x=train_ds)

        return random_forest_model


def launch_eval(model_name="baseline"):
    commercial_offers = prepare_commercial_offers()
    class_names = [co.name for co in commercial_offers]

    _, _, test_features_ds_pd, test_labels_ds_pd = prepare_data()

    if model_name == "baseline":

        test_features_ds_pd = test_features_ds_pd.drop(
            ["city_elevation", "is_resident", "upper_bound"], axis=1)

        test_labels_ds_pd["cluster"] = test_labels_ds_pd.apply(
            best_offer, axis=1)

        test_year_projection = project_over_year(
            test_features_ds_pd, commercial_offers)

        cm = confusion_matrix(
            test_labels_ds_pd["cluster"], test_year_projection["cluster"])

        # Log the confusion matrix as an image summary
        figure = training_camp_2022.view.clustering.plot_cm_with_labels(
            cm, class_names=class_names)
        training_camp_2022.view.clustering.show()

    elif model_name == "random_forest":
        rf_test_df = test_features_ds_pd.groupby(by=["user"]).first()
        rf_test_df["len"] = test_features_ds_pd[["user", "age"]].groupby(
            by=["user"]).count()

        test_year_projection = project_over_year(
            test_features_ds_pd, commercial_offers)

        # rf_test_df = pd.concat([
        #     rf_test_df[[
        #         "city_density", "city_elevation", "is_resident",
        #         "gender", "age", "upper_bound"]],
        #     test_year_projection[["f1", "f2", "f3", "cluster"]]], axis=1)
        rf_test_df = pd.concat([
            rf_test_df[["gender", "age"]],
            test_year_projection[["f1", "f2", "f3", "cluster"]]], axis=1)

        test_ds = tfdf.keras.pd_dataframe_to_tf_dataset(
            rf_test_df, label="cluster")

        test_pred = model.predict(test_ds)
        cluster_pred = np.argmax(test_pred, axis=1)

        cm = confusion_matrix(rf_test_df["cluster"], cluster_pred)

        # Log the confusion matrix as an image summary
        figure = training_camp_2022.view.clustering.plot_cm_with_labels(
            cm, class_names=class_names)


if __name__ == "__main__":
    import training_camp_2022.experiments.dataset_generator
    training_camp_2022.experiments.dataset_generator.generate_datasets(
        num_users=1000)
    model = launch_train(model_name="baseline")
    launch_eval(model_name="baseline")

