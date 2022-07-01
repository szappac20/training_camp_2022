import training_camp_2022.data.window_generator
import training_camp_2022.models.multi_step_dense
import training_camp_2022.experiments.energy_consumptions_prediction


def test_output_shape():
    train_features_ds_pd, train_labels_ds_pd, \
        validation_features_ds_pd, validation_labels_ds_pd, \
        test_features_ds_pd, test_labels_ds_pd = \
            training_camp_2022.experiments.\
                energy_consumptions_prediction.prepare_data()

    train_df = \
        training_camp_2022.experiments.energy_consumptions_prediction.\
            impute_data(train_features_ds_pd)
    val_df = \
        training_camp_2022.experiments.energy_consumptions_prediction.\
            impute_data(validation_features_ds_pd)
    test_df = \
        training_camp_2022.experiments.energy_consumptions_prediction.\
            impute_data(test_features_ds_pd)

    quadrimestral_window = \
        training_camp_2022.data.window_generator.WindowGenerator(
            input_width=6, label_width=1, shift=1,
            train_df=train_df, val_df=val_df, test_df=test_df,
            label_columns=["f1", "f2", "f3"])

    multi_step_dense = training_camp_2022.models.multi_step_dense.MultiStepDense(
        num_units=3)
    multi_step_dense_history = \
        multi_step_dense.compile_and_fit(quadrimestral_window)

    # inputs, labels = monthly_window.example
    # print(inputs, labels)

    for example_inputs, example_labels in quadrimestral_window.train.take(3):
        print(f'Inputs shape (batch, time, features): {example_inputs.shape}')
        print(f'Labels shape (batch, time, features): {example_labels.shape}')

        """
        print("Inputs")
        print(*example_inputs)
        print("Labels")
        print(*example_labels)
        """

        predictions = multi_step_dense(example_inputs)
        print(predictions.shape)


if __name__ == "__main__":
    test_output_shape()
