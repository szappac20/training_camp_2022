import pandas as pd
import matplotlib.pyplot as plt

import training_camp_2022.data.energy_bill_generator
import training_camp_2022.data.commercial_offer
import training_camp_2022.data.energy_bill_generator


def test_history_consumptions():
    mono_oraria = training_camp_2022.data.commercial_offer.CommercialOffer(
        price_dict={"f1": 0.28, "f2": 0.28, "f3": 0.28}, name="mono_oraria")

    serale = training_camp_2022.data.commercial_offer.CommercialOffer(
        price_dict={"f1": 0.31, "f2": 0.26, "f3": 0.26}, name="serale")

    notturna = training_camp_2022.data.commercial_offer.CommercialOffer(
        price_dict={"f1": 0.56, "f2": 0.56, "f3": 0.}, name="notturna")

    num_users = 2
    commercial_offers = [mono_oraria, serale, notturna]
    # features_1_df, features_2_df, labels_1_df, labels_2_df = \
    #     training_camp_2022.data.energy_bill_generator.bill_generator.run(
    # num_users=num_users)

    bill_generator = \
        training_camp_2022.data.energy_bill_generator.EnergyBillGenerator(
            commercial_offers=[mono_oraria, serale, notturna])
    features_1_df, features_2_df, labels_1_df, labels_2_df = \
        bill_generator.run(num_users=num_users)


def test_create_fake_consumption_f1_f2_f3():
    history_months = 24
    upper_bound = pd.DataFrame(data=[3.]*history_months)
    city_density = 100.
    city_elevation = 1000.
    is_resident = 1
    age = 40
    test_consumptions = training_camp_2022.data.energy_bill_generator.\
        create_fake_consumption_f1_f2_f3(
            history_length=history_months, upper_bound=upper_bound,
            city_density=city_density, city_elevation=city_elevation,
            is_resident=is_resident, age=age)

    plt.show()


if __name__ == "__main__":
    # test_history_consumptions()
    test_create_fake_consumption_f1_f2_f3()
