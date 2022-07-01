import numpy as np
import pandas as pd

import training_camp_2022.data.commercial_offer


def test_compute_yearly_cost():
    mono_oraria = training_camp_2022.data.commercial_offer.CommercialOffer(
        price_dict={"f1": 0.28, "f2": 0.28, "f3": 0.28}, name="mono_oraria")

    bills = pd.DataFrame(
        data={
            "month": range(1, 13), "year": 2021,
            "f1": np.random.randint(0, 70, 12),
            "f2": np.random.randint(0, 70, 12),
            "f3": np.random.randint(0, 70, 12),
        })

    cost = mono_oraria.compute_yearly_cost(bills)

    print(cost)


if __name__ == "__main__":
    test_compute_yearly_cost()
