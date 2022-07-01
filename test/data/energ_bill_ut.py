import json
import pprint
import os
import training_camp_2022.config
import training_camp_2022.data.energy_bill


def test_energy_bill_instance():
    """
    Test the creation of an energy bill from a json source file
    """
    json_path = os.path.join(
        training_camp_2022.config.data_path, "test_energy_bill.json")
    with open(json_path, "rb") as fp:
        energy_bill_dict = json.loads(fp.read())

    pprint.pprint(energy_bill_dict)
    # pprint.pprint(energy_bill_dict["customData"])

    test_energy_bill = \
        training_camp_2022.data.energy_bill.EnergyBill.from_dict(
            src_dict=energy_bill_dict)

    err_msg = "Wrong instance of energy bill"
    cond = isinstance(
        test_energy_bill, training_camp_2022.data.energy_bill.EnergyBill)
    assert cond, err_msg


if __name__ == "__main__":
    test_energy_bill_instance()
