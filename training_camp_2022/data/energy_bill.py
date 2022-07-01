import json
import training_camp_2022.data.energy_customer


class EnergyBill(object):
    def __init__(
            self, customer, energy_bill_id, energy_data
    ):
        """
        Constructor of energyBill object

        Args:
            customer (src.data.energy_customer.EnergyCustomer):
            energy_bill_id (str): unique identifier of the bill
            energy_data (dict): collection of the energy data

        """
        self.customer = customer
        self.id = energy_bill_id

    @staticmethod
    def from_json():
        pass

    def to_json(self):
        pass

    @staticmethod
    def from_dict(src_dict):
        """
        Instance af on EnergyBill object from a given dictionary

        Args:
            src_dict (dict):

        """
        fiscal_id = src_dict["fiscalID"]
        customer_id = src_dict["bamData"]["items"].keys()[0]
        customer_dict = src_dict["bamData"]["items"][customer_id]

        energy_customer = training_camp_2022.data.energy_customer.EnergyCustomer(
            fiscal_id=fiscal_id, customer_dict=customer_dict)

        energy_data = src_dict["customData"]

        return EnergyBill(energy_customer, energy_data)

    def to_dict(self):
        pass
