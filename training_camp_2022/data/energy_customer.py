
class EnergyCustomer(object):
    """

    Args:
        fiscal_id (str): fiscal code (unique identifier of the customer)
                         Length is 16
        customer_dict (dict):
    """
    def __init__(
            self, fiscal_id, customer_dict):
        self.id = customer_dict["id"]
        self.fiscal_id = fiscal_id

        self.emails = customer_dict["emails"]
        self.addresses = customer_dict["addresses"]
        self.phoneNumbers = customer_dict["phoneNumbers"]
