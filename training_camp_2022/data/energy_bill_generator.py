import codicefiscale
import string
import pprint

import os
import numpy as np
import scipy.stats
import pandas as pd
import datetime
import calendar
import random

import statsmodels.tsa.arima_process

import training_camp_2022.data.italian_names
import training_camp_2022.config
import training_camp_2022.data.commercial_offer
import training_camp_2022.models.yearly_wave
import training_camp_2022.models.vector_autoregressive

pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", None)


def create_ar(corr_coeff, seed=None):
    if seed is not None:
        np.random.seed(123)

    ar = np.random.randn(3, 3)
    symm_ar = np.dot(ar, ar.transpose())
    symm_ar = corr_coeff * symm_ar / (4. * np.max(symm_ar))
    np.fill_diagonal(symm_ar, corr_coeff)
    return symm_ar


def create_ar_def_pos(corr_coeff):
    ar = create_ar(corr_coeff)
    while np.any(np.linalg.eigvals(ar) <= 0):
        print("Entering here")
        ar = create_ar(corr_coeff)

    return ar


ar_1 = create_ar_def_pos(0.1)
ar_12 = create_ar_def_pos(0.9)
# print("AR_1 positive:", np.all(np.linalg.eigvals(ar_1) > 0))
# print("AR_12 positive:", np.all(np.linalg.eigvals(ar_12) > 0))


var_params = [ar_1] + [np.zeros(shape=(3, 3))] * 10 + [ar_12]


def create_fake_consumption_f1(
        history_length, upper_bound, city_density,
        city_elevation, is_resident, age):
    """
    Consumption in high load period


    The baseline is an ar process with a lag 12
    (with coefficient is about 0.7)

    """

    ar = np.array([
        1., -.1, .0, .0, .0, .0, .0, .0, .0, .0, .0, .0,
        -np.random.normal(loc=0.7, scale=0.3)])

    f1 = statsmodels.tsa.arima_process.arma_generate_sample(
        ar=ar, ma=[1.], nsample=history_length)

    # Standard deviation is proportional to upper_bound
    std_f1 = 1.2 * np.random.normal(loc=upper_bound)
    mean_f1 = (
            15. * np.random.normal(loc=upper_bound) +
            40. * city_elevation / 2000. +
            20. * (age - 58.) / 40.)

    f1 = f1 * std_f1 + mean_f1

    if is_resident == 0:
        f1 = np.random.normal(loc=0.6, scale=0.1) * f1
    f1 = f1.round(0)
    f1 = np.clip(f1, a_min=0., a_max=None)
    return f1


def create_fake_consumption_f2(
        history_length, upper_bound, city_density,
        city_elevation, is_resident, age):
    """
    Consumption in early morning and evening periods
    """
    ar = np.array([
        1., -.1, .0, .0, .0, .0, .0, .0, .0, .0, .0, .0,
        -np.random.normal(loc=0.7, scale=0.3)])

    f2 = statsmodels.tsa.arima_process.arma_generate_sample(
        ar=ar, ma=[1.], nsample=history_length)

    # Standard deviation is proportional to upper_bound
    std_f2 = 1.2 * np.random.normal(loc=upper_bound)
    mean_f2 = (
            15. * np.random.normal(loc=upper_bound) +
            50. * city_elevation / 2000. +
            20. * (age - 58.) / 40.)

    f2 = f2 * std_f2 + mean_f2

    if is_resident == 0:
        f2 = np.random.normal(loc=0.6, scale=0.1) * f2
    f2 = f2.round(0)
    f2 = np.clip(f2, a_min=0., a_max=None)
    return f2


def create_fake_consumption_f3(
        history_length, upper_bound, city_density,
        city_elevation, is_resident, age):
    """
    Consumption in nightly and holiday period
    """

    ar = np.array([
        1., -.1, .0, .0, .0, .0, .0, .0, .0, .0, .0, .0,
        -np.random.normal(loc=0.7, scale=0.3)])

    f3 = statsmodels.tsa.arima_process.arma_generate_sample(
        ar=ar, ma=[1.], nsample=history_length)

    # Standard deviation is proportional to upper_bound
    std_f3 = 1.2 * np.random.normal(loc=upper_bound)
    mean_f3 = (
            15. * np.random.normal(loc=upper_bound) +
            50. * city_elevation / 2000. +
            30. * (58. - age) / 40.)

    f3 = f3 * std_f3 + mean_f3

    if is_resident == 0:
        f3 = np.random.normal(loc=0.6, scale=0.1) * f3
    f3 = f3.round(0)
    f3 = np.clip(f3, a_min=0., a_max=None)
    return f3


def create_fake_consumption_f1_f2_f3(
        history_length, upper_bound, city_density,
        city_elevation, is_resident, age):
    """
    Consumptions in the three different periods at the same time
    """
    init_month = datetime.datetime(year=2020, month=7, day=1)

    # TODO Insert more weight on summer or winter
    peak_months = random.choices(
        range(1, 13), weights=[3, 3, 2, 1, 1, 2, 3, 3, 2, 1, 1, 2], k=3)
    var_init = training_camp_2022.models.yearly_wave.generate_yearly_wave(
        init_month=init_month, time_steps=12, peak_months=np.array(peak_months))

    mean_f1 = (
        15. * np.random.normal(
            loc=upper_bound.loc[0], scale=0.3*upper_bound.iloc[0]) +
        50. * city_elevation / 2000. +
        20. * (age - 58.) / 40.)

    mean_f2 = (
        15. * np.random.normal(
            loc=upper_bound.iloc[0], scale=0.3*upper_bound.iloc[0]) +
        50. * city_elevation / 2000. +
        20. * (age - 58.) / 40.)

    mean_f3 = (
        15. * np.random.normal(
            loc=upper_bound.iloc[0], scale=0.3*upper_bound.iloc[0]) +
        50. * city_elevation / 2000. +
        18. * (58. - age) / 40.)
    mu = np.array([mean_f1, mean_f2, mean_f3]).clip(min=0.1)

    # Standard deviation is proportional to upper_bound
    sigma = 8. * np.diag(
        np.multiply(
            np.random.normal(
                loc=upper_bound.iloc[0],
                scale=0.05*upper_bound.iloc[0], size=3),
            min(abs(mu))/mu))

    var_init["f1"] = var_init["f1"] * sigma[0][0] + mean_f1
    var_init["f2"] = var_init["f2"] * sigma[1][1] + mean_f2
    var_init["f3"] = var_init["f3"] * sigma[2][2] + mean_f3

    var_model = \
        training_camp_2022.models.vector_autoregressive.VectorAutoregressive(
            dimension=3, order_p=12, var_params=var_params,
            mu=mu, sigma=sigma, variable_names=["f1", "f2", "f3"])

    consumptions = var_model.generate_sample(
        n_samples=1, var_init=var_init, time_steps=history_length+12)

    if bool(is_resident) is False:
        consumptions = np.random.normal(loc=0.6, scale=0.1) * consumptions
    consumptions = consumptions.round(0)
    consumptions = consumptions.clip(lower=0.)
    consumptions = consumptions.droplevel("sample", axis=0)

    return consumptions.iloc[-history_length:].values


class EnergyBillGenerator(object):
    def __init__(self, commercial_offers, keep_prob=0.2):
        """
        Args:
            commercial_offers (list): list of commercial offers
        """
        italian_tows_path = os.path.join(
            training_camp_2022.config.data_path, "comuni_italiani.csv")
        self.italian_tows_pd = pd.read_csv(
            italian_tows_path, sep=";", encoding="latin1")

        self.bill_features = [
            "month", "year", "user", "city_density", "city_elevation",
            "is_resident", "gender", "age", "upper_bound", "f1", "f2", "f3"]

        self.commercial_offers = commercial_offers
        self.commercial_offers_names = []
        self.scoring_labels = ["user"]
        for offer in self.commercial_offers:
            self.commercial_offers_names.append(offer.name)
            self.scoring_labels.append(offer.name)

        self.keep_prob = keep_prob

    def offer_fitting(self, row):
        """
        Compute the fitting probability of different commercial offer
        with respect to realized costs

        Args:
            row : a row of a DataFrame containing the costs for each offer

        Returns the probabilities of fitting a commercial offer
        """

        costs = row[self.commercial_offers_names]
        prob = (np.sum(costs)) / costs

        # Linear fitting
        # prob = prob / np.sum(prob)

        expon = 5.
        # Exponential fitting
        prob = prob ** expon / np.sum(prob ** expon)

        return prob

    def run(self, num_users, seed=None):
        """
        Run the generator in order to build a large number
        of (fake) energy bills
        Every user has a sequence of bills (of variable length)

        Args:
            num_users (int):

        Returns a list of energy bills
        """
        if seed is not None:
            np.random.seed(123)

        # gender = random.choice(["F", "M"])
        # name = ""
        # if gender == "F":
        #     name = random.choice(
        #         training_camp_2022.data.italian_names.female_names)
        # elif gender == "M":
        #     name = random.choice(
        #         training_camp_2022.data.italian_names.male_names)

        # birthdate = (
        #     datetime.datetime(1920, 1, 1) +
        #     datetime.timedelta(days=np.random.randint(0, 365*80)))
        # surname = random.choice(
        #     training_camp_2022.data.italian_names.surnames)

        # birthtown = random.choice(self.italian_tows)

        # codicefiscale.codicefiscale(
        #     surname=surname, name=name, sex=gender, birthdate=birthdate,
        #     birthplace="Torino")

        # For each user we may have a random number of bills
        # (in the last year)
        # past_range = pd.date_range(end="6/1/2018", periods=12, freq="MS")
        # future_range = pd.date_range(start="7/1/2018", periods=12, freq="MS")

        past_years = 1
        incoming_years = 1
        # Bills covering two years
        history_months = 12 * (past_years + incoming_years)
        init_dates = pd.date_range(
            end="6/1/2023", periods=history_months, freq="MS")
        end_dates = []
        for i in range(history_months):
            d = init_dates[i]
            end_dates.append(datetime.datetime(
                year=d.year, month=d.month,
                day=calendar.monthrange(d.year, d.month)[1]))

        past_features_dataset = pd.DataFrame(columns=self.bill_features)
        future_features_dataset = pd.DataFrame(columns=self.bill_features)

        past_labels_dataset = pd.DataFrame(columns=self.scoring_labels)
        future_labels_dataset = pd.DataFrame(columns=self.scoring_labels)

        for user in range(num_users):
            if (user+1) % (num_users//10) == 0:
                print("Generating user", user, num_users)
            consumptions = pd.DataFrame(
                data={"month": init_dates.month, "year": init_dates.year})

            # Selection of the city includes the number of inhabitants
            # as weights of the random choice
            num_towns = len(self.italian_tows_pd)
            idx = random.choices(
                range(num_towns), weights=self.italian_tows_pd["Popolazione"],
                k=1)[0]
            city = self.italian_tows_pd.loc[idx]

            consumptions["user"] = user
            consumptions["city_density"] = city["Densità"]
            consumptions["city_elevation"] = city["Altitudine"]
            is_resident = random.choices([0, 1], weights=(40, 60))[0]
            consumptions["is_resident"] = is_resident
            consumptions["gender"] = random.choice(["M", "F"])
            age = np.random.randint(18, 98)
            consumptions["age"] = age
            consumptions["upper_bound"] = random.choices(
                [1.5, 3., 4.5, 6., 7], weights=(10, 80, 5, 4, 1))[0]

            consumptions[["f1", "f2", "f3"]] = create_fake_consumption_f1_f2_f3(
                    history_length=history_months,
                    upper_bound=consumptions["upper_bound"],
                    city_density=city["Densità"],
                    city_elevation=city["Altitudine"],
                    is_resident=is_resident, age=age)

            """
            consumptions["f1"] = create_fake_consumption_f1(
                history_length=history_months,
                upper_bound=consumptions["upper_bound"],
                city_density=city["Densità"],
                city_elevation=city["Altitudine"],
                is_resident=is_resident, age=age)
            consumptions["f2"] = create_fake_consumption_f2(
                history_length=history_months,
                upper_bound=consumptions["upper_bound"],
                city_density=city["Densità"],
                city_elevation=city["Altitudine"],
                is_resident=is_resident, age=age)
            consumptions["f3"] = create_fake_consumption_f3(
                history_length=history_months,
                upper_bound=consumptions["upper_bound"],
                city_density=city["Densità"],
                city_elevation=city["Altitudine"],
                is_resident=is_resident, age=age)
            """

            past_consumptions = consumptions.iloc[:12 * past_years]
            incoming_consumptions = consumptions.iloc[12 * past_years:]

            past_best_offer = pd.DataFrame(data={"user": [user]})
            past_best_offer[["f1", "f2", "f3"]] = \
                past_consumptions[["f1", "f2", "f3"]].sum()

            future_best_offer = pd.DataFrame(data={"user": [user]})
            future_best_offer[["f1", "f2", "f3"]] = \
                incoming_consumptions[["f1", "f2", "f3"]].sum()

            for offer in self.commercial_offers:
                past_best_offer[offer.name] = past_best_offer.apply(
                    offer.compute_yearly_cost, axis=1)
                future_best_offer[offer.name] = future_best_offer.apply(
                    offer.compute_yearly_cost, axis=1)

            fitt_prob = [
                f"prob_{name}" for name in self.commercial_offers_names]
            past_best_offer[fitt_prob] = \
                past_best_offer.apply(
                    self.offer_fitting, axis=1, result_type="expand")
            future_best_offer[fitt_prob] = \
                future_best_offer.apply(
                    self.offer_fitting, axis=1, result_type="expand")

            if self.keep_prob < 1.:
                # Randomly delete a variable number of months in the past
                drop_mask = np.ones(12 * past_years)
                while sum(drop_mask) == (12 * past_years):
                    drop_mask = scipy.stats.bernoulli.rvs(
                        p=(1-self.keep_prob), size=12 * past_years)

                drop_mask = np.arange(12 * past_years)[drop_mask.astype(bool)]
                past_consumptions = past_consumptions.drop(drop_mask)

            past_features_dataset = pd.concat(
                [past_features_dataset, past_consumptions], ignore_index=True)
            future_features_dataset = pd.concat(
                [future_features_dataset, incoming_consumptions],
                ignore_index=True)

            past_labels_dataset = pd.concat(
                [past_labels_dataset, past_best_offer], ignore_index=True)
            future_labels_dataset = pd.concat(
                [future_labels_dataset, future_best_offer], ignore_index=True)

        return \
            past_features_dataset, future_features_dataset,\
            past_labels_dataset, future_labels_dataset


class EnergyBillJsonGenerator(object):
    def __init__(self):
        italian_tows_path = os.path.join(
            training_camp_2022.config.data_path, "comuni_italiani.csv")
        self.italian_tows_pd = pd.read_csv(
            italian_tows_path, sep=";", encoding="latin1")
        # self.italian_tows = italian_tows_pd["Denominazione in italiano"]

    def run(self, num_bills):
        """
        Run the generator in order to build a given number of energy bills

        Args:
            num_bills:

        Returns a list of energy bills
        """

        bills = []
        bill_dict = {}
        for n in range(num_bills):
            gender = random.choice(["F", "M"])
            name = ""
            if gender == "F":
                name = random.choice(
                    training_camp_2022.data.italian_names.female_names)
            elif gender == "M":
                name = random.choice(
                    training_camp_2022.data.italian_names.male_names)

            birthdate = (
                datetime.datetime(1920, 1, 1) +
                datetime.timedelta(days=np.random.randint(0, 365*80)))
            surname = random.choice(
                training_camp_2022.data.italian_names.surnames)

            birth_index = np.random.randint(0, len(self.italian_tows_pd))
            residence_index = np.random.randint(0, len(self.italian_tows_pd))
            birthplace = self.italian_tows_pd.loc[birth_index][
                "Denominazione in italiano"]

            fiscal_id = codicefiscale.codicefiscale.encode(
                surname=surname, name=name, sex=gender, birthdate=birthdate,
                birthplace=birthplace)
            letters_numbers = string.ascii_lowercase + string.digits
            item_id = '-'.join([
                ''.join(random.choice(string.digits) for _ in range(8)),
                ''.join(random.choice(string.digits) for _ in range(4)),
                ''.join(random.choice(letters_numbers) for _ in range(4)),
                ''.join(random.choice(letters_numbers) for _ in range(4)),
                ''.join(random.choice(letters_numbers) for _ in range(12))
            ])

            instance_id = '-'.join([
                ''.join(random.choice(letters_numbers) for _ in range(8)),
                ''.join(random.choice(letters_numbers) for _ in range(4)),
                ''.join(random.choice(letters_numbers) for _ in range(4)),
                ''.join(random.choice(letters_numbers) for _ in range(4)),
                ''.join(random.choice(letters_numbers) for _ in range(12))
            ])

            bill_dict = {
                "source": "energy-simulazione",
                "sourceVersion": "1.28.70",
                "eventType": "simulazione",
                "eventCategory": "ENERG",
                "fiscalID": fiscal_id,
                "cluster": "on-prem",
                "eventTime": "????",
                "instanceID": instance_id,
                "stepName": "Simulazione",
                "data": {"cpq_codvista": "ENERG"},
                "contextData": self.generate_context_data(),
                "bamData": {
                    "items": self.generate_item(
                        item_id, birth_index, residence_index, gender,
                        name, surname, birthdate, fiscal_id),
                    "info": {
                        "lastUpdate": "????"
                    },
                },
                "customData": self.generate_consumptions(),
                "technicalData": self.generate_technical_data(),
                "products": [
                    {
                        "id": "????",
                        "name": "Poste Dual Sostenibile",
                        "productStepId": "????",
                        "catalogViewCode": "????"
                    }
                ],
            }

        bills.append(bill_dict)

#        pprint.pprint(bill_dict, sort_dicts=False)

        return bills

    def generate_item(
            self, item_id, birth_index, residence_index, domicilio_index,
            gender, name, surname, birthdate, fiscal_id):
        item_dict = {
            item_id: {
                "addresses": {},
                "advancedEletronicSignatures": [],
                "attributes": [],
                "bankAccounts": [],
                "classification": {
                  "code": "R",
                  "value": "RETAIL"
                },
                "emails": [
                    {
                        "email": (
                            name.lower() + "." + surname.lower() +
                            "@test.com"),
                        "PEC": False,
                        "type": {
                            "code": "2",
                            "value": "EMAIL"
                        },
                        "customData": {
                            "dataModified": True,
                            "asyncCertification": True
                        }
                    }
                ],
                "id": item_id,
                "idAUC": "????",
                "legalForm": {
                  "code": "PF",
                  "value": "PERSONA FISICA"
                },
                "legalInfo": self.generate_legal_info(),
                "legalNature": "PF",
                "personInfo": self.generate_personal_info(
                    birth_index=birth_index, residence_index=residence_index,
                    gender=gender, firstname=name, lastname=surname,
                    birthdate=birthdate),
                "phoneNumbers": [
                    {
                        "type": {
                            "code": "6",
                            "value": "CELLULARE"
                        },
                        "countryCode": {
                            "code": "IT",
                            "value": "+39"
                        },
                        "areaCode": random.choice(
                            ["347", "346", "340", "328", "320", "338", "335"]),
                        "number": ''.join(
                            random.choice(string.digits) for _ in range(7)),
                        "customData": {
                          "dataModified": True,
                          "asyncCertification": True
                        }
                    }
                ],
                "privacy": self.generate_privacy(
                    pf1_value=True, pf2_value=True,
                    pf3_value=True, pf4_value=True),
                "questionnaires": [],
                "relatedCustomer": [],
                "taxData": {
                    "taxCode": fiscal_id,
                    "taxResidences": [
                        {
                            "country": {
                                "code": "IT",
                                "value": "ITALIA"
                            },
                            "nif": fiscal_id,
                            "flagNoNif": False
                        }
                    ],
                    "customData": {
                        "newTaxCode": fiscal_id
                    }
                },
                "type": {
                    "code": "A",
                    "value": "ACTUAL"
                },
                "customData": {},
            },
        }

        return item_dict

    def generate_addresses(
            self, residence_index, domicilio_index=None):
        residence_city_row = self.italian_tows_pd.loc[residence_index]

        addresses_res = {}
        addresses_res["attributes"] = []
        addresses_res["certificationDate"] = "2022-01-27T17:30:18Z"
        addresses_res["city"] = {
            "code": residence_city_row["Codice Catastale del comune"],
            "value": residence_city_row["Denominazione in italiano"]}
        addresses_res["country"] = {
            "code": "IT",
            "value": "ITALIA"}
        addresses_res["province"] = {
            "code": residence_city_row["Sigla automobilistica"],
            "value": residence_city_row[
                "Denominazione dell'Unità territoriale " +
                "sovracomunale\n(valida a fini statistici)"]}
        addresses_res["type"] = {"code": "2", "value": "RESIDENZA"}
        addresses_res["zipCode"] = {
            "code": residence_city_row["CAP_INIT"],
            "value": residence_city_row["CAP_INIT"]
        },
        addresses_list = [addresses_res]

        if domicilio_index is not None:
            domicilio_city_row = self.italian_tows_pd.loc[domicilio_index]
            addresses_dom = {}
            addresses_dom["attributes"] = []
            addresses_dom["certificationDate"] = "2022-01-27T17:30:18Z"
            addresses_dom["city"] = {
                "code": domicilio_city_row["Codice Catastale del comune"],
                "value": domicilio_city_row["Denominazione in italiano"]}
            addresses_dom["country"] = {
                "code": "IT",
                "value": "ITALIA"}
            addresses_dom["province"] = {
                "code": domicilio_city_row["Sigla automobilistica"],
                "value": domicilio_city_row[
                    "Denominazione dell'Unità territoriale " +
                    "sovracomunale\n(valida a fini statistici)"]}
            addresses_dom["type"] = {"code": "6", "value": "DOMICILIO"}
            addresses_list.append(addresses_dom)

        return addresses_list

    def generate_personal_info(
            self, birth_index, gender,
            firstname, lastname, birthdate):
        """
        Generate the key personalInfo

        """
        birth_row = self.italian_tows_pd.loc[birth_index]
        birthplace = birth_row["Denominazione in italiano"]
        codeplace = birth_row["Codice Catastale del comune"]
        gender_value = "FEMMINA"
        if gender == "M":
            gender_value = "MASCHIO"

        personal_info_dict = {
            "birthCity": {
                "code": codeplace,
                "value": birthplace,
            },
            "birthCountry": {
                "code": "IT",
                "value": "ITALIA"
            },
            "birthProvince": {
                "code": birth_row["Sigla automobilistica"],
                "value": birth_row[
                    "Denominazione dell'Unità territoriale " +
                    "sovracomunale\n(valida a fini statistici)"]
            },
            "citizenshipNoUSA": False,
            "citizenships": [
                {
                    "country": {
                        "code": "IT",
                        "value": "ITALIA"
                    }
                }
            ],
            "digitalIdentity": {},
            "economicClassification": [],
            "firstName": firstname,
            "gender": {
                "code": gender,
                "value": gender_value
            },
            "lastName": lastname,
            "customData": {
                "customerCodes": [
                ],
            },
            "birthDate": birthdate.strftime("%Y-%m-%d"),
        }

        return personal_info_dict

    def generate_legal_info(self):
        """
        Generate the key legalInfo

        """
        legal_info_dict = {
            "attributes": [],
            "businessActivities": [
                {
                    "type": "SAE",
                    "code": "0600",
                    "description": "FAMIGLIE CONSUMATRICI"
                },
                {
                    "type": "RAE",
                    "code": "0000",
                    "description": "NON VALORIZZABILE"
                },
                {
                    "type": "CIAE",
                    "code": "6093",
                    "description": "ALTRE PERSONE FISICHE"
                }
            ],
            "salesPoints": [],
            "subscriptions": [],
            "websites": []
            }

        return legal_info_dict

    def generate_context_data(self):
        """
        Generate the key contextData
        """
        letters_numbers = string.ascii_lowercase + string.digits
        operator_id = ''.join(random.choice(string.digits) for _ in range(8))
        abi = "00000"

        context_data_dict = {
            "stepID": "energy-simulazione",
            "workstation": str(np.random.randint(10)).zfill(2),
            "office": abi,
            "operation": "????",
            "sessionID": "????",
            "correlationID": "????",
            "channel": random.choice(["UP"]),
            "operatorID": operator_id,
            "roleID": "????"
        }

        return context_data_dict

    def generate_privacy(
            self, pf1_value, pf2_value, pf3_value, pf4_value):

        privacy_dict = {
            "allRequiredConsentsExpressed": True,
            "consents": [
                {
                    "type": "Consenso",
                    "code": "PF1",
                    "value": pf1_value
                },
                {
                    "type": "Consenso",
                    "code": "PF2",
                    "value": pf2_value
                },
                {
                    "type": "Consenso",
                    "code": "PF3",
                    "value": pf3_value
                },
                {
                    "type": "Consenso",
                    "code": "PF4",
                    "value": pf4_value
                }
            ]
        }

        return privacy_dict

    def generate_consumptions(self):
        custom_data_dict = {
            "energy": {},
            "documentId": "????",
            "setResultCalled": True,
            "cte": {
                "dataScadenza": "9999-12-31",
                "prezzoUnico": "????",
                "dataInizioValidita": "????",
                "dataFineValidita": "9999-12-31"
            },
            "dossierCustomerRef": "????",
            "dossierNumber ": "????"
        }

        return custom_data_dict

    def generate_technical_data(self):
        technical_data_dict = {
            "rispostaOCR": {},
            "energy": {
                "indirizzoFornituraCIVICO_mod": "Y",
                "indirizzoFornituraCAP_mod": "Y",
                "indirizzoFornituraCITTA_mod": "Y",
                "indirizzoFornituraVIA_mod": "Y",
                "categoriaDUtilizzo_mod": "Y",
                "dataAttivazioneFornitura_mod": "Y",
                "dataFineConsumo_mod": "Y",
                "provinciaFornOCR_mod": "Y",
                "consumoAnnuo_mod": "Y"
            }
        }

        return technical_data_dict
