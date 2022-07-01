import datetime

import numpy as np
import pandas as pd
import statsmodels.tsa.arima_process
import statsmodels.tsa.arima.model

import training_camp_2022.models.vector_autoregressive


def test_sample_generation():
    np.random.seed(12345)
    arparams = np.array([.75, -.25])
    maparams = np.array([.65, .35])
    ar = np.r_[1, -arparams] # add zero-lag and negate
    ma = np.r_[1, maparams] # add zero-lag
    y = statsmodels.tsa.arima_process.arma_generate_sample(ar, ma, 250)
    model = statsmodels.tsa.arima.model.ARIMA(
        endog=y, exog=None, order=(2, 0, 2), trend="n").fit()
    print(y.shape)


def test_var_generate_sample():
    """

    :return:
    """
    dimension = 3
    sigma = 5. * np.eye(dimension)
    mu = 20. * np.array([1., 20., 30.])
    var_params = [0.5 * np.eye(dimension), -0.1 * np.eye(dimension)]
    time_steps = 2000
    variable_names = ["f1", "f2", "f3"]

    test_var = \
        training_camp_2022.models.vector_autoregressive.VectorAutoregressive(
            dimension=dimension, order_p=2, var_params=var_params,
            mu=mu, sigma=sigma, variable_names=variable_names)

    time_steps_init = 2
    var_init = pd.DataFrame(
        data=mu.reshape((1, -1)).repeat(repeats=time_steps_init, axis=0),
        index=pd.date_range(
            start=datetime.date(year=2019, month=1, day=1),
            periods=time_steps_init, freq="MS"),
        columns=variable_names)

    test_var.generate_sample(
        n_samples=1, var_init=var_init, time_steps=time_steps)
    # print(test_var.data)

    results = test_var.fit(2, trend="c")
    print(results.summary())
    params = results.params
    print(params.iloc[:dimension+1])
    print(params.iloc[dimension+1:2*dimension+1])
    print(results.sigma_u)

    import pprint
    pprint.pprint(vars(results))


def test_var_statsmodels():

    import statsmodels.api as sm
    from statsmodels.tsa.api import VAR
    from statsmodels.tsa.base.datetools import dates_from_str

    mdata = sm.datasets.macrodata.load_pandas().data

    # prepare the dates index
    dates = mdata[['year', 'quarter']].astype(int).astype(str)
    quarterly = dates["year"] + "Q" + dates["quarter"]
    quarterly = dates_from_str(quarterly)

    mdata = mdata[['realgdp', 'realcons', 'realinv']]
    mdata.index = pd.DatetimeIndex(quarterly)
    mdata = mdata.asfreq("3M")

    data = np.log(mdata).diff().dropna()

    # make a VAR model
    model = VAR(data)
    print(data)
    print(model)


if __name__ == "__main__":
    # test_sample_generation()
    test_var_generate_sample()
    # test_var_statsmodels()
