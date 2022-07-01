import numpy as np
import pandas as pd
import statsmodels.tsa.api


class VectorAutoregressive(object):
    def __init__(
            self, data=None, dimension=None, order_p=None, var_params=None,
            mu=None, sigma=None, freq="MS", variable_names=None):
        """

        Args:
            data :
            dimension (int): total number of variables described by
                             the autoregressive process
            order_p (int|list): number of lags
            var_params (list): list of autoregressive matrix
            mu (np.array): intercept of the process
            sigma (np.ndarray): covariance matrix of the error term
                                The shape must be [dimension, dimension]
            freq (str): frequency
            variable_names (list): labels of the variables described by
                                   the autoregressive process
        """

        self.data = data

        # Dimension of the Auto-Regressive Process
        self.dimension = dimension
        self.freq = freq
        self.order_p = order_p
        self.var_params = np.array(var_params[::-1])

        self.mu = np.zeros(self.dimension)
        if mu is not None:
            self.mu = mu

        self.sigma = np.eye(self.dimension)
        if sigma is not None:
            self.sigma = sigma

        self.variable_names = [f"x_{i}" for i in range(self.dimension)]
        """Name of the autoregressive variables"""
        if variable_names is not None:
            self.variable_names = variable_names

    def generate_sample(self, n_samples, time_steps, var_init):
        """
        Generate n_samples realization of a vector auto-regressive process

        Args:
            n_samples (int): number of process to be generated
            time_steps (int): number of time-steps of the process
                              to be generated
            var_init (pd.DataFrame): initial values of the process
                                     to be generated

        Returns:
            a pandas.DataFrame
        """

        var_ts = pd.date_range(
            start=var_init.index[0], periods=time_steps, freq=self.freq)

        index = pd.MultiIndex.from_product(
            [var_ts, range(n_samples)], names=["date", "sample"])

        samples = pd.DataFrame(
            data=0., index=index, columns=self.variable_names)
        time_steps_init = var_init.shape[0]

        samples.loc[var_ts[:time_steps_init]] = (
            np.repeat(var_init.values, repeats=n_samples, axis=0))

        if self.mu is not None:
            samples.loc[var_ts[:time_steps_init]] -= np.repeat(
                self.mu.reshape(1, -1),
                repeats=(n_samples*time_steps_init), axis=0)

        shocks = np.zeros(shape=(time_steps, n_samples, self.dimension))

        shocks[1:, :] = np.random.multivariate_normal(
            mean=np.zeros(self.dimension), cov=self.sigma,
            size=(time_steps-1, n_samples), tol=1e-6)

        for time_step in range(time_steps_init, time_steps):
            ar_ = samples.loc[var_ts[time_step - self.order_p:time_step], :]
            ar_ = ar_.values.reshape((self.order_p, n_samples, self.dimension))

            samples.loc[var_ts[time_step], :] = (
                np.sum(np.matmul(ar_, self.var_params), axis=0) +
                shocks[time_step])

        if self.mu is not None:
            samples += np.repeat(
                self.mu.reshape(1, -1),
                repeats=(n_samples*time_steps), axis=0)

        """
            if self._trend == "c":
                arx_ts[time_step] += self._const * (1 - np.sum(self._ar))
        """
        self.data = samples.query("sample == 0").droplevel("sample")

        return samples

    def fit(self, maxlags=None, trend="c"):
        """

        Args:
            maxlags (int|list):
            trend (str):

        Returns:
            the result statsmodels object
        """
        if maxlags is None:
            maxlags = self.order_p

        # Make a VAR model
        sm_model = statsmodels.tsa.api.VAR(self.data)
        results = sm_model.fit(maxlags=maxlags, trend=trend)

        return results
