import numpy as np
import pandas as pd
import datetime
import matplotlib.pyplot as plt


def generate_yearly_wave(
        init_month, time_steps, peak_months, exponent=6, freq="MS",
        seed=None):
    """

    Args:
        init_month (datetime.date):
        time_steps (int):
        peak_months (np.array):
        exponent (float):
        freq (str):
        seed :

    Returns:
        a Pandas DataFrame with three periodic signals
    """
    if seed is not None:
        np.random.seed(123)

    index = pd.date_range(start=init_month, periods=time_steps, freq=freq)
    yearly_wave_pd = pd.DataFrame(index=index)

    wave_signal = np.cos([
        (month.month-peak_months) / 12 * np.pi for month in index])
    yearly_wave_pd["f1"] = (
        (wave_signal[:, 0]**exponent - 0.5) * 2.
        + np.random.normal(scale=0.1, size=12))
    yearly_wave_pd["f2"] = (
        (wave_signal[:, 1]**exponent - 0.5) * 2.
        + np.random.normal(scale=0.1, size=12))
    yearly_wave_pd["f3"] = (
        (wave_signal[:, 2]**exponent - 0.5) * 2.
        + np.random.normal(scale=0.1, size=12))

    return yearly_wave_pd


if __name__ == "__main__":

    test_wave = generate_yearly_wave(
        init_month=datetime.date(year=2019, month=1, day=1),
        time_steps=12, peak_months=np.random.randint(1, 13, 3))
    test_wave.plot()
    plt.show()
