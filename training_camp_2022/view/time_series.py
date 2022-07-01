import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import datetime


def plot_consumptions(consumptions):

    ax = consumptions[["f1", "f2", "f3"]].plot.bar(stacked=True)
    xtl = [item.get_text()[:7] for item in ax.get_xticklabels()]
    _ = ax.set_xticklabels(xtl)
    plt.show()
    return ax


def plot_consumptions_users(consumptions):
    def plot_clustered_stacked(
            dfall, labels=None, title="multiple stacked bar plot",
            H="/", **kwargs):
        """
        Given a list of dataframes, with identical columns and index,
        create a clustered stacked bar plot.
        labels is a list of the names of the dataframe, used for the legend
        title is a string for the title of the plot
        H is the hatch used for identification of the different dataframe
        """

        n_df = len(dfall)
        n_col = len(dfall[0].columns)
        n_ind = len(dfall[0].index)
        axe = plt.subplot(111)

        # For each data frame
        for df in dfall:
            axe = df.plot(
                kind="bar", linewidth=0, stacked=True, ax=axe,
                legend=False, grid=False, **kwargs)
            # make bar plots

        # get the handles we want to modify
        h, l = axe.get_legend_handles_labels()
        # len(h) = n_col * n_df
        for i in range(0, n_df * n_col, n_col):
            for j, pa in enumerate(h[i:i+n_col]):
                # for each index
                for rect in pa.patches:
                    rect.set_x(
                        rect.get_x() + 1 / float(n_df + 1) * i / float(n_col))
                    rect.set_hatch(H * int(i / n_col))
                    rect.set_width(1 / float(n_df + 1))

        axe.set_xticks((np.arange(0, 2 * n_ind, 2) + 1 / float(n_df + 1)) / 2.)
        axe.set_xticklabels(df.index, rotation=90)
        xtl = [item.get_text()[:7] for item in axe.get_xticklabels()]
        _ = axe.set_xticklabels(xtl)

        axe.set_title(title)

        # Add invisible data to add another legend
        n = []
        for i in range(n_df):
            n.append(axe.bar(0, 0, color="gray", hatch=H * i))

        l1 = axe.legend(h[:n_col], l[:n_col], loc=[1.01, 0.5])
        if labels is not None:
            l2 = plt.legend(n, labels, loc=[1.01, 0.1])
        axe.add_artist(l1)
        return axe

    df_list = []
    num_users = len(set(consumptions["user"]))
    for user in range(num_users):
        tmp_df = consumptions[["f1", "f2", "f3"]][consumptions["user"] == user]
        tmp_df["date"] = consumptions.apply(
            lambda row: datetime.datetime(
                year=row.year, month=row.month, day=1), axis=1)
        tmp_df = tmp_df.set_index("date")
        df_list.append(tmp_df)

    plot_clustered_stacked(df_list, range(len(df_list)))


def comparing_performance(val_performance, test_performance, metric_index):

    fig = plt.figure(figsize=(12, 8))

    x = np.arange(len(test_performance))
    width = 0.3
    val_mae = [v[metric_index] for v in val_performance.values()]
    test_mae = [v[metric_index] for v in test_performance.values()]

    plt.ylabel("Mean Absolute Error [kWh]")
    plt.bar(x - 0.17, val_mae, width, label="Validation")
    plt.bar(x + 0.17, test_mae, width, label="Test")
    plt.xticks(ticks=x, labels=test_performance.keys(), rotation=45)
    _ = plt.legend()

    return fig


def show():
    plt.show()
