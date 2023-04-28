# -------------------------------------------------------------------------------
# different plot style functions called from the main one depending on parameter
# -------------------------------------------------------------------------------

# See mapping user plot structure keywords to corresponding functions in the end of this file

import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.dates import DateFormatter, date2num, num2date
from matplotlib.figure import Figure
from pandas import DataFrame, Timedelta

from . import utils


def plot_vs_time(
    data_channel: DataFrame, fig: Figure, ax: Axes, plot_info: dict, color=None
):
    # -------------------------------------------------------------------------
    # plot this data vs time
    # -------------------------------------------------------------------------

    # need to plot this way, and not data_position.plot(...) because the datetime column is of type Timestamp
    # plotting this way, to_pydatetime() converts it to type datetime which is needed for DateFormatter
    # changing the type of the column itself with the table does not work
    data_channel = data_channel.sort_values("datetime")

    res_col = color
    all_col = (
        color
        if plot_info["resampled"] == "no" or plot_info["parameter"] == "event_rate"
        else "darkgray"
    )

    if plot_info["resampled"] != "only":
        ax.plot(
            data_channel["datetime"].dt.to_pydatetime(),
            data_channel[plot_info["parameter"]],
            zorder=0,
            color=all_col,
        )

    # -------------------------------------------------------------------------
    # plot resampled average
    # -------------------------------------------------------------------------

    if plot_info["resampled"] != "no":
        # unless event rate - already resampled and counted in some time window
        if not plot_info["parameter"] == "event_rate":
            # resample in given time window, as start pick the first timestamp in table
            resampled = (
                data_channel.set_index("datetime")
                .resample(plot_info["time_window"], origin="start")
                .mean(numeric_only=True)
            )
            # will have datetime as index after resampling -> put back
            resampled = resampled.reset_index()
            # the timestamps in the resampled table will start from the first timestamp, and go with sampling intervals
            # I want to shift them by half sampling window, so that the resampled value is plotted in the middle time window in which it was calculated
            resampled["datetime"] = (
                resampled["datetime"] + Timedelta(plot_info["time_window"]) / 2
            )

            ax.plot(
                resampled["datetime"].dt.to_pydatetime(),
                resampled[plot_info["parameter"]],
                color=res_col,
                zorder=1,
                marker="o",
                linestyle="-",
            )

    # -------------------------------------------------------------------------
    # beautification
    # -------------------------------------------------------------------------

    # --- time ticks/labels on x-axis
    min_x = date2num(data_channel.iloc[0]["datetime"])
    max_x = date2num(data_channel.iloc[-1]["datetime"])
    time_points = np.linspace(min_x, max_x, 10)
    labels = [num2date(time).strftime("%Y\n%m/%d\n%H:%M") for time in time_points]

    # set ticks
    ax.set_xticks(time_points)
    ax.set_xticklabels(labels)

    # --- set labels
    fig.supxlabel("UTC Time")
    y_label = (
        f"{plot_info['label']}, {plot_info['unit_label']}"
        if plot_info["unit_label"] == "%"
        else f"{plot_info['label']} [{plot_info['unit_label']}]"
    )
    fig.supylabel(y_label)


def par_vs_ch(
    data_channel: DataFrame, fig: Figure, ax: Axes, plot_info: dict, color=None
):
    if len(data_channel[plot_info["parameter"]].unique()) > 1:
        utils.logger.error(
            "\033[91mYou are trying to plot multiple values for a given channel.\nThis is not possible, there should be only one unique value! Try again.\033[0m"
        )
        return

    # -------------------------------------------------------------------------
    # plot data vs channel ID
    # -------------------------------------------------------------------------
    # trick to get a correct position of channels, independently from the 'channel' entry
    # (everything was ok when using 'fcid'; but using 'rawid' as 'channel', we loose the possibility to order channels over x-axis in a decent way)
    map_dict = utils.MAP_DICT
    location = data_channel["location"].unique()[0]
    position = data_channel["position"].unique()[0]
    ax.scatter(
        map_dict[str(location)][str(position)],
        data_channel[plot_info["parameter"]].unique()[0],
        color=color,
    )

    # -------------------------------------------------------------------------
    # beautification
    # -------------------------------------------------------------------------
    ax.set_ylabel("")
    ax.set_xlabel("")

    # --- set labels
    fig.supxlabel("Channel ID")
    y_label = (
        f"{plot_info['label']}, {plot_info['unit_label']}"
        if plot_info["unit_label"] == "%"
        else f"{plot_info['label']} [{plot_info['unit_label']}]"
    )
    fig.supylabel(y_label)


def plot_histo(
    data_channel: DataFrame, fig: Figure, ax: Axes, plot_info: dict, color=None
):
    # --- histo range
    # take full range if not specified
    x_min = (
        plot_info["range"][0]
        if plot_info["range"][0]
        else data_channel[plot_info["parameter"]].min()
    )
    x_max = (
        plot_info["range"][1]
        if plot_info["range"][1]
        else data_channel[plot_info["parameter"]].max()
    )

    # --- bin width
    bwidth = {"keV": 2.5}
    bin_width = bwidth[plot_info["unit"]] if plot_info["unit"] in bwidth else 1

    # Compute number of bins
    if bin_width:
        bin_edges = (
            np.arange(x_min, x_max + bin_width, bin_width / 5)
            if plot_info["unit_label"] == "%"
            else np.arange(x_min, x_max + bin_width, bin_width)
        )
    # bin_width is never zero? unless is somehow zero in par-settings.json but why would it be?
    else:
        bin_edges = 50

    # -------------------------------------------------------------------------
    # Plot histogram
    data_channel[plot_info["parameter"]].plot.hist(
        bins=bin_edges,
        range=[x_min, x_max],
        histtype="step",
        linewidth=1.5,
        ax=ax,
        color=color,
    )

    # -------------------------------------------------------------------------
    ax.set_yscale("log")
    x_label = (
        f"{plot_info['label']}, {plot_info['unit_label']}"
        if plot_info["unit_label"] == "%"
        else f"{plot_info['label']} [{plot_info['unit_label']}]"
    )
    fig.supxlabel(x_label)

    # saving x,y data into output files
    ch_dict = {
        "values": {},
        "mean": "",
        "plot_info": plot_info,
        "timestamp": {},
    }

    return ch_dict


def plot_scatter(
    data_channel: DataFrame, fig: Figure, ax: Axes, plot_info: dict, color=None
):
    # plot data
    ax.scatter(
        data_channel["datetime"].dt.to_pydatetime(),
        data_channel[plot_info["parameter"]],
        color=color,
        # useful if there are overlapping points (but more difficult to see light colour points...)
        # facecolors='none',
        # edgecolors=color,
    )

    # --- time ticks/labels on x-axis
    ax.xaxis.set_major_formatter(DateFormatter("%Y\n%m/%d\n%H:%M"))

    fig.supxlabel("UTC Time")
    y_label = (
        f"{plot_info['label']}, {plot_info['unit_label']}"
        if plot_info["unit_label"] == "%"
        else f"{plot_info['label']} [{plot_info['unit_label']}]"
    )
    fig.supylabel(y_label)

    # saving x,y data into output files
    ch_dict = {
        "values": {"all": data_channel[plot_info["parameter"]], "resampled": []},
        "mean": "",
        "plot_info": plot_info,
        "timestamp": {
            "all": data_channel["datetime"].dt.to_pydatetime(),
            "resampled": [],
        },
    }

    return ch_dict


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# UNDER CONSTRUCTION!!!
def plot_heatmap(
    data_channel: DataFrame, fig: Figure, ax: Axes, plot_info: dict, color=None
):
    # some plotting settings
    xbin = int(
        (
            (
                data_channel.iloc[-1]["datetime"] - data_channel.iloc[0]["datetime"]
            ).total_seconds()
            * 1.5
        )
        / 1e3
    )
    if plot_info["parameter"] in ["energies", "energy_in_pe"]:
        col_map = "magma"
        ymin = 0
        ymax = 10
        ybin = 100
    if plot_info["parameter"] == "trigger_pos":
        col_map = "viridis"
        ymin = -200
        ymax = 10000
        ybin = 100

    # to plot spms data, we need a new dataframe with numeric-datetime column and 'unrolled'-list values column
    # new_df = pd.DataFrame(columns=['datetime', plot_info["parameter"]])
    new_df = pd.DataFrame()
    for _, row in data_channel.iterrows():
        for value in row[plot_info["parameter"]]:
            # remove nan entries for simplicity (and since we have the possibility here)
            if value is np.nan:
                continue
            new_row = [[row["datetime"], value]]
            new_row = pd.DataFrame(
                new_row, columns=["datetime", plot_info["parameter"]]
            )
            new_df = pd.concat([new_df, new_row], ignore_index=True, axis=0)

    x_values = pd.to_numeric(new_df["datetime"].dt.to_pydatetime()).values
    y_values = new_df[plot_info["parameter"]]

    # plot data

    h, xedges, yedges = np.histogram2d(
        x_values,
        y_values,
        bins=[xbin, ybin],
        range=[
            [data_channel.iloc[0]["datetime"], data_channel.iloc[-1]["datetime"]],
            [ymin, ymax],
        ],
    )
    # cmap = copy(fig.get_cmap(col_map))
    # cmap.set_bad(cmap(0))

    ax.pcolor(xedges, yedges, h.T, cmap=col_map)  # norm=mpl.colors.LogNorm(),

    # TO DO: add major locators (pay attention when you have only one point!)

    # set date format
    # --- time ticks/labels on x-axis
    min_x = date2num(data_channel.iloc[0]["datetime"])
    max_x = date2num(data_channel.iloc[-1]["datetime"])
    time_points = np.linspace(min_x, max_x, 10)
    labels = [num2date(time).strftime("%Y\n%m/%d\n%H:%M") for time in time_points]

    # set ticks
    ax.set_xticks(time_points)
    ax.set_xticklabels(labels)

    fig.supxlabel("UTC Time")
    fig.supylabel(f"{plot_info['label']} [{plot_info['unit_label']}]")

    # saving x,y data into output files
    ch_dict = {
        "values": {"all": {}, "resampled": []},
        "mean": "",
        "plot_info": plot_info,
        "timestamp": {
            "all": {},
            "resampled": [],
        },
    }

    return ch_dict


# -------------------------------------------------------------------------------
# mapping user keywords to plot style functions
# -------------------------------------------------------------------------------

PLOT_STYLE = {
    "vs time": plot_vs_time,
    "vs ch": par_vs_ch,
    "histogram": plot_histo,
    "scatter": plot_scatter,
    "heatmap": plot_heatmap,
}
