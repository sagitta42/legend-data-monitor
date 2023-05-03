# -------------------------------------------------------------------------------
# different status map functions called from the main one depending on parameter
# -------------------------------------------------------------------------------

# See mapping user plot structure keywords to corresponding functions in the end of this file


import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
from pandas import DataFrame, Timedelta, concat
from tabulate import tabulate


from . import utils


def status_plot(subsystem, data_analysis: DataFrame, plot_info: dict, pdf: PdfPages):
    # -------------------------------------------------------------------------
    # plot a map with statuses of channels
    # -------------------------------------------------------------------------
    utils.logger.info("\33[95m~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\33[0m")
    utils.logger.info("\33[95m~~~ S T A T U S  M A P : %s\33[0m", plot_info["title"])
    utils.logger.info("\33[95m~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\33[0m")

    data_analysis = data_analysis.sort_values(["location", "position"])

    # get threshold values
    low_thr = plot_info["limits"][0]
    high_thr = plot_info["limits"][1]
    utils.logger.debug(
        "...low threshold for "
        + plot_info["parameter"]
        + " set at: "
        + str(low_thr)
        + " "
        + plot_info["unit_label"]
    )
    utils.logger.debug(
        "...high threshold for "
        + plot_info["parameter"]
        + " set at: "
        + str(high_thr)
        + " "
        + plot_info["unit_label"]
    )

    # define the title of the status map
    plot_title = f"{plot_info['subsystem']} - "
    if low_thr is not None or high_thr is not None:
        if low_thr is None and high_thr is not None:
            plot_title += (
                f"{plot_info['parameter']} > {high_thr} {plot_info['unit_label']}"
            )
        if low_thr is not None and high_thr is None:
            plot_title += (
                f"{plot_info['parameter']} < {low_thr} {plot_info['unit_label']}"
            )
        if low_thr is not None and high_thr is not None:
            plot_title += f"{plot_info['parameter']} < {low_thr} {plot_info['unit_label']} || {plot_info['parameter']} > {high_thr} {plot_info['unit_label']}"
    if low_thr is None and high_thr is None:
        plot_title += f"{plot_info['parameter']} (no checks)"

    new_dataframe = DataFrame()
    # loop over individual channels (otherwise, the problematic timestamps apply to all detectors, even the OK ones) and create a summary dataframe
    for channel in data_analysis["channel"].unique():
        # select one block of DataFrame
        data_per_ch = data_analysis.loc[data_analysis["channel"] == channel]
        # let's save some info (they could be lost after resampling, or wrongly averaged - this keeps us safe from similar bugs)
        name = (data_per_ch["name"].unique())[0]
        location = (data_per_ch["location"].unique())[0]
        position = (data_per_ch["position"].unique())[0]
        # if not the event rate, study the status looking at the resample values
        if not plot_info["parameter"] == "event_rate":
            data_per_ch = (
                data_per_ch.set_index("datetime")
                .resample(plot_info["time_window"], origin="start")
                .mean(numeric_only=True)
            )
            data_per_ch = data_per_ch.reset_index()
            data_per_ch["datetime"] = (
                data_per_ch["datetime"] + Timedelta(plot_info["time_window"]) / 2
            )

        status = 0  # -> OK detector (update it if it's out of threshold)
        # get timestamp where the interval is out of threshold
        out_thr_datetimes = np.array([], dtype="datetime64")
        if low_thr is not None or high_thr is not None:
            if low_thr is None and high_thr is not None:
                if (data_per_ch[plot_info["parameter"]] > high_thr).any():
                    status = 1  # -> problematic detector
                    out_thr_datetimes = np.append(
                        out_thr_datetimes,
                        data_per_ch.loc[
                            data_per_ch[plot_info["parameter"]] > high_thr, "datetime"
                        ].values,
                    )

            if low_thr is not None and high_thr is None:
                if (data_per_ch[plot_info["parameter"]] < low_thr).any():
                    status = 1  # -> problematic detector
                    out_thr_datetimes = np.append(
                        out_thr_datetimes,
                        data_per_ch.loc[
                            data_per_ch[plot_info["parameter"]] < low_thr, "datetime"
                        ].values,
                    )

            if low_thr is not None and high_thr is not None:
                if (data_per_ch[plot_info["parameter"]] < low_thr).any() or (
                    data_per_ch[plot_info["parameter"]] > high_thr
                ).any():
                    status = 1  # -> problematic detector
                    out_thr_datetimes = np.append(
                        out_thr_datetimes,
                        data_per_ch.loc[
                            data_per_ch[plot_info["parameter"]] > high_thr, "datetime"
                        ].values,
                    )
                    out_thr_datetimes = np.append(
                        out_thr_datetimes,
                        data_per_ch.loc[
                            data_per_ch[plot_info["parameter"]] < low_thr, "datetime"
                        ].values,
                    )

            # create a new row in the new dataframe with essential info (ie: channel, name, location, position, status)
            new_row = [[channel, name, location, position, status]]
            new_df = DataFrame(
                new_row, columns=["channel", "name", "location", "position", "status"]
            )
            new_dataframe = concat([new_dataframe, new_df], ignore_index=True, axis=0)

        # print message with timestamps where the detector is out of threshold
        if len(out_thr_datetimes) != 0:
            out_thr_datetimes = [
                str(time).replace("T", " ")[:-10] for time in out_thr_datetimes
            ]
            utils.logger.warning(
                "\033[93mChannel %s (str. %s, pos. %s) is out of threshold at:\n%s\033[0m",
                channel,
                location,
                position,
                out_thr_datetimes,
            )

    # --------------------------------------------------------------------------------------------------------------------------
    # include OFF channels and see what is their status
    off_channels = subsystem.channel_map[subsystem.channel_map["status"] == "off"][
        "channel"
    ].unique()

    if len(off_channels) != 0:
        for channel in off_channels:
            # check if the channel is already in the status dataframe; if not, add a new row for it
            if channel not in new_dataframe["channel"].values:
                status_info = subsystem.channel_map[
                    subsystem.channel_map["channel"] == channel
                ]["status"].iloc[0]

                # get status info
                if status_info == "off":
                    status = 3
                if status_info == "AC":
                    status = (
                        2  # not there at the moment, leave it as a future option...
                    )

                # get position within the array + other necessary info
                name = subsystem.channel_map.loc[
                    subsystem.channel_map["channel"] == channel
                ]["name"].iloc[0]
                location = subsystem.channel_map.loc[
                    subsystem.channel_map["channel"] == channel
                ]["location"].iloc[0]
                position = subsystem.channel_map.loc[
                    subsystem.channel_map["channel"] == channel
                ]["position"].iloc[0]

                # define new row for not-ON detectors
                new_row = [[channel, name, location, position, status]]
                new_df = DataFrame(
                    new_row,
                    columns=["channel", "name", "location", "position", "status"],
                )
                # add the new row to the dataframe (order?)
                new_dataframe = concat(
                    [new_dataframe, new_df], ignore_index=True, axis=0
                )

    # --------------------------------------------------------------------------------------------------------------------------
    # sort the dataframe according to channel ID number
    new_dataframe = new_dataframe.sort_values("channel")
    new_dataframe = new_dataframe.reset_index()
    new_dataframe = new_dataframe.drop(
        columns=["index"]
    )  # somehow, an index column appears: remove it

    # create a pivot with necessary info
    result = new_dataframe.pivot(index="position", columns="location", values="status")

    # create a fancier copy of the pivot table for the printing
    # (only for a given level of the logger, to save time in making the prettier copy)
    if utils.logger.getEffectiveLevel() is utils.logging.DEBUG:
        output_result = result.fillna("")
        output_result = output_result.replace(3.0, "-")
        output_result = output_result.replace(2.0, "\033[93m-\033[0m")
        output_result = output_result.replace(1.0, "\033[91mX\033[0m")
        output_result = output_result.replace(0.0, "\033[94m\u2713\033[0m")
        # convert to dataframe (to use 'tabulate' library)
        df = DataFrame(output_result.to_records())

        output_result = tabulate(
            df, headers="keys", tablefmt="psql", showindex=False, stralign="center"
        )
        utils.logger.debug(
            "Status map summary for " + plot_info["parameter"] + ":\n%s", output_result
        )

    # --------------------------------------------------------------------------------------------------------------------------
    # create the figure
    fig = plt.figure(num=None, figsize=(8, 12), dpi=80, facecolor="w", edgecolor="k")
    sns.set(font_scale=1)

    # create labels for dets
    labels = new_dataframe.pivot(index="position", columns="location", values="name")

    # labels definition (AFTER having included OFF detectors too)
    # LOCATION:
    x_axis_labels = [f"S{no}" for no in sorted(new_dataframe["location"].unique())]
    # POSITION:
    y_axis_labels = [
        no
        for no in range(
            min(new_dataframe["position"].unique()),
            max(new_dataframe["position"].unique() + 1),
        )
    ]

    # to account for empty strings: not a good idea actually...
    # In L60, there are S1,S2,S7,S8: do we really want to display 4 empty strings, i.e. S3-S6? There is no need!
    # x_axis_labels = [f"S{no}" for no in range(min(new_dataframe["location"].unique()), max(new_dataframe["location"].unique()+1))]

    # status map (0=OK, 1=X, 2=AC, 3=OFF)
    custom_cmap = ["#318CE7", "#CC0000", "#F7AB60", "#D0D0D0", "#FFFFFF"]

    # create the heatmap
    status_map = sns.heatmap(
        data=result,
        annot=labels,
        annot_kws={"size": 6},
        vmin=0,
        vmax=len(custom_cmap),
        yticklabels=y_axis_labels,
        xticklabels=x_axis_labels,
        fmt="s",
        cmap=custom_cmap,
        cbar=True,
        cbar_kws={"shrink": 0.5},
        linewidths=1,
        linecolor="white",
        square=True,
        rasterized=True,
    )

    # set user defined heatmap colours
    colorbar = status_map.collections[0].colorbar
    colorbar.set_ticks([1 / 2, 3 / 2, 5 / 2, 7 / 2])
    colorbar.set_ticklabels(["OK", "X", "AC", "OFF"])

    plt.tick_params(
        axis="both",
        which="major",
        # labelsize=20,
        labelbottom=False,
        bottom=False,
        top=False,
        labeltop=True,
    )
    plt.yticks(rotation=0)
    plt.title(plot_title)
    pdf.savefig(bbox_inches="tight")

    # returning the figure
    return fig
