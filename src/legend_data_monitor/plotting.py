import shelve

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from pandas import DataFrame
from seaborn import color_palette

from . import analysis_data, plot_styles, status_plot, subsystem, utils

# -------------------------------------------------------------------------

# global variable to be filled later with colors based on number of channels
COLORS = []

# -------------------------------------------------------------------------
# main plotting function(s)
# -------------------------------------------------------------------------

# plotting function that makes subsystem plots
# feel free to write your own one using Dataset, Subsystem and ParamData objects
# for example, this structure won't work to plot one parameter VS the other


def make_subsystem_plots(subsystem: subsystem.Subsystem, plots: dict, plt_path: str):
    pdf = PdfPages(plt_path + "-" + subsystem.type + ".pdf")
    out_dict = {}

    # for param in subsys.parameters:
    for plot_title in plots:
        utils.logger.info(
            "\33[95m~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\33[0m"
        )
        utils.logger.info(f"\33[95m~~~ P L O T T I N G : {plot_title}\33[0m")
        utils.logger.info(
            "\33[95m~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\33[0m"
        )

        # --- original plot settings provided in json
        # - parameter of interest
        # - event type all/pulser/phy/Klines
        # - variation (bool)
        # - time window (for event rate or vs time plot)
        plot_settings = plots[plot_title]

        # --- defaults
        # default time window None if not parameter event rate will be accounted for in AnalysisData,
        # here need to account for plot style vs time (None for all others)
        if "time_window" not in plot_settings:
            plot_settings["time_window"] = None
        # same, here need to account for unit label %
        if "variation" not in plot_settings:
            plot_settings["variation"] = False
        # !? this is not needed because is checked in AnalysisData
        # if "cuts" not in plot_settings:
        #     plot_settings["cuts"] = []

        # -------------------------------------------------------------------------
        # set up analysis data
        # -------------------------------------------------------------------------

        # --- AnalysisData:
        # - select parameter of interest
        # - subselect type of events (pulser/phy/all/klines)
        # - calculate variation from mean, if asked
        data_analysis = analysis_data.AnalysisData(
            subsystem.data, selection=plot_settings
        )
        # cuts will be loaded but not applied; for our purposes, need to apply the cuts right away
        # currently only K lines cut is used, and only data after cut is plotted -> just replace
        data_analysis.data = data_analysis.apply_all_cuts()
        utils.logger.debug(data_analysis.data)

        # -------------------------------------------------------------------------
        # set up plot info
        # -------------------------------------------------------------------------

        # --- color settings using a pre-defined palette
        # num colors needed = max number of channels per string
        # - find number of unique positions in each string
        # - get maximum occurring
        if plot_settings["plot_structure"] == "per cc4":
            if (
                data_analysis.data.iloc[0]["cc4_id"] is None
                or data_analysis.data.iloc[0]["cc4_channel"] is None
            ):
                if subsystem.type in ["spms", "pulser"]:
                    utils.logger.error(
                        "\033[91mPlotting per CC4 is not available for %s. Try again!\033[0m",
                        subsystem.type,
                    )
                    exit()
                else:
                    utils.logger.error(
                        "\033[91mPlotting per CC4 is not available because CC4 ID or/and CC4 channel are 'None'.\nTry again!\033[0m"
                    )
                    exit()
            # ...if cc4 are present, group by them
            max_ch_per_string = (
                data_analysis.data.groupby("cc4_id")["cc4_channel"].nunique().max()
            )
        else:
            max_ch_per_string = (
                data_analysis.data.groupby("location")["position"].nunique().max()
            )
        global COLORS
        COLORS = color_palette("hls", max_ch_per_string).as_hex()

        # --- information needed for plot structure
        # ! currently "parameters" is one parameter !
        # subject to change if one day want to plot multiple in one plot
        plot_info = {
            "title": plot_title,
            "subsystem": subsystem.type,
            "locname": {"geds": "string", "spms": "fiber", "pulser": "aux"}[
                subsystem.type
            ],
            "unit": utils.PLOT_INFO[plot_settings["parameters"]]["unit"],
            "plot_style": plot_settings["plot_style"],
        }

        # --- information needed for plot style
        plot_info["parameter"] = plot_settings[
            "parameters"
        ]  # could be multiple in the future!
        plot_info["label"] = utils.PLOT_INFO[plot_info["parameter"]]["label"]
        # unit label should be % if variation was asked
        plot_info["unit_label"] = (
            "%" if plot_settings["variation"] else plot_info["unit"]
        )
        plot_info["cuts"] = plot_settings["cuts"] if "cuts" in plot_settings else ""
        # time window might be needed fort he vs time function
        plot_info["time_window"] = plot_settings["time_window"]
        # threshold values are needed for status map; might be needed for plotting limits on canvas too
        if subsystem.type != "pulser":
            plot_info["limits"] = (
                utils.PLOT_INFO[plot_info["parameter"]]["limits"][subsystem.type][
                    "variation"
                ]
                if plot_settings["variation"]
                else utils.PLOT_INFO[plot_info["parameter"]]["limits"][subsystem.type][
                    "absolute"
                ]
            )

        # -------------------------------------------------------------------------
        # call chosen plot structure
        # -------------------------------------------------------------------------

        # choose plot function based on user requested structure e.g. per channel or all ch together
        plot_structure = PLOT_STRUCTURE[plot_settings["plot_structure"]]
        utils.logger.debug("Plot structure: " + plot_settings["plot_structure"])

        # writing data_analys and plot_info to file to be later plotted by the dashboard
        out_dict["data"] = data_analysis
        out_dict["plot_info"] = plot_info

        # plotting
        par_dict_content = plot_structure(data_analysis, plot_info, pdf)

        # For some reason, after some plotting functions the index is set to "channel".
        # We need to set it back otherwise status_plot.py gets crazy and everything crashes.
        data_analysis.data = data_analysis.data.reset_index()
        # saving dataframe
        par_dict_content["df_" + plot_info["subsystem"]] = data_analysis

        # -------------------------------------------------------------------------
        # call status plot
        # -------------------------------------------------------------------------
        if "status" in plot_settings and plot_settings["status"]:
            if subsystem.type == "pulser":
                utils.logger.debug(
                    "Thresholds are not enabled for pulser! Use you own eyes to do checks there"
                )
            else:
                status_fig = status_plot.status_plot(
                    subsystem, data_analysis, plot_info, pdf
                )
                # saving status map figure
                par_dict_content["map_" + plot_info["subsystem"]] = status_fig


        # saving PARAMETER DICT in the dictionary that will be stored in the shelve object
        # event type key is already there
        if plot_settings["event_type"] in out_dict.keys():
            #  check if the parameter is already there (without this, previous inspected parameters are overwritten)
            if (
                plot_info["parameter"]
                not in out_dict[plot_settings["event_type"]].keys()
            ):
                out_dict[plot_settings["event_type"]][
                    plot_info["parameter"]
                ] = par_dict_content
        # event type key is NOT there
        else:
            # empty dictionary (not filled yet)
            if len(out_dict.keys()) == 0:
                out_dict = {
                    plot_settings["event_type"]: {
                        plot_info["parameter"]: par_dict_content
                    }
                }
            # the dictionary already contains something (but for another event type selection)
            else:
                out_dict[plot_settings["event_type"]] = {
                    plot_info["parameter"]: par_dict_content
                }

    # save in shelve object
    out_file = shelve.open(plt_path)
    out_file["monitoring"] = out_dict
    out_file.close()

    # save in pdf object
    pdf.close()

    utils.logger.info(
        f"All plots saved in: \33[4m{plt_path}-{subsystem.type}.pdf\33[0m"
    )


# -------------------------------------------------------------------------------
# different plot structure functions, defining figures and subplot layouts
# -------------------------------------------------------------------------------

# See mapping user plot structure keywords to corresponding functions in the end of this file


def plot_per_ch(data_analysis, plot_info, pdf):
    # --- choose plot function based on user requested style e.g. vs time or histogram
    plot_style = plot_styles.PLOT_STYLE[plot_info["plot_style"]]
    utils.logger.debug("Plot style: " + plot_info["plot_style"])

    par_dict = {}
    data_analysis.data = data_analysis.data.sort_values(["location", "position"])

    # -------------------------------------------------------------------------------

    # separate figure for each string/fiber ("location")
    for location, data_location in data_analysis.data.groupby("location"):
        utils.logger.debug(f"... {plot_info['locname']} {location}")

        # -------------------------------------------------------------------------------
        # create plot structure: 1 column, N rows with subplot for each channel
        # -------------------------------------------------------------------------------

        # number of channels in this string/fiber
        numch = len(data_location["channel"].unique())
        # create corresponding number of subplots for each channel, set constrained layout to accommodate figure suptitle
        fig, axes = plt.subplots(
            nrows=numch,
            ncols=1,
            figsize=(10, numch * 3),
            sharex=True,
            constrained_layout=True,
        )  # , sharey=True)
        # in case of pulser, axes will be not a list but one axis -> convert to list
        if numch == 1:
            axes = [axes]

        # -------------------------------------------------------------------------------
        # plot
        # -------------------------------------------------------------------------------

        ax_idx = 0
        # plot one channel on each axis, ordered by position
        for position, data_channel in data_location.groupby("position"):
            utils.logger.debug(f"...... position {position}")

            # plot selected style on this axis
            ch_dict = plot_style(
                data_channel, fig, axes[ax_idx], plot_info, color=COLORS[ax_idx]
            )

            # --- add summary to axis
            # name, position and mean are unique for each channel - take first value
            t = data_channel.iloc[0][
                ["channel", "position", "name", plot_info["parameter"] + "_mean"]
            ]
            if t["channel"] not in par_dict.keys():
                par_dict[str(t["channel"])] = ch_dict

            text = (
                t["name"]
                + "\n"
                + f"channel {t['channel']}\n"
                + f"position {t['position']}\n"
                + (
                    f"mean {round(t[plot_info['parameter']+'_mean'],3)} [{plot_info['unit']}]"
                    if t[plot_info["parameter"] + "_mean"] is not None
                    else ""
                )  # handle with care mean='None' situations
            )
            axes[ax_idx].text(1.01, 0.5, text, transform=axes[ax_idx].transAxes)

            # add grid
            axes[ax_idx].grid("major", linestyle="--")
            # remove automatic y label since there will be a shared one
            axes[ax_idx].set_ylabel("")

            # plot line at 0% for variation
            if plot_info["unit_label"] == "%":
                axes[ax_idx].axhline(y=0, color="gray", linestyle="--")

            ax_idx += 1

        # -------------------------------------------------------------------------------

        fig.suptitle(f"{plot_info['subsystem']} - {plot_info['title']}", y=1.15)
        if plot_info["subsystem"] == "pulser":
            axes[0].set_title("")
        else:
            axes[0].set_title(f"{plot_info['locname']} {location}")

        plt.savefig(pdf, format="pdf", bbox_inches="tight")
        # figures are retained until explicitly closed; close to not consume too much memory
        plt.close()

    return par_dict


def plot_per_cc4(data_analysis, plot_info, pdf):
    if plot_info["subsystem"] == "pulser":
        utils.logger.error(
            "\033[91mPlotting per CC4 is not available for the pulser channel.\nTry again with a different plot structure!\033[0m"
        )
        exit()
    # --- choose plot function based on user requested style e.g. vs time or histogram
    plot_style = plot_styles.PLOT_STYLE[plot_info["plot_style"]]
    utils.logger.debug("Plot style: " + plot_info["plot_style"])

    par_dict = {}

    # --- create plot structure
    # number of cc4s
    no_cc4_id = len(data_analysis.data["cc4_id"].unique())
    # set constrained layout to accommodate figure suptitle
    fig, axes = plt.subplots(
        no_cc4_id,
        figsize=(10, no_cc4_id * 3),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )

    # -------------------------------------------------------------------------------
    # create label of format hardcoded for geds sXX-pX-chXXX-name-CC4channel
    # -------------------------------------------------------------------------------
    labels = data_analysis.data.groupby("channel").first()[
        ["name", "position", "location", "cc4_channel", "cc4_id"]
    ]
    labels["channel"] = labels.index
    labels["label"] = labels[
        ["location", "position", "channel", "name", "cc4_channel"]
    ].apply(lambda x: f"s{x[0]}-p{x[1]}-ch{str(x[2]).zfill(3)}-{x[3]}-{x[4]}", axis=1)
    # put it in the table
    data_analysis.data = data_analysis.data.set_index("channel")
    data_analysis.data["label"] = labels["label"]

    # -------------------------------------------------------------------------------
    # plot
    # -------------------------------------------------------------------------------

    data_analysis.data = data_analysis.data.sort_values(
        ["cc4_id", "cc4_channel", "label"]
    )
    # new subplot for each string
    ax_idx = 0
    for cc4_id, data_cc4_id in data_analysis.data.groupby("cc4_id"):
        utils.logger.debug(f"... CC4 {cc4_id}")

        # new color for each channel
        col_idx = 0
        labels = []
        for label, data_channel in data_cc4_id.groupby("label"):
            cc4_channel = (label.split("-"))[-1]
            utils.logger.debug(f"...... channel {cc4_channel}")
            ch_dict = plot_style(
                data_channel, fig, axes[ax_idx], plot_info, COLORS[col_idx]
            )
            labels.append(label)
            col_idx += 1

            channel = ((label.split("-")[2]).split("ch")[-1]).lstrip("0")
            if channel not in par_dict.keys():
                par_dict[channel] = ch_dict

        # add grid
        axes[ax_idx].grid("major", linestyle="--")
        # beautification
        axes[ax_idx].set_title(f"{plot_info['locname']} {cc4_id}")
        axes[ax_idx].set_ylabel("")
        axes[ax_idx].legend(labels=labels, loc="center left", bbox_to_anchor=(1, 0.5))

        # plot the position of the two K lines
        if plot_info["cuts"] == "K lines":
            axes[ax_idx].axhline(y=1460.822, color="gray", linestyle="--")
            axes[ax_idx].axhline(y=1524.6, color="gray", linestyle="--")

        # plot line at 0% for variation
        if plot_info["unit_label"] == "%":
            axes[ax_idx].axhline(y=0, color="gray", linestyle="--")
        ax_idx += 1

    # -------------------------------------------------------------------------------
    fig.suptitle(f"{plot_info['subsystem']} - {plot_info['title']}", y=1.15)
    # fig.supylabel(f'{plotdata.param.label} [{plotdata.param.unit_label}]') # --> plot style
    plt.savefig(pdf, format="pdf", bbox_inches="tight")
    # figures are retained until explicitly closed; close to not consume too much memory
    plt.close()

    return par_dict


# technically per location

def plot_per_string(data_analysis, plot_info, pdf, *string):
    if plot_info["subsystem"] == "pulser":
        utils.logger.error(
            "\033[91mPlotting per string is not available for the pulser channel.\nTry again with a different plot structure!\033[0m"
        )
        exit()

    # --- choose plot function based on user requested style e.g. vs time or histogram
    plot_style = plot_styles.PLOT_STYLE[plot_info["plot_style"]]
    if not string:
        utils.logger.debug("Plot style: " + plot_info["plot_style"])

    par_dict = {}

    # -------------------------------------------------------------------------------
    # create label of format hardcoded for geds pX-chXXX-name
    # -------------------------------------------------------------------------------

    labels = data_analysis.data.groupby("channel").first()[["name", "position"]]
    labels["channel"] = labels.index
    labels["label"] = labels[["position", "channel", "name"]].apply(
        lambda x: f"p{x[0]}-ch{str(x[1]).zfill(3)}-{x[2]}", axis=1
    )
    # put it in the table
    data_analysis.data = data_analysis.data.set_index("channel")
    data_analysis.data["label"] = labels["label"]
    data_analysis.data = data_analysis.data.sort_values("label")

    # -------------------------------------------------------------------------------
    # plot
    # -------------------------------------------------------------------------------

    data_analysis.data = data_analysis.data.sort_values(["location", "label"])
    # new subplot for each string
    for location, data_location in data_analysis.data.groupby("location"):
        # if string number specified, this function is being called by external code
        # and is not invoked by make_subsystem_plots()
        if string:
            if string[0] != location:
                continue
            max_ch_per_string = (
                data_analysis.data.groupby("location")["position"].nunique().max()
            )
            global COLORS
            COLORS = color_palette("hls", max_ch_per_string).as_hex()

        # otherwise just go on with standard code
        else:
            utils.logger.debug(f"... {plot_info['locname']} {location}")
        # create one different figure per string
        fig, axes = plt.subplots(figsize=(15, 5))

        # new color for each channel
        col_idx = 0
        labels = []
        for label, data_channel in data_location.groupby("label"):
            ch_dict = plot_style(
                data_channel, fig, axes, plot_info, color=COLORS[col_idx]
            )
            labels.append(label)
            col_idx += 1
            channel = ((label.split("-")[1]).split("ch")[-1]).lstrip("0")
            if channel not in par_dict.keys():
                par_dict[channel] = ch_dict

        # add grid
        axes.grid("major", linestyle="--")
        # beautification
        axes.set_title(f"{plot_info['locname']} {location}")
        axes.set_ylabel("")
        axes.legend(labels=labels, loc="center left", bbox_to_anchor=(1, 0.5))

        # plot the position of the two K lines
        if plot_info["title"] == "K lines":
            axes.axhline(y=1460.822, color="gray", linestyle="--")
            axes.axhline(y=1524.6, color="gray", linestyle="--")

        # plot line at 0% for variation
        if plot_info["unit_label"] == "%":
            axes.axhline(y=0, color="gray", linestyle="--")

        # -------------------------------------------------------------------------------
        fig.suptitle(f"{plot_info['subsystem']} - {plot_info['title']}")
        # fig.supylabel(f'{plotdata.param.label} [{plotdata.param.unit_label}]') # --> plot style
        if not string:
            plt.savefig(pdf, format="pdf", bbox_inches="tight")
        # figures are retained until explicitly closed; close to not consume too much memory
        plt.close()

    if string:
        return fig
    else:
        return par_dict


def plot_array(data_analysis, plot_info, pdf):
    if plot_info["subsystem"] != "geds":
        utils.logger.error(
            "\033[91mPlotting per array is not available for the spms or pulser channel.\nTry again with geds!\033[0m"
        )
        exit()

    import matplotlib.patches as mpatches

    # --- choose plot function based on user requested style
    plot_style = plot_styles.PLOT_STYLE[plot_info["plot_style"]]
    utils.logger.debug("Plot style: " + plot_info["plot_style"])

    par_dict = {}

    # --- create plot structure
    fig, axes = plt.subplots(
        1,  # no of location
        figsize=(10, 3),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )

    # -------------------------------------------------------------------------------
    # create label of format hardcoded for geds sX-pX-chXXX-name
    # -------------------------------------------------------------------------------
    labels = data_analysis.data.groupby("channel").first()[
        ["name", "location", "position"]
    ]
    labels["channel"] = labels.index
    labels["label"] = labels[["location", "position", "channel", "name"]].apply(
        lambda x: f"s{x[0]}-p{x[1]}-ch{str(x[2]).zfill(3)}-{x[3]}", axis=1
    )
    # put it in the table
    data_analysis.data = data_analysis.data.set_index("channel")
    data_analysis.data["label"] = labels["label"]
    data_analysis.data = data_analysis.data.sort_values("label")

    # -------------------------------------------------------------------------------
    # plot
    # -------------------------------------------------------------------------------
    data_analysis.data = data_analysis.data.sort_values(["location", "label"])

    # one color for each string
    col_idx = 0
    # some lists to fill with info, string by string
    labels = []
    channels = []
    legend = []

    # group by string
    for location, data_location in data_analysis.data.groupby("location"):
        utils.logger.debug(f"... {plot_info['locname']} {location}")

        values_per_string = []  # y values - in each string
        channels_per_string = []  # x values - in each string
        # group by channel
        for label, data_channel in data_location.groupby("label"):
            ch_dict = plot_style(data_channel, fig, axes, plot_info, COLORS[col_idx])

            channel = ((label.split("-")[2]).split("ch")[-1]).lstrip("0")
            if channel not in par_dict.keys():
                par_dict[channel] = ch_dict

            labels.append(label)
            channels.append(int(channel))
            values_per_string.append(ch_dict["values"])
            channels_per_string.append(int(channel))

        # get average of plotted parameter per string (print horizontal line)
        avg_of_string = sum(values_per_string) / len(values_per_string)
        axes.hlines(
            y=avg_of_string,
            xmin=min(channels_per_string),
            xmax=max(channels_per_string),
            color="k",
            linestyle="-",
            linewidth=1,
        )
        utils.logger.debug(f"..... average: {round(avg_of_string, 2)}")

        # get legend entry (print string + colour)
        legend.append(
            mpatches.Patch(
                color=COLORS[col_idx],
                label=f"s{location} - avg: {round(avg_of_string, 2)} {plot_info['unit_label']}",
            )
        )

        # LAST thing to update
        col_idx += 1

    # -------------------------------------------------------------------------------
    # add legend
    axes.legend(
        loc=(1.04, 0.0),
        ncol=1,
        frameon=True,
        facecolor="white",
        framealpha=0,
        handles=legend,
    )
    # add grid
    axes.grid("major", linestyle="--")
    # beautification
    axes.ylabel = None
    axes.xlabel = None
    # add x labels
    axes.set_xticks(channels)
    axes.set_xticklabels(labels, fontsize=6)
    # rotate x labels
    plt.xticks(rotation=70, ha="right")
    # title/label
    fig.supxlabel("")
    fig.suptitle(f"{plot_info['subsystem']} - {plot_info['title']}", y=1.15)

    # -------------------------------------------------------------------------------
    plt.savefig(pdf, format="pdf", bbox_inches="tight")
    plt.close()

    return par_dict


# -------------------------------------------------------------------------------
# SiPM specific structures
# -------------------------------------------------------------------------------


def plot_per_fiber_and_barrel(data_analysis: DataFrame, plot_info: dict, pdf: PdfPages):
    if plot_info["subsystem"] != "spms":
        utils.logger.error(
            "\033[91mPlotting per fiber-barrel is available ONLY for spms.\nTry again!\033[0m"
        )
        exit()
    # here will be a function plotting SiPMs with:
    # - one figure for top and one for bottom SiPMs
    # - each figure has subplots with N columns and M rows where N is the number of fibers, and M is the number of positions (top/bottom -> 2)
    # this function will only work for SiPMs requiring a columns 'barrel' in the channel map
    # add a check in config settings check to make sure geds are not called with this structure to avoid crash
    pass


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# UNDER CONSTRUCTION!!!
def plot_per_barrel_and_position(
    data_analysis: DataFrame, plot_info: dict, pdf: PdfPages
):
    if plot_info["subsystem"] != "spms":
        utils.logger.error(
            "\033[91mPlotting per barrel-position is available ONLY for spms.\nTry again!\033[0m"
        )
        exit()
    # here will be a function plotting SiPMs with:
    # - one figure for each barrel-position combination (IB-top, IB-bottom, OB-top, OB-bottom) = 4 figures in total

    plot_style = plot_styles.PLOT_STYLE[plot_info["plot_style"]]
    utils.logger.debug("Plot style: " + plot_info["plot_style"])

    par_dict = {}

    # re-arrange dataframe to separate location: from location=[IB-015-016] to location=[IB] & fiber=[015-016]
    data_analysis.data["fiber"] = (
        data_analysis.data["location"].str.split("-").str[1].str.join("")
        + "-"
        + data_analysis.data["location"].str.split("-").str[2].str.join("")
    )
    data_analysis.data["location"] = (
        data_analysis.data["location"].str.split("-").str[0].str.join("")
    )

    # -------------------------------------------------------------------------------
    # create label of format hardcoded for geds pX-chXXX-name
    # -------------------------------------------------------------------------------

    labels = data_analysis.data.groupby("channel").first()[
        ["name", "position", "location", "fiber"]
    ]
    labels["channel"] = labels.index
    labels["label"] = labels[
        ["position", "location", "fiber", "channel", "name"]
    ].apply(lambda x: f"{x[0]}-{x[1]}-{x[2]}-ch{str(x[3]).zfill(3)}-{x[4]}", axis=1)
    # put it in the table
    data_analysis.data = data_analysis.data.set_index("channel")
    data_analysis.data["label"] = labels["label"]
    data_analysis.data = data_analysis.data.sort_values("label")

    data_analysis.data = data_analysis.data.sort_values(["location", "label"])

    # separate figure for each barrel ("location"= IB, OB)...
    for location, data_location in data_analysis.data.groupby("location"):
        utils.logger.debug(f"... {location} barrel")
        # ...and position ("position"= bottom, top)
        for position, data_position in data_location.groupby("position"):
            utils.logger.debug(f"..... {position}")

            # -------------------------------------------------------------------------------
            # create plot structure: M columns, N rows with subplots for each channel
            # -------------------------------------------------------------------------------

            # number of channels in this barrel
            if location == "IB":
                num_rows = 3
                num_cols = 3
            if location == "OB":
                num_rows = 4
                num_cols = 5
            # create corresponding number of subplots for each channel, set constrained layout to accommodate figure suptitle
            fig, axes = plt.subplots(
                nrows=num_rows,
                ncols=num_cols,
                figsize=(10, num_rows * 3),
                sharex=True,
                constrained_layout=True,
            )  # , sharey=True)

            # -------------------------------------------------------------------------------
            # plot
            # -------------------------------------------------------------------------------

            data_position = data_position.reset_index()
            channel = data_position["channel"].unique()
            det_idx = 0
            col_idx = 0
            labels = []
            for ax_row in axes:
                for (
                    axes
                ) in ax_row:  # this is already the Axes object (no need to add ax_idx)
                    # plot one channel on each axis, ordered by position
                    data_position = data_position[
                        data_position["channel"] == channel[col_idx]
                    ]  # get only rows for a given channel

                    # plotting...
                    if data_position.empty:
                        det_idx += 1
                        continue

                    ch_dict = plot_style(
                        data_position, fig, axes, plot_info, color=COLORS[det_idx]
                    )
                    labels.append(data_position["label"])

                    if channel[det_idx] not in par_dict.keys():
                        par_dict[channel[det_idx]] = ch_dict

                    """text = (
                        data_position.iloc[0]["name"]
                        + "\n"
                        + f"channel {channel[col_idx]}\n"
                        + f"position {t['position']}\n"
                        + f"mean {round(t[plot_info['parameter']+'_mean'],3)} [{plot_info['unit']}]"
                    )"""
                    text = str(channel[col_idx])
                    # axes.text(1.01, 0.5, text, transform=axes.transAxes)
                    axes.set_title(label=text, loc="center")

                    # add grid
                    axes.grid("major", linestyle="--")
                    # remove automatic y label since there will be a shared one
                    axes.set_ylabel("")

                    det_idx += 1
                    col_idx += 1

            fig.suptitle(f"{plot_info['subsystem']} - {plot_info['title']}", y=1.15)
            # fig.supylabel(f'{plotdata.param.label} [{plotdata.param.unit_label}]') # --> plot style
            plt.savefig(pdf, format="pdf", bbox_inches="tight")
            # figures are retained until explicitly closed; close to not consume too much memory
            plt.close()

    return par_dict


# -------------------------------------------------------------------------------
# mapping user keywords to plot style functions
# -------------------------------------------------------------------------------

PLOT_STRUCTURE = {
    "per channel": plot_per_ch,
    "per cc4": plot_per_cc4,
    "per string": plot_per_string,
    "array": plot_array,
    "per fiber": plot_per_fiber_and_barrel,
    "per barrel": plot_per_barrel_and_position,
}
