import io
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


def make_subsystem_plots(
    subsystem: subsystem.Subsystem, plots: dict, plt_path: str, saving=None
):
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
        # range for parameter
        if "range" not in plot_settings:
            plot_settings["range"] = [None, None]
        # add saving info + plot where we save things
        plot_settings["saving"] = saving
        plot_settings["plt_path"] = plt_path

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
            "locname": {"geds": "string", "spms": "fiber", "pulser": "puls"}[
                subsystem.type
            ],
            "unit": utils.PLOT_INFO[plot_settings["parameters"]]["unit"],
            "plot_style": plot_settings["plot_style"],
        }

        # information for having the resampled or all entries (needed only for 'vs time' style option)
        plot_info["resampled"] = (
            plot_settings["resampled"] if "resampled" in plot_settings else ""
        )

        if plot_settings["plot_style"] == "vs time":
            if plot_info["resampled"] == "":
                plot_info["resampled"] = "also"
                utils.logger.warning(
                    "\033[93mNo 'resampled' option was specified. Both resampled and all entries will be plotted (otherwise you can try again using the option 'no', 'only', 'also').\033[0m"
                )
        else:
            if plot_info["resampled"] != "":
                utils.logger.warning(
                    "\033[93mYou're using the option 'resampled' for a plot style that does not need it. For this reason, that option will be ignored.\033[0m"
                )

        # --- information needed for plot style
        plot_info["label"] = utils.PLOT_INFO[plot_settings["parameters"]]["label"]
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
                utils.PLOT_INFO[plot_settings["parameters"]]["limits"][subsystem.type][
                    "variation"
                ]
                if plot_settings["variation"]
                else utils.PLOT_INFO[plot_settings["parameters"]]["limits"][
                    subsystem.type
                ]["absolute"]
            )
        plot_info["parameter"] = (
            plot_settings["parameters"] + "_var"
            if plot_info["unit_label"] == "%"
            else plot_settings["parameters"]
        )  # could be multiple in the future!
        plot_info["param_mean"] = plot_settings["parameters"] + "_mean"
        plot_info["range"] = plot_settings["range"]

        # -------------------------------------------------------------------------
        # call chosen plot structure
        # -------------------------------------------------------------------------

        # choose plot function based on user requested structure e.g. per channel or all ch together
        plot_structure = PLOT_STRUCTURE[plot_settings["plot_structure"]]
        utils.logger.debug("Plot structure: " + plot_settings["plot_structure"])

        # plotting
        plot_structure(data_analysis.data, plot_info, pdf)

        # For some reason, after some plotting functions the index is set to "channel".
        # We need to set it back otherwise status_plot.py gets crazy and everything crashes.
        data_analysis.data = data_analysis.data.reset_index()

        # -------------------------------------------------------------------------
        # saving dataframe + plot info
        # -------------------------------------------------------------------------

        par_dict_content = {}

        # saving dataframe data for each parameter
        par_dict_content["df_" + plot_info["subsystem"]] = data_analysis.data
        par_dict_content["plot_info"] = plot_info

        # -------------------------------------------------------------------------
        # call status plot
        # -------------------------------------------------------------------------

        if "status" in plot_settings and plot_settings["status"]:
            if subsystem.type == "pulser":
                utils.logger.debug(
                    "Thresholds are not enabled for pulser! Use you own eyes to do checks there"
                )
            else:
                _ = status_plot.status_plot(
                    subsystem, data_analysis.data, plot_info, pdf
                )

        # -------------------------------------------------------------------------
        # save results
        # -------------------------------------------------------------------------

        # building a dictionary with dataframe/plot_info to be later stored in a shelve object
        if saving is not None:
            out_dict = utils.build_out_dict(
                plot_settings, plot_info, par_dict_content, out_dict, saving, plt_path
            )

    # save in shelve object, overwriting the already existing file with new content (either completely new or new bunches)
    if saving is not None:
        out_file = shelve.open(plt_path + f"-{subsystem.type}")
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


def plot_per_ch(data_analysis: DataFrame, plot_info: dict, pdf: PdfPages):
    # --- choose plot function based on user requested style e.g. vs time or histogram
    plot_style = plot_styles.PLOT_STYLE[plot_info["plot_style"]]
    utils.logger.debug("Plot style: " + plot_info["plot_style"])

    data_analysis = data_analysis.sort_values(["location", "position"])

    # -------------------------------------------------------------------------------

    # separate figure for each string/fiber ("location")
    for location, data_location in data_analysis.groupby("location"):
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
        axes = [axes] if numch == 1 else axes

        # -------------------------------------------------------------------------------
        # plot
        # -------------------------------------------------------------------------------

        ax_idx = 0
        # plot one channel on each axis, ordered by position
        for position, data_channel in data_location.groupby("position"):
            utils.logger.debug(f"...... position {position}")
            # define what colors are needed
            # if this function is not called by makes_subsystem_plot() need to define colors locally
            # to be included in a separate function to be called every time (maybe in utils?)
            max_ch_per_string = (
                data_analysis.groupby("location")["position"].nunique().max()
            )
            global COLORS
            COLORS = color_palette("hls", max_ch_per_string).as_hex()

            # plot selected style on this axis
            _ = plot_style(
                data_channel, fig, axes[ax_idx], plot_info, color=COLORS[ax_idx]
            )

            # --- add summary to axis
            # name, position and mean are unique for each channel - take first value
            t = data_channel.iloc[0][
                ["channel", "position", "name", plot_info["param_mean"]]
            ]

            text = (
                t["name"]
                + "\n"
                + f"channel {t['channel']}\n"
                + f"position {t['position']}\n"
                + (
                    f"mean {round(t[plot_info['param_mean']],3)} [{plot_info['unit']}]"
                    if t[plot_info["param_mean"]] is not None
                    else ""
                )  # handle with care mean='None' situations
            )
            axes[ax_idx].text(1.01, 0.5, text, transform=axes[ax_idx].transAxes)

            # add grid
            axes[ax_idx].grid("major", linestyle="--")
            axes[ax_idx].set_axisbelow(True)
            # remove automatic y label since there will be a shared one
            axes[ax_idx].set_ylabel("")

            # plot line at 0% for variation
            if plot_info["unit_label"] == "%":
                axes[ax_idx].axhline(y=0, color="gray", linestyle="--")

            ax_idx += 1

        # -------------------------------------------------------------------------------
        if plot_info["subsystem"] == "pulser":
            y_title = 1.05
            axes[0].set_title("")
        else:
            y_title = 1.01
            axes[0].set_title(f"{plot_info['locname']} {location}")
        fig.suptitle(f"{plot_info['subsystem']} - {plot_info['title']}", y=y_title)

        if pdf:
            plt.savefig(pdf, format="pdf", bbox_inches="tight")
            # figures are retained until explicitly closed; close to not consume too much memory
            plt.close()

    return fig


def plot_per_cc4(data_analysis: DataFrame, plot_info: dict, pdf: PdfPages):
    if plot_info["subsystem"] == "pulser":
        utils.logger.error(
            "\033[91mPlotting per CC4 is not available for the pulser channel.\nTry again with a different plot structure!\033[0m"
        )
        exit()
    # --- choose plot function based on user requested style e.g. vs time or histogram
    plot_style = plot_styles.PLOT_STYLE[plot_info["plot_style"]]
    utils.logger.debug("Plot style: " + plot_info["plot_style"])

    # --- create plot structure
    # number of cc4s
    no_cc4_id = len(data_analysis["cc4_id"].unique())
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
    labels = data_analysis.groupby("channel").first()[
        ["name", "position", "location", "cc4_channel", "cc4_id"]
    ]
    labels["channel"] = labels.index
    labels["label"] = labels[
        ["location", "position", "channel", "name", "cc4_channel"]
    ].apply(lambda x: f"s{x[0]}-p{x[1]}-ch{str(x[2]).zfill(3)}-{x[3]}-{x[4]}", axis=1)
    # put it in the table
    data_analysis = data_analysis.set_index("channel")
    data_analysis["label"] = labels["label"]

    # -------------------------------------------------------------------------------
    # plot
    # -------------------------------------------------------------------------------

    data_analysis = data_analysis.sort_values(["cc4_id", "cc4_channel", "label"])
    # new subplot for each string
    ax_idx = 0
    for cc4_id, data_cc4_id in data_analysis.groupby("cc4_id"):
        utils.logger.debug(f"... CC4 {cc4_id}")
        # set colors
        max_ch_per_cc4 = data_analysis.groupby("cc4_id")["cc4_channel"].nunique().max()
        global COLORS
        COLORS = color_palette("hls", max_ch_per_cc4).as_hex()

        # new color for each channel
        col_idx = 0
        labels = []
        for label, data_channel in data_cc4_id.groupby("label"):
            cc4_channel = (label.split("-"))[-1]
            utils.logger.debug(f"...... channel {cc4_channel}")
            _ = plot_style(data_channel, fig, axes[ax_idx], plot_info, COLORS[col_idx])
            labels.append(label)
            col_idx += 1

        # add grid
        axes[ax_idx].grid("major", linestyle="--")
        axes[ax_idx].set_axisbelow(True)
        # beautification
        axes[ax_idx].set_title(f"CC4 {cc4_id}")
        axes[ax_idx].set_ylabel("")
        axes[ax_idx].legend(labels=labels, loc="center left", bbox_to_anchor=(1, 0.5))

        # plot the position of the two K lines
        if plot_info["parameter"] == "K_events":
            axes[ax_idx].axhline(y=1460.822, color="gray", linestyle="--")
            axes[ax_idx].axhline(y=1524.6, color="gray", linestyle="--")

        # plot line at 0% for variation
        if plot_info["unit_label"] == "%":
            axes[ax_idx].axhline(y=0, color="gray", linestyle="--")
        ax_idx += 1

    # -------------------------------------------------------------------------------
    y_title = 1.05 if plot_info["subsystem"] == "pulser" else 1.01
    fig.suptitle(f"{plot_info['subsystem']} - {plot_info['title']}", y=y_title)
    # if no pdf is specified, then the function is not being called by make_subsystem_plots()
    if pdf:
        plt.savefig(pdf, format="pdf", bbox_inches="tight")
        # figures are retained until explicitly closed; close to not consume too much memory
        plt.close()

    return fig


def plot_per_string(data_analysis: DataFrame, plot_info: dict, pdf: PdfPages):
    # --- choose plot function based on user requested style e.g. vs time or histogram
    plot_style = plot_styles.PLOT_STYLE[plot_info["plot_style"]]
    utils.logger.debug("Plot style: " + plot_info["plot_style"])

    # --- create plot structure
    # number of strings/fibers
    no_location = len(data_analysis["location"].unique())
    # set constrained layout to accommodate figure suptitle
    fig, axes = plt.subplots(
        no_location,
        figsize=(10, no_location * 3),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    # in case of pulser, axes will be not a list but one axis -> convert to list
    axes = [axes] if no_location == 1 else axes

    # -------------------------------------------------------------------------------
    # create label of format hardcoded for geds pX-chXXX-name
    # -------------------------------------------------------------------------------

    labels = data_analysis.groupby("channel").first()[["name", "position"]]
    labels["channel"] = labels.index
    labels["label"] = labels[["position", "channel", "name"]].apply(
        lambda x: f"p{x[0]}-ch{str(x[1]).zfill(3)}-{x[2]}", axis=1
    )
    # put it in the table
    data_analysis = data_analysis.set_index("channel")
    data_analysis["label"] = labels["label"]

    # -------------------------------------------------------------------------------
    # plot
    # -------------------------------------------------------------------------------

    data_analysis = data_analysis.sort_values(["location", "position"])
    # new subplot for each string
    ax_idx = 0
    for location, data_location in data_analysis.groupby("location"):
        # define what colors are needed
        # if this function is not called by makes_subsystem_plot() need to define colors
        # to be included in a separate function to be called every time (maybe in utils?)
        max_ch_per_string = (
            data_analysis.groupby("location")["position"].nunique().max()
        )
        global COLORS
        COLORS = color_palette("hls", max_ch_per_string).as_hex()

        utils.logger.debug(f"... {plot_info['locname']} {location}")

        # new color for each channel
        col_idx = 0
        labels = []
        # groupby "label" will sort in alphabetical order (1, 10, 11, 2, 3, ...) -> group by position
        for _, data_channel in data_location.groupby("position"):
            plot_style(data_channel, fig, axes[ax_idx], plot_info, COLORS[col_idx])
            # label is the same for all entries for the same channel
            labels.append(data_channel["label"].iloc[0])
            col_idx += 1

        # add grid
        axes[ax_idx].grid("major", linestyle="--")
        axes[ax_idx].set_axisbelow(True)
        # beautification
        axes[ax_idx].set_title(f"{plot_info['locname']} {location}")
        axes[ax_idx].set_ylabel("")
        axes[ax_idx].legend(labels=labels, loc="center left", bbox_to_anchor=(1, 0.5))

        # plot the position of the two K lines
        if plot_info["parameter"] == "K_events":
            axes[ax_idx].axhline(y=1460.822, color="gray", linestyle="--")
            axes[ax_idx].axhline(y=1524.6, color="gray", linestyle="--")

        # plot line at 0% for variation
        if plot_info["unit_label"] == "%":
            axes[ax_idx].axhline(y=0, color="gray", linestyle="--")
        ax_idx += 1

    # -------------------------------------------------------------------------------
    y_title = 1.05 if plot_info["subsystem"] == "pulser" else 1.01
    fig.suptitle(f"{plot_info['subsystem']} - {plot_info['title']}", y=y_title)

    # if no pdf is specified, then the function is not being called by make_subsystem_plots()
    if pdf:
        plt.savefig(pdf, format="pdf", bbox_inches="tight")
        # figures are retained until explicitly closed; close to not consume too much memory
        plt.close()

    return fig


def plot_array(data_analysis: DataFrame, plot_info: dict, pdf: PdfPages):
    if plot_info["subsystem"] == "spms":
        utils.logger.error(
            "\033[91mPlotting per array is not available for the spms.\nTry again!\033[0m"
        )
        exit()

    import matplotlib.patches as mpatches

    # --- choose plot function based on user requested style
    plot_style = plot_styles.PLOT_STYLE[plot_info["plot_style"]]
    utils.logger.debug("Plot style: " + plot_info["plot_style"])

    # --- create plot structure
    fig, axes = plt.subplots(
        1,  # no of location
        figsize=(10, 3),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )

    # -------------------------------------------------------------------------------
    # plot
    # -------------------------------------------------------------------------------
    data_analysis = data_analysis.sort_values(["location", "position"])

    # one color for each string
    col_idx = 0
    # some lists to fill with info, string by string
    labels = []
    channels = []
    legend = []

    # group by string
    for location, data_location in data_analysis.groupby("location"):
        utils.logger.debug(f"... {plot_info['locname']} {location}")

        max_ch_per_string = (
            data_analysis.groupby("location")["position"].nunique().max()
        )
        global COLORS
        COLORS = color_palette("hls", max_ch_per_string).as_hex()

        values_per_string = []  # y values - in each string
        channels_per_string = []  # x values - in each string
        # group by channel
        for position, data_channel in data_location.groupby("position"):
            plot_style(data_channel, fig, axes, plot_info, COLORS[col_idx])

            map_dict = utils.MAP_DICT
            location = data_channel["location"].unique()[0]

            # same detector for all dataframe since one channel
            labels.append(data_channel["name"].iloc[0])
            channels.append(map_dict[str(location)][str(position)])
            values_per_string.append(data_channel[plot_info["parameter"]].unique()[0])
            channels_per_string.append(map_dict[str(location)][str(position)])

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
    # set the grid behind the points
    axes.set_axisbelow(True)
    # beautification
    axes.ylabel = None
    axes.xlabel = None
    # add x labels
    axes.set_xticks(channels)
    axes.set_xticklabels(labels, fontsize=5)
    # rotate x labels
    plt.xticks(rotation=90, ha="center")
    # title/label
    fig.supxlabel("")
    fig.suptitle(f"{plot_info['subsystem']} - {plot_info['title']}", y=1.05)

    # -------------------------------------------------------------------------------
    # if no pdf is specified, then the function is not being called by make_subsystem_plots()
    if pdf:
        plt.savefig(pdf, format="pdf", bbox_inches="tight")
        # figures are retained until explicitly closed; close to not consume too much memory
        plt.close()

    # return fig


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
    data_analysis["fiber"] = (
        data_analysis["location"].str.split("-").str[1].str.join("")
        + "-"
        + data_analysis["location"].str.split("-").str[2].str.join("")
    )
    data_analysis["location"] = (
        data_analysis["location"].str.split("-").str[0].str.join("")
    )

    # -------------------------------------------------------------------------------
    # create label of format hardcoded for geds pX-chXXX-name
    # -------------------------------------------------------------------------------

    labels = data_analysis.groupby("channel").first()[
        ["name", "position", "location", "fiber"]
    ]
    labels["channel"] = labels.index
    labels["label"] = labels[
        ["position", "location", "fiber", "channel", "name"]
    ].apply(lambda x: f"{x[0]}-{x[1]}-{x[2]}-ch{str(x[3]).zfill(3)}-{x[4]}", axis=1)
    # put it in the table
    data_analysis = data_analysis.set_index("channel")
    data_analysis["label"] = labels["label"]

    data_analysis = data_analysis.sort_values(["location", "position"])

    # separate figure for each barrel ("location"= IB, OB)...
    for location, data_location in data_analysis.groupby("location"):
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

                    # set label as title for each axes
                    text = (
                        data_position["label"][0][4:]
                        if position == "top"
                        else data_position["label"][0][7:]
                    )
                    axes.set_title(label=text, loc="center")

                    # add grid
                    axes.grid("major", linestyle="--")
                    axes.set_axisbelow(True)
                    # remove automatic y label since there will be a shared one
                    axes.set_ylabel("")

                    det_idx += 1
                    col_idx += 1

            fig.suptitle(
                f"{plot_info['subsystem']} - {plot_info['title']}\n{position} {location}",
                y=1.15,
            )
            # fig.supylabel(f'{plotdata.param.label} [{plotdata.param.unit_label}]') # --> plot style
            plt.savefig(pdf, format="pdf", bbox_inches="tight")
            # figures are retained until explicitly closed; close to not consume too much memory
            plt.close()

            with io.BytesIO() as buf:
                fig.savefig(buf, bbox_inches="tight")
                buf.seek(0)
                par_dict[f"figure_plot_{location}_{position}"] = buf.getvalue()

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
