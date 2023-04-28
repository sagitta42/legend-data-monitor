import glob
import importlib.resources
import json
import logging
import os
import re
import shelve
import sys

# for getting DataLoader time range
from datetime import datetime, timedelta

from pandas import DataFrame, concat

# -------------------------------------------------------------------------

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# stream handler (console)
stream_handler = logging.StreamHandler()
stream_handler.setLevel(logging.DEBUG)

# format
formatter = logging.Formatter("%(asctime)s:  %(message)s")
stream_handler.setFormatter(formatter)
# file_handler.setFormatter(formatter)

# add to logger
logger.addHandler(stream_handler)

# -------------------------------------------------------------------------

pkg = importlib.resources.files("legend_data_monitor")

# load dictionary with plot info (= units, thresholds, label, ...)
with open(pkg / "settings" / "par-settings.json") as f:
    PLOT_INFO = json.load(f)

# which parameter belongs to which tier
with open(pkg / "settings" / "parameter-tiers.json") as f:
    PARAMETER_TIERS = json.load(f)

# which lh5 parameters are needed to be loaded from lh5 to calculate them
with open(pkg / "settings" / "special-parameters.json") as f:
    SPECIAL_PARAMETERS = json.load(f)

# dictionary map (helpful when we want to map channels based on their location/position)
with open(pkg / "settings" / "map-channels.json") as f:
    MAP_DICT = json.load(f)

# convert all to lists for convenience
for param in SPECIAL_PARAMETERS:
    if isinstance(SPECIAL_PARAMETERS[param], str):
        SPECIAL_PARAMETERS[param] = [SPECIAL_PARAMETERS[param]]

# -------------------------------------------------------------------------
# Subsystem related functions (for getting channel map & status)
# -------------------------------------------------------------------------


def get_query_times(**kwargs):
    """
    Get time ranges for DataLoader query from user input, as well as first timestamp for channel map/status query.

    Available kwargs:

    Available kwargs:
    dataset=
        dict with the following keys:
            - 'path' [str]: < move description here from get_data() >
            - 'version' [str]: < move description here from get_data() >
            - 'type' [str]: < move description here > ! not possible for multiple types now!
            - the following keys depending in time selection mode (choose one)
                1) 'start' : <start datetime>, 'end': <end datetime> where <datetime> input is of format 'YYYY-MM-DD hh:mm:ss'
                2) 'window'[str]: time window in the past from current time point, format: 'Xd Xh Xm' for days, hours, minutes
                2) 'timestamps': str or list of str in format 'YYYYMMDDThhmmssZ'
                3) 'runs': int or list of ints for run number(s)  e.g. 10 for r010
    Or input kwargs separately path=, ...; start=&end=, or window=, or timestamps=, or runs=

    Designed in such a way to accommodate Subsystem init kwargs. A bit cumbersome and can probably be done better.

    Path, version, and type only needed because channel map and status cannot be queried by run directly,
        so we need these to look up first timestamp in data path to run.

    >>> get_query_times(..., start='2022-09-28 08:00:00', end='2022-09-28 09:30:00')
    {'timestamp': {'start': '20220928T080000Z', 'end': '20220928T093000Z'}}, '20220928T080000Z'

    >> get_query_times(..., runs=27)
    ({'run': ['r027']}, '20220928T091135Z')
    """
    # get time range query for DataLoader
    timerange = get_query_timerange(**kwargs)

    first_timestamp = ""
    # get first timestamp in case keyword is timestamp
    if "timestamp" in timerange:
        if "start" in timerange["timestamp"]:
            first_timestamp = timerange["timestamp"]["start"]
        else:
            first_timestamp = min(timerange["timestamp"])
    # look in path to find first timestamp if keyword is run
    else:
        # currently only list of runs and not 'start' and 'end', so always list
        # find earliest run, format rXXX
        first_run = min(timerange["run"])

        # --- get dsp filelist of this run
        # if setup= keyword was used, get dict; otherwise kwargs is already the dict we need
        path_info = kwargs["dataset"] if "dataset" in kwargs else kwargs

        # format to search /path_to_prod-ref[/v06.00]/generated/tier/**/phy/**/r027 (version might not be there)
        glob_path = os.path.join(
            path_info["path"],
            path_info["version"],
            "generated",
            "tier",
            "**",
            path_info["type"],
            "**",
            first_run,
            "*.lh5",
        )
        dsp_files = glob.glob(glob_path)
        # find earliest
        dsp_files.sort()
        first_file = dsp_files[0]
        # extract timestamp
        first_timestamp = get_key(first_file)

    return timerange, first_timestamp


def get_query_timerange(**kwargs):
    """
    Get DataLoader compatible time range.

    Available kwargs:

    dataset=
        dict with the following keys depending in time selection mode (choose one)
            1) 'start' : <start datetime>, 'end': <end datetime> where <datetime> input is of format 'YYYY-MM-DD hh:mm:ss'
            2) 'window'[str]: time window in the past from current time point, format: 'Xd Xh Xm' for days, hours, minutes
            2) 'timestamps': str or list of str in format 'YYYYMMDDThhmmssZ'
            3) 'runs': int or list of ints for run number(s)  e.g. 10 for r010
    Or enter kwargs separately start=&end=, or window=, or timestamp=, or runs=

    Designed in such a way to accommodate Subsystem init kwargs. A bit cumbersome and can probably be done better.

    >>> get_query_timerange(start='2022-09-28 08:00:00', end='2022-09-28 09:30:00')
    {'timestamp': {'start': '20220928T080000Z', 'end': '20220928T093000Z'}}

    >>> get_query_timerange(window='1d 5h 0m')
    {'timestamp': {'end': '20230220T114337Z', 'start': '20230219T064337Z'}}

    >>> get_query_timerange(timestamps=['20220928T080000Z', '20220928093000Z'])
    {'timestamp': ['20220928T080000Z', '20220928093000Z']}
    >>> get_query_timerange(timestamps='20220928T080000Z')
    {'timestamp': ['20220928T080000Z']}

    >> get_query_timerange(runs=[9,10])
    {'run': ['r009', 'r010']}
    >>> get_query_timerange(runs=10)
    {'run': ['r010']}

    >>> get_query_timerange(dataset={'start': '2022-09-28 08:00:00', 'end':'2022-09-28 09:30:00'})
    {'timestamp': {'start': '20220928T080000Z', 'end': '20220928T093000Z'}}
    """
    # if dataset= kwarg was provided, get the 'selection' part
    # otherwise kwargs itself is already the dict we need with start= & end=; or timestamp= etc
    user_selection = kwargs["dataset"] if "dataset" in kwargs else kwargs

    # -------------------------------------------------------------------------
    #  in these cases, DataLoader will be called with (timestamp >= ...) and (timestamp <= ...)
    # -------------------------------------------------------------------------
    if "start" in user_selection:
        time_range = {"timestamp": {}}
        for point in ["start", "end"]:
            try:
                time_range["timestamp"][point] = datetime.strptime(
                    user_selection[point], "%Y-%m-%d %H:%M:%S"
                ).strftime("%Y%m%dT%H%M%SZ")
            except ValueError:
                logger.error("\033[91mInvalid date format!\033[0m")
                return

    elif "window" in user_selection:
        time_range = {"timestamp": {}}
        time_range["timestamp"]["end"] = datetime.now().strftime("%Y%m%dT%H%M%SZ")
        try:
            days, hours, minutes = re.split(r"d|h|m", user_selection["window"])[
                :-1
            ]  # -1 for trailing ''
        except ValueError:
            logger.error("\033[91mInvalid window format!\033[0m")
            return

        dt = timedelta(days=int(days), hours=int(hours), minutes=int(minutes))
        time_range["timestamp"]["start"] = (
            datetime.strptime(time_range["timestamp"]["end"], "%Y%m%dT%H%M%SZ") - dt
        ).strftime("%Y%m%dT%H%M%SZ")

    # -------------------------------------------------------------------------
    #  in these cases, DataLoader will be called with (timestamp/run == ...) or (timestamp/run == ...)
    # -------------------------------------------------------------------------
    elif "timestamps" in user_selection:
        time_range = {"timestamp": []}
        time_range["timestamp"] = (
            user_selection["timestamps"]
            if isinstance(user_selection["timestamps"], list)
            else [user_selection["timestamps"]]
        )

    elif "runs" in user_selection:
        runs = (
            user_selection["runs"]
            if isinstance(user_selection["runs"], list)
            else [user_selection["runs"]]
        )
        # check validity
        for run in runs:
            if not isinstance(run, int):
                logger.error("\033[91mInvalid run format!\033[0m")
                return

        # format rXXX for DataLoader
        time_range = {"run": []}
        time_range["run"] = ["r" + str(run).zfill(3) for run in runs]

    else:
        logger.error(
            "\033[91mInvalid time selection. Choose among: runs, timestamps, window, start+end - try again!\033[0m"
        )
        return

    return time_range


# -------------------------------------------------------------------------
# Plotting related functions
# -------------------------------------------------------------------------


def check_plot_settings(conf: dict):
    from . import plot_styles, plotting

    options = {
        "plot_structure": plotting.PLOT_STRUCTURE.keys(),
        "plot_style": plot_styles.PLOT_STYLE.keys(),
    }

    for subsys in conf["subsystems"]:
        for plot in conf["subsystems"][subsys]:
            # settings for this plot
            plot_settings = conf["subsystems"][subsys][plot]

            # check if all necessary fields for param settings were provided
            for field in options:
                # if this field is not provided by user, tell them to provide it
                # (if optional to provided, will have been set with defaults before calling set_defaults())
                if field not in plot_settings:
                    logger.error(
                        f"\033[91mProvide {field} in plot settings of '{plot}' for {subsys}!\033[0m"
                    )
                    logger.error(
                        "\033[91mAvailable options: {}\033[0m".format(
                            ",".join(options[field])
                        )
                    )
                    return False

                # check if the provided option is valid
                opt = plot_settings[field]

                if opt not in options[field]:
                    logger.error(
                        f"\033[91mOption {opt} provided for {field} in plot settings of '{plot}' for {subsys} does not exist!\033[0m"
                    )
                    logger.error(
                        "\033[91mAvailable options: {}\033[0m".format(
                            ",".join(options[field])
                        )
                    )
                    return False

            # if vs time was provided, need time window
            if (
                plot_settings["plot_style"] == "vs time"
                and "time_window" not in plot_settings
            ):
                logger.error(
                    "\033[91mYou chose plot style 'vs time' and did not provide 'time_window'!\033[0m"
                )
                return False

    return True


def make_output_paths(config: dict, user_time_range: dict) -> str:
    """
    Get a dict and return a dict. The function defines output paths and create directories accordingly.

    To use when you want a specific output structure of the following type: [...]/prod-ref/{version}/generated/plt/phy/{period}/{run}
    This does not work if you select more types (eg. both cal and phy) or timestamp intervals (but just runs).
    It can be used for run summary plots, eg during stable data taking.
    """
    logger.info("Setting up plotting...")

    if "output" not in config:
        logger.error(
            "\033[91mProvide output folder path in your config field 'output'!\033[0m"
        )
        return

    # general output path
    try:
        make_dir(config["output"])
    except Exception:
        logger.error(f"\033[91mCannot make output folder {config['output']}\033[0m")
        logger.error("\033[91mMaybe you don't have rights to create this path?\033[0m")
        return

    # create subfolders for the fixed path
    logger.info("config[output]:" + config["output"])
    version_dir = os.path.join(config["output"], config["dataset"]["version"])
    generated_dir = os.path.join(version_dir, "generated")
    plt_dir = os.path.join(generated_dir, "plt")
    # 'phy' or 'cal' if one of the two is specified; if both are specified, store data in 'cal_phy/'
    if isinstance(config["dataset"]["type"], list):
        type_dir = os.path.join(plt_dir, "cal_phy")
    else:
        type_dir = os.path.join(plt_dir, config["dataset"]["type"])
    # period info
    period_dir = os.path.join(type_dir, config["dataset"]["period"]) + "/"

    # output subfolders
    make_dir(version_dir)
    make_dir(generated_dir)
    make_dir(plt_dir)
    make_dir(type_dir)
    make_dir(period_dir)

    return period_dir


def make_dir(dir_path):
    """Check if directory exists, and if not, make it."""
    message = "Output directory " + dir_path
    if not os.path.isdir(dir_path):
        os.mkdir(dir_path)
        message += " (created)"
    logger.info(message)


def get_multiple_run_id(user_time_range: dict) -> str:
    time_range = list(user_time_range.values())[0]
    name_time = "{}".format("_".join(time_range))
    return name_time


def get_time_name(user_time_range: dict) -> str:
    """Get a name for each available time selection.

    careful handling of folder name depending on the selected time range. The possibilities are:
      1) user_time_range = {'timestamp': {'start': '20220928T080000Z', 'end': '20220928T093000Z'}} => start + end
              -> folder: 20220928T080000Z_20220928T093000Z/
      2) user_time_range = {'timestamp': ['20230207T103123Z']} => one key
              -> folder: 20230207T103123Z/
      3) user_time_range = {'timestamp': ['20230207T103123Z', '20230207T141123Z', '20230207T083323Z']} => multiple keys
              -> get min/max and use in the folder name
              -> folder: 20230207T083323Z_20230207T141123Z/
      4) user_time_range = {'run': ['r010']} => one run
              -> folder: r010/
      5) user_time_range = {'run': ['r010', 'r014']} => multiple runs
              -> folder: r010_r014/
    """
    name_time = ""
    if "timestamp" in user_time_range.keys():
        time_range = list(user_time_range.values())[0]
        if "start" in time_range:
            name_time += time_range["start"] + "_" + time_range["end"]
        else:
            if len(time_range) == 1:
                name_time += time_range[0]
            else:
                timestamp_range = [
                    datetime.strptime(string, "%Y%m%dT%H%M%SZ").timestamp()
                    for string in time_range
                ]
                min_idx = timestamp_range.index(min(timestamp_range))
                max_idx = timestamp_range.index(max(timestamp_range))
                name_time += time_range[min_idx] + "_" + time_range[max_idx]

    elif "run" in user_time_range.keys():
        name_time = get_multiple_run_id(user_time_range)

    else:
        logger.error("\033[91mInvalid time selection!\033[0m")
        return

    return name_time


def get_timestamp(filename):
    # Assumes that the timestamp is in the format YYYYMMDDTHHMMSSZ
    return filename.split("-")[-2]


def get_run_name(config, user_time_range: dict) -> str:
    """Get the run ID given start/end timestamps."""
    # this is the root directory to search in the timestamps
    main_folder = os.path.join(
        config["dataset"]["path"], config["dataset"]["version"], "generated/tier"
    )

    # start/end timestamps of the selected time range of interest
    start_timestamp = user_time_range["timestamp"]["start"]
    end_timestamp = user_time_range["timestamp"]["end"]

    run_list = []  # this will be updated with the run ID

    # start to look for timestamps inside subfolders
    def search_for_timestamp(folder):
        run_id = ""
        for subfolder in os.listdir(folder):
            subfolder_path = os.path.join(folder, subfolder)
            if os.path.isdir(subfolder_path):
                files = sorted(glob.glob(os.path.join(subfolder_path, "*")))
                for i, file in enumerate(files):
                    if (
                        get_timestamp(files[i - 1])
                        <= start_timestamp
                        <= get_timestamp(file)
                    ) or (
                        get_timestamp(files[i - 1])
                        <= end_timestamp
                        <= get_timestamp(file)
                    ):
                        run_id = file.split("/")[-2]
                        run_list.append(run_id)
                        break

                if len(run_list) == 0:
                    search_for_timestamp(subfolder_path)
                else:
                    break
        return

    search_for_timestamp(main_folder)

    if len(run_list) == 0:
        logger.error(
            "\033[91mThe selected timestamps were not find anywhere. Try again with another time range!\033[0m"
        )
        sys.exit()
    if len(run_list) > 1:
        return get_multiple_run_id(user_time_range)

    return run_list[0]


def get_all_plot_parameters(subsystem: str, config: dict):
    """Get list of all parameters needed for all plots for given subsystem."""
    all_parameters = []
    if subsystem in config["subsystems"]:
        for plot in config["subsystems"][subsystem]:
            parameters = config["subsystems"][subsystem][plot]["parameters"]
            if isinstance(parameters, str):
                all_parameters.append(parameters)
            else:
                all_parameters += parameters

            # check if there is any QC entry; if so, add it to the list of parameters to load
            if "quality_cuts" in config["subsystems"][subsystem][plot]:
                all_parameters.append(
                    config["subsystems"][subsystem][plot]["quality_cuts"]
                )

    return all_parameters


def get_key(dsp_fname: str) -> str:
    """Extract key from lh5 filename."""
    return re.search(r"-\d{8}T\d{6}Z", dsp_fname).group(0)[1:]


# -------------------------------------------------------------------------
# Config file related functions (for building files)
# -------------------------------------------------------------------------


def add_config_entries(
    config: dict,
    file_keys: str,
    prod_path: str,
    prod_config: dict,
) -> dict:
    """Add missing information (output, dataset) to the configuration file. This function is generally used during automathic data production, where the initiali config file has only the 'subsystem' entry."""
    # Get the keys
    with open(file_keys) as f:
        keys = f.readlines()
    # Remove newline characters from each line using strip()
    keys = [key.strip() for key in keys]
    # get only keys of timestamps
    timestamp = [key.split("-")[-1] for key in keys]

    # Get the experiment
    experiment = (keys[0].split("-"))[0].upper()

    # Get the period
    period = (keys[0].split("-"))[1]

    # Get the run
    run = (keys[0].split("-"))[2]

    # Get the version
    if "dataset" in config.keys():
        if "version" in config["dataset"].keys():
            version = config["dataset"]["version"]
        else:
            version = (
                (prod_path.split("/"))[-2]
                if prod_path.endswith("/")
                else (prod_path.split("/"))[-1]
            )
        if "type" in config["dataset"].keys():
            type = config["dataset"]["type"]
        else:
            logger.error("\033[91mYou need to provide data type! Try again.\033[0m")
            sys.exit()
        if "path" in config["dataset"].keys():
            path = config["dataset"]["path"]
        else:
            logger.error(
                "\033[91mYou need to provide path to lh5 files! Try again.\033[0m"
            )
            sys.exit()
    else:
        # get phy/cal lists
        phy_keys = [key for key in keys if "phy" in key]
        cal_keys = [key for key in keys if "cal" in key]
        if len(phy_keys) == 0 and len(cal_keys) == 0:
            logger.error("\033[91mNo keys to load. Try again.\033[0m")
            return
        if len(phy_keys) != 0 and len(cal_keys) == 0:
            type = "phy"
        if len(phy_keys) == 0 and len(cal_keys) != 0:
            type = "cal"
            logger.error("\033[91mcal is still under development! Try again.\033[0m")
            return
        if len(phy_keys) != 0 and len(cal_keys) != 0:
            type = ["cal", "phy"]
            logger.error(
                "\033[91mBoth cal and phy are still under development! Try again.\033[0m"
            )
            return
        # Get the production path
        path = (
            prod_path.split("prod-ref")[0] + "prod-ref"
            if prod_path.split("prod-ref")[0].endswith("/")
            else prod_path.split("prod-ref")[0] + "/prod-ref"
        )

    if "output" in config.keys():
        prod_path = config["output"]

    # create the dataset dictionary
    dataset_dict = {
        "experiment": experiment,
        "period": period,
        "version": version,
        "path": path,
        "type": type,
        "run": run,
        "timestamps": timestamp,
    }

    more_info = {"output": prod_path, "dataset": dataset_dict}

    # 'saving' and 'subsystem' info must be already there
    config.update(more_info)

    # let's make a check that everything we need is inside the config, otherwise exit
    if not all(key in config for key in ["output", "dataset", "saving", "subsystems"]):
        logger.error(
            '\033[91mThere are missing entries among ["output", "dataset", "saving", "subsystems"] in the config file (found keys: %s). Try again and check you start with "output" and "dataset" info!\033[0m',
            config.keys(),
        )
        sys.exit()

    return config


# -------------------------------------------------------------------------
# Saving related functions
# -------------------------------------------------------------------------


def build_out_dict(
    plot_settings: list,
    plot_info: list,
    par_dict_content: dict,
    out_dict: dict,
    saving: str,
    plt_path: str,
):
    """Build the output dictionary based on the input 'saving' option."""
    # we overwrite the object with a new one
    if saving == "overwrite":
        out_dict = save_dict(plot_settings, plot_info, par_dict_content, out_dict)

    # we retrieve the already existing shelve object, and we append new things to it; the parameter here is fixed
    if saving == "append":
        # the file does not exist, so first we create it and then, at the next step, we'll append things
        if not os.path.exists(plt_path + "-" + plot_info["subsystem"] + ".dat"):
            # logger.warning(
            #    "\033[93mYou selected 'append' when saving, but the file with already saved data does not exist. For this reason, it will be created first.\033[0m"
            # )
            out_dict = save_dict(plot_settings, plot_info, par_dict_content, out_dict)

        # the file exists, so we are going to append data
        else:
            logger.info(
                "There is already a file containing output data. Appending new data to it right now..."
            )
            # open already existing shelve file
            with shelve.open(plt_path + "-" + plot_info["subsystem"], "r") as shelf:
                old_dict = dict(shelf)

            # the parameter is there
            parameter = (
                plot_info["parameter"].split("_var")[0]
                if "_var" in plot_info["parameter"]
                else plot_info["parameter"]
            )
            if old_dict["monitoring"]["pulser"][parameter]:
                # get already present df
                old_df = old_dict["monitoring"]["pulser"][parameter][
                    "df_" + plot_info["subsystem"]
                ]
                old_df = check_level0(old_df)
                # get new df (plot_info object is the same as before, no need to get it and update it)
                new_df = par_dict_content["df_" + plot_info["subsystem"]]
                # concatenate the two dfs (channels are no more grouped; not a problem)
                merged_df = DataFrame.empty
                merged_df = concat([old_df, new_df], ignore_index=True, axis=0)
                merged_df = merged_df.reset_index()
                merged_df = check_level0(merged_df)
                # re-order content in order of channels/timestamps
                merged_df = merged_df.sort_values(["channel", "datetime"])

                # redefine the dict containing the df and plot_info
                par_dict_content = {}
                par_dict_content["df_" + plot_info["subsystem"]] = merged_df
                par_dict_content["plot_info"] = plot_info

                # saved the merged df as usual
                out_dict = save_dict(
                    plot_settings, plot_info, par_dict_content, old_dict["monitoring"]
                )
                # we need to save it, otherwise when looping over the next parameter we lose the appended info for the already inspected parameter
                out_file = shelve.open(plt_path + "-" + plot_info["subsystem"])
                out_file["monitoring"] = out_dict
                out_file.close()

    return out_dict


def save_dict(
    plot_settings: list, plot_info: list, par_dict_content: dict, out_dict: dict
):
    """Create a dictionary with the correct format for being saved in the final shelve object."""
    parameter = (
        plot_info["parameter"].split("_var")[0]
        if "_var" in plot_info["parameter"]
        else plot_info["parameter"]
    )
    # event type key is already there
    if plot_settings["event_type"] in out_dict.keys():
        out_dict[plot_settings["event_type"]][parameter] = par_dict_content
    # event type key is NOT there
    else:
        # empty dictionary (not filled yet)
        if len(out_dict.keys()) == 0:
            out_dict = {plot_settings["event_type"]: {parameter: par_dict_content}}
        # the dictionary already contains something (but for another event type selection)
        else:
            out_dict[plot_settings["event_type"]] = {parameter: par_dict_content}

    return out_dict


def check_level0(dataframe: DataFrame) -> DataFrame:
    """Check if a dataframe contains the 'level_0' column. If so, remove it."""
    if "level_0" in dataframe.columns:
        dataframe = dataframe.drop(columns=["level_0"])
    return dataframe
