import json
import re

from . import plotting, subsystem, utils


def control_plots(user_config_path: str):
    """Set the configuration file and the output paths when a user config file is provided. The function to generate plots is then automatically called."""
    # -------------------------------------------------------------------------
    # Read user settings
    # -------------------------------------------------------------------------
    with open(user_config_path) as f:
        config = json.load(f)

    # check validity of plot settings
    valid = utils.check_plot_settings(config)
    if not valid:
        return

    # -------------------------------------------------------------------------
    # Define PDF file basename
    # -------------------------------------------------------------------------

    # Format: l200-p02-{run}-{data_type}; One pdf/log/shelve file for each subsystem

    try:
        data_types = (
            [config["dataset"]["type"]]
            if isinstance(config["dataset"]["type"], str)
            else config["dataset"]["type"]
        )

        plt_basename = "{}-{}-".format(
            config["dataset"]["experiment"].lower(),
            config["dataset"]["period"],
        )
    except (KeyError, TypeError):
        # means something about dataset is wrong -> print Subsystem.get_data doc
        utils.logger.error(
            "\033[91mSomething is missing or wrong in your 'dataset' field of the config. You can see the format here under 'dataset=':\033[0m"
        )
        utils.logger.info("\033[91m%s\033[0m", subsystem.Subsystem.get_data.__doc__)
        return

    user_time_range = utils.get_query_timerange(dataset=config["dataset"])
    # will be returned as None if something is wrong, and print an error message
    if not user_time_range:
        return

    # create output folders for plots
    period_dir = utils.make_output_paths(config, user_time_range)
    # get correct time info for subfolder's name
    name_time = (
        utils.get_run_name(config, user_time_range)
        if "timestamp" in user_time_range.keys()
        else utils.get_time_name(user_time_range)
    )
    output_paths = period_dir + name_time + "/"
    utils.make_dir(output_paths)
    if not output_paths:
        return

    # we don't care here about the time keyword timestamp/run -> just get the value
    plt_basename += name_time
    plt_path = output_paths + plt_basename
    plt_path += "-{}".format("_".join(data_types))

    # plot
    generate_plots(config, plt_path)


def auto_control_plots(
    plot_config: str, file_keys: str, prod_path: str, prod_config: str
):
    """Set the configuration file and the output paths when a config file is provided during automathic plot production."""
    # -------------------------------------------------------------------------
    # Read user settings
    # -------------------------------------------------------------------------
    with open(plot_config) as f:
        config = json.load(f)

    # check validity of plot settings
    valid = utils.check_plot_settings(config)
    if not valid:
        return

    # -------------------------------------------------------------------------
    # Add missing information (output, dataset) to the config
    # -------------------------------------------------------------------------
    config = utils.add_config_entries(config, file_keys, prod_path, prod_config)

    # -------------------------------------------------------------------------
    # Define PDF file basename
    # -------------------------------------------------------------------------
    # Format: l200-p02-{run}-{data_type}; One pdf/log/shelve file for each subsystem

    try:
        data_types = (
            [config["dataset"]["type"]]
            if isinstance(config["dataset"]["type"], str)
            else config["dataset"]["type"]
        )
        plt_basename = "{}-{}-".format(
            config["dataset"]["experiment"].lower(),
            config["dataset"]["period"],
        )
    except (KeyError, TypeError):
        # means something about dataset is wrong -> print Subsystem.get_data doc
        utils.logger.error(
            "\033[91mSomething is missing or wrong in your 'dataset' field of the config. You can see the format here under 'dataset=':\033[0m"
        )
        utils.logger.info("\033[91m%s\033[0m", subsystem.Subsystem.get_data.__doc__)
        return

    user_time_range = utils.get_query_timerange(dataset=config["dataset"])
    # will be returned as None if something is wrong, and print an error message
    if not user_time_range:
        return

    # create output folders for plots
    period_dir = utils.make_output_paths(config, user_time_range)
    # get correct time info for subfolder's name
    name_time = config["dataset"]["run"]
    output_paths = period_dir + name_time + "/"
    utils.make_dir(output_paths)
    if not output_paths:
        return

    # we don't care here about the time keyword timestamp/run -> just get the value
    plt_basename += name_time
    plt_path = output_paths + plt_basename
    plt_path += "-{}".format("_".join(data_types))

    # plot
    generate_plots(config, plt_path)


def generate_plots(config: dict, plt_path: str):
    """Generate plots once the config file is set and once we provide the path and name in which store results."""
    # -------------------------------------------------------------------------
    # Get pulser first - needed to flag pulser events
    # -------------------------------------------------------------------------

    # get saving option
    if "saving" in config:
        saving = config["saving"]
    else:
        saving = None

    # some output messages, just to warn the user...
    if saving is None:
        utils.logger.warning(
            "\033[93mData will not be saved, but the pdf will be.\033[0m"
        )
    elif saving == "append":
        utils.logger.warning(
            "\033[93mYou're going to append new data to already existing data. If not present, you first create the output file as a very first step.\033[0m"
        )
    elif saving == "overwrite":
        utils.logger.warning(
            "\033[93mYou have accepted to overwrite already generated files, there's no way back until you manually stop the code NOW!\033[0m"
        )
    else:
        utils.logger.error(
            "\033[91mThe selected saving option in the config file is wrong. Try again with 'overwrite', 'append' or nothing!\033[0m"
        )
        exit()

    # put it in a dict, so that later, if pulser is also wanted to be plotted, we don't have to load it twice
    subsystems = {"pulser": subsystem.Subsystem("pulser", dataset=config["dataset"])}
    # get list of all parameters needed for all requested plots, if any
    parameters = utils.get_all_plot_parameters("pulser", config)
    # get data for these parameters and time range given in the dataset
    # (if no parameters given to plot, baseline and wfmax will always be loaded to flag pulser events anyway)
    subsystems["pulser"].get_data(parameters)
    utils.logger.debug(subsystems["pulser"].data)

    # -------------------------------------------------------------------------

    # What subsystems do we want to plot?
    subsystems_to_plot = list(config["subsystems"].keys())

    for system in subsystems_to_plot:
        # -------------------------------------------------------------------------
        # set up subsystem
        # -------------------------------------------------------------------------

        # set up if wasn't already set up (meaning, not pulser, previously already set up)
        if system not in subsystems:
            # Subsystem: knows its channel map & software status (on/off channels)
            subsystems[system] = subsystem.Subsystem(system, dataset=config["dataset"])
            # get list of parameters needed for all requested plots, if any
            parameters = utils.get_all_plot_parameters(system, config)
            # get data for these parameters and dataset range
            subsystems[system].get_data(parameters)
            utils.logger.debug(subsystems[system].data)
            # flag pulser events for future parameter data selection
            subsystems[system].flag_pulser_events(subsystems["pulser"])

        # -------------------------------------------------------------------------
        # make subsystem plots
        # -------------------------------------------------------------------------

        # - set up log file for each system
        # file handler
        file_handler = utils.logging.FileHandler(plt_path + "-" + system + ".log")
        file_handler.setLevel(utils.logging.DEBUG)
        # add to logger
        utils.logger.addHandler(file_handler)

        plotting.make_subsystem_plots(
            subsystems[system], config["subsystems"][system], plt_path, saving
        )

        # -------------------------------------------------------------------------
        # beautification of the log file
        # -------------------------------------------------------------------------
        # Read the log file into a string
        with open(plt_path + "-" + system + ".log") as f:
            log_text = f.read()
        # Define a regular expression pattern to match escape sequences for color codes
        pattern = re.compile(r"\033\[[0-9;]+m")
        # Remove the color codes from the log text using the pattern
        clean_text = pattern.sub("", log_text)
        # Write the cleaned text to a new file
        with open(plt_path + "-" + system + ".log", "w") as f:
            f.write(clean_text)
