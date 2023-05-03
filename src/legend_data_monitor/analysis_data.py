import os
import shelve

import numpy as np
import pandas as pd
from pandas import DataFrame, concat

# needed to know which parameters are not in DataLoader
# but need to be calculated, such as event rate
from . import cuts, utils

# -------------------------------------------------------------------------


class AnalysisData:
    """
    Object containing information for a data subselected from Subsystem data based on given criteria.

    sub_data [DataFrame]: subsystem data

    Available kwargs:
        selection=
            dict with the following contents:
                - 'parameters' [str or list of str]: parameter(s) of interest e.g. 'baseline'
                - 'event_type' [str]: event type, options: pulser/phy/all
                - 'cuts' [str or list of str]: [optional] cuts to apply to data (will be loaded but not applied immediately)
                - 'variation' [bool]: [optional] keep absolute value of parameter (False) or calculate % variation from mean (True).
                    Default: False
                - 'time_window' [str]: [optional] time window in which to calculate event rate, in case that's the parameter of interest.
                    Format: time_window='NA', where N is integer, and A is M for months, D for days, T for minutes, and S for seconds.
                    Default: None
        Or input kwargs directly parameters=, event_type=, cuts=, variation=, time_window=

        To apply a single cut, use data_after_cut = ldm.apply_cut(<analysis_data>)
        To apply all cuts, use data_after_all_cuts = <analysis_data>.apply_all_cuts()
            where <analysis_data> is the AnalysisData object you created.
    """

    def __init__(self, sub_data: pd.DataFrame, **kwargs):
        # if selection= was provided, take the dict
        # if kwargs were used directly, kwargs itself is already our dict
        # need to do .copy() or else modifies original config!
        analysis_info = (
            kwargs["selection"].copy() if "selection" in kwargs else kwargs.copy()
        )

        # -------------------------------------------------------------------------
        # validity checks
        # -------------------------------------------------------------------------

        # defaults
        if "time_window" not in analysis_info:
            analysis_info["time_window"] = None
        if "variation" not in analysis_info:
            analysis_info["variation"] = False
        if "cuts" not in analysis_info:
            analysis_info["cuts"] = []
        if "plt_path" not in analysis_info:
            analysis_info["saving"] = analysis_info["plt_path"] = None

        # convert single parameter input to list for convenience
        for input in ["parameters", "cuts"]:
            if isinstance(analysis_info[input], str):
                analysis_info[input] = [analysis_info[input]]

        if analysis_info["event_type"] != "all" and "flag_pulser" not in sub_data:
            utils.logger.error(
                f"\033[91mYour subsystem data does not have a pulser flag! We need it to subselect event type {analysis_info['event_type']}\033[0m"
            )
            utils.logger.error(
                "\033[91mRun the function <subsystem>.flag_pulser_events(<pulser>) first, where <subsystem> is your Subsystem object, \033[0m"
                + "\033[91mand <pulser> is a Subsystem object of type 'pulser', which already has it data loaded with <pulser>.get_data(); then create AnalysisData object.\033[0m"
            )
            return

        # cannot do event rate and another parameter at the same time
        # since event rate is calculated in windows
        if (
            "event_rate" in analysis_info["parameters"]
            and len(analysis_info["parameters"]) > 1
        ):
            utils.logger.error(
                "\033[91mCannot get event rate and another parameter at the same time!\n \
                Event rate has to be calculated based on time windows, so the other parameter has to be thrown away.\
                Contact developers if you want, for example, to keep that parameter, but look at mean in the windows of event rate.\033[0m"
            )
            return

        # time window must be provided for event rate
        if (
            analysis_info["parameters"][0] == "event_rate"
            and not analysis_info["time_window"]
        ):
            utils.logger.error(
                "\033[91mProvide argument <time_window> in which to take the event rate!\033[0m"
            )
            utils.logger.error("\033[91m%s\033[0m", self.__doc__)
            return

        self.parameters = analysis_info["parameters"]
        self.evt_type = analysis_info["event_type"]
        self.time_window = analysis_info["time_window"]
        self.variation = analysis_info["variation"]
        self.cuts = analysis_info["cuts"]
        self.saving = analysis_info["saving"]
        self.plt_path = analysis_info["plt_path"]

        # -------------------------------------------------------------------------
        # subselect data
        # -------------------------------------------------------------------------

        # always get basic parameters
        params_to_get = [
            "timestamp",
            "datetime",
            "channel",
            "name",
            "location",
            "position",
            "cc4_id",
            "cc4_channel",
            "daq_crate",
            "daq_card",
            "HV_card",
            "HV_channel",
            "det_type",
            "status",
        ]
        # pulser flag is present only if subsystem.flag_pulser_events() was called
        # needed to subselect phy/pulser events
        if "flag_pulser" in sub_data:
            params_to_get.append("flag_pulser")

        # if special parameter, get columns needed to calculate it
        for param in self.parameters:
            # check if the parameter is within the par-settings.json file
            if param in utils.PLOT_INFO.keys():
                # check if it is a special parameter
                if param in utils.SPECIAL_PARAMETERS:
                    # ignore if none are needed
                    params_to_get += (
                        utils.SPECIAL_PARAMETERS[param]
                        if utils.SPECIAL_PARAMETERS[param]
                        else []
                    )
                else:
                    # otherwise just load it
                    params_to_get.append(param)
            # the parameter does not exist
            else:
                utils.logger.error(
                    "\033[91m'%s' either does not exist in 'par-settings.json' or you misspelled the parameter's name. TRY AGAIN.\033[0m",
                    param,
                )
                exit()

        # avoid repetition
        params_to_get = list(np.unique(params_to_get))

        # check if there are the corresponding columns in the dataframe; otherwise, exit
        if set(params_to_get).issubset(sub_data.columns):
            self.data = sub_data[params_to_get].copy()
        else:
            utils.logger.error(
                "\033[91mOne/more entry/entries among %s is/are not present in the dataframe. TRY AGAIN.\033[0m",
                params_to_get,
            )
            exit()

        # -------------------------------------------------------------------------

        # selec phy/puls/all events
        bad = self.select_events()
        if bad:
            return

        # calculate if special parameter
        self.special_parameter()

        # calculate channel mean
        self.channel_mean()

        # calculate variation if needed - only works after channel mean
        self.calculate_variation()

        # -------------------------------------------------------------------------

        self.data = self.data.sort_values(["channel", "datetime"])

    def select_events(self):
        # do we want to keep all, phy or pulser events?
        if self.evt_type == "pulser":
            utils.logger.info("... keeping only pulser events")
            self.data = self.data[self.data["flag_pulser"]]
        elif self.evt_type == "phy":
            utils.logger.info("... keeping only physical (non-pulser) events")
            self.data = self.data[~self.data["flag_pulser"]]
        elif self.evt_type == "K_lines":
            utils.logger.info("... selecting K lines in physical (non-pulser) events")
            self.data = self.data[~self.data["flag_pulser"]]
            energy = utils.SPECIAL_PARAMETERS["K_events"][0]
            self.data = self.data[
                (self.data[energy] > 1430) & (self.data[energy] < 1575)
            ]
        elif self.evt_type == "all":
            utils.logger.info("... keeping all (pulser + non-pulser) events")
        else:
            utils.logger.error("\033[91mInvalid event type!\033[0m")
            utils.logger.error("\033[91m%s\033[0m", self.__doc__)
            return "bad"

    def special_parameter(self):
        for param in self.parameters:
            if param == "wf_max_rel":
                # calculate wf max relative to baseline
                self.data["wf_max_rel"] = self.data["wf_max"] - self.data["baseline"]
            elif param == "event_rate":
                # ! sorry need to jump through a lot of hoops here ! bare with me....

                # --- count number of events in given time windows
                # - count() returns count of rows for each column - redundant, same value in each (unless we have NaN)
                # just want one column 'event_rate' -> pick 'channel' since it's never NaN, so correct count; rename to event rate
                # - this is now a resampled dataframe with column event rate, and multiindex channel, datetime -> put them back as columns with reset index
                event_rate = (
                    self.data.set_index("datetime")
                    .groupby("channel")
                    .resample(self.time_window, origin="start")
                    .count()["channel"]
                    .to_frame(name="event_rate")
                    .reset_index()
                )

                # divide event count in each time window by sampling window in seconds to get Hz
                dt_seconds = get_seconds(self.time_window)
                event_rate["event_rate"] = event_rate["event_rate"] / dt_seconds

                # --- get rid of last value
                # as the data range does not equally divide by the time window, the count in the last "window" will be smaller
                # as it corresponds to, in reality, smaller window
                # since we divided by the window, the rate then will appear as smaller
                # it's too complicated to fix that, so I will just get rid of the last row
                event_rate = event_rate.iloc[:-1]

                # --- shift timestamp
                # the resulting table will start with the first timestamp of the original table
                # I want to shift the time values by the half the time window, so that the event rate value corresponds to the middle of the time window
                event_rate["datetime"] = (
                    event_rate["datetime"] + pd.Timedelta(self.time_window) / 2
                )

                # --- now have to jump through hoops to put back in location position and name
                # - group original table by channel and pick first occurrence to get the channel map (ignore other columns)
                # - reindex to match event rate table index
                # - put the columns in with concat
                event_rate = event_rate.set_index("channel")
                self.data = pd.concat(
                    [
                        event_rate,
                        self.data.groupby("channel")
                        .first()
                        .reindex(event_rate.index)[["name", "location", "position"]],
                    ],
                    axis=1,
                )
                # put the channel back as column
                self.data = self.data.reset_index()
            elif param == "FWHM":
                self.data = self.data.reset_index()

                # calculate FWHM for each channel (substitute 'param' column with it)
                channel_fwhm = (
                    self.data.groupby("channel")[utils.SPECIAL_PARAMETERS[param][0]]
                    .apply(
                        lambda x: 2.355
                        * np.sqrt(np.mean((x - np.mean(x, axis=0)) ** 2, axis=0))
                    )
                    .reset_index(name="FWHM")
                )

                # join the calculated RMS values to the original dataframe
                self.data = self.data.merge(channel_fwhm, on="channel")

                # put channel back in
                self.data.reset_index()
            elif param == "K_events":
                self.data = self.data.reset_index()
                self.data = self.data.rename(
                    columns={utils.SPECIAL_PARAMETERS[param][0]: "K_events"}
                )
            elif param == "AoE_Custom":
                self.data["AoE_Custom"] = self.data["A_max"] / self.data["cuspEmax"]

    def channel_mean(self):
        """
        Get mean value of each parameter of interest in each channel in the first 10% of the dataset.

        Ignore in case of SiPMs, as each entry is a list of values, not a single value.
        """
        utils.logger.info("... getting channel mean")
        # series with index channel, columns of parameters containing mean of each channel;
        # the mean is performed over the first 10% interval of the full time range specified in the config file

        # get mean (only for non-list parameters; in that case, add a new column with None values):
        # check if we are looking at SiPMs -> do not get mean because entries are usually lists
        # ToDo: need to iterate over the parameters (some of them could be lists, others not)

        # congratulations, it's a sipm!
        if self.is_spms():
            channels = (self.data["channel"]).unique()
            channel_mean = pd.DataFrame(
                {"channel": channels, self.parameters[0]: [None] * len(channels)}
            )
            channel_mean = channel_mean.set_index("channel")
        # otherwise, it's either the pulser or geds
        else:
            if self.saving is None or self.saving == "overwrite":
                # get the dataframe for timestamps below 10% of data present in the selected time window
                self_data_time_cut = cut_dataframe(self.data)
                # create a column with the mean of the cut dataframe (cut in the time window of interest)
                channel_mean = self_data_time_cut.groupby("channel").mean(
                    numeric_only=True
                )[self.parameters]

            if self.saving == "append":
                subsys = self.get_subsys()
                # the file does not exist, so we get the mean as usual
                if not os.path.exists(self.plt_path + "-" + subsys + ".dat"):
                    self_data_time_cut = cut_dataframe(self.data)
                    # create a column with the mean of the cut dataframe (cut in the time window of interest)
                    channel_mean = self_data_time_cut.groupby("channel").mean(
                        numeric_only=True
                    )[self.parameters]

                # the file exist: we have to combine previous data with new data, and re-compute the mean over the first 10% of data (that now, are more than before)
                else:
                    # open already existing shelve file
                    with shelve.open(self.plt_path + "-" + subsys, "r") as shelf:
                        old_dict = dict(shelf)
                    # get old dataframe (we are interested only in the column with mean values)
                    old_df = old_dict["monitoring"][self.evt_type][self.parameters[0]][
                        "df_" + subsys
                    ]
                    """
                    # to use in the future for a more refined version of updated mean values...

                    # if previously we chose to plot % variations, we do not have anymore the absolute values to use when computing this new mean;
                    # what we can do, is to get absolute values starting from the mean and the % values present in the old dataframe'
                    # Later, we need to put these absolute values in the corresponding parameter column
                    if self.variation:
                        old_df[self.parameters[0]] = (old_df[self.parameters[0]] / 100 + 1) * old_df[self.parameters[0] + "_mean"]

                    merged_df = concat([old_df, self.data], ignore_index=True, axis=0)
                    # remove 'level_0' column (if present)
                    merged_df = utils.check_level0(merged_df)
                    merged_df = merged_df.reset_index()

                    self_data_time_cut = cut_dataframe(merged_df)

                    # ...still we have to re-compute the % variations of previous time windows because now the mean estimate is different!!!
                    """
                    # a column of mean values
                    mean_df = old_df[self.parameters[0] + "_mean"]
                    # a column of channels
                    channels = old_df["channel"]
                    # two columns: one of channels, one of mean values
                    channel_mean = concat(
                        [channels, mean_df], ignore_index=True, axis=1
                    ).rename(columns={0: "channel", 1: self.parameters[0]})
                    # drop potential duplicate rows
                    channel_mean = channel_mean.drop_duplicates(subset=["channel"])
                    # set 'channel' column as index
                    channel_mean = channel_mean.set_index("channel")

            # FWHM mean is meaningless -> drop (special parameter for SiPMs); no need to get previous mean values for these parameters
            if "FWHM" in self.parameters:
                channel_mean.drop("FWHM", axis=1)
            if "K_events" in self.parameters:
                channel_mean.drop("K_events", axis=1)

        # rename columns to be param_mean
        channel_mean = channel_mean.rename(
            columns={param: param + "_mean" for param in self.parameters}
        )
        # add it as column for convenience - repeating redundant information, but convenient
        self.data = self.data.set_index("channel")
        self.data = pd.concat(
            [self.data, channel_mean.reindex(self.data.index)], axis=1
        )
        # put channel back in
        self.data = self.data.reset_index()

    def calculate_variation(self):
        """
        Add a new column containing the percentage variation of a given parameter.

        The new column is called '<parameter>_var'.
        There is still the <parameter> column containing absolute values.
        There is only the <parameter> column if variation is set to False.
        """
        if self.variation:
            utils.logger.info("... calculating % variation from the mean")
            for param in self.parameters:
                # % variation: subtract mean from value for each channel
                self.data[param + "_var"] = (
                    self.data[param] / self.data[param + "_mean"] - 1
                ) * 100  # %

    def apply_all_cuts(self) -> DataFrame:
        data_after_cuts = self.data.copy()
        for cut in self.cuts:
            data_after_cuts = cuts.apply_cut(data_after_cuts, cut)
        return data_after_cuts

    def is_spms(self) -> bool:
        """Return True if 'location' (=fiber) and 'position' (=top, bottom) are strings."""
        if isinstance(self.data.iloc[0]["location"], str) and isinstance(
            self.data.iloc[0]["position"], str
        ):
            return True
        else:
            return False

    def is_geds(self) -> bool:
        """Return True if 'location' (=string) and 'position' are NOT strings."""
        if not self.is_spms():
            return True
        else:
            False

    def is_pulser(self) -> bool:
        """Return True if 'location' (=string) and 'position' are NOT strings."""
        if self.is_geds():
            if (
                self.data.iloc[0]["location"] == 0
                and self.data.iloc[0]["position"] == 0
            ):
                return True
            else:
                return False
        else:
            return False

    def get_subsys(self) -> str:
        """Return 'pulser', 'geds' or 'spms'."""
        if self.is_pulser():
            return "pulser"
        if self.is_spms():
            return "spms"
        if self.is_geds():
            return "geds"


# -------------------------------------------------------------------------
# helper function
# -------------------------------------------------------------------------


def get_seconds(time_window: str):
    """
    Convert sampling format used for DataFrame.resample() to int representing seconds.

    Needed for event rate calculation.

    >>> get_seconds('30T')
    1800
    """
    # correspondence of symbol to seconds, T = minutes
    str_to_seconds = {"S": 1, "T": 60, "H": 60 * 60, "D": 24 * 60 * 60}
    # unit of this time window
    time_unit = time_window[-1]

    return int(time_window.rstrip(time_unit)) * str_to_seconds[time_unit]


def cut_dataframe(data: DataFrame) -> DataFrame:
    """Get mean value of the parameters under study over the first 10% of data present in the selected time range."""
    min_datetime = data["datetime"].min()  # first timestamp
    max_datetime = data["datetime"].max()  # last timestamp
    duration = max_datetime - min_datetime
    ten_percent_duration = duration * 0.1
    thr_datetime = min_datetime + ten_percent_duration  # 10% timestamp
    # get only the rows for datetimes before the 10% of the specified time range
    return data.loc[data["datetime"] < thr_datetime]
