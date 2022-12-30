import logging

import numpy as np
import pandas as pd

from activitysim.core import config, inject, logit, pipeline, simulate, tracing
from activitysim.core.util import assign_in_place

from .ldt_pattern import LDT_PATTERN
from .util import estimation

logger = logging.getLogger(__name__)


@inject.step()
def ldt_scheduling(
    longdist_tours, persons_merged, land_use, network_los, chunk_size, trace_hh_id
):
    """
    This model schedules the start/end times for actors are on a trip
    in a given day (only those in the longdist_tours csv). This specifically estimates
    the start hour for begin trips, the end hour for end trips, and both the start/hour
    hour for complete trips.

    - *Configuration File*: `ldt_scheduling.yaml`
    - *Core Table*: `longdist_tours`
    - *Result Field*: `ldt_start_hour & ldt_end_hour`
    - *Result dtype*: `int8`
    """
    trace_label = "ldt_scheduling"
    model_settings_file_name = "ldt_scheduling.yaml"
    actor_column_name = "actor_type"
    start_colname = "ldt_start_hour"
    end_colname = "ldt_end_hour"

    # preliminary estimation steps
    model_settings = config.read_model_settings(model_settings_file_name)
    estimator = estimation.manager.begin_estimation(trace_label)
    constants = config.get_model_constants(model_settings)  # constants shared by all

    # merging in global constants
    categories = config.get_global_constants()
    constants.update(categories)

    # converting parameters to dataframes
    ldt_tours = longdist_tours.to_frame()
    land_use = land_use.to_frame()
    logger.info("Running %s with %d tours" % (trace_label, ldt_tours.shape[0]))

    # merging in the persons_merged data into ldt_tours for estimation
    persons_merged = persons_merged.to_frame()
    ldt_tours_merged = pd.merge(
        ldt_tours,
        persons_merged,
        left_on="person_id",
        right_index=True,
        how="left",
        suffixes=("", "_r"),
    )

    # read in settings for all estimation categories (start, end, complete)
    spec_categories = model_settings.get("SPEC_CATEGORIES", {})
    nest_spec = config.get_logit_model_settings(model_settings)  # MNL

    # create a dictionary to translate estimated complete tour choices to
    # start and end hours
    # earliest start hour is 5, latest end hour is 23, minimum duration is 2
    # duration = end hour - start hour
    complete_tour_translation = {}
    i = 5
    x = 0
    while i <= 23:
        if i > 18:
            break
        j = i
        while j <= 23:
            if j - i >= 2:
                complete_tour_translation[x] = [i, j]
                x += 1
            j += 1
        i += 1
    complete_tour_translation = pd.Series(complete_tour_translation)

    # map patterns in settings to their respective enums
    patterns = {
        "complete": LDT_PATTERN.COMPLETE,
        "begin": LDT_PATTERN.BEGIN,
        "end": LDT_PATTERN.END,
    }

    # read in and create time/dist skims between TAZs and internal TAZs
    model_area_key = model_settings.get("MODEL_AREA_KEY", None)
    segment_key = model_settings.get("SEGMENT_KEY", None)
    time_key = model_settings.get("TIME_KEY", None)
    dist_key = model_settings.get("DIST_KEY", None)

    assert (
        model_area_key is not None
        and segment_key is not None
        and time_key is not None
        and dist_key is not None
    )

    times, dists = get_skims(
        network_los, land_use, dist_key, time_key, segment_key, model_area_key
    )

    # get the TAZs that don't have possible destinations for a 2 hour complete ldt
    # --can't travel there/back and spend the min time in the destination
    # merge them into the choosers dataframe to prevent these 2 hour complete ldt tours
    invalid_tazs = []
    for taz in times.index:
        if (
            land_use.loc[taz][model_area_key] == 1
            and 2 * np.where(dists.loc[taz] >= 50, times.loc[taz], 9999).min()
            + constants["min_time_in_dest"]
            >= 119
        ):
            invalid_tazs.append(taz)
    ldt_tours_merged["no_2_hour_tour"] = ldt_tours_merged["home_zone_id"].isin(
        invalid_tazs
    )

    # lists to append all results to
    starts_list = []
    ends_list = []

    # estimate schedules for households/persons separately
    for actor_type, tours_segment in ldt_tours_merged.groupby(actor_column_name):
        # the segment to estimate on
        choosers = tours_segment

        logger.info(
            "ldt_scheduling actor_type '%s' (%s tours)"
            % (
                actor_type,
                len(choosers.index),
            )
        )

        # if there are no choosers, set default to -1
        if choosers.empty:
            starts_list.append(
                pd.Series(-1, index=tours_segment.index, name=start_colname).to_frame()
            )
            ends_list.append(
                pd.Series(-1, index=tours_segment.index, name=end_colname).to_frame()
            )
            continue

        # estimate schedules for begin/end/complete tours separately
        for category_settings in spec_categories:
            # see function documentation for category_num translation
            category_num = patterns.get(
                category_settings["NAME"].lower().replace("_tour", "")
            )
            # limiting scheduling to the current tour pattern
            subset = choosers[
                (choosers["ldt_pattern"] & LDT_PATTERN.COMPLETE) == category_num
            ]
            if len(subset) == 0:
                continue
            # name of the current tour pattern beign estimated
            category_name = category_settings["NAME"]

            # logic for complete tour pattern scheduling
            if category_num == LDT_PATTERN.COMPLETE:
                # read in specification for complete tour pattern scheduling
                model_spec = simulate.read_model_spec(
                    file_name=category_settings["SPEC"]
                )
                coefficients_df = simulate.read_model_coefficients(category_settings)
                model_spec = simulate.eval_coefficients(
                    model_spec, coefficients_df, estimator
                )

                nest_spec = config.get_logit_model_settings(model_settings)

                if estimator:
                    estimator.write_model_settings(
                        model_settings, model_settings_file_name
                    )
                    estimator.write_spec(model_settings)
                    estimator.write_coefficients(coefficients_df, model_settings)
                    estimator.write_choosers(choosers)

                # apply the specified multinomial logit model
                choices = simulate.simple_simulate(
                    choosers=subset,
                    spec=model_spec,
                    nest_spec=nest_spec,
                    locals_d=constants,
                    chunk_size=chunk_size,
                    trace_label=trace_label,
                    trace_choice_name="ldt_scheduling_"
                    + actor_type
                    + "_"
                    + category_name,
                    estimator=estimator,
                )

                if estimator:
                    estimator.write_choices(choices)
                    choices = estimator.get_survey_values(
                        choices,
                        "persons",
                        "ldt_scheduling_" + actor_type + "_" + category_name,
                    )
                    estimator.write_override_choices(choices)
                    estimator.end_estimation()

                # get the start/end hours associated with the estimated values
                starts_ends = complete_tour_translation.loc[choices.values]

                start_choices = pd.DataFrame(
                    starts_ends.apply(lambda x: x[0]).values,
                    index=subset.index,
                    columns=[start_colname],
                )
                end_choices = pd.DataFrame(
                    starts_ends.apply(lambda x: x[1]).values,
                    index=subset.index,
                    columns=[end_colname],
                )

                starts_list.append(start_choices)
                ends_list.append(end_choices)
            # logic for start/end pattern scheduling
            else:
                # get the freuqencies for the specific category type
                constants = config.get_model_constants(category_settings)

                # sampling probabilities for the current tour pattern (start/end)
                df = pd.DataFrame(
                    index=subset.index, columns=["hour_" + str(x) for x in range(24)]
                )
                for i in range(24):
                    key = "hour_" + str(i)
                    df[key] = constants[key]
                choices, _ = logit.make_choices(df, trace_choosers=trace_hh_id)

                # merge in scheduled values to the respective tour pattern (start/end)
                if category_num == LDT_PATTERN.BEGIN:
                    if isinstance(choices, pd.Series):
                        choices = choices.to_frame(start_colname)
                    starts_list.append(choices)
                elif category_num == LDT_PATTERN.END:
                    if isinstance(choices, pd.Series):
                        choices = choices.to_frame(end_colname)
                    ends_list.append(choices)
                else:
                    raise ValueError(f"BUG, bad category_num {category_num}")

    # merge in start hours if there are any, default -1
    if len(starts_list) != 0:
        starts_df = (
            pd.concat(starts_list)
            .reindex(ldt_tours_merged.index)
            .fillna({start_colname: -1}, downcast="infer")
        )
        assign_in_place(ldt_tours, starts_df)

    # merge in end hours if there are any, default -1
    if len(ends_list) != 0:
        ends_df = (
            pd.concat(ends_list)
            .reindex(ldt_tours_merged.index)
            .fillna({end_colname: -1}, downcast="infer")
        )
        assign_in_place(ldt_tours, ends_df)

    tracing.print_summary(
        "ldt_scheduling all tour type start times",
        ldt_tours[start_colname],
        value_counts=True,
    )

    tracing.print_summary(
        "ldt_scheduling all tour type end times",
        ldt_tours[end_colname],
        value_counts=True,
    )

    # update ldt_tours
    pipeline.replace_table("longdist_tours", ldt_tours)

    # merge start/end hours into households/persons for use in the internal mode model
    households = pipeline.get_table("households")
    persons = pipeline.get_table("persons")

    # why do we need to merge to households/persons to get internal stuff to run?
    households[start_colname] = starts_df.reindex(households.index).fillna(
        {start_colname: -1}, downcast="infer"
    )
    households[end_colname] = ends_df.reindex(households.index).fillna(
        {end_colname: -1}, downcast="infer"
    )

    persons[start_colname] = starts_df.reindex(persons.index).fillna(
        {start_colname: -1}, downcast="infer"
    )
    persons[end_colname] = ends_df.reindex(persons.index).fillna(
        {end_colname: -1}, downcast="infer"
    )

    pipeline.replace_table("households", households)
    pipeline.replace_table("persons", persons)

    if trace_hh_id:
        tracing.trace_df(
            ldt_tours,
            label=trace_label,
            slicer="tour_id",
            index_label="tour_id",
            warn_if_empty=True,
        )


def get_skims(network_los, land_use, dist_key, time_key, segment_key, model_area_key):
    """
    This function returns the time and distance skims for
    """
    from ...core.skim_dataset import SkimDataset
    from ...core.skim_dictionary import SkimDict

    skim_dict = network_los.get_default_skim_dict()
    if isinstance(skim_dict, SkimDict):
        skims = skim_dict.skim_data._skim_data
        key_dict = network_los.get_default_skim_dict().skim_dim3
        time_val = key_dict[time_key][segment_key]
        time_skim = skims[time_val]

        dist_val = key_dict[dist_key][segment_key]
        dist_skim = skims[dist_val]
        internal_tazs = land_use[land_use[model_area_key] == 1].index

        return pd.DataFrame(
            time_skim[:, internal_tazs - 1],
            index=land_use.index,
            columns=internal_tazs,
        ), pd.DataFrame(
            dist_skim[:, internal_tazs - 1],
            index=land_use.index,
            columns=internal_tazs,
        )
    elif isinstance(skim_dict, SkimDataset):
        time_array = skim_dict.dataset[time_key].sel(time_period=segment_key)
        dist_array = skim_dict.dataset[dist_key].sel(time_period=segment_key)
        internal_tazs = land_use[land_use[model_area_key] == 1].index
        time_df = pd.DataFrame(
            time_array.isel(dtaz=internal_tazs),
            index=time_array.coords["otaz"],
            columns=internal_tazs,
        )
        dist_df = pd.DataFrame(
            dist_array.isel(dtaz=internal_tazs),
            index=time_array.coords["otaz"],
            columns=internal_tazs,
        )
        return time_df, dist_df
    else:
        raise NotImplementedError(type(skim_dict))
