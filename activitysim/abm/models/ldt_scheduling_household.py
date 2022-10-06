# ActivitySim
# See full license in LICENSE.txt

import logging

from activitysim.core import config, expressions, inject, pipeline, simulate, tracing, logit

from .util import estimation

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


@inject.step()
def ldt_scheduling_household(households, households_merged, chunk_size, trace_hh_id):
    """
    This model schedules the start/end times for households that were assigned
    to be beginning and/or ending a tour on a given day
        - 0/complete: schedules both start/end of tour
        - 1/begin: schedules beginning of tour
        - 2/end: schedules end of tour
        - 3/away: does not schedule
        - 4/notour/-1/no lDT generated - does not schedule
    """
    trace_label = "ldt_scheduling_households"
    model_settings_file_name = "ldt_scheduling.yaml"

    choosers = households_merged.to_frame()

    # limiting ldt_scheduling_households to patterns whose begin/end times can be scheduled - complete/begin/end
    choosers = choosers[choosers["ldt_pattern_household"].isin([0, 1, 2])]
    logger.info("Running %s with %d hosueholds", trace_label, len(choosers))

    # preliminary estimation steps
    model_settings = config.read_model_settings(model_settings_file_name)
    estimator = estimation.manager.begin_estimation("ldt_scheduling_households")

    constants = config.get_model_constants(model_settings)

    # preprocessor - doesn't import anything
    preprocessor_settings = model_settings.get("preprocessor", None)
    if preprocessor_settings:

        locals_d = {}
        if constants is not None:
            locals_d.update(constants)

        expressions.assign_columns(
            df=choosers,
            model_settings=preprocessor_settings,
            locals_dict=locals_d,
            trace_label=trace_label,
        )

    # reads in the three different specifications needed for scheduling LDTs -
    # complete (specification and coefficient), end, and start (constants)
    spec_categories = model_settings.get("SPEC_CATEGORIES", {})

    # convert households to DF and set default start/end hours to -1
    households = households.to_frame()
    households["ldt_start_hour"] = -1
    households["ldt_end_hour"] = -1

    # matching each possible start-end combination to an index starting from 0
    # predictions will all be index values and can be translated to actual start/end
    # times using this dictionary
    complete_tour_translation = {}
    i = 5
    x = 0
    while i <= 23:
        j = i
        while j <= 23:
            complete_tour_translation[x] = [i, j]
            j += 1
            x += 1
        i += 1
    complete_tour_translation = pd.Series(complete_tour_translation)

    # do prediction for each of the categories encoded on ldt_scheduling yaml
    for category_settings in spec_categories:
        # see function documentation for category_num translation
        category_num = int(category_settings["CATEGORY"])
        # limiting scheduling to the current tour pattern
        subset = choosers[choosers["ldt_pattern_household"] == category_num]
        # name of the current tour pattern being estimated
        category_name = category_settings["NAME"]

        # logic for complete tour pattern scheduling
        if category_num == 0:
            # read in specification for complete tour pattern scheduling
            model_spec = simulate.read_model_spec(file_name=category_settings["SPEC"])
            coefficients_df = simulate.read_model_coefficients(category_settings)
            model_spec = simulate.eval_coefficients(model_spec, coefficients_df, estimator)

            nest_spec = config.get_logit_model_settings(model_settings)

            if estimator:
                estimator.write_model_settings(model_settings, model_settings_file_name)
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
                trace_choice_name="ldt_scheduling_household_" + category_name,
                estimator=estimator,
            )

            if estimator:
                estimator.write_choices(choices)
                choices = estimator.get_survey_values(
                    choices, "households", "ldt_scheduling_household_" + category_name
                )
                estimator.write_override_choices(choices)
                estimator.end_estimation()

            # get the start/end hours associated with the estimated values
            starts_ends = complete_tour_translation.loc[choices.values]
            print(starts_ends)
            # merge start/end hours to the households file
            households.loc[choices.index, "ldt_start_hour"] = starts_ends.apply(lambda x: x[0]).values
            households.loc[choices.index, "ldt_end_hour"] = starts_ends.apply(lambda x: x[1]).values
            # go to next pattern after complete tour estimation is finished
            continue
        constants = config.get_model_constants(category_settings)

        # sampling probabilities for the current tour pattern (start/end)
        df = pd.DataFrame(index=subset.index, columns=["hour_" + str(x) for x in range(24)])
        for i in range(24):
            key = "hour_" + str(i)
            df[key] = constants[key]
        choices, _ = logit.make_choices(df)

        print(choices.shape)

        # merge in scheduled values to the respective tour pattern (start/end)
        if category_num == 1:
            households.loc[choices.index, "ldt_start_hour"] = choices
        else:
            households.loc[choices.index, "ldt_end_hour"] = choices

    # merging into the final_households csv
    pipeline.replace_table("households", households)

    tracing.print_summary(
        "ldt_scheduling_household_start_hours",
        households[households["ldt_start_hour"] != -1]["ldt_start_hour"],
        value_counts=True
    )

    tracing.print_summary(
        "ldt_scheduling_household_end_hours",
        households[households["ldt_end_hour"] != -1]["ldt_end_hour"],
        value_counts=True
    )

    if trace_hh_id:
        tracing.trace_df(households, label=trace_label, warn_if_empty=True)

    # merging into longdist_tours csv
    longdist_tours = pipeline.get_table("longdist_tours")

    # merge start/end hours for households only, -2 for persons
    longdist_tours["ldt_start_hour"] = (
        np.where(longdist_tours["actor_type"] == "household",
                 households.loc[longdist_tours["household_id"], "ldt_start_hour"], -2)
    )
    longdist_tours["ldt_end_hour"] = (
        np.where(longdist_tours["actor_type"] == "household",
                 households.loc[longdist_tours["household_id"], "ldt_end_hour"], -2)
    )

    pipeline.replace_table("longdist_tours", longdist_tours)
