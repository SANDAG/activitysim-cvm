# ActivitySim
# See full license in LICENSE.txt

import logging

import numpy as np
import pandas as pd

from activitysim.core import (
    config,
    expressions,
    inject,
    logit,
    pipeline,
    simulate,
    tracing,
)

from .ldt_pattern import LDT_PATTERN
from .util import estimation

logger = logging.getLogger(__name__)


@inject.step()
def ldt_scheduling_person(
    persons, persons_merged, chunk_size, trace_hh_id, network_los
):
    """
    This model schedules the start/end times for persons that were assigned
    to be beginning and/or ending a tour on a given day
        - 0/complete: schedules both start/end of tour
        - 1/begin: schedules beginning of tour
        - 2/end: schedules end of tour
        - 3/away: does not schedule
        - 4/notour/-1/no lDT generated - does not schedule
    """
    trace_label = "ldt_scheduling_person"
    model_settings_file_name = "ldt_scheduling.yaml"

    choosers = persons_merged.to_frame()

    # limiting ldt_scheduling_persons to applicable patterns - complete, begin, or end
    choosers = choosers[(choosers["ldt_pattern_person"] & LDT_PATTERN.COMPLETE) > 0]
    logger.info("Running %s with %d persons", trace_label, len(choosers))

    # preliminary estimation steps
    model_settings = config.read_model_settings(model_settings_file_name)
    estimator = estimation.manager.begin_estimation("ldt_scheduling_person")

    constants = config.get_model_constants(model_settings)

    # preprocessor - adds nothing
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

    # convert persons file to DF and set default start/end hours to -1
    persons = persons.to_frame()
    persons["ldt_start_hour"] = -1
    persons["ldt_end_hour"] = -1

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

    patterns = {
        "complete": LDT_PATTERN.COMPLETE,
        "begin": LDT_PATTERN.BEGIN,
        "end": LDT_PATTERN.END,
    }

    # do prediction for each of the categories encoded on ldt_scheduling yaml
    for category_settings in spec_categories:
        # see function documentation for category_num translation
        category_num = patterns.get(
            category_settings["NAME"].lower().replace("_tour", "")
        )
        # limiting scheduling to the current tour pattern
        subset = choosers[
            (choosers["ldt_pattern_person"] & LDT_PATTERN.COMPLETE) == category_num
        ]
        # name of the current tour pattern beign estimated
        category_name = category_settings["NAME"]

        # logic for complete tour pattern scheduling
        if category_num == LDT_PATTERN.COMPLETE:
            # read in specification for complete tour pattern scheduling
            model_spec = simulate.read_model_spec(file_name=category_settings["SPEC"])
            coefficients_df = simulate.read_model_coefficients(category_settings)
            model_spec = simulate.eval_coefficients(
                model_spec, coefficients_df, estimator
            )

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
                trace_choice_name="ldt_scheduling_person_" + category_name,
                estimator=estimator,
            )

            if estimator:
                estimator.write_choices(choices)
                choices = estimator.get_survey_values(
                    choices, "persons", "ldt_scheduling_person_" + category_name
                )
                estimator.write_override_choices(choices)
                estimator.end_estimation()

            # get the start/end hours associated with the estimated values
            starts_ends = complete_tour_translation.loc[choices.values]
            # merge start/end hours to the persons file
            persons.loc[subset.index, "ldt_start_hour"] = starts_ends.apply(
                lambda x: x[0]
            ).values
            persons.loc[subset.index, "ldt_end_hour"] = starts_ends.apply(
                lambda x: x[1]
            ).values

        else:

            constants = config.get_model_constants(category_settings)

            # sampling probabilities for the current tour pattern (start/end)
            df = pd.DataFrame(
                index=choosers.index, columns=["hour_" + str(x) for x in range(24)]
            )
            for i in range(24):
                key = "hour_" + str(i)
                df[key] = constants[key]
            choices, _ = logit.make_choices(df, trace_choosers=trace_hh_id)

            # merge in scheduled values to the respective tour pattern (start/end)
            if category_num == LDT_PATTERN.BEGIN:
                persons.loc[subset.index, "ldt_start_hour"] = choices
            elif category_num == LDT_PATTERN.END:
                persons.loc[subset.index, "ldt_end_hour"] = choices
            else:
                raise ValueError(f"BUG, bad category_num {category_num}")

    persons["ldt_start_period"] = network_los.skim_time_period_label(
        persons["ldt_start_hour"],
        fillna=0,
    )
    persons["ldt_end_period"] = network_los.skim_time_period_label(
        persons["ldt_end_hour"],
        fillna=0,
    )

    # merging into persons
    pipeline.replace_table("persons", persons)

    tracing.print_summary(
        "ldt_scheduling_person_start_hours",
        persons[persons["ldt_start_hour"] != -1]["ldt_start_hour"],
        value_counts=True,
    )

    tracing.print_summary(
        "ldt_scheduling_person_end_hours",
        persons[persons["ldt_end_hour"] != -1]["ldt_end_hour"],
        value_counts=True,
    )

    if trace_hh_id:
        tracing.trace_df(persons, label=trace_label, warn_if_empty=True)

    # merging into longdist_tours
    longdist_tours = pipeline.get_table("longdist_tours")

    def fill_in(x, colname):
        if x == -1:
            return -1
        return persons.loc[x, colname]

    longdist_tours["ldt_start_hour"] = np.where(
        longdist_tours["actor_type"] == "person",
        longdist_tours["person_id"].apply(lambda x: fill_in(x, "ldt_start_hour")),
        longdist_tours["ldt_start_hour"],
    )
    longdist_tours["ldt_end_hour"] = np.where(
        longdist_tours["actor_type"] == "person",
        longdist_tours["person_id"].apply(lambda x: fill_in(x, "ldt_end_hour")),
        longdist_tours["ldt_end_hour"],
    )

    longdist_tours["ldt_start_period"] = network_los.skim_time_period_label(
        longdist_tours["ldt_start_hour"],
        fillna=0,
    )
    longdist_tours["ldt_end_period"] = network_los.skim_time_period_label(
        longdist_tours["ldt_end_hour"],
        fillna=0,
    )

    pipeline.replace_table("longdist_tours", longdist_tours)

    if trace_hh_id:
        tracing.trace_df(persons, label=trace_label)
