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
def ldt_scheduling(longdist_tours, persons_merged, chunk_size, trace_hh_id):
    trace_label = "ldt_scheduling"
    model_settings_file_name = "ldt_scheduling.yaml"
    segment_column_name = "tour_type"

    # preliminary estimation steps
    model_settings = config.read_model_settings(model_settings_file_name)
    estimator = estimation.manager.begin_estimation(trace_label)
    constants = config.get_model_constants(model_settings)  # constants shared by all

    # merging in global constants
    category_file_name = model_settings.get("CATEGORY_CONSTANTS", None)
    if category_file_name is not None:
        categories = config.read_settings_file(category_file_name)
        constants.update(categories)

    # converting parameters to dataframes
    ldt_tours = longdist_tours.to_frame()
    logger.info("Running %s with %d tours" % (trace_label, ldt_tours.shape[0]))
    
    persons_merged = persons_merged.to_frame()
    ldt_tours_merged = pd.merge(
        ldt_tours,
        persons_merged,
        left_on="person_id",
        right_index=True,
        how="left",
        suffixes=("", "_r"),
    )
    
    model_spec = simulate.read_model_spec(
        file_name=model_settings["SPEC"]
    )  # reading in generic model spec
    nest_spec = config.get_logit_model_settings(model_settings)  # MNL
    
    choices_list = []
    for tour_purpose, tours_segment in ldt_tours_merged.groupby(segment_column_name):
        if tour_purpose.startswith("longdist_"):
            tour_purpose = tour_purpose[9:]
        tour_purpose = tour_purpose.lower()

        choosers = tours_segment
        
        logger.info(
            "ldt_internal_external tour_type '%s' (%s tours)"
            % (
                tour_purpose,
                len(choosers.index),
            )
        )
        
        if choosers.empty:
            choices_list.append(
                pd.Series(-1, index=tours_segment.index, name=colname).to_frame()
            )
            continue
        
        coefficients_df = simulate.get_segment_coefficients(model_settings, tour_purpose)
        category_spec = simulate.eval_coefficients(
            model_spec, coefficients_df, estimator
        )

        if estimator:
            estimator.write_model_settings(model_settings, model_settings_file_name)
            estimator.write_spec(model_settings)
            estimator.write_coefficients(coefficients_df, model_settings)
            estimator.write_choosers(choosers)
        
        choices = simulate.simple_simulate(
            choosers=choosers,
            spec=category_spec,
            nest_spec=nest_spec,
            locals_d=constants,
            chunk_size=chunk_size,
            trace_label=tracing.extend_trace_label(trace_label, tour_purpose),
            trace_choice_name=colname,
            estimator=estimator,
        )
        
        if isinstance(choices, pd.Series):
            choices = choices.to_frame(colname)
        
        choices = choices.reindex(tours_segment.index).fillna(
            {colname: LDT_IE_NULL}, downcast="infer"
        )
        
        tracing.print_summary(
            "ldt_internal_external %s choices" % tour_purpose,
            choices[colname],
            value_counts=True,
        )
        
        choices_list.append(choices)
        
    choices_df = pd.concat(choices_list)
    
    tracing.print_summary(
        "ldt_internal_external all tour type choices",
        choices_df[colname],
        value_counts=True,
    )
    
    assign_in_place(ldt_tours, choices_df)
    
    pipeline.replace_table("longdist_tours", ldt_tours)
    
    if trace_hh_id:
        tracing.trace_df(
            ldt_tours,
            label=trace_label,
            slicer="tour_id",
            index_label="tour_id",
            warn_if_empty=True,
        )

@inject.step()
def ldt_scheduling_person(persons, persons_merged, chunk_size, trace_hh_id):
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

    pipeline.replace_table("longdist_tours", longdist_tours)
    
    if trace_hh_id:
        tracing.trace_df(persons, label=trace_label)
