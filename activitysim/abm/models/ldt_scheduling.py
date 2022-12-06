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
from activitysim.core.util import assign_in_place, reindex

from .ldt_pattern import LDT_PATTERN
from .util import estimation

logger = logging.getLogger(__name__)

@inject.step()
def ldt_scheduling(longdist_tours, persons_merged, chunk_size, trace_hh_id):
    """
    This model schedules the start/end times for actors that were assigned
    to be beginning and/or ending a tour on a given day (only those in
    the longdist_tours csv)
        - 0/complete: schedules both start/end of tour
        - 1/begin: schedules beginning of tour
        - 2/end: schedules end of tour
        - 3/away: does not schedule
        - 4/notour/-1/no lDT generated - does not schedule
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

    spec_categories = model_settings.get("SPEC_CATEGORIES", {})
    nest_spec = config.get_logit_model_settings(model_settings)  # MNL
    
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
    
    patterns = {
        "complete": LDT_PATTERN.COMPLETE,
        "begin": LDT_PATTERN.BEGIN,
        "end": LDT_PATTERN.END,
    }

    starts_list = []
    ends_list = []
    for actor_type, tours_segment in ldt_tours_merged.groupby(actor_column_name):
        choosers = tours_segment
        
        logger.info(
            "ldt_scheduling actor_type '%s' (%s tours)"
            % (
                actor_type,
                len(choosers.index),
            )
        )
        
        if choosers.empty:
            starts_list.append(
                pd.Series(-1, index=tours_segment.index, name=start_colname).to_frame()
            )
            ends_list.append(
                pd.Series(-1, index=tours_segment.index, name=end_colname).to_frame()
            )
            continue
        
        for category_settings in spec_categories:
            # see function documentation for category_num translation
            category_num = patterns.get(
                category_settings["NAME"].lower().replace("_tour", "")
            )
            # limiting scheduling to the current tour pattern
            subset = choosers[
                (choosers["ldt_pattern"] & LDT_PATTERN.COMPLETE) == category_num
            ]
            if(len(subset) == 0):
                continue
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
                    trace_choice_name="ldt_scheduling_" + actor_type + "_" + category_name,
                    estimator=estimator,
                )

                if estimator:
                    estimator.write_choices(choices)
                    choices = estimator.get_survey_values(
                        choices, "persons", "ldt_scheduling_" + actor_type + "_" + category_name
                    )
                    estimator.write_override_choices(choices)
                    estimator.end_estimation()
                    
                # get the start/end hours associated with the estimated values
                starts_ends = complete_tour_translation.loc[choices.values]
                
                start_choices = pd.DataFrame(starts_ends.apply(lambda x: x[0]).values, index=subset.index, columns=[start_colname])
                end_choices = pd.DataFrame(starts_ends.apply(lambda x: x[1]).values, index=subset.index, columns=[end_colname])
                
                starts_list.append(start_choices)
                ends_list.append(end_choices)                

            else:
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
    if len(starts_list) != 0:   
        starts_df = pd.concat(starts_list).reindex(ldt_tours_merged.index).fillna(
            {start_colname: -1}, downcast="infer"
        )
        assign_in_place(ldt_tours, starts_df)
    
    if len(ends_list) != 0:
        ends_df = pd.concat(ends_list).reindex(ldt_tours_merged.index).fillna(
            {end_colname: -1}, downcast="infer"
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
    
    pipeline.replace_table("longdist_tours", ldt_tours)
    
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
    
    # TODO: fix logic for trip start hour for end/complete trips
    trips = pipeline.get_table("longdist_trips")
    trips["ldt_start_hour"] = starts_df.loc[trips.longdist_tour_id].values
    trips["ldt_start_hour"] = np.where(trips["purpose"] == "travel_out", -1, trips["ldt_start_hour"])
    pipeline.replace_table("longdist_trips", trips)
 
    if trace_hh_id:
        tracing.trace_df(
            ldt_tours,
            label=trace_label,
            slicer="tour_id",
            index_label="tour_id",
            warn_if_empty=True,
        )