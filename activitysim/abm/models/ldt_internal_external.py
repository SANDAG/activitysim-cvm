# ActivitySim
# See full license in LICENSE.txt

import logging

import numpy as np
import pandas as pd

from activitysim.core import config, expressions, inject, pipeline, simulate, tracing
from activitysim.core.util import assign_in_place, reindex

from .util import estimation

logger = logging.getLogger(__name__)

LDT_IE_INTERNAL = 0
LDT_IE_EXTERNAL = 1
LDT_IE_NULL = -1

@inject.step()
def ldt_internal_external(
    longdist_tours, persons_merged, chunk_size, trace_hh_id
):
    """
    This model determines if a person on an LDT is going/will go/is at an internal location (within Ohio/0)
    or at an external location (outside of Ohio/1)
    """
    trace_label = "ldt_internal_external"
    colname = "internal_external"
    model_settings_file_name = "ldt_internal_external.yaml"
    segment_column_name = "tour_type"

    # preliminary estimation steps
    model_settings = config.read_model_settings(model_settings_file_name)
    estimator = estimation.manager.begin_estimation(trace_label)
    constants = config.get_model_constants(model_settings)  # constants shared by all

    # merging in global constants
    category_file_name = model_settings.get("CATEGORY_CONSTANTS", None)
    if category_file_name is not None:
        categories = config.get_model_constants(
            config.read_model_settings(category_file_name)
        )
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
