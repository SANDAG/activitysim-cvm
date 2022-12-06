# ActivitySim
# See full license in LICENSE.txt

import logging

import numpy as np
from activitysim.abm.models.ldt_internal_external import LDT_IE_EXTERNAL

from activitysim.core import config, expressions, inject, pipeline, simulate, tracing, los
from activitysim.core.util import assign_in_place, reindex

import pandas as pd

from .util import estimation

logger = logging.getLogger(__name__)


@inject.step()
def ldt_external_mode_choice(
    longdist_tours, persons_merged, network_los, chunk_size, trace_hh_id
):
    """
    This model determines if a person on an LDT is going/will go/is at an internal location (within Ohio/0)
    or at an external location (outside of Ohio/1)
    """

    trace_label = "ldt_external_mode_choice"
    colname = "external_tour_mode"
    model_settings_file_name = "ldt_external_mode_choice.yaml"
    segment_column_name = "tour_type"

    # preliminary estimation steps
    model_settings = config.read_model_settings(model_settings_file_name)
    estimator = estimation.manager.begin_estimation(trace_label)
    
    constants = {}
    constants.update(config.get_model_constants(model_settings))  # constants shared by all
    
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
    nest_spec = config.get_logit_model_settings(model_settings)  # NL

    # setup skims and skim keys
    orig_col = "home_zone_id"
    dest_col = "external_destchoice"

    skim_dict = network_los.get_default_skim_dict()

    odt_skim_stack_wrapper = skim_dict.wrap_3d(
        orig_key=orig_col, dest_key=dest_col, dim3_key="ldt_start_hour"
    )
    dot_skim_stack_wrapper = skim_dict.wrap_3d(
        orig_key=dest_col, dest_key=orig_col, dim3_key="ldt_start_hour"
    )
    od_skim_wrapper = skim_dict.wrap(orig_col, dest_col)
    
    skims = {
        "odt_skims": odt_skim_stack_wrapper,
        "dot_skims": dot_skim_stack_wrapper,
        "od_skims": od_skim_wrapper,
    }

    if network_los.zone_system == los.THREE_ZONE:
        # fixme - is this a lightweight object?

        tvpb = network_los.tvpb

        tvpb_logsum_odt = tvpb.wrap_logsum(
            orig_key=orig_col,
            dest_key=dest_col,
            tod_key="out_period",
            segment_key="demographic_segment",
            cache_choices=True,
            trace_label=trace_label,
            tag="tvpb_logsum_odt",
        )
        tvpb_logsum_dot = tvpb.wrap_logsum(
            orig_key=dest_col,
            dest_key=orig_col,
            tod_key="in_period",
            segment_key="demographic_segment",
            cache_choices=True,
            trace_label=trace_label,
            tag="tvpb_logsum_dot",
        )

        skims.update(
            {"tvpb_logsum_odt": tvpb_logsum_odt, "tvpb_logsum_dot": tvpb_logsum_dot}
        )

        # TVPB constants can appear in expressions
        if model_settings.get("use_TVPB_constants", True):
            constants.update(
                network_los.setting("TVPB_SETTINGS.tour_mode_choice.CONSTANTS")
            )
            
    # add skim keys to constants dictionary so that they can be accessed in 
    # specification csv expressions
    constants.update(skims)
    
    choices_list = []
    for tour_purpose, tours_segment in ldt_tours_merged.groupby(segment_column_name):
        if tour_purpose.startswith("longdist_"):
            tour_purpose = tour_purpose[9:]
        tour_purpose = tour_purpose.lower()
        
        if network_los.zone_system == los.THREE_ZONE:
            tvpb_logsum_odt.extend_trace_label(tour_purpose)
            tvpb_logsum_dot.extend_trace_label(tour_purpose)
            
        choosers = tours_segment[tours_segment.internal_external == LDT_IE_EXTERNAL]
        
        logger.info(
            "ldt_external_tour_mode_choice tour_type '%s' (%s tours)"
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
        nest_category_spec = simulate.eval_nest_coefficients(nest_spec, coefficients_df, estimator)
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
            nest_spec=nest_category_spec,
            locals_d=constants,
            chunk_size=chunk_size,
            skims=skims,
            trace_label=tracing.extend_trace_label(trace_label, tour_purpose),
            trace_choice_name=colname,
            estimator=estimator
        )
        
        alts = category_spec.columns
        choices = choices.map(
            dict(list(zip(list(range(len(alts))), alts)))
        )
        
        if isinstance(choices, pd.Series):
            choices = choices.to_frame(colname)
        
        choices = choices.reindex(tours_segment.index).fillna(
            {colname: -1}, downcast="infer"
        )
        
        tracing.print_summary(
            "ldt_external_tour_mode %s choices" % tour_purpose,
            choices[colname],
            value_counts=True,
        )
        
        choices_list.append(choices)
        
    choices_df = pd.concat(choices_list)
    
    tracing.print_summary(
        "ldt_external_mode_choice all tour type choices",
        choices_df[colname],
        value_counts=True,
    )
    
    assign_in_place(ldt_tours, choices_df)
    
    pipeline.replace_table("longdist_tours", ldt_tours)
    
    trips = pipeline.get_table("longdist_trips")
    trips["mode"] = choices_df.loc[trips.longdist_tour_id.values].values
    pipeline.replace_table("longdist_trips", trips)
    
    if trace_hh_id:
        tracing.trace_df(
            ldt_tours,
            label=trace_label,
            slicer="tour_id",
            index_label="tour_id",
            warn_if_empty=True,
        )

