# ActivitySim
# See full license in LICENSE.txt

import logging

import numpy as np
import pandas as pd

from ...core import config, inject, logit, pipeline, tracing
from .util import estimation
from .ldt_internal_external import LDT_IE_EXTERNAL

from activitysim.core.util import assign_in_place, reindex

logger = logging.getLogger(__name__)

@inject.step()
def ldt_external_destchoice(
    longdist_tours, persons_merged, chunk_size, trace_hh_id
):
    """
    This model determines the destination of those traveling externally based on a probability distribution.
    """
    trace_label = "ldt_external_destchoice"
    colname = "external_destchoice"
    model_settings_file_name = "ldt_external_destchoice.yaml"
    segment_column_name = "tour_type"
    
    # preliminary estimation steps
    model_settings = config.read_model_settings(model_settings_file_name)
    estimator = estimation.manager.begin_estimation("ldt_external_destchoice")
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
    
    external_probabilities_file_path = config.config_file_path(model_settings.get("REGION_PROBABILITIES"))
    external_probabilities = pd.read_csv(external_probabilities_file_path, index_col=0)
    
    region_categories = model_settings.get(
        "REGION_CATEGORIES", {}
    )  # reading in category-specific things
    
    choices_list = []
    for tour_purpose, tours_segment in ldt_tours_merged.groupby(segment_column_name):
        if tour_purpose.startswith("longdist_"):
            tour_purpose = tour_purpose[9:]
        tour_purpose = tour_purpose.lower()
        
        choosers = tours_segment[tours_segment.internal_external == LDT_IE_EXTERNAL]

        if choosers.empty:
            choices_list.append(
                pd.Series(-1, index=tours_segment.index, name=colname).to_frame()
            )
            continue
        
        region_choices_list = []
        for region_category in region_categories:
            region = region_category["NAME"]

            region_choosers = choosers[choosers["LDTdistrict"] == region]

            logger.info(
                "ldt_external_destchoice tour_type '%s' region '%s' (%s tours)"
                % (
                    tour_purpose,
                    region,
                    len(region_choosers.index),
                )
            )
            
            if region_choosers.empty:
                choices_list.append(
                    pd.Series(-1, index=region_choosers.index, name=colname).to_frame()
                )
                continue

            if estimator:
                estimator.write_model_settings(model_settings, model_settings_file_name)
                estimator.write_spec(model_settings)
                # estimator.write_coefficients(coefficients_df, model_settings)
                estimator.write_choosers(choosers)

            prob_list = np.zeros(len(external_probabilities))

            for i, taz in enumerate(external_probabilities.index):
                prob_list[i] = external_probabilities.loc[taz][region]
            # prob_list[-1] = 1 - np.sum(prob_list[:-1])
            
            pr = np.broadcast_to(prob_list, (len(region_choosers.index), len(external_probabilities)))
            df = pd.DataFrame(pr, index=region_choosers.index, columns=external_probabilities.index)

            choices, _ = logit.make_choices(df, trace_choosers=trace_hh_id)
            df = df.reset_index()

            if estimator:
                estimator.write_choices(choices)
                choices = estimator.get_survey_values(choices, "persons", colname)
                estimator.write_override_choices(choices)
                estimator.end_estimation()

            destinations = pd.DataFrame(
                data=pd.Series(data=external_probabilities.index)[choices].values,
                index=region_choosers.index,
                columns=[colname]
            )

            destinations = destinations.reindex(region_choosers.index)
            
            region_choices_list.append(destinations)
        
        region_choices = pd.concat(region_choices_list)
        region_choices = region_choices.reindex(tours_segment.index).fillna(
            {colname: -1}, downcast="infer"
        )
        choices_list.append(region_choices)
            
    choices_df = pd.concat(choices_list)

    tracing.print_summary(
        "ldt_external_destchoice of all tour types",
        choices_df[choices_df[colname] != -1][colname],
        describe=True
    )

    assign_in_place(ldt_tours, choices_df)

    pipeline.replace_table("longdist_tours", ldt_tours)
    
    trips = pipeline.get_table("longdist_trips")
    trips["destination"] = np.where((trips["purpose"] == "travel_out") & (trips["internal_external"] == "EXTERNAL"), choices_df.loc[trips.longdist_tour_id].iloc[:, 0], trips["destination"])
    trips["origin"] = np.where((trips["purpose"] == "travel_home") & (trips["internal_external"] == "EXTERNAL"), choices_df.loc[trips.longdist_tour_id].iloc[:, 0], trips["origin"])
    pipeline.replace_table("longdist_trips", trips)
    
    if trace_hh_id:
        tracing.trace_df(
            ldt_tours,
            label=trace_label,
            slicer="tour_id",
            index_label="tour_id",
            warn_if_empty=True,
        )
