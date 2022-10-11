# ActivitySim
# See full license in LICENSE.txt

import logging

import numpy as np
import pandas as pd

from ...core import config, inject, logit, pipeline, tracing
from .util import estimation
from .ldt_internal_external import LDT_IE_EXTERNAL

logger = logging.getLogger(__name__)


def ldt_external_destchoice_households(households, households_merged, trace_hh_id):
    trace_label = "ldt_external_destchoice_household"
    colname = "external_destchoice"
    model_settings_file_name = "ldt_external_destchoice.yaml"

    # preliminary estimation steps
    model_settings = config.read_model_settings(model_settings_file_name)
    estimator = estimation.manager.begin_estimation("ldt_external_destchoice_household")
    constants = config.get_model_constants(model_settings)  # constants shared by all

    households = households.to_frame()
    choosers = households_merged.to_frame()

    choosers = choosers[choosers["on_hh_ldt"]]
    choosers = choosers[choosers["internal_external"] == LDT_IE_EXTERNAL]

    choosers[colname] = -1
    households[colname] = -1

    spec_categories = model_settings.get(
        "SPEC_CATEGORIES", {}
    )  # reading in category-specific things
    
    external_probabilities_file_path = config.config_file_path(model_settings.get("REGION_PROBABILITIES"))
    external_probabilities = pd.read_csv(external_probabilities_file_path, index_col=0)

    for category_settings in spec_categories:
        region = category_settings["NAME"]  # Central, NE, NW, SE, SW
        print(f"Estimating LDT region {region}")

        region_choosers = choosers[choosers["LDTdistrict"] == region]

        if estimator:
            estimator.write_model_settings(model_settings, model_settings_file_name)
            estimator.write_spec(model_settings)
            # estimator.write_coefficients(coefficients_df, model_settings)
            estimator.write_choosers(choosers)

        constants = config.get_model_constants(category_settings)
        
        prob_list = np.zeros(len(external_probabilities))

        for i, taz in enumerate(external_probabilities.index):
            prob_list[i] = external_probabilities.loc[taz][region]
        # prob_list[-1] = 1 - np.sum(prob_list[:-1])
        
        pr = np.broadcast_to(prob_list, (len(region_choosers.index), len(external_probabilities)))
        df = pd.DataFrame(pr, index=region_choosers.index, columns=external_probabilities.index)

        choices, _ = logit.make_choices(df)
        df = df.reset_index()

        if estimator:
            estimator.write_choices(choices)
            choices = estimator.get_survey_values(choices, "households", colname)
            estimator.write_override_choices(choices)
            estimator.end_estimation()

        households.loc[choices.index, colname] = (
            pd.Series(data=external_probabilities.index)[choices].values
        )
        choosers.loc[choices.index, colname] = (
            pd.Series(data=external_probabilities.index)[choices].values
        )

    pipeline.replace_table("households", households)

    tracing.print_summary(
        trace_label,
        choosers[colname],
        value_counts=True,
    )


def ldt_external_destchoice_persons(persons, persons_merged, trace_hh_id):
    trace_label = "ldt_external_destchoice_persons"
    colname = "external_destchoice"
    model_settings_file_name = "ldt_external_destchoice.yaml"

    # preliminary estimation steps
    model_settings = config.read_model_settings(model_settings_file_name)
    estimator = estimation.manager.begin_estimation("ldt_external_destchoice_persons")
    constants = config.get_model_constants(model_settings)  # constants shared by all

    persons = persons.to_frame()
    choosers = persons_merged.to_frame()

    choosers = choosers[choosers["on_person_ldt"]]
    choosers = choosers[choosers["internal_external"] == LDT_IE_EXTERNAL]

    choosers[colname] = -1
    persons[colname] = -1

    spec_categories = model_settings.get(
        "SPEC_CATEGORIES", {}
    )  # reading in category-specific things
    
    external_probabilities_file_path = config.config_file_path(model_settings.get("REGION_PROBABILITIES"))
    external_probabilities = pd.read_csv(external_probabilities_file_path, index_col=0)

    for category_settings in spec_categories:
        region = category_settings["NAME"]
        print(f"Estimating LDT region {region}")

        region_choosers = choosers[choosers["LDTdistrict"] == region]

        if estimator:
            estimator.write_model_settings(model_settings, model_settings_file_name)
            estimator.write_spec(model_settings)
            # estimator.write_coefficients(coefficients_df, model_settings)
            estimator.write_choosers(choosers)

        constants = config.get_model_constants(category_settings)

        prob_list = np.zeros(len(external_probabilities))

        for i, taz in enumerate(external_probabilities.index):
            prob_list[i] = external_probabilities.loc[taz][region]
        # prob_list[-1] = 1 - np.sum(prob_list[:-1])
        
        pr = np.broadcast_to(prob_list, (len(region_choosers.index), len(external_probabilities)))
        df = pd.DataFrame(pr, index=region_choosers.index, columns=external_probabilities.index)

        choices, _ = logit.make_choices(df)
        df = df.reset_index()

        if estimator:
            estimator.write_choices(choices)
            choices = estimator.get_survey_values(choices, "persons", colname)
            estimator.write_override_choices(choices)
            estimator.end_estimation()

        persons.loc[choices.index, colname] = (
            pd.Series(data=external_probabilities.index)[choices].values
        )
        choosers.loc[choices.index, colname] = (
            pd.Series(data=external_probabilities.index)[choices].values
        )

    pipeline.replace_table("persons", persons)

    tracing.print_summary(
        trace_label,
        choosers[colname],
        value_counts=True,
    )


@inject.step()
def ldt_external_destchoice(
    households, households_merged, persons, persons_merged, chunk_size, trace_hh_id
):
    """
    This model determines the destination of those traveling externally based on a probability distribution.
    """
    ldt_external_destchoice_households(households, households_merged, trace_hh_id)
    ldt_external_destchoice_persons(persons, persons_merged, trace_hh_id)

    persons = pipeline.get_table("persons")
    households = pipeline.get_table("households")
    longdist_tours = pipeline.get_table("longdist_tours")

    def fill_in(x):
        if x == -1:
            return -1
        return persons.loc[x, "external_destchoice"]

    longdist_tours["external_destchoice"] = -1
    longdist_tours["external_destchoice"] = np.where(
        longdist_tours["actor_type"] == "person",
        longdist_tours["person_id"].apply(fill_in),
        longdist_tours["external_destchoice"],
    )
    longdist_tours["external_destchoice"] = np.where(
        longdist_tours["actor_type"] == "household",
        households.loc[longdist_tours["household_id"], "external_destchoice"],
        longdist_tours["external_destchoice"],
    )

    pipeline.replace_table("longdist_tours", longdist_tours)
