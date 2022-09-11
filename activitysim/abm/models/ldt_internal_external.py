# ActivitySim
# See full license in LICENSE.txt

from inspect import trace
import logging

from activitysim.core import config, expressions, inject, pipeline, simulate, tracing, logit

from .util import estimation

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


@inject.step()
def ldt_internal_external(persons, persons_merged, households, households_merged, chunk_size, trace_hh_id):
    """
    This model determines if a person on an LDT is going/will go/is at an internal location (within Ohio/0)
    or at an external location (outside of Ohio/1)
    """

    trace_label = "ldt_internal_external"
    colname = "internal_external"
    model_settings_file_name = "ldt_internal_external.yaml"

    # preliminary estimation steps
    model_settings = config.read_model_settings(model_settings_file_name)
    estimator = estimation.manager.begin_estimation("ldt_scheduling_person")
    constants = config.get_model_constants(model_settings)  # constants shared by all

    # converting parameters to dataframes
    hh_full = households_merged.to_frame()
    persons_full = persons_merged.to_frame()
    households = households.to_frame()
    persons = persons.to_frame()

    # setting default value for internal_external choice to -1
    households[colname] = -1
    persons[colname] = -1

    spec_categories = model_settings.get("SPEC_CATEGORIES", {})  # reading in category-specific things
    model_spec = simulate.read_model_spec(file_name=model_settings["SPEC"])  # reading in generic model spec
    nest_spec = config.get_logit_model_settings(model_settings)  # all are MNL

    for category_settings in spec_categories:  # iterate through all the categories
        category_name = category_settings["NAME"]  # HOUSEHOLD, WORKRELATED, OTHER
        full_name = colname + "_" + category_name
        actor_type = category_settings["TYPE"]  # household or person

        if actor_type == "household":
            choosers = hh_full
            logger.info("Running %s with %d households", full_name, len(choosers))
        else:
            choosers = persons_full
            logger.info("Running %s with %d persons", full_name, len(choosers))

        choosers = choosers[choosers["on_ldt"]]

        # preprocessor - doesn't add anything
        preprocessor_settings = category_settings.get("preprocessor", None)
        if preprocessor_settings:
            locals_d = {}
            if constants is not None:
                locals_d.update(constants)

            expressions.assign_columns(
                df=choosers,
                model_settings=preprocessor_settings,
                locals_dict=locals_d,
                trace_label=full_name,
            )

        # reading in specific category coefficients & evaluate them
        coefficients_df = simulate.read_model_coefficients(category_settings)
        category_spec = simulate.eval_coefficients(model_spec, coefficients_df, estimator)

        if estimator:
            estimator.write_model_settings(category_settings, model_settings_file_name)
            estimator.write_spec(category_settings)
            estimator.write_coefficients(coefficients_df, category_settings)
            estimator.write_choosers(choosers)

        # run the multinomial logit models for the current category
        choices = simulate.simple_simulate(
            choosers=choosers,
            spec=category_spec,
            nest_spec=nest_spec,
            locals_d=constants,
            chunk_size=chunk_size,
            trace_label=trace_label,
            trace_choice_name=full_name,
            estimator=estimator,
        )

        if estimator:
            estimator.write_choices(choices)
            choices = estimator.get_survey_values(
                choices, actor_type, full_name
            )
            estimator.write_override_choices(choices)
            estimator.end_estimation()

        # merge in results into relevant df
        if actor_type == "household":
            households.loc[choices.index, colname] = choices.values
        else:
            persons.loc[choices.index, colname] = choices.values

    tracing.print_summary(
        "ldt_internal_external_households",
        households.internal_external,
        value_counts=True
    )

    tracing.print_summary(
        "ldt_internal_external_persons",
        persons.internal_external,
        value_counts=True
    )

    # merging into final csvs
    pipeline.replace_table("households", households)
    pipeline.replace_table("persons", persons)

    # merging into longdist csv
    longdist_tours = pipeline.get_table("longdist_tours")

    longdist_tours[colname] = (
        np.where(longdist_tours["actor_type"] == "household",
                 households.loc[longdist_tours["household_id"], colname], -2)
    )

    # have temp persons_temp df to prevent np.where from throwing an error
    # since all people have a person_id, this will never be used
    persons_temp = persons
    persons_temp.loc[-1, colname] = -2

    longdist_tours[colname] = (
        np.where(longdist_tours["actor_type"] == "person",
                 persons_temp.loc[longdist_tours["person_id"], colname], longdist_tours[colname])
    )

    print(longdist_tours)

    pipeline.replace_table("longdist_tours", longdist_tours)
