# ActivitySim
# See full license in LICENSE.txt

import logging

from activitysim.core import config, expressions, inject, pipeline, simulate, tracing, logit

from .util import estimation

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


@inject.step()
def ldt_pattern_person(persons, persons_merged, chunk_size, trace_hh_id):
    """
    This model gives each LDT individual one of the possible LDT categories for a given day --
        - complete tour (start and end tour on same day)
        - begin tour
        - end tour
        - away on tour
        - no tour
    """
    trace_label = "ldt_pattern_person"
    model_settings_file_name = "ldt_pattern_person.yaml"

    choosers_full = persons_merged.to_frame()
    # limiting ldt_pattern_person to ldt individuals
    # choosers = choosers[choosers.ldt_tour_gen_persons_WORKRELATED | choosers.ldt_tour_gen_persons_OTHER]
    logger.info("Running %s with %d persons", trace_label, len(choosers_full))

    # necessary to run
    model_settings = config.read_model_settings(model_settings_file_name)
    estimator = estimation.manager.begin_estimation("ldt_pattern_person")

    constants = config.get_model_constants(model_settings)

    # - preprocessor
    preprocessor_settings = model_settings.get("preprocessor", None)
    if preprocessor_settings:
        locals_d = {}
        if constants is not None:
            locals_d.update(constants)

        expressions.assign_columns(
            df=choosers_full,
            model_settings=preprocessor_settings,
            locals_dict=locals_d,
            trace_label=trace_label,
        )

    # model_spec = simulate.read_model_spec(file_name=model_settings["SPEC"])
    # model_spec = simulate.eval_coefficients(model_spec, coefficients_df, estimator)

    spec_purposes = model_settings.get("SPEC_PURPOSES", {})

    # nest_spec = config.get_logit_model_settings(model_settings)

    persons = persons.to_frame()
    persons["ldt_pattern"] = -1

    for purpose_settings in spec_purposes:

        purpose_name = purpose_settings["NAME"]
        choosers = choosers_full[choosers_full["ldt_tour_gen_persons_" + purpose_name]]

        constants = config.get_model_constants(purpose_settings)

        # coefficients_df = simulate.read_model_coefficients(model_settings)

        if estimator:
            estimator.write_model_settings(model_settings, model_settings_file_name)
            estimator.write_spec(model_settings)
            # estimator.write_coefficients(coefficients_df, model_settings)
            estimator.write_choosers(choosers)

        # calculating complementary probability
        notour_prob = 1 - constants["COMPLETE"] - constants["BEGIN"] - constants["END"] - constants["AWAY"]

        # sampling probabilities
        df = pd.DataFrame(index=choosers.index, columns=["complete", "begin", "end", "away", "none"])
        df["complete"], df["begin"], df["end"], df["away"], df["none"] = (
            constants["COMPLETE"], constants["BEGIN"], constants["END"], constants["AWAY"], notour_prob
        )
        print(constants["COMPLETE"])
        print(notour_prob)
        # _ is the random value used to make the monte carlo draws
        choices, _ = logit.make_choices(df)

        if estimator:
            estimator.write_choices(choices)
            choices = estimator.get_survey_values(
                choices, "persons", "ldt_pattern_" + purpose_name
            )
            estimator.write_override_choices(choices)
            estimator.end_estimation()

        colname = "ldt_pattern"
        # making one ldt pattern field instead of segmenting by person/household currently
        persons.loc[choices.index, "ldt_pattern"] = choices

        pipeline.replace_table("persons", persons)

    tracing.print_summary(
        colname,
        persons[colname],
        value_counts=True,
    )
