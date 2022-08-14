# ActivitySim
# See full license in LICENSE.txt

import logging

from activitysim.core import config, expressions, inject, pipeline, simulate, tracing, logit

from .util import estimation

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


@inject.step()
def ldt_pattern_household(households, households_merged, chunk_size, trace_hh_id):
    """
    This model gives each LDT household one of the possible LDT categories for a given day --
        - complete tour (start and end tour on same day)
        - begin tour
        - end tour
        - away on tour
        - no tour
    """
    trace_label = "ldt_pattern_household"
    model_settings_file_name = "ldt_pattern_household.yaml"

    choosers = households_merged.to_frame()
    # if we want to limit choosers, we can do so here
    # limiting ldt_pattern_household to households that go on LDTs
    choosers = choosers[choosers.ldt_tour_gen_household]
    logger.info("Running %s with %d persons", trace_label, len(choosers))

    # necessary to run
    model_settings = config.read_model_settings(model_settings_file_name)
    estimator = estimation.manager.begin_estimation("ldt_pattern_household")

    constants = config.get_model_constants(model_settings)

    # - preprocessor
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

    # model_spec = simulate.read_model_spec(file_name=model_settings["SPEC"])
    # coefficients_df = simulate.read_model_coefficients(model_settings)
    # model_spec = simulate.eval_coefficients(model_spec, coefficients_df, estimator)

    # nest_spec = config.get_logit_model_settings(model_settings)

    # base estimator
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
    # _ is the random value used to make the monte carlo draws
    choices, _ = logit.make_choices(df)

    # overwriting estimator
    if estimator:
        estimator.write_choices(choices)
        choices = estimator.get_survey_values(
            choices, "households", "ldt_pattern_household"
        )
        estimator.write_override_choices(choices)
        estimator.end_estimation()

    # setting -1 to non-LDT households
    households = households.to_frame()
    households["ldt_pattern_household"] = (
        choices.reindex(households.index).fillna(-1)
    )

    # adding some convenient fields
    households["on_ldt"] = np.where(households["ldt_pattern_household"].isin([-1, 4]), False, True)

    # merging into households
    pipeline.replace_table("households", households)

    tracing.print_summary(
        "ldt_pattern_household",
        households.ldt_pattern_household,
        value_counts=True
    )

    if trace_hh_id:
        tracing.trace_df(households, label=trace_label, warn_if_empty=True)

    # init log dist trip table
    hh_making_longdist_tours_patterns = (
        households[households["ldt_tour_gen_household"]]["ldt_pattern_household"]
        .astype(int)
    )
    longdist_tours = pipeline.get_table("longdist_tours")
    longdist_tours = (
        pd.merge(longdist_tours, hh_making_longdist_tours_patterns,
                 how="left", left_on="household_id", right_index=True)
    )
    pipeline.replace_table("longdist_tours", longdist_tours)
