# ActivitySim
# See full license in LICENSE.txt.
import logging

from activitysim.core import config, expressions, inject, pipeline, simulate, tracing
from ...core.util import reindex_i

from .util import estimation
from .ldt_tour_gen import process_longdist_tours

logger = logging.getLogger(__name__)


@inject.step()
def ldt_tour_gen_household(households, households_merged, chunk_size, trace_hh_id):
    """
    This model predicts whether a household will go on an LDT trip over a 2 week period
    """

    trace_label = "ldt_tour_gen_household"
    model_settings_file_name = "ldt_tour_gen_household.yaml"

    choosers = households_merged.to_frame()
    # if we want to limit choosers, we can do so here
    # choosers = choosers[choosers.workplace_zone_id > -1]
    logger.info("Running %s with %d persons", trace_label, len(choosers))

    model_settings = config.read_model_settings(model_settings_file_name)
    estimator = estimation.manager.begin_estimation("ldt_tour_gen_household")

    constants = config.get_model_constants(model_settings)

    # - preprocessor - adds accessiblity to choosers sample
    preprocessor_settings = model_settings.get("preprocessor", None)
    if preprocessor_settings:
        expressions.assign_columns(
            df=choosers,
            model_settings=preprocessor_settings,
            trace_label=trace_label,
        )

    model_spec = simulate.read_model_spec(file_name=model_settings["SPEC"])
    coefficients_df = simulate.read_model_coefficients(model_settings)
    model_spec = simulate.eval_coefficients(model_spec, coefficients_df, estimator)

    nest_spec = config.get_logit_model_settings(model_settings)

    if estimator:
        estimator.write_model_settings(model_settings, model_settings_file_name)
        estimator.write_spec(model_settings)
        estimator.write_coefficients(coefficients_df, model_settings)
        estimator.write_choosers(choosers)

    choices = simulate.simple_simulate(
        choosers=choosers,
        spec=model_spec,
        nest_spec=nest_spec,
        locals_d=constants,
        chunk_size=chunk_size,
        trace_label=trace_label,
        trace_choice_name="ldt_tour_gen_household",
        estimator=estimator,
    )

    if estimator:
        estimator.write_choices(choices)
        choices = estimator.get_survey_values(
            choices, "households", "ldt_tour_gen_household"
        )
        estimator.write_override_choices(choices)
        estimator.end_estimation()

    households = households.to_frame()
    households["ldt_tour_gen_HOUSEHOLD"] = (
        choices.reindex(households.index).fillna(0).astype(bool)
    )

    pipeline.replace_table("households", households)

    tracing.print_summary(
        "ldt_tour_gen_HOUSEHOLD",
        households.ldt_tour_gen_HOUSEHOLD,
        value_counts=True,
    )

    if trace_hh_id:
        tracing.trace_df(households, label=trace_label, warn_if_empty=True)

    # init log dist trip table
    hh_making_longdist_tours = households[households["ldt_tour_gen_HOUSEHOLD"]]
    tour_counts = (
        hh_making_longdist_tours[["ldt_tour_gen_HOUSEHOLD"]]
        .astype(int)
        .rename(
            columns={"ldt_tour_gen_HOUSEHOLD": "longdist_household"}
        )
    )
    longdist_tours = process_longdist_tours(
        households, tour_counts, "longdist_household"
    )
    pipeline.extend_table("longdist_tours", longdist_tours)
