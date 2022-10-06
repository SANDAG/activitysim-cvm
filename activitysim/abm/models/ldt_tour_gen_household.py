# ActivitySim
# See full license in LICENSE.txt.
import logging

from activitysim.core import config, expressions, inject, pipeline, simulate, tracing

from .util import estimation

logger = logging.getLogger(__name__)


@inject.step()
def ldt_tour_gen_household(households, households_merged, chunk_size, trace_hh_id):
    """
    This model predicts whether a household will go on an LDT trip over a 2 week period.

    - *Configuration File*: `ldt_tour_gen_household.yaml`
    - *Core Table*: `households`
    - *Result Field*: `ldt_tour_gen_household`
    - *Result dtype*: `bool`
    """

    trace_label = "ldt_tour_gen_household"
    model_settings_file_name = "ldt_tour_gen_household.yaml"

    choosers = households_merged.to_frame()
    # if we want to limit choosers, we can do so here
    # choosers = choosers[choosers.workplace_zone_id > -1]
    logger.info("Running %s with %d households", trace_label, len(choosers))

    model_settings = config.read_model_settings(model_settings_file_name)
    estimator = estimation.manager.begin_estimation("ldt_tour_gen_household")

    # reading in some category constants
    constants = config.get_model_constants(model_settings)

    category_file_name = model_settings.get("CATEGORY_CONSTANTS", {})
    categories = {}
    if category_file_name is not None:
        categories = config.get_model_constants(
            config.read_model_settings(category_file_name)
        )
    constants.update(categories)

    # preprocessor - adds accessiblity of chooser origin for use in estimation
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

    # reading in model specification/coefficients
    model_spec = simulate.read_model_spec(file_name=model_settings["SPEC"])
    coefficients_df = simulate.read_model_coefficients(model_settings)
    model_spec = simulate.eval_coefficients(model_spec, coefficients_df, estimator)

    nest_spec = config.get_logit_model_settings(model_settings)

    if estimator:
        estimator.write_model_settings(model_settings, model_settings_file_name)
        estimator.write_spec(model_settings)
        estimator.write_coefficients(coefficients_df, model_settings)
        estimator.write_choosers(choosers)

    # running the tour gen multinomial logit model
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

    # merging in tour gen results to households df
    households = households.to_frame()
    households["ldt_tour_gen_household"] = (
        choices.reindex(households.index).fillna(0).astype(bool)
    )

    # merging into final_households csv
    pipeline.replace_table("households", households)

    tracing.print_summary(
        "ldt_tour_gen_household",
        choices,
        value_counts=True,
    )

    if trace_hh_id:
        tracing.trace_df(households, label=trace_label, warn_if_empty=True)
