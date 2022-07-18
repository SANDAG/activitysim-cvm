# ActivitySim
# See full license in LICENSE.txt.
import logging

from activitysim.core import config, expressions, inject, pipeline, simulate, tracing

from .util import estimation

logger = logging.getLogger(__name__)


@inject.step()
def long_distance_trip_gen_person(persons_merged, persons, households,
                  chunk_size, trace_hh_id):
    """

    """

    trace_label = "long_distance_trip_gen_person"
    model_settings_file_name = "long_distance_trip_gen_person.yaml"

    choosers = persons_merged.to_frame()
    # if we want to limit choosers, we can do so here
    #choosers = choosers[choosers.workplace_zone_id > -1]
    logger.info("Running %s with %d persons", trace_label, len(choosers))

    model_settings = config.read_model_settings(model_settings_file_name)
    estimator = estimation.manager.begin_estimation("long_distance_trip_gen_person")

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
        trace_choice_name="long_distance_trip_gen_person",
        estimator=estimator,
    )

    if estimator:
        estimator.write_choices(choices)
        choices = estimator.get_survey_values(
            choices, "persons", "long_distance_trip_gen_person"
        )
        estimator.write_override_choices(choices)
        estimator.end_estimation()

    persons = persons_merged.to_frame()
    persons_merged["long_distance_trip_gen_persons"] = (
        choices.reindex(persons_merged.index).fillna(0).astype(bool)
    )

    pipeline.replace_table("persons_merged", persons_merged)

    tracing.print_summary(
        "long_distance_trip_gen_persons",
        persons_merged.long_distance_trip_gen_persons,
        value_counts=True,
    )

    if trace_hh_id:
        tracing.trace_df(persons_merged, label=trace_label, warn_if_empty=True)
