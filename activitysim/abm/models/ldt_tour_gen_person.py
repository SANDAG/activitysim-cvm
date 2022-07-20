# ActivitySim
# See full license in LICENSE.txt.
import logging

from activitysim.core import config, expressions, inject, pipeline, simulate, tracing

from .util import estimation
from .ldt_tour_gen import process_longdist_tours

logger = logging.getLogger(__name__)

@inject.step()
def ldt_tour_gen_person(persons, persons_merged, 
                  chunk_size, trace_hh_id):
    """

    """

    trace_label = "ldt_tour_gen_person"
    model_settings_file_name = "ldt_tour_gen_person.yaml"

    choosers = persons_merged.to_frame()
    logger.info("Running %s with %d persons", trace_label, len(choosers))

    model_settings = config.read_model_settings(model_settings_file_name)
    estimator = estimation.manager.begin_estimation("ldt_tour_gen_person")

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
    spec_purposes = model_settings.get('SPEC_PURPOSES', {})
    
    # needs to be outside the loop so we do it only once
    persons = persons.to_frame()
        
    for purpose_settings in spec_purposes:
    
        purpose_name = purpose_settings['NAME']
        
        coefficients_df = simulate.read_model_coefficients(purpose_settings)
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
            trace_choice_name="ldt_tour_gen_person_" + purpose_name,
            estimator=estimator,
        )

        if estimator:
            estimator.write_choices(choices)
            choices = estimator.get_survey_values(
                choices, "persons", "ldt_tour_gen_person_" + purpose_name
            )
            estimator.write_override_choices(choices)
            estimator.end_estimation()

        colname = "ldt_tour_gen_persons_" + purpose_name
        persons[colname] = (
            choices.reindex(persons.index).fillna(0).astype(bool)
        )

        pipeline.replace_table("persons", persons)

        tracing.print_summary(
            colname,
            persons[colname],
            value_counts=True,
        )

        if trace_hh_id:
            tracing.trace_df(persons, label=trace_label, warn_if_empty=True)

        ####
        # init log dist trip table
        persons_making_longdist_tours = persons[persons[colname]]
        tour_counts = (
            persons_making_longdist_tours[[colname]]
                .astype(int)
                .rename(
                columns={colname: f"longdist_{purpose_name.lower()}"}
            )
        )
        longdist_tours = process_longdist_tours(
            persons, tour_counts, f"longdist_{purpose_name.lower()}"
        )
        pipeline.extend_table("longdist_tours", longdist_tours)
