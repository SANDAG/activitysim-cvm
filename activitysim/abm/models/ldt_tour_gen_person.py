# ActivitySim
# See full license in LICENSE.txt.
import logging

from activitysim.core import config, expressions, inject, pipeline, simulate, tracing

from .util import estimation

logger = logging.getLogger(__name__)


@inject.step()
def ldt_tour_gen_person(persons, persons_merged, chunk_size, trace_hh_id):
    """
    This model determines whether a person goes on an LDT trip
    (whether for a work-related or other purpose) over a period of 2 weeks

    - *Configuration File*: `ldt_tour_gen_person.yaml`
    - *Core Table*: `persons`
    - *Result Fields*: `ldt_tour_gen_person_*` (one per purpose)
    - *Result dtype*: `bool`
    """

    trace_label = "ldt_tour_gen_person"
    model_settings_file_name = "ldt_tour_gen_person.yaml"

    # convert persons_merged to dataframe to use as fully-specified choosers file
    choosers = persons_merged.to_frame()
    logger.info("Running %s with %d persons", trace_label, len(choosers))

    # preliminary estimation steps
    model_settings = config.read_model_settings(model_settings_file_name)
    estimator = estimation.manager.begin_estimation("ldt_tour_gen_person")

    # reading in category constants
    constants = config.get_model_constants(model_settings)

    # merging in global constants
    categories = config.get_global_constants()
    constants.update(categories)

    # preprocessor - adds nothing
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

    # reading in the two specs for each individual ldt type
    model_spec = simulate.read_model_spec(file_name=model_settings["SPEC"])
    spec_purposes = model_settings.get("SPEC_PURPOSES", {})

    # needs to be outside the loop so we do it only once
    persons = persons.to_frame()

    # run estimation for all purposes (OTHER and WORKRELATED) using their respective settings
    for purpose_settings in spec_purposes:
        # the specified purpose name
        purpose_name = purpose_settings["NAME"]

        # read in the specfic purpose model coefficients
        coefficients_df = simulate.read_model_coefficients(purpose_settings)
        # need to differentiate the model_spec read in and the one used for each purpose, need to redeclare
        model_spec_purpose = simulate.eval_coefficients(
            model_spec, coefficients_df, estimator
        )

        nest_spec = config.get_logit_model_settings(model_settings)  # MNL

        if estimator:
            estimator.write_model_settings(model_settings, model_settings_file_name)
            estimator.write_spec(model_settings)
            estimator.write_coefficients(coefficients_df, model_settings)
            estimator.write_choosers(choosers)

        # run the multinomial logit models for the current ldt type
        choices = simulate.simple_simulate(
            choosers=choosers,
            spec=model_spec_purpose,
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

        # merging choices into the person csv
        colname = "ldt_tour_gen_person_" + purpose_name
        persons[colname] = choices.reindex(persons.index).fillna(0).astype(bool)

        pipeline.replace_table("persons", persons)

        tracing.print_summary(
            colname,
            choices,
            value_counts=True,
        )

        if trace_hh_id:
            tracing.trace_df(persons, label=trace_label, warn_if_empty=True)
