# ActivitySim
# See full license in LICENSE.txt

import logging

from activitysim.core import config, expressions, inject, pipeline, simulate, tracing, logit

from .util import estimation

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


@inject.step()
def ldt_scheduling_person(persons, persons_merged, chunk_size, trace_hh_id):
    """
    This model schedules the times for applicable LDT patterns
        - 0/complete: schedules both start/end of tour
        - 1/begin: schedules beginning of tour
        - 2/end: schedules end of tour
        - 3/away: does not schedule
        - 4/notour/-1/no lDT generated - does not schedule
    """
    trace_label = "ldt_scheduling_person"
    model_settings_file_name = "ldt_scheduling.yaml"

    choosers = persons_merged.to_frame()
    # if we want to limit choosers, we can do so here
    # limiting ldt_scheduling_persons to applicable patterns - complete, begin, or end
    choosers = choosers[choosers["ldt_pattern"].isin([0, 1, 2])]
    logger.info("Running %s with %d persons", trace_label, len(choosers))

    # necessary to run
    model_settings = config.read_model_settings(model_settings_file_name)
    estimator = estimation.manager.begin_estimation("ldt_scheduling_person")

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

    spec_categories = model_settings.get("SPEC_CATEGORIES", {})

    persons = persons.to_frame()
    persons["ldt_start_hour"] = -1
    persons["ldt_end_hour"] = -1

    complete_tour_translation = {}
    i = 5
    x = 0
    while i <= 23:
        j = i
        while j <= 23:
            complete_tour_translation[x] = [i, j]
            j += 1
            x += 1
        i += 1
    complete_tour_translation = pd.Series(complete_tour_translation)

    for category_settings in spec_categories:
        category_num = int(category_settings["CATEGORY"])
        subset = choosers[choosers["ldt_pattern"] == category_num]
        category_name = category_settings["NAME"]

        if category_num == 0:
            model_spec = simulate.read_model_spec(file_name=category_settings["SPEC"])
            coefficients_df = simulate.read_model_coefficients(category_settings)
            model_spec = simulate.eval_coefficients(model_spec, coefficients_df, estimator)

            nest_spec = config.get_logit_model_settings(model_settings)

            if estimator:
                estimator.write_model_settings(model_settings, model_settings_file_name)
                estimator.write_spec(model_settings)
                estimator.write_coefficients(coefficients_df, model_settings)
                estimator.write_choosers(choosers)

            choices = simulate.simple_simulate(
                choosers=subset,
                spec=model_spec,
                nest_spec=nest_spec,
                locals_d=constants,
                chunk_size=chunk_size,
                trace_label=trace_label,
                trace_choice_name="ldt_scheduling_person_" + category_name,
                estimator=estimator,
            )

            if estimator:
                estimator.write_choices(choices)
                choices = estimator.get_survey_values(
                    choices, "persons", "ldt_scheduling_person_" + category_name
                )
                estimator.write_override_choices(choices)
                estimator.end_estimation()

            starts_ends = complete_tour_translation.loc[choices.values]
            persons.loc[choices.index, "ldt_start_hour"] = starts_ends.apply(lambda x: x[0]).values
            persons.loc[choices.index, "ldt_end_hour"] = starts_ends.apply(lambda x: x[1]).values

            continue

        constants = config.get_model_constants(category_settings)

        # sampling probabilities
        df = pd.DataFrame(index=choosers.index, columns=["hour_" + str(x) for x in range(24)])
        for i in range(24):
            key = "hour_" + str(i)
            df[key] = constants[key]
        choices, _ = logit.make_choices(df)

        if category_num == 1:
            persons.loc[choices.index, "ldt_start_hour"] = choices
        else:
            persons.loc[choices.index, "ldt_end_hour"] = choices

    # merging into persons
    pipeline.replace_table("persons", persons)

    tracing.print_summary(
        "ldt_scheduling_person_starts",
        persons.ldt_start_hour,
        value_counts=True
    )

    tracing.print_summary(
        "ldt_scheduling_person_ends",
        persons.ldt_end_hour,
        value_counts=True
    )

    if trace_hh_id:
        tracing.trace_df(persons, label=trace_label, warn_if_empty=True)

    # merging into longdist_tours
    longdist_tours = pipeline.get_table("longdist_tours")

    # making a version of persons with valid lookup values for a person_id of -1
    persons_temp = persons
    persons_temp.loc[-1, "ldt_start_hour"] = -2
    persons_temp.loc[-1, "ldt_end_hour"] = -2

    # merging start/end for persons; -2 lookup value would never be used since all entries with person_id of -1
    # is necessarily a household and thus not a person -- defaults to existing value
    longdist_tours["ldt_start_hour"] = (
        np.where(longdist_tours["actor_type"] == "person",
                 persons_temp.loc[longdist_tours["person_id"], "ldt_start_hour"], longdist_tours["ldt_start_hour"])
    )
    longdist_tours["ldt_end_hour"] = (
        np.where(longdist_tours["actor_type"] == "person",
                 persons_temp.loc[longdist_tours["person_id"], "ldt_end_hour"], longdist_tours["ldt_end_hour"])
    )

    pipeline.replace_table("longdist_tours", longdist_tours)
