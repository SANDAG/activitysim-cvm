# ActivitySim
# See full license in LICENSE.txt

import logging

from activitysim.core import config, expressions, inject, pipeline, simulate, tracing, logit

from .util import estimation
from .ldt_tour_gen import process_longdist_tours

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
    temp = False

    for purpose_settings in spec_purposes:
        purpose_name = purpose_settings["NAME"]
        colname = "ldt_pattern_person_" + purpose_name

        # default value
        persons[colname] = -1
        choosers_full[colname] = -1

        # only consider people who are predicted to go on LDT tour over 2 week period
        choosers = choosers_full[choosers_full["ldt_tour_gen_person_" + purpose_name]]
        # only consider people who aren't scheduled to go on household LDT
        choosers = choosers[choosers["ldt_pattern_household"].isin([-1, 4])]

        if temp:
            # only consider people who aren't scheduled to go on work LDT when scheduling other LDT
            choosers = choosers[choosers["ldt_pattern_person_WORKRELATED"].isin([-1, 4])]

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

        # _ is the random value used to make the monte carlo draws
        choices, _ = logit.make_choices(df)

        if estimator:
            estimator.write_choices(choices)
            choices = estimator.get_survey_values(
                choices, "persons", colname
            )
            estimator.write_override_choices(choices)
            estimator.end_estimation()

        # making one ldt pattern field instead of segmenting by person/household currently
        persons.loc[choices.index, colname] = choices
        # adding it to choosers for downstream integrity check
        choosers_full.loc[choices.index, colname] = choices

        temp = True

        tracing.print_summary(
            colname,
            persons[colname],
            value_counts=True,
        )

    # adding convenient fields
    # whether or not person is scheduled to be on LDT trip
    persons["on_ldt"] = np.where(persons["ldt_pattern_person_WORKRELATED"].isin([0, 1, 2, 3]), True, False)
    persons["on_ldt"] = (
        np.where(~persons["on_ldt"], persons["ldt_pattern_person_OTHER"].isin([0, 1, 2, 3]), persons["on_ldt"])
    )

    # -1 is no LDT trip (whether a trip was not generated/not scheduled), 0 is work releated, 1 is other
    persons["ldt_purpose"] = np.where(persons["on_ldt"], 1, -1)
    persons["ldt_purpose"] = (
        np.where(persons["ldt_pattern_person_WORKRELATED"].isin([0, 1, 2, 3]), 0, persons["ldt_purpose"])
    )

    # -1 is no LDT trip (whether a trip was not generated/not scheduled), others match up to the pattern for a
    # person's specified ldt_purpose (excluding 4, which means no scheduled LDT--changed to -1)
    persons["ldt_pattern"] = np.where(persons["on_ldt"], 0, -1)
    persons["ldt_pattern"] = (
        np.where(persons["ldt_purpose"] == 0, persons["ldt_pattern_person_WORKRELATED"], persons["ldt_pattern"])
    )
    persons["ldt_pattern"] = (
        np.where(persons["ldt_purpose"] == 1, persons["ldt_pattern_person_OTHER"], persons["ldt_pattern"])
    )

    pipeline.replace_table("persons", persons)

    process_person_tours(persons, "workrelated", 0)
    process_person_tours(persons, "other", 1)


def process_person_tours(persons, purpose: str, purpose_num: int):
    persons_making_longdist_tours = persons[persons["ldt_purpose"] == purpose_num]
    tour_counts = (
        persons_making_longdist_tours[["on_ldt"]]
        .astype(int)
        .rename(
            columns={"on_ldt": f"longdist_person_{purpose}"}
        )
    )

    print(tour_counts)

    longdist_tours_person = process_longdist_tours(
        persons, tour_counts, "longdist"
    )

    longdist_tours_person = (
        pd.merge(longdist_tours_person, persons[["ldt_pattern"]],
                 how="left", left_on="person_id", right_index=True)
    )

    pipeline.extend_table("longdist_tours", longdist_tours_person)
