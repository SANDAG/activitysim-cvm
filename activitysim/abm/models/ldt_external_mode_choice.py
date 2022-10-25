# ActivitySim
# See full license in LICENSE.txt

import logging

import numpy as np
from activitysim.abm.models.ldt_internal_external import LDT_IE_EXTERNAL

from activitysim.core import config, expressions, inject, pipeline, simulate, tracing, los



from .util import estimation

logger = logging.getLogger(__name__)


@inject.step()
def ldt_external_mode_choice(
    persons, persons_merged, households, households_merged, network_los, chunk_size, trace_hh_id
):
    """
    This model determines if a person on an LDT is going/will go/is at an internal location (within Ohio/0)
    or at an external location (outside of Ohio/1)
    """

    trace_label = "ldt_external_mode_choice"
    colname = "external_mode_choice"
    model_settings_file_name = "ldt_external_mode_choice.yaml"

    # preliminary estimation steps
    model_settings = config.read_model_settings(model_settings_file_name)
    estimator = estimation.manager.begin_estimation(trace_label)
    constants = config.get_model_constants(model_settings)  # constants shared by all

    # converting parameters to dataframes
    hh_full = households_merged.to_frame()
    persons_full = persons_merged.to_frame()
    households = households.to_frame()
    persons = persons.to_frame()

    # setting default value for internal_external choice to -1
    households[colname] = -1
    persons[colname] = -1

    spec_categories = model_settings.get(
        "SPEC_CATEGORIES", {}
    )  # reading in category-specific things
    model_spec = simulate.read_model_spec(
        file_name=model_settings["SPEC"]
    )  # reading in generic model spec
    nest_spec = config.get_logit_model_settings(model_settings)  # NL
    
    # setup skims and skim keys
    orig_col = "home_zone_id"
    dest_col = "external_destchoice"

    skim_dict = network_los.get_default_skim_dict()

    odt_skim_stack_wrapper = skim_dict.wrap_3d(
        orig_key=orig_col, dest_key=dest_col, dim3_key="ldt_start_hour"
    )
    dot_skim_stack_wrapper = skim_dict.wrap_3d(
        orig_key=dest_col, dest_key=orig_col, dim3_key="ldt_start_hour"
    )
    od_skim_wrapper = skim_dict.wrap(orig_col, dest_col)
    
    skims = {
        "odt_skims": odt_skim_stack_wrapper,
        "dot_skims": dot_skim_stack_wrapper,
        "od_skims": od_skim_wrapper,
    }

    if network_los.zone_system == los.THREE_ZONE:
        # fixme - is this a lightweight object?

        tvpb = network_los.tvpb

        tvpb_logsum_odt = tvpb.wrap_logsum(
            orig_key=orig_col,
            dest_key=dest_col,
            tod_key="out_period",
            segment_key="demographic_segment",
            cache_choices=True,
            trace_label=trace_label,
            tag="tvpb_logsum_odt",
        )
        tvpb_logsum_dot = tvpb.wrap_logsum(
            orig_key=dest_col,
            dest_key=orig_col,
            tod_key="in_period",
            segment_key="demographic_segment",
            cache_choices=True,
            trace_label=trace_label,
            tag="tvpb_logsum_dot",
        )

        skims.update(
            {"tvpb_logsum_odt": tvpb_logsum_odt, "tvpb_logsum_dot": tvpb_logsum_dot}
        )

        # TVPB constants can appear in expressions
        if model_settings.get("use_TVPB_constants", True):
            constants.update(
                network_los.setting("TVPB_SETTINGS.tour_mode_choice.CONSTANTS")
            )
            
    # add skim keys to constants dictionary so that they can be accessed in 
    # specification csv expressions
    constants.update(skims)
    
    patterns = {
        "WORKRELATED": "ldt_tour_gen_person_WORKRELATED",
        "OTHER": "ldt_tour_gen_person_OTHER"
    }

    for category_settings in spec_categories:  # iterate through all the categories
        category_name = category_settings["NAME"]  # HOUSEHOLD, WORKRELATED, OTHER
        full_name = colname + "_" + category_name
        actor_type = category_name.lower() # household or person

        if actor_type == "household":
            # need ldt_pattern field for the specification
            choosers = hh_full.rename(columns={"ldt_pattern_household": "ldt_pattern"})
            # only consider people going externally
            choosers = choosers[choosers["internal_external"] == LDT_IE_EXTERNAL]
            # only consider people who are on household ldts
            choosers = choosers[choosers["on_hh_ldt"]]
            logger.info("Running %s with %d households", full_name, len(choosers))
        else:
            # need ldt_pattern field for the specification
            choosers = persons_full.rename(
                columns={"ldt_pattern_person": "ldt_pattern"}
            )
            # only consider people who are on person ldts
            choosers = choosers[choosers["on_person_ldt"]]
            # only consider people going externally
            choosers = choosers[choosers["internal_external"] == LDT_IE_EXTERNAL]
            # restrict to current category
            choosers = choosers[choosers[patterns.get(category_name)]]
            logger.info("Running %s with %d persons", full_name, len(choosers))

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
        nest_category_spec = simulate.eval_nest_coefficients(nest_spec, coefficients_df, estimator)
        category_spec = simulate.eval_coefficients(
            model_spec, coefficients_df, estimator
        )

        if estimator:
            estimator.write_model_settings(category_settings, model_settings_file_name)
            estimator.write_spec(category_settings)
            estimator.write_coefficients(coefficients_df, category_settings)
            estimator.write_choosers(choosers)

        # run the nested logit models for the current category
        choices = simulate.simple_simulate(
            choosers=choosers,
            spec=category_spec,
            nest_spec=nest_category_spec,
            locals_d=constants,
            chunk_size=chunk_size,
            skims=skims,
            trace_label=trace_label,
            trace_choice_name=full_name,
            estimator=estimator,
        )
        
        alts = category_spec.columns
        choices = choices.map(
            dict(list(zip(list(range(len(alts))), alts)))
        )

        if estimator:
            estimator.write_choices(choices)
            choices = estimator.get_survey_values(choices, actor_type, full_name)
            estimator.write_override_choices(choices)
            estimator.end_estimation()

        # merge in results into relevant df
        if actor_type == "household":
            households.loc[choices.index, colname] = choices.values
        else:
            persons.loc[choices.index, colname] = choices.values

        # print out summary of estimated internal_external choices for the current category
        tracing.print_summary(
            full_name, choices, value_counts=True
        )

    # merging into final csvs
    pipeline.replace_table("households", households)
    pipeline.replace_table("persons", persons)

    # merging into longdist csv
    longdist_tours = pipeline.get_table("longdist_tours")

    # merging into longdist_tours with a custom function to handle cases
    # where we can't index on person_id (it is -1)
    def fill_in(x):
        if x == -1:
            return -1
        return persons.loc[x, colname]

    longdist_tours[colname] = -1
    longdist_tours[colname] = np.where(
        longdist_tours["actor_type"] == "person",
        longdist_tours["person_id"].apply(fill_in),
        longdist_tours[colname]
    )
    longdist_tours[colname] = np.where(
        longdist_tours["actor_type"] == "household",
        households.loc[longdist_tours["household_id"], colname],
        longdist_tours[colname]
    )
    
    pipeline.replace_table("longdist_tours", longdist_tours)

