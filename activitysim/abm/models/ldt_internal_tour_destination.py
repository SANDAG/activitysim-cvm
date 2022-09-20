# ActivitySim
# See full license in LICENSE.txt.
import logging

import pandas as pd
from activitysim.core import config, inject, pipeline, tracing
from activitysim.core.util import assign_in_place

from .util import estimation, tour_destination  # noqa: F401

logger = logging.getLogger(__name__)


@inject.step()
def ldt_tour_destination(
    longdist_tours, persons_merged, network_los, chunk_size, trace_hh_id
):

    """
    Given the tour generation from the above, each tour needs to have a
    destination, so in this case tours are the choosers (with the associated
    person that's making the tour)
    """

    trace_label = "ldt_internal_tour_destination"
    model_settings_file_name = "ldt_internal_tour_destination.yaml"
    model_settings = config.read_model_settings(model_settings_file_name)

    logsum_column_name = model_settings.get("DEST_CHOICE_LOGSUM_COLUMN_NAME")
    want_logsums = logsum_column_name is not None

    sample_table_name = model_settings.get("DEST_CHOICE_SAMPLE_TABLE_NAME")
    want_sample_table = (
        config.setting("want_dest_choice_sample_tables")
        and sample_table_name is not None
    )

    # choosers are tours - in a sense tours are choosing their destination
    ldt_tours = longdist_tours.to_frame()
    ldt_tours = ldt_tours[
        ldt_tours.tour_category == 1 #
    ]  # filter tours travel on model day

    persons_merged = persons_merged.to_frame()

    # - if no ldt tours
    if ldt_tours.shape[0] == 0:
        tracing.no_results("ldt_tour_destination")
        return

    # TODO create ldt_tour_segment before this model step
    ldt_tours = pd.merge(
        ldt_tours.person_id,
        persons_merged[['NAICSP02', 'student_status', 'AGE']],
        on = 'person_id',
        how = 'left'
    )

    ldt_tours = ldt_tours.rename(columns = {'NAICSP02' : 'worker_industry'})
    day_hours = [6,7,8,9,10,11,12,13,14,15,16,17,18,19,20] # TO DO check
    overnight_hours = [21,22,23,0,1,2,3,4,5] # TO DO check
    # household tours are sub-segmented only by daytrip / overnight
    ldt_tours.loc[(ldt_tours['tour_type'] == 'longdist_household') & (
        ldt_tours['ldt_start_hour'].isin(day_hours)), 'ldt_tour_segment'] = 'household_day'
    ldt_tours.loc[(ldt_tours['tour_type'] == 'longdist_household') & (
        ldt_tours['ldt_start_hour'].isin(overnight_hours)), 'ldt_tour_segment'] = 'household_day'

    # individual work related tours are sub-segmented by daytrip / overnight and worker industry category
    for i in range(max(ldt_tours['worker_industry'])):
        ldt_tours.loc[(ldt_tours['tour_type'] == 'longdist_person_workrelated') & (
                ldt_tours['worker_industry'] == i) & (
                          ldt_tours['ldt_start_hour'].isin(day_hours)), 'ldt_tour_segment'] = 'workrelated_day_' + str(i)

        ldt_tours.loc[(ldt_tours['tour_type'] == 'longdist_person_workrelated') & (
                ldt_tours['worker_industry'] == i) & (
                          ldt_tours['ldt_start_hour'].isin(overnight_hours)), 'ldt_tour_segment'] = 'workrelated_overnight_' + str(i)


    # - Individual other tours are sub-segmented by daytrip / overnight and student status
    ldt_tours.loc[(ldt_tours['tour_type'] == 'longdist_person_other') & (
            ldt_tours['student_status'] == 1) & (
                    ldt_tours['ldt_start_hours'].isin(day_hours).isin(
                        day_hours)) & (
                            ldt_tours['age'] > 18), 'ldt_tour_segment'] = 'other_day_student'

    ldt_tours.loc[(ldt_tours['tour_type'] == 'longdist_person_other') & (
            ldt_tours['student_status'] == 1) & (
                      ldt_tours['ldt_start_hours'].isin(day_hours).isin(
                          overnight_hours)) & (
                            ldt_tours['age'] > 18), 'ldt_tour_segment'] = 'other_overnight_student'

    ldt_tours.loc[(ldt_tours['tour_type'] == 'longdist_person_other') & (
            ldt_tours['student_status'] == 0) & (
                      ldt_tours['ldt_start_hours'].isin(day_hours).isin(
                          day_hours)) & (
                            ldt_tours['age'] > 18), 'ldt_tour_segment'] = 'other_day_nonstudent'

    ldt_tours.loc[(ldt_tours['tour_type'] == 'longdist_person_other') & (
            ldt_tours['student_status'] == 0) & (
                      ldt_tours['ldt_start_hours'].isin(day_hours).isin(
                          overnight_hours)) & (
                            ldt_tours['age'] > 18), 'ldt_tour_segment'] = 'other_overnight_nonstudent'



    #ldt_tours

    # TODO
    # estimator = estimation.manager.begin_estimation("ldt_tour_destination")
    # if estimator:
    #     estimator.write_coefficients(model_settings=model_settings)
    #     # estimator.write_spec(model_settings, tag='SAMPLE_SPEC')
    #     estimator.write_spec(model_settings, tag="SPEC")
    #     estimator.set_alt_id(model_settings["ALT_DEST_COL_NAME"])
    #     estimator.write_table(
    #         inject.get_injectable("size_terms"), "size_terms", append=False
    #     )
    #     estimator.write_table(
    #         inject.get_table("land_use").to_frame(), "landuse", append=False
    #     )
    #     estimator.write_model_settings(model_settings, model_settings_file_name)

    choices_df, save_sample_df = tour_destination.run_tour_destination(
        ldt_tours,
        persons_merged,
        want_logsums,
        want_sample_table,
        model_settings,
        network_los,
        None,  # estimator
        chunk_size,
        trace_hh_id,
        trace_label,
    )

    # TODO
    # if estimator:
    #     estimator.write_choices(choices_df.choice)
    #     choices_df.choice = estimator.get_survey_values(
    #         choices_df.choice, "tours", "destination"
    #     )
    #     estimator.write_override_choices(choices_df.choice)
    #     estimator.end_estimation()

    # add column as we want joint_tours table for tracing.
    ldt_tours["destination"] = choices_df.choice
    assign_in_place(ldt_tours, ldt_tours[["destination"]])
    pipeline.replace_table("longdist_tours", ldt_tours)

    if want_logsums:
        ldt_tours[logsum_column_name] = choices_df["logsum"]
        assign_in_place(ldt_tours, ldt_tours[[logsum_column_name]])

    tracing.print_summary("destination", ldt_tours.destination, describe=True)

    if want_sample_table:
        assert len(save_sample_df.index.get_level_values(0).unique()) == len(choices_df)
        # save_sample_df.set_index(model_settings['ALT_DEST_COL_NAME'], append=True, inplace=True)
        pipeline.extend_table(sample_table_name, save_sample_df)

    if trace_hh_id:
        tracing.trace_df(ldt_tours, label="longdist_destination.ldt_tours")
