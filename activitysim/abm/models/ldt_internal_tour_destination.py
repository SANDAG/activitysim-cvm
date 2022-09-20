# ActivitySim
# See full license in LICENSE.txt.
import logging

from activitysim.core import config, inject, pipeline, tracing
from activitysim.core.util import assign_in_place

from .util import estimation, tour_destination  # noqa: F401

logger = logging.getLogger(__name__)


@inject.step()
def ldt_internal_tour_destination(
    longdist_tours,
    persons_merged,
    network_los,
    chunk_size,
    trace_hh_id,
    households_merged,
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
    pipeline.get_rn_generator().add_channel("longdist_tours", ldt_tours)

    ldt_tours = ldt_tours[
        ldt_tours.ldt_pattern >= 1
    ]  # filter tours travel on model day

    persons_merged = persons_merged.to_frame()
    hh = households_merged.to_frame()

    # - if no joint tours
    if ldt_tours.shape[0] == 0:
        tracing.no_results("ldt_tour_destination")
        return

    # TODO create ldt_tour_segment before this model step

    ldt_tours = ldt_tours.join(
        persons_merged[["NAICSP02"]], on="person_id", rsuffix="__"
    )

    seg1 = ldt_tours["tour_type"].apply(lambda x: x.split("_")[-1])
    import numpy as np

    seg2 = np.where(ldt_tours["ldt_pattern"] == 1, "day", "overnight")  # FIXME

    ldt_tours["seg1"] = seg1
    ldt_tours["seg2"] = seg2

    def seg3_(row):
        if row["seg1"] == "household":
            return f"{row['seg1']}_{row['seg2']}"
        elif row.seg1 == "workrelated":
            return f"{row['seg1']}_{row['seg2']}_{np.clip(row['NAICSP02'], 1, 17):.0f}"
        else:
            return f"{row['seg1']}_{row['seg2']}_student"

    ldt_tours["ldt_tour_segment"] = ldt_tours.apply(seg3_, axis=1)

    # ldt_tours["in_period"] = network_los.skim_time_period_label(
    #     ldt_tours['ldt_end_hour'],
    #     fillna=0,
    # )
    # ldt_tours["out_period"] = network_los.skim_time_period_label(
    #     ldt_tours['ldt_start_hour'],
    #     fillna=0,
    # )

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
