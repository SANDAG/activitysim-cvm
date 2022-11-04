import glob
import os

import numpy as np
import openmatrix
import sharrow as sh

from activitysim.core import config, inject
from activitysim.core.los import LOS_SETTINGS_FILE_NAME


@inject.custom_step()
def skims_preprocess(data_dir):
    """
    Precompute the gross travel time
    """
    los_settings = config.read_settings_file(LOS_SETTINGS_FILE_NAME, mandatory=True)

    skim_settings = los_settings["taz_skims"]
    if isinstance(skim_settings, str):
        skims_omx_fileglob = skim_settings
    else:
        skims_omx_fileglob = skim_settings.get("omx", None)
        skims_omx_fileglob = skim_settings.get("files", skims_omx_fileglob)
    skims_filenames = glob.glob(os.path.join(data_dir, skims_omx_fileglob))
    index_names = ("otaz", "dtaz", "time_period")
    indexes = None
    time_period_breaks = los_settings.get("skim_time_periods", {}).get("periods")
    time_periods_raw = los_settings.get("skim_time_periods", {}).get("labels")
    time_periods = np.unique(time_periods_raw)
    time_period_sep = "__"

    skims_zarr = skim_settings.get("zarr", None)
    if skims_zarr is None:
        raise ValueError(
            f"skims_preprocess requires taz_skims.zarr in {LOS_SETTINGS_FILE_NAME}"
        )

    time_window = los_settings.get("skim_time_periods", {}).get("time_window")
    period_minutes = los_settings.get("skim_time_periods", {}).get("period_minutes")
    n_periods = int(time_window / period_minutes)

    tp_map = {}
    tp_imap = {}
    label = time_periods_raw[0]
    i = 0
    for t in range(n_periods):
        if t in time_period_breaks:
            i = time_period_breaks.index(t)
            label = time_periods_raw[i]
        tp_map[t + 1] = label
        tp_imap[t + 1] = i
    tp_map[-1] = tp_map[1]
    tp_imap[-1] = tp_imap[1]

    # raw_omx = los_settings["taz_skims"]["omx"]
    # if isinstance(raw_omx, str):
    #     raw_omx = [raw_omx]

    omxs = [
        openmatrix.open_file(skims_filename, mode="r")
        for skims_filename in skims_filenames
    ]
    try:
        if isinstance(time_periods, (list, tuple)):
            time_periods = np.asarray(time_periods)
        ds = sh.dataset.from_omx_3d(
            omxs,
            index_names=index_names,
            indexes=indexes,
            time_periods=time_periods,
            time_period_sep=time_period_sep,
        )

        # Gross time is the total overall travel time, used to determine schedule feasibility.
        ds["AIR_GROSS_TIME"] = ds["AIR_IVT"] + ds["AIR_FWT"] + ds["AIR_DRV"]
        ds["ICDT_GROSS_TIME"] = (
            ds["ICDT_IVT"] + ds["ICDT_TWT"] + ds["ICDT_DRV"] + ds["ICDT_XWK"]
        )
        ds["ICWT_GROSS_TIME"] = (
            ds["ICWT_IVT"]
            + ds["ICWT_TWT"]
            + ds["ICWT_AWK"]
            + ds["ICWT_EWK"]
            + ds["ICWT_XWK"]
        )
        ds["ICRDT_GROSS_TIME"] = (
            ds["ICRDT_IVT"] + ds["ICRDT_TWT"] + ds["ICRDT_DRV"] + ds["ICRDT_XWK"]
        )
        ds["ICRWT_GROSS_TIME"] = (
            ds["ICRWT_IVT"]
            + ds["ICRWT_TWT"]
            + ds["ICRWT_AWK"]
            + ds["ICRWT_EWK"]
            + ds["ICRWT_XWK"]
        )

        ds = ds.drop_vars(
            [
                "ICRWT_HFRQ",
                "ICRDT_HFRQ",
            ]
        )
        ds.attrs["time_period_map"] = tp_map
        ds.attrs["time_period_imap"] = tp_imap

        cache_dir = config.get_cache_dir()
        print(f"cache_dir = {os.path.join(cache_dir, skims_zarr)}")
        ds.to_zarr_with_attr(os.path.join(cache_dir, skims_zarr), mode="w")

    finally:
        for f in omxs:
            f.close()

    print("done")
