import logging

import numpy as np
import pandas as pd

from ...core.util import reindex
from .util.canonical_ids import set_tour_index
from .util.tour_frequency import create_tours

logger = logging.getLogger(__name__)


def set_longdist_tour_index(tours):
    """
    The new index values are stable based on the person_id, tour_type, and tour_num.
    The existing index is ignored and replaced.

    This gives us a stable (predictable) tour_id with tours in canonical order
    (when tours are sorted by tour_id, tours for each person
    of the same type will be adjacent and in increasing tour_type_num order)

    It also simplifies attaching random number streams to tours that are stable
    (even across simulations)

    Parameters
    ----------
    tours : DataFrame
        Tours dataframe to reindex.
    """

    tour_num_col = "tour_type_num"
    # changed tours types to be more specific
    possible_tours = ["longdist_household1", "longdist_person_workrelated1", "longdist_person_other1"]
    possible_tours_count = len(possible_tours)

    assert tour_num_col in tours.columns

    # create string tour_id corresonding to keys in possible_tours (e.g. 'work1', 'j_shopping2')
    tours["tour_id"] = tours.tour_type + tours[tour_num_col].map(str)

    # map recognized strings to ints
    tours.tour_id = tours.tour_id.replace(
        to_replace=possible_tours, value=list(range(possible_tours_count))
    )

    # convert to numeric - shouldn't be any NaNs - this will raise error if there are
    tours.tour_id = pd.to_numeric(tours.tour_id, errors="raise").astype(np.int64)

    tours.tour_id = (
        (tours.household_id * possible_tours_count * 100)
        + tours.person_id * 10
        + tours.tour_id
    )

    # if tours.tour_id.duplicated().any():
    #     print("\ntours.tour_id not unique\n%s" % tours[tours.tour_id.duplicated(keep=False)])
    #     print(tours[tours.tour_id.duplicated(keep=False)][['survey_tour_id', 'tour_type', 'tour_category']])
    assert not tours.tour_id.duplicated().any()

    tours.set_index("tour_id", inplace=True, verify_integrity=True)

    # we modify tours in place, but return the dataframe for the convenience of the caller
    return tours


def process_longdist_tours(df, tour_counts, tour_category):
    """
    This method processes a tour_counts column and turns out a DataFrame that
    represents the long distance tours that were generated.

    Parameters
    ----------
    df: pandas.DataFrame
        persons or household table containing
    tour_counts : pandas.Series
        Matches the df, a tour frequency column
    tour_category : str
        A label for the type of tours

    Returns
    -------
    tours : DataFrame
        An example of a tours DataFrame is supplied as a comment in the
        source code - it has an index which is a unique tour identifier,
        a person_id column, and a tour type column which comes from the
        column names of the alternatives DataFrame supplied above.
    """

    tours = create_tours(tour_counts, tour_category=tour_category)

    if "household_id" in df.columns:  # only person dfs have a household_id column; processing persons here
        tours["household_id"] = reindex(df.household_id, tours.person_id)
        tours["origin"] = reindex(df.home_zone_id, tours.person_id)
    else:  # processing households here
        # TODO get smart about this, don't just assume we're in households...
        # wouldn't even people living alone technically be in households unless want to segment by GQ/household
        # create_tours returns ids as person_id, need to reassignto household_id
        tours["household_id"] = tours["person_id"]
        tours["origin"] = reindex(df.home_zone_id, tours.household_id)
        tours["person_id"] = -1   # hh tours don't use person ids
        # number of participants = hhsize since hh ldt tours are necessarily joint
        # (can remove if participation stage implemented)
        tours["number_of_participants"] = reindex(df.hhsize, tours.household_id)

    # assign stable (predictable) tour_id
    set_longdist_tour_index(tours)

    return tours
