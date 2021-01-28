#!/usr/bin/env python
# coding=utf-8
import geojson
import logging
from datetime import datetime, date
from typing import Union
from numbers import Real
from pyproj import Geod

import pandas as pd
import numpy as np

N = Real
L = logging.getLogger(__name__)  # noqa


def add_flag_metadata(standard_name, long_name=None):
    def dec(fn):
        fn.standard_name = standard_name
        fn.long_name = long_name
        return fn
    return dec


def isfixedlength(lst : Union[list, tuple],
                  length : int
                  ) -> bool:
    if not isinstance(lst, (list, tuple)):
        raise ValueError('Required: list/tuple, Got: {}'.format(type(lst)))

    if len(lst) != length:
        raise ValueError(
            'Incorrect list/tuple length for {}. Required: {}, Got: {}'.format(
                lst,
                length,
                len(lst)
            )
        )

    return True


def isnan(v):
    return (
        v is None or
        v is np.nan or
        v is np.ma.masked
    )


def mapdates(dates):
    if hasattr(dates, 'dtype') and np.issubdtype(dates.dtype, np.datetime64):
        # numpy datetime objects
        return dates.astype('datetime64[ns]')
    else:
        try:
            # Finally try unix epoch seconds
            return pd.to_datetime(dates, unit='s').values.astype('datetime64[ns]')
        except Exception:
            # strings work here but we don't advertise that
            return np.array(dates, dtype='datetime64[ns]')


def check_timestamps(times : np.ndarray,
                     max_time_interval : N = None):
    """Sanity checks for timestamp arrays

    Checks that the times supplied are in monotonically increasing
    chronological order, and optionally that time intervals between
    measurements do not exceed a value `max_time_interval`.  Note that this is
    not a QARTOD test, but rather a utility test to make sure times are in the
    proper order and optionally do not have large gaps prior to processing the
    data.

    Args:
        times: Input array of timestamps
        max_time_interval: The interval between values should not exceed this
            value. [optional]
    """

    time_diff = np.diff(times)
    sort_diff = np.diff(sorted(times))
    # Check if there are differences between sorted and unsorted, and then
    # see if if there are any duplicate times.  Then check that none of the
    # diffs exceeds the sorted time.
    zero = np.array(0, dtype=time_diff.dtype)
    if not np.array_equal(time_diff, sort_diff) or np.any(sort_diff == zero):
        return False
    elif (max_time_interval is not None and
          np.any(sort_diff > max_time_interval)):
        return False
    else:
        return True


def dict_update(d, u):
    # http://stackoverflow.com/a/3233356
    from collections.abc import Mapping
    for k, v in u.items():
        if isinstance(d, Mapping):
            if isinstance(v, Mapping):
                r = dict_update(d.get(k, {}), v)
                d[k] = r
            else:
                d[k] = u[k]
        else:
            d = {k: u[k] }
    return d


def cf_safe_name(name):
    import re
    if isinstance(name, str):
        if re.match('^[0-9_]', name):
            # Add a letter to the front
            name = "v_{}".format(name)
        return re.sub(r'[^_a-zA-Z0-9]', "_", name)

    raise ValueError('Could not convert "{}" to a safe name'.format(name))


class GeoNumpyDateEncoder(geojson.GeoJSONEncoder):

    def default(self, obj):
        """If input object is an ndarray it will be converted into a list
        """
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.generic):
            return obj.item()
        # elif isinstance(obj, pd.Timestamp):
        #     return obj.to_pydatetime().isoformat()
        elif isinstance(obj, (datetime, date)):
            return obj.isoformat()
        elif np.isnan(obj):
            return None

        return geojson.factory.GeoJSON.to_instance(obj)


def great_circle_distance(lat_arr, lon_arr):
    dist = np.ma.zeros(lon_arr.size, dtype=np.float64)
    g = Geod(ellps='WGS84')
    _, _, dist[1:] = g.inv(lon_arr[:-1], lat_arr[:-1], lon_arr[1:], lat_arr[1:])
    return dist


def distance_from_target(lat, lon, target_lat, target_lon):
    g = Geod(ellps='WGS84')
    _, _, dist_to_target = g.inv(lon, lat, target_lon, target_lat)
    dist_to_target = np.ma.masked_invalid(dist_to_target.astype(np.float64))
    return dist_to_target
