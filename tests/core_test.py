from __future__ import annotations

import datetime as dt

import pytest
import numpy as np

from torch_time import datetime, unixtime


def test_unixtime() -> None:
    now = dt.datetime.now()
    assert unixtime([now]).item() == np.array(now, dtype="datetime64[ns]").astype(np.int64)
    date_strings = ["2025-03-06 00:00:00", "2025-03-06 12:00:00"]
    expect = [1741219200000000000, 1741262400000000000]
    assert unixtime(date_strings).tolist() == expect
    assert (
        unixtime([dt.datetime.strptime(d, "%Y-%m-%d %H:%M:%S") for d in date_strings]).tolist()
        == expect
    )
    assert unixtime(expect).tolist() == expect


@pytest.mark.parametrize(
    "year, month, day, hour, minute, second",
    [
        (2025, 3, 6, 0, 0, 0),
        (2025, 3, 6, 12, 0, 0),
    ],
)
def test_datetime_record(
    year: int, month: int, day: int, hour: int, minute: int, second: int
) -> None:
    x = datetime(
        [dt.datetime(year, month, day, hour, minute, second).strftime("%Y-%m-%d %H:%M:%S")]
    )
    assert x.Y.item() == year
    assert x.M.item() == month
    assert x.D.item() == day
    assert x.h.item() == hour
    assert x.m.item() == minute
    assert x.s.item() == second
