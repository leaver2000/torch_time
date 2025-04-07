from __future__ import annotations

import datetime as dt

import pytest
import numpy as np
from torch_time import datetime, unixtime
import torch
from numpy.typing import ArrayLike
import pandas as pd


def assert_equal(a: ArrayLike, b: ArrayLike):
    dtype = None
    if not isinstance(a, torch.Tensor):
        if isinstance(b, torch.Tensor):
            dtype = b.dtype
        a = torch.as_tensor(a, dtype=dtype)
    if not isinstance(b, torch.Tensor):
        b = torch.as_tensor(b, dtype=a.dtype)

    assert a.shape == b.shape
    assert a.dtype == b.dtype
    assert (a == b).all()


def test_unixtime() -> None:
    now = dt.datetime.now()
    assert unixtime([now]).item() == np.array(now, dtype="datetime64[ns]").astype(np.int64)
    date_strings = ["2025-03-06 00:00:00", "2025-03-06 12:00:00"]
    date_objects = [dt.datetime.strptime(d, "%Y-%m-%d %H:%M:%S") for d in date_strings]
    date_array = np.array(date_objects, dtype="datetime64[ns]")
    expect = [1741219200000000000, 1741262400000000000]
    assert_equal(unixtime(date_strings), expect)
    assert_equal(unixtime(date_objects), expect)
    assert_equal(unixtime(date_array), expect)
    assert_equal(unixtime(date_array.astype(np.int64)), expect)


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


def test_with_pandas() -> None:
    expect = pd.date_range("2000-03-06 00:00:00", "2025-03-06 12:00:00", freq="D")
    date = datetime(expect)
    assert_equal(date.Y, expect.year)
    assert_equal(date.M, expect.month)
    assert_equal(date.D, expect.day)
    assert_equal(date.h, expect.hour)
    assert_equal(date.m, expect.minute)
    assert_equal(date.s, expect.second)
    assert_equal(date.ns, expect.nanosecond)
    assert_equal(unixtime(expect), unixtime(expect.values))

    rec = date.numpy()
    assert isinstance(rec, np.recarray)
    assert rec.shape == (len(expect),)

    item = rec[0]
    assert isinstance(item, np.record)

    assert item.Y == expect.year[0]
    assert item.M == expect.month[0]
    assert item.D == expect.day[0]
    assert item.h == expect.hour[0]
    assert item.m == expect.minute[0]
    assert item.s == expect.second[0]
    assert item.ns == expect.nanosecond[0]
