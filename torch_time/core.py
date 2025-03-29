from __future__ import annotations

__all__ = [
    "Datetime",
    "unixtime",
    "datetime",
    "julian_day",
    "julian_date",
    "julian_century",
    "julian_day_time",
]

import datetime as _datetime
from typing import Final, Literal, Mapping, Sequence, NamedTuple, Self

import numpy as np
import torch
from numpy.typing import ArrayLike

# ..... { types } .....
type Pair[T] = tuple[T, T]
type Precision = Literal["s", "ms", "us", "ns"]
type NestedSequence[T] = Sequence[NestedSequence[T]] | Sequence[T]
type UnixTimestampLike = ArrayLike | NestedSequence[_datetime.datetime]
type DatetimeLike = UnixTimestampLike | Datetime


# ..... { constant } .....
SECONDS_PER_MINUTE = 60
MINUTES_PER_HOUR = 60
HOURS_PER_DAY = 24
SECONDS_PER_HOUR = SECONDS_PER_MINUTE * MINUTES_PER_HOUR
SECONDS_PER_DAY = SECONDS_PER_HOUR * HOURS_PER_DAY

_time_table: Final[Mapping[str, int]]
_time_table = {"ns": 1_000_000_000, "us": 1_000_000, "ms": 1_000, "s": 1}


class Datetime(NamedTuple):
    Y: torch.Tensor
    M: torch.Tensor
    D: torch.Tensor
    h: torch.Tensor
    m: torch.Tensor
    s: torch.Tensor
    ns: torch.Tensor

    def double(self) -> Self:
        return self.__class__(*map(lambda x: x.double(), self))


def unixtime(x: UnixTimestampLike, /, precision: Precision | None = None) -> torch.Tensor:
    """Convert a sequence of datetime like objects or unix timestamps to a tensor of unix timestamps.

    Args:
        x: A sequence of datetime objects or unix timestamps
        precision: The precision of the unix timestamps

    Returns:
        A tensor of unix timestamps

    Examples:
    >>> date_strings = ["2025-03-06 00:00:00", "2025-03-06 12:00:00"]
    >>> expect = [1741219200000000000, 1741262400000000000]
    >>> unixtime(date_strings).tolist() == expect
    >>> unixtime([datetime.datetime.strptime(d, "%Y-%m-%d %H:%M:%S") for d in date_strings]).tolist() == expect
    >>> unixtime(expect).tolist() == expect
    """
    if isinstance(x, torch.Tensor):
        return x.clone()
    x = np.asarray(x)
    if np.isdtype(x.dtype, np.str_):
        x = x.astype(f"datetime64[{precision or 'ns'}]")
    elif np.isdtype(x.dtype, np.object_):
        x = x.astype(f"datetime64[{precision or 'ns'}]")

    return torch.from_numpy(x.astype(np.int64))


def datetime(dt: DatetimeLike, /, precision: Precision | None = None) -> Datetime:
    """
    Extract year, month, day, hour, minute, second from timestamps using only PyTorch operations.
    No loops, no NumPy, no datetime library.

    Args:
        timestamps: A tensor of Unix timestamps (seconds since epoch)

    Returns:
        A dictionary containing tensors for each datetime component

    References:
        - claude.ai chat

    >>> import datetime
    >>> from wxvit.utils.time import datetime_record
    >>> now = datetime.datetime.now(datetime.timezone.utc)
    >>> now
    datetime.datetime(2025, 3, 7, 5, 24, 29, 226463, tzinfo=datetime.timezone.utc)
    >>> datetime_record(now)
    {'Y': tensor(2025), 'M': tensor(3), 'D': tensor(7), 'h': tensor(5), 'm': tensor(24), 's': tensor(29), 'ns': tensor(0)}
    """
    global SECONDS_PER_MINUTE, MINUTES_PER_HOUR, HOURS_PER_DAY
    if isinstance(dt, Datetime):
        return dt

    ut = unixtime(dt, precision)

    factor = _time_table[precision or "ns"]
    if precision == "ns" or precision is None:
        ns = ut % 1_000
        ut //= factor
    elif precision in ("us", "ms", "s"):
        ns = torch.zeros_like(ut)
        ut //= factor
    elif precision is not None:
        raise ValueError(f"Invalid precision: {precision}")

    s = ut % SECONDS_PER_MINUTE
    ut = ut // SECONDS_PER_MINUTE

    m = ut % MINUTES_PER_HOUR
    ut = ut // MINUTES_PER_HOUR

    h = ut % HOURS_PER_DAY
    days_since_epoch = ut // HOURS_PER_DAY

    # Calculate date components
    # First convert to days since 0000-03-01 (to simplify leap year calculations)
    # 719468 is days from 0000-03-01 to 1970-01-01
    days_since_0000_03_01 = days_since_epoch + 719468

    # The era is 400 years
    era = (days_since_0000_03_01 >= 0).long() * (days_since_0000_03_01 // 146097)
    day_of_era = days_since_0000_03_01 - era * 146097  # [0, 146096]

    # The year of era (0-399)
    year_of_era = (
        day_of_era - day_of_era // 1460 + day_of_era // 36524 - day_of_era // 146096
    ) // 365

    Y = year_of_era + era * 400  # Calculate year
    day_of_year = day_of_era - (
        365 * year_of_era + year_of_era // 4 - year_of_era // 100 + year_of_era // 400
    )

    # Shift from Mar-Feb to Jan-Dec (month = m + 2)
    f = (5 * day_of_year + 2) // 153  # [0, 11]
    D = day_of_year - (153 * f + 2) // 5 + 1  # [1, 31]
    M = f + 3 - (f >= 10).long() * 12  # [1, 12]
    Y = Y + (M <= 2).long()  # Adjust the year for Jan/Feb

    return Datetime(Y, M, D, h, m, s, ns).double()


def julian_day_time(dt: DatetimeLike, /, precision: Precision | None = None) -> Pair[torch.Tensor]:
    """
    Convert a datetime components or a Unix timestamp to a tuple of tensors containing a Julian Day
    and time fraction.

    Args:
        components: A tensor of datetime components or a tensor of Unix timestamps
        precision: The precision of the Unix timestamp

    Returns:
        A tuple of tensors containing a Julian Day and time fraction

    References:
        - https://github.com/seanredmond/juliandate/blob/main/juliandate/juliandate.py
    """

    Y, M, D, h, m, s, ns = datetime(dt, precision)

    f = ((M - 14) / 12).ceil()
    ymd = (
        ((1461 * (Y + 4800 + f)) / 4).ceil()
        + ((367 * (M - 2 - 12 * f)) / 12).ceil()
        - ((3 * ((Y + 4900 + f) / 100).ceil()) / 4).ceil()
        + D
        - 32075
    )

    hms = (h * 3600 + m * 60 + (s + (ns / 1e9))) / 86400.0
    hms -= 0.5  # julian day starts at noon

    return ymd, hms


def julian_day(dt: DatetimeLike, /, precision: Precision | None = None) -> torch.Tensor:
    """
    Calculate the Julian Day from a Unix Timestamp, datetime64
    """
    ymd, _ = julian_day_time(dt, precision)
    return ymd


def julian_date(dt: DatetimeLike, /, precision: Precision | None = None) -> torch.Tensor:
    """
    Calculate the Julian Date from a Unix Timestamp, datetime64

    The Julian date (JD) of any instant is the Julian day number plus the fraction of a day since
    the preceding noon in Universal Time. Julian dates are expressed as a Julian day number with a
    decimal fraction added.

    Args:
        dt: A tensor of datetime components or a tensor of Unix timestamps
        precision: The precision of the Unix timestamp

    Returns:
        A tensor of Julian Dates
    """
    ymd, hms = julian_day_time(dt, precision)

    return ymd + hms


def julian_century(dt: DatetimeLike, /, precision: Precision | None = None) -> torch.Tensor:
    """
    Calculate the Julian Century from a Unix timestamp.
    """

    JD = julian_date(dt, precision)
    return (JD - 2451545) / 36525
