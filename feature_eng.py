"""Calendar and cyclic time feature engineering for pandas DataFrames."""

from __future__ import annotations

from collections.abc import Hashable
from datetime import datetime, time, timedelta

import numpy as np
import pandas as pd


class FeatureEngineer:
    """Create day-of-week and time-of-day features from tabular data."""

    SECONDS_PER_DAY = 24 * 60 * 60
    DAYS_OF_WEEK = (
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
        "Sunday",
    )
    RESERVED_TIME_NAMES = frozenset(
        {
            "time_fraction",
            "time_sin",
            "time_cos",
            "time period",
            "month",
            "day",
        }
    )

    def __init__(self, data: pd.DataFrame):
        if not isinstance(data, pd.DataFrame):
            raise TypeError("data must be a pandas DataFrame")
        self.data = data.copy()

    def _resolve_column_name(self, column: Hashable) -> Hashable:
        if column in self.data.columns:
            return column
        if str(column) in self.data.columns:
            return str(column)
        if isinstance(column, str) and column.isdigit() and int(column) in self.data.columns:
            return int(column)
        raise KeyError(f"column {column!r} is not in the data")

    def find_day_column(self) -> str:
        """Find the unique column whose name contains 'day'."""
        # 1. Exact match check
        for exact in ("day", "day_name", "day_of_week", "weekday"):
            matches = [
                col
                for col in self.data.columns
                if isinstance(col, str) and col.lower() == exact
            ]
            if matches:
                return matches[0]

        # 2. Substring match, excluding already generated one-hot day columns
        day_prefixes = tuple(f"day_{d.lower()}" for d in self.DAYS_OF_WEEK)
        columns = [
            column
            for column in self.data.columns
            if isinstance(column, str)
            and "day" in column.lower()
            and not column.lower().startswith(day_prefixes)
            and not column.lower().startswith("day_")
        ]
        if not columns:
            raise ValueError("no column with 'day' in its name was found")
        if len(columns) > 1:
            raise ValueError(f"multiple day columns found: {columns!r}")
        return columns[0]

    def add_day_one_hot(
        self,
        column: Hashable | None = None,
        drop_original: bool = False,
    ) -> pd.DataFrame:
        """Append one-hot columns for a day column, preserving the row index."""
        if not isinstance(drop_original, bool):
            raise TypeError("drop_original must be a bool")

        original_column = None
        prefix = "day"

        if column is None:
            try:
                found_col = self.find_day_column()
                original_column = found_col
                prefix = str(found_col)
                day_source = self.data[found_col]
            except ValueError:
                # No day column found: derive from DatetimeIndex or datetime index
                dt_index = None
                if isinstance(self.data.index, pd.DatetimeIndex):
                    dt_index = self.data.index
                elif not self.data.empty:
                    try:
                        dt_index = pd.to_datetime(self.data.index, errors="raise")
                    except Exception:
                        dt_index = None

                if dt_index is not None and isinstance(dt_index, pd.DatetimeIndex):
                    day_source = pd.Series(dt_index.day_name(), index=self.data.index)
                    prefix = "day"
                else:
                    raise ValueError(
                        "no column with 'day' in its name was found and index is not datetime-like"
                    )
        else:
            resolved = self._resolve_column_name(column)
            original_column = resolved
            prefix = str(resolved)
            source_col = self.data[resolved]

            if pd.api.types.is_datetime64_any_dtype(source_col):
                day_source = source_col.dt.day_name()
            else:
                day_source = source_col

        # If source has standard day names, enforce all 7 days for split consistency
        sample_non_na = day_source.dropna()
        if not sample_non_na.empty and sample_non_na.astype(str).isin(self.DAYS_OF_WEEK).all():
            cat_source = pd.Categorical(day_source, categories=list(self.DAYS_OF_WEEK))
            encoded = pd.get_dummies(
                cat_source,
                prefix=prefix,
                prefix_sep="_",
                dummy_na=False,
                dtype=int,
            )
            encoded.index = self.data.index
        else:
            encoded = pd.get_dummies(
                day_source,
                prefix=prefix,
                prefix_sep="_",
                dummy_na=False,
                dtype=int,
            )

        overlapping_columns = encoded.columns.intersection(self.data.columns)
        if not overlapping_columns.empty:
            raise ValueError(
                f"one-hot output columns already exist: "
                f"{overlapping_columns.tolist()!r}"
            )

        result = pd.concat([self.data.copy(), encoded], axis=1)
        if drop_original and original_column is not None and original_column in result.columns:
            result = result.drop(columns=original_column)
        return result

    def _time_seconds(self, time_column: Hashable | None) -> pd.Series:
        if time_column is None:
            if isinstance(self.data.index, pd.DatetimeIndex):
                values = pd.Series(self.data.index, index=self.data.index)
            else:
                dt_index = None
                if not self.data.empty:
                    try:
                        dt_index = pd.to_datetime(self.data.index, errors="raise")
                    except Exception:
                        dt_index = None

                if dt_index is not None and isinstance(dt_index, pd.DatetimeIndex):
                    values = pd.Series(dt_index, index=self.data.index)
                else:
                    candidates = [
                        column
                        for column in self.data.columns
                        if isinstance(column, str)
                        and column.lower() not in self.RESERVED_TIME_NAMES
                        and not column.lower().startswith("day_")
                        and (
                            pd.api.types.is_datetime64_any_dtype(self.data[column])
                            or any(
                                name in column.lower()
                                for name in ("timestamp", "datetime", "time")
                            )
                            or column == "Unnamed: 0"
                        )
                    ]
                    # Filter candidates to ensure they contain datetime or time values
                    valid_candidates = []
                    for cand in candidates:
                        cand_s = self.data[cand]
                        if pd.api.types.is_datetime64_any_dtype(cand_s):
                            valid_candidates.append(cand)
                        else:
                            try:
                                pd.to_datetime(cand_s.iloc[:5])
                                valid_candidates.append(cand)
                            except Exception:
                                pass

                    if len(valid_candidates) != 1:
                        raise ValueError(
                            "specify time_column, or provide a DatetimeIndex or "
                            "exactly one time-like column"
                        )
                    values = self.data[valid_candidates[0]]
        else:
            resolved = self._resolve_column_name(time_column)
            values = self.data[resolved]

        # Vectorized calculation for DatetimeIndex or datetime64 series
        if isinstance(values, pd.DatetimeIndex):
            return pd.Series(
                (
                    values.hour * 3600
                    + values.minute * 60
                    + values.second
                    + values.microsecond / 1_000_000
                ).astype(float),
                index=self.data.index,
            )
        if pd.api.types.is_datetime64_any_dtype(values):
            return (
                values.dt.hour * 3600
                + values.dt.minute * 60
                + values.dt.second
                + values.dt.microsecond / 1_000_000
            ).astype(float)

        def seconds(value: object) -> float:
            if pd.isna(value):
                return np.nan
            if isinstance(value, (pd.Timestamp, datetime, np.datetime64)):
                timestamp = pd.Timestamp(value)
                return (
                    timestamp.hour * 3600
                    + timestamp.minute * 60
                    + timestamp.second
                    + timestamp.microsecond / 1_000_000
                )
            if isinstance(value, time):
                return (
                    value.hour * 3600
                    + value.minute * 60
                    + value.second
                    + value.microsecond / 1_000_000
                )
            if isinstance(value, (pd.Timedelta, timedelta, np.timedelta64)):
                return pd.Timedelta(value).total_seconds() % self.SECONDS_PER_DAY
            if isinstance(value, str):
                try:
                    timestamp = pd.Timestamp(value)
                except (TypeError, ValueError):
                    try:
                        return (
                            pd.to_timedelta(value).total_seconds()
                            % self.SECONDS_PER_DAY
                        )
                    except (TypeError, ValueError):
                        pass
                else:
                    return (
                        timestamp.hour * 3600
                        + timestamp.minute * 60
                        + timestamp.second
                        + timestamp.microsecond / 1_000_000
                    )
            raise TypeError(
                f"unsupported time value {value!r}; use a datetime, time, "
                "timedelta, or parseable time string"
            )

        return values.map(seconds).astype(float)

    def add_time_fraction(
        self,
        time_column: Hashable | None = None,
        output_column: str = "time_fraction",
    ) -> pd.DataFrame:
        """Append seconds since midnight divided by the total seconds in a day."""
        if output_column in self.data.columns:
            raise ValueError(f"output column {output_column!r} already exists")

        result = self.data.copy()
        result[output_column] = (
            self._time_seconds(time_column) / self.SECONDS_PER_DAY
        ).to_numpy()
        return result

    def add_cyclic_time(
        self,
        time_column: Hashable | None = None,
        sin_column: str = "time_sin",
        cos_column: str = "time_cos",
    ) -> pd.DataFrame:
        """Append sine and cosine encodings of time-of-day."""
        if sin_column == cos_column:
            raise ValueError("sin_column and cos_column must be different")
        if sin_column in self.data.columns or cos_column in self.data.columns:
            raise ValueError("a time encoding output column already exists")

        angle = 2 * np.pi * self._time_seconds(time_column) / self.SECONDS_PER_DAY
        result = self.data.copy()
        result[sin_column] = np.sin(angle).to_numpy()
        result[cos_column] = np.cos(angle).to_numpy()
        return result

