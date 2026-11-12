"""Calendar and cyclic time feature engineering for pandas DataFrames."""

from datetime import datetime, time, timedelta

import numpy as np
import pandas as pd


class FeatureEngineer:
    """Create day-of-week and time-of-day features from tabular data."""

    SECONDS_PER_DAY = 24 * 60 * 60

    def __init__(self, data: pd.DataFrame):
        if not isinstance(data, pd.DataFrame):
            raise TypeError("data must be a pandas DataFrame")
        self.data = data.copy()

    def find_day_column(self) -> str:
        """Find the unique column whose name contains 'day'."""
        columns = [
            column
            for column in self.data.columns
            if isinstance(column, str) and "day" in column.lower()
        ]
        if not columns:
            raise ValueError("no column with 'day' in its name was found")
        if len(columns) > 1:
            raise ValueError(f"multiple day columns found: {columns!r}")
        return columns[0]

    def add_day_one_hot(
        self,
        column: str | None = None,
        drop_original: bool = False,
    ) -> pd.DataFrame:
        """Append one-hot columns for a day column, preserving the row index."""
        if not isinstance(drop_original, bool):
            raise TypeError("drop_original must be a bool")
        column = self.find_day_column() if column is None else column
        if column not in self.data.columns:
            raise KeyError(f"day column {column!r} is not in the data")

        result = self.data.copy()
        encoded = pd.get_dummies(
            result[column],
            prefix=str(column),
            prefix_sep="_",
            dummy_na=False,
            dtype=int,
        )
        overlapping_columns = encoded.columns.intersection(result.columns)
        if not overlapping_columns.empty:
            raise ValueError(
                f"one-hot output columns already exist: "
                f"{overlapping_columns.tolist()!r}"
            )
        result = pd.concat([result, encoded], axis=1)
        if drop_original:
            result = result.drop(columns=column)
        return result

    def _time_seconds(self, time_column: str | None) -> pd.Series:
        if time_column is None:
            if isinstance(self.data.index, pd.DatetimeIndex):
                values = pd.Series(self.data.index, index=self.data.index)
            else:
                candidates = [
                    column
                    for column in self.data.columns
                    if isinstance(column, str)
                    and any(
                        name in column.lower()
                        for name in ("time", "timestamp", "datetime")
                    )
                ]
                if len(candidates) != 1:
                    raise ValueError(
                        "specify time_column, or provide a DatetimeIndex or "
                        "exactly one time-like column"
                    )
                values = self.data[candidates[0]]
        else:
            if time_column not in self.data.columns:
                raise KeyError(f"time column {time_column!r} is not in the data")
            values = self.data[time_column]

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
        time_column: str | None = None,
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
        time_column: str | None = None,
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
