"""Fit and apply standard scaling to forecasting CSV features."""

from __future__ import annotations

from collections.abc import Hashable
from os import PathLike

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


class DataNormalizer:
    """Load a CSV or DataFrame and scale features while keeping the target separate."""

    def __init__(
        self,
        csv_path: str | PathLike | pd.DataFrame,
        target_column: Hashable,
        index_col: int | str | None = None,
        parse_dates: bool = False,
        numeric_only: bool = True,
    ):
        if isinstance(csv_path, pd.DataFrame):
            self.data = csv_path.copy()
        else:
            self.data = pd.read_csv(
                csv_path,
                index_col=index_col,
                parse_dates=parse_dates,
            )
        self.target_column = target_column
        self.numeric_only = numeric_only
        self.feature_scaler = StandardScaler()
        self.feature_columns: pd.Index | None = None

        if not self.data.columns.is_unique:
            raise ValueError("CSV must have unique column names")
        self._split_target(self.data)

    def _resolve_target_column(self, data: pd.DataFrame) -> Hashable:
        if self.target_column in data.columns:
            return self.target_column
        if str(self.target_column) in data.columns:
            return str(self.target_column)
        if (
            isinstance(self.target_column, str)
            and self.target_column.isdigit()
            and int(self.target_column) in data.columns
        ):
            return int(self.target_column)
        raise KeyError(f"target column {self.target_column!r} is not in the data")

    def _split_target(self, data: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
        if not isinstance(data, pd.DataFrame):
            raise TypeError("data must be a pandas DataFrame")
        if not data.columns.is_unique:
            raise ValueError("data must have unique column names")

        target_col = self._resolve_target_column(data)
        target = data[target_col]
        if not pd.api.types.is_numeric_dtype(target.dtype):
            raise TypeError("target column must be numeric")

        features = data.drop(columns=target_col)
        if self.numeric_only:
            features = features.select_dtypes(include="number")
        else:
            if not all(pd.api.types.is_numeric_dtype(dtype) for dtype in features.dtypes):
                raise TypeError("all feature columns must be numeric")

        if features.shape[1] == 0:
            raise ValueError(
                "data must contain at least one feature besides the target"
            )

        target_arr = target.to_numpy(dtype=float)
        if not np.isfinite(target_arr).all():
            raise ValueError("target column must not contain missing or infinite values")

        features_arr = features.to_numpy(dtype=float)
        if not np.isfinite(features_arr).all():
            raise ValueError("feature columns must not contain missing or infinite values")

        return features, target

    def fit_transform(
        self, train_data: pd.DataFrame | None = None
    ) -> tuple[pd.DataFrame, pd.Series]:
        """Fit on training features, then return scaled features and target.

        When ``train_data`` is omitted, the data loaded in the constructor is
        used as training data.
        """
        data = self.data if train_data is None else train_data
        features, target = self._split_target(data)
        self.feature_scaler = StandardScaler()
        scaled = self.feature_scaler.fit_transform(features)
        self.feature_columns = features.columns.copy()

        X = pd.DataFrame(scaled, index=features.index, columns=features.columns)
        return X, target.copy()

    def transform(self, data: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
        """Scale new data with the training scaler and return its target."""
        if self.feature_columns is None:
            raise RuntimeError("call fit_transform with training data before transform")

        features, target = self._split_target(data)
        if not features.columns.equals(self.feature_columns):
            raise ValueError(
                "data feature columns must match the fitted training columns"
            )

        scaled = self.feature_scaler.transform(features)
        X = pd.DataFrame(scaled, index=features.index, columns=features.columns)
        return X, target.copy()
