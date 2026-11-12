"""Fit and apply standard scaling to forecasting CSV features."""

from os import PathLike

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


class DataNormalizer:
    """Load a CSV and scale features while keeping the target separate."""

    def __init__(
        self,
        csv_path: str | PathLike,
        target_column: str,
        index_col: int | str | None = None,
        parse_dates: bool = False,
    ):
        self.data = pd.read_csv(
            csv_path,
            index_col=index_col,
            parse_dates=parse_dates,
        )
        self.target_column = target_column
        self.feature_scaler = StandardScaler()
        self.feature_columns: pd.Index | None = None

        if not self.data.columns.is_unique:
            raise ValueError("CSV must have unique column names")
        self._split_target(self.data)

    def _split_target(self, data: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
        if not isinstance(data, pd.DataFrame):
            raise TypeError("data must be a pandas DataFrame")
        if not data.columns.is_unique:
            raise ValueError("data must have unique column names")
        if self.target_column not in data.columns:
            raise KeyError(f"target column {self.target_column!r} is not in the data")

        features = data.drop(columns=self.target_column)
        target = data[self.target_column]
        if features.shape[1] == 0:
            raise ValueError(
                "data must contain at least one feature besides the target"
            )
        if not all(pd.api.types.is_numeric_dtype(dtype) for dtype in features.dtypes):
            raise TypeError("all feature columns must be numeric")
        if not pd.api.types.is_numeric_dtype(target.dtype):
            raise TypeError("target column must be numeric")

        values = data.to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError("data must not contain missing or infinite values")
        return features, target

    def fit_transform(
        self, train_data: pd.DataFrame | None = None
    ) -> tuple[pd.DataFrame, pd.Series]:
        """Fit on training features, then return scaled features and target.

        When ``train_data`` is omitted, the CSV loaded by the constructor is
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
