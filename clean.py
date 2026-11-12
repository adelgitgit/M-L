"""Data-cleaning helpers for tabular time-series data."""

from collections.abc import Hashable
from numbers import Integral

import numpy as np
import pandas as pd


class Cleaner:
    """Apply Kalman filtering or KNN imputation to numeric DataFrame columns."""

    def __init__(self, data: pd.DataFrame):
        if not isinstance(data, pd.DataFrame):
            raise TypeError("data must be a pandas DataFrame")
        self.data = data.copy()

    def make_kalman_filter(
        self,
        process_variance: float = 1e-5,
        measurement_variance: float = 1e-2,
    ) -> pd.DataFrame:
        """Filter numeric columns with a scalar, local-level Kalman filter.

        Missing values after the first observation are filled with the filter's
        prediction. Leading missing values and entirely missing columns remain
        missing because there is no prior observation to estimate them from.
        """
        process_variance = float(process_variance)
        measurement_variance = float(measurement_variance)
        if not np.isfinite(process_variance) or process_variance < 0:
            raise ValueError("process_variance must be a finite, non-negative number")
        if not np.isfinite(measurement_variance) or measurement_variance <= 0:
            raise ValueError("measurement_variance must be a finite, positive number")

        result = self.data.copy()
        numeric_columns = result.select_dtypes(include="number").columns

        for column in numeric_columns:
            observations = result[column].to_numpy(dtype=float, na_value=np.nan)
            if np.isinf(observations).any():
                raise ValueError(f"column {column!r} contains infinite values")
            filtered = np.full(observations.shape, np.nan, dtype=float)
            observed_indices = np.flatnonzero(~np.isnan(observations))
            if observed_indices.size == 0:
                continue

            first = observed_indices[0]
            state = observations[first]
            covariance = measurement_variance
            filtered[first] = state

            for index in range(first + 1, len(observations)):
                covariance += process_variance
                observation = observations[index]
                if not np.isnan(observation):
                    gain = covariance / (covariance + measurement_variance)
                    state += gain * (observation - state)
                    covariance *= 1 - gain
                filtered[index] = state

            result[column] = filtered

        return result

    def make_knn_imputer(
        self,
        n_neighbors: int = 5,
        auto_impute: bool = True,
        bad_row_indices: pd.Index | None = None,
        column: Hashable | None = None,
    ) -> pd.DataFrame:
        """Impute missing numeric values using good rows as KNN neighbors.

        ``bad_row_indices`` contains pandas index labels for rows that should
        not be used as neighbors. When ``column`` is provided, only that column
        is changed at the bad rows; without explicit bad-row indices, its
        missing values select the rows to impute. If ``auto_impute`` is false,
        return a copy unchanged and preserve all missing values.
        """
        if isinstance(n_neighbors, bool) or not isinstance(n_neighbors, Integral):
            raise TypeError("n_neighbors must be a positive integer")
        if n_neighbors < 1:
            raise ValueError("n_neighbors must be a positive integer")
        n_neighbors = int(n_neighbors)
        if not isinstance(auto_impute, bool):
            raise TypeError("auto_impute must be a bool")
        if bad_row_indices is not None and not isinstance(bad_row_indices, pd.Index):
            raise TypeError("bad_row_indices must be a pandas Index or None")
        if bad_row_indices is not None:
            unknown_indices = bad_row_indices[~bad_row_indices.isin(self.data.index)]
            if not unknown_indices.empty:
                raise ValueError(
                    f"bad_row_indices contains labels not present in the data: "
                    f"{unknown_indices.tolist()!r}"
                )
        if column is not None:
            if column not in self.data.columns:
                raise KeyError(f"column {column!r} is not present in the data")
            if column not in self.data.select_dtypes(include="number").columns:
                raise TypeError(f"column {column!r} must be numeric")

        result = self.data.copy()
        if not auto_impute:
            return result

        numeric_columns = result.select_dtypes(include="number").columns
        if len(numeric_columns) == 0:
            return result

        numeric_data = result.loc[:, numeric_columns]
        if np.isinf(numeric_data.to_numpy(dtype=float, na_value=np.nan)).any():
            raise ValueError("numeric columns contain infinite values")
        if not numeric_data.isna().any().any():
            if column is None or bad_row_indices is None:
                return result

        if bad_row_indices is None:
            bad_rows = numeric_data[column].isna().to_numpy()
        else:
            bad_rows = result.index.isin(bad_row_indices)
        good_rows = ~bad_rows
        if column is None and not numeric_data.isna().any().any():
            return result
        if column is not None and not bad_rows.any():
            return result
        if not good_rows.any():
            raise ValueError("KNN imputation requires at least one good row")

        good_numeric_data = numeric_data.loc[good_rows]
        columns_to_impute = good_numeric_data.columns[good_numeric_data.notna().any()]
        if len(columns_to_impute) == 0:
            raise ValueError("KNN imputation requires numeric values in good rows")
        if column is not None and column not in columns_to_impute:
            raise ValueError(f"column {column!r} has no observed values in good rows")

        try:
            from sklearn.impute import KNNImputer
        except ImportError as error:
            raise ImportError(
                "make_knn_imputer requires scikit-learn; install it with "
                "'pip install scikit-learn'"
            ) from error

        imputer = KNNImputer(n_neighbors=n_neighbors)
        imputer.fit(good_numeric_data.loc[:, columns_to_impute])
        query_data = numeric_data.loc[:, columns_to_impute].copy()
        if column is not None:
            column_position = columns_to_impute.get_loc(column)
            query_data.iloc[bad_rows, column_position] = np.nan
            imputed = imputer.transform(query_data)
            result.iloc[bad_rows, result.columns.get_loc(column)] = imputed[
                bad_rows, column_position
            ]
        else:
            imputed = imputer.transform(query_data)
            result.loc[:, columns_to_impute] = imputed
        return result

