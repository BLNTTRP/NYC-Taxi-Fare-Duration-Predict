import pandas as pd
import numpy as np
from scipy.spatial.distance import mahalanobis
from scipy.stats import chi2
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.cluster import DBSCAN
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer
from sklearn.base import BaseEstimator, TransformerMixin


class TimedeltaConverter(BaseEstimator, TransformerMixin):
    """Converts timedelta columns to seconds and passes through other columns."""

    def fit(self, X, y=None):
        self.feature_names_in_ = X.columns
        return self

    def transform(self, X):
        X = X.copy()
        for col in X.columns:
            if np.issubdtype(X[col].dtype, np.timedelta64):
                X[col] = X[col].dt.total_seconds()
        return X

    def get_feature_names_out(self, input_features=None):
        return self.feature_names_in_


class DynamicPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self, targets_num=None, targets_date=None):
        self.targets_num = targets_num or []
        self.targets_date = targets_date or []
        self.preprocessor = None

    def fit(self, X, y=None):
        X = X.copy()

        # Normalize missing values in store_and_fwd_flag
        if "store_and_fwd_flag" in X.columns:
            X["store_and_fwd_flag"] = X["store_and_fwd_flag"].fillna("N")

        # Explicitly define what to treat as categorical
        cat_cols = [
            col
            for col in X.columns
            if (X[col].dtype == "object" or X[col].dtype.name == "category")
            and col not in self.targets_num + self.targets_date
        ]

        # For numerical columns, exclude targets and dates
        num_cols = [
            col
            for col in X.select_dtypes(include=np.number).columns
            if col not in self.targets_num + self.targets_date
            and col not in ["PULocationID", "DOLocationID"]
        ]

        passthrough_cols = self.targets_num + self.targets_date

        for loc_col in ["PULocationID", "DOLocationID", "weather"]:
            if loc_col in X.columns:
                passthrough_cols.append(loc_col)
                if loc_col in cat_cols:
                    # Remove from categorical if already present (for weather)
                    cat_cols.remove(loc_col)

        self.cat_cols_ = cat_cols
        self.num_cols_ = num_cols
        self.passthrough_cols_ = passthrough_cols

        self.preprocessor = ColumnTransformer(
            transformers=[
                (
                    "cat",
                    OneHotEncoder(
                        handle_unknown="ignore", sparse_output=False, drop="if_binary"
                    ),
                    self.cat_cols_,
                ),
                (
                    "num",
                    Pipeline(
                        [
                            ("timedelta_to_seconds", TimedeltaConverter()),
                            ("scaler", StandardScaler()),
                        ]
                    ),
                    self.num_cols_,
                ),
                ("pass", "passthrough", self.passthrough_cols_),
            ],
            remainder="drop",
            verbose_feature_names_out=False,  # Simplified output
        )

        self.preprocessor.fit(X)
        return self

    def transform(self, X):
        X_transformed = self.preprocessor.transform(X)

        if hasattr(self.preprocessor, "get_feature_names_out"):
            feature_names = self.preprocessor.get_feature_names_out()
        else:
            feature_names = self.cat_cols_ + self.num_cols_ + self.passthrough_cols_

        # Use infer_objects to restore numeric types where possible
        df = pd.DataFrame(X_transformed, columns=feature_names, index=X.index)
        df = pd.DataFrame(X_transformed, columns=feature_names, index=X.index)
        df = df.infer_objects()
        # Round only float columns to 4 decimals (or as needed)
        float_cols = df.select_dtypes(include=["float"]).columns
        df[float_cols] = df[float_cols].round(4)
        return df


class TripDurationAdder(BaseEstimator, TransformerMixin):
    def __init__(self, add_total_seconds=True):
        self.add_total_seconds = add_total_seconds

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()

        X["trip_duration"] = X["tpep_dropoff_datetime"] - X["tpep_pickup_datetime"]
        if self.add_total_seconds:
            X["trip_time_seconds"] = X["trip_duration"].dt.total_seconds()
        return X


class PickupTimeFeatures(BaseEstimator, TransformerMixin):
    """
    Extracts time-based features from pickup datetime.
    Adds hour, day of week, and weekend indicator.
    """

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()

        # Basic time-based features
        X["pickup_hour"] = X["tpep_pickup_datetime"].dt.hour
        X["pickup_dayofweek"] = X["tpep_pickup_datetime"].dt.dayofweek
        X["is_weekend"] = X["pickup_dayofweek"].isin([5, 6]).astype(int)

        return X


class OutlierCleaner(BaseEstimator, TransformerMixin):

    def __init__(
        self,
        targets_num=["fare_amount", "trip_time_seconds"],
        targets_cat=["PULocationID", "DOLocationID"],
        targets_date=["tpep_pickup_datetime", "tpep_dropoff_datetime"],
        epsilon_barrier=10**-8,
        mahalanobis_threshold=0.98,
        dbscan_eps=0.05,
        dbscan_min_samples=30,
    ):
        self.targets_num = targets_num
        self.targets_cat = targets_cat
        self.targets_date = targets_date
        self.epsilon_barrier = epsilon_barrier
        self.mahalanobis_threshold = mahalanobis_threshold
        self.dbscan_eps = dbscan_eps
        self.dbscan_min_samples = dbscan_min_samples

    def fit(self, X, y=None):
        self.scaler_ = StandardScaler()

        # Safer selection of existing numeric target columns
        X_num = X[self.targets_num].apply(pd.to_numeric, errors="coerce")

        if X_num.empty:
            raise ValueError(
                f"No numeric target columns found in input: expected {self.targets_num}"
            )
        X_num_log = np.log(X_num + self.epsilon_barrier)

        # Drop rows with NaNs after log
        before_drop = len(X_num_log)
        X_num_log = X_num_log.dropna()
        after_drop = len(X_num_log)

        X_scaled = self.scaler_.fit_transform(X_num_log)

        self.mean_ = np.mean(X_scaled, axis=0)
        self.cov_ = np.cov(X_scaled, rowvar=False)
        self.cov_inv_ = np.linalg.pinv(self.cov_)
        self.dof_ = X_scaled.shape[1]
        self.cutoff_ = chi2.ppf(self.mahalanobis_threshold, df=self.dof_)
        return self

    def _mahalanobis_mask(self, X_scaled):
        diff = X_scaled - self.mean_
        left = np.dot(diff, self.cov_inv_)
        mdist = np.sqrt(np.sum(left * diff, axis=1))
        return mdist < np.sqrt(self.cutoff_)

    def _dbscan_mask(self, X_scaled):
        dbscan = DBSCAN(eps=self.dbscan_eps, min_samples=self.dbscan_min_samples)
        labels = dbscan.fit_predict(X_scaled)
        return labels != -1

    def transform(self, X, y=None):
        X_clean = X.copy()

        # === Mahalanobis filtering ===
        X_num = X_clean[self.targets_num].apply(pd.to_numeric, errors="coerce")

        # Mask valid values: strictly positive and not NaN
        valid_mask = (X_num > 0).all(axis=1) & X_num.notna().all(axis=1)
        X_num_valid = X_num[valid_mask]

        if X_num_valid.empty:
            print("[Warning] No valid rows for Mahalanobis filtering.")
            return X_clean.reset_index(drop=True)

        with np.errstate(invalid="ignore"):
            X_num_log = np.log(X_num_valid + self.epsilon_barrier)

        X_scaled = self.scaler_.transform(X_num_log)
        mask_mahal = self._mahalanobis_mask(X_scaled)

        # === Reconstruct values ===
        X_scaled_inv = pd.DataFrame(
            self.scaler_.inverse_transform(X_scaled),
            index=X_num_log.index,
            columns=self.targets_num,
        )
        X_original_scale = np.exp(X_scaled_inv) - self.epsilon_barrier
        X_clean.loc[X_num_log.index, self.targets_num] = X_original_scale

        # === DBSCAN filtering ===
        X_date = pd.to_datetime(X_clean[self.targets_date[0]]) - pd.to_datetime(
            X_clean[self.targets_date[1]]
        )
        X_clean["duration_secs"] = X_date.dt.total_seconds().fillna(0)

        columns_to_use = self.targets_num + ["duration_secs"]
        X_dbscan = X_clean[columns_to_use].select_dtypes(include=[np.number])

        if X_dbscan.empty:
            mask_dbscan = pd.Series(True, index=X_clean.index)
        else:
            scaler = StandardScaler()
            X_dbscan_scaled = scaler.fit_transform(X_dbscan)
            mask_dbscan_raw = self._dbscan_mask(X_dbscan_scaled)
            mask_dbscan = pd.Series(mask_dbscan_raw, index=X_dbscan.index)

        # === Final mask ===
        valid_idx = X_num_log.index.intersection(mask_dbscan.index)
        final_mask = pd.Series(False, index=X_clean.index)
        final_mask.loc[valid_idx] = mask_mahal & mask_dbscan.loc[valid_idx]

        if final_mask.sum() == 0:
            print("[Warning] All rows dropped — skipping OutlierCleaner.")
            return X_clean.reset_index(drop=True)

        X_clean.drop(columns=["duration_secs"], inplace=True, errors="ignore")
        return X_clean.loc[final_mask]


class SI_personalized(SimpleImputer):
    """
    Custom imputer that skips datetime columns.
    Only imputes non-datetime columns using the specified strategy.
    """

    def __init__(self, strategy="most_frequent", fill_value=None, copy=True):
        super().__init__(strategy=strategy, fill_value=fill_value, copy=copy)

    def fit(self, X, y=None):
        # Save column info
        self.datetime_cols_ = X.select_dtypes(include=["datetime64[ns]"]).columns
        self.non_datetime_cols_ = X.columns.difference(self.datetime_cols_)

        # Fit only on non-datetime columns
        return super().fit(X[self.non_datetime_cols_], y)

    def transform(self, X):
        # Impute only non-datetime columns
        transformed = super().transform(X[self.non_datetime_cols_])
        imputed_df = pd.DataFrame(
            transformed,
            columns=self.non_datetime_cols_,
            index=X.index,
        ).infer_objects()

        # Add back the datetime columns in original order
        return pd.concat([imputed_df, X[self.datetime_cols_]], axis=1)[X.columns]


def build_pipeline():
    return Pipeline(
        steps=[
            ("imputer", SI_personalized()),
            ("trip_duration_adder", TripDurationAdder()),
            ("extract_hour", PickupTimeFeatures()),
            ("outlier_cleaner", OutlierCleaner()),
            (
                "dynamic_preprocessor",
                DynamicPreprocessor(
                    targets_num=["fare_amount", "trip_time_seconds"],
                    targets_date=[
                        "tpep_pickup_datetime",
                        "tpep_dropoff_datetime",
                        "trip_duration",
                    ],
                ),
            ),
        ]
    )


# Wrapper function to run the pipeline
def run_pipeline(df: pd.DataFrame) -> pd.DataFrame:
    """
    Runs the preprocessing pipeline on a raw taxi trip DataFrame.
    Returns the cleaned DataFrame with engineered features.
    """
    pipeline = build_pipeline()
    return pipeline.fit_transform(df)
