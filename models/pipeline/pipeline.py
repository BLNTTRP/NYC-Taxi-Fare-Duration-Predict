#     features = {
#        "PULocationID": int(PULocationID),
#       "DOLocationID": int(DOLocationID),
#      "trip_distance":trip_distance,
#     "pickup_hour": pick_hour,
##     "is_weekend": 0,
#  "pickup_date": pickup_date,
#  "weather": weather
# }
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from datetime import datetime
from sklearn.utils.validation import check_is_fitted
import joblib

class OneHotWeather(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass  # encoder will be initialized in fit()

    def fit(self, X, y=None):
        if "weather" in X.columns:
            self.encoder_ = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
            self.encoder_.fit(X[["weather"]])
        else:
            self.encoder_ = None
        return self

    def transform(self, X):
        check_is_fitted(self, "encoder_")  # <-- required for sklearn compliance
        X = X.copy()

        if self.encoder_ is not None:
            X_weather_encoded = pd.DataFrame(
                self.encoder_.transform(X[["weather"]]),
                index=X.index,
                columns=self.encoder_.get_feature_names_out(["weather"]),
            )
            X = pd.concat([X.drop(columns="weather"), X_weather_encoded], axis=1)

        return X
    
class Num_Scaler(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.num_features = None
        self.scaler_ = StandardScaler()
    
    def fit(self, X, y=None):
        self.num_features = X.select_dtypes(include=["int64", "float64"]).columns
        self.num_features = self.num_features.difference(['PULocationID', 'DOLocationID'])
        self.scaler_.fit(X[self.num_features])
        return self
    
    def transform(self, X, y=None):
        X_copy = X.copy()
        X_copy[self.num_features] = self.scaler_.transform(X_copy[self.num_features])
        return X_copy


class PickupTimeFeatures(BaseEstimator, TransformerMixin):
    """
    Extracts time-based features from pickup datetime.
    Adds hour, day of week, and weekend indicator.
    """

    def fit(self, X, y=None):
        self.is_fitted_ = True
        return self

    def transform(self, X):
        X = X.copy()

        # Basic time-based features
        
        X["pickup_hour"] = pd.to_datetime(datetime.now()).hour
        X["pickup_dayofweek"] = pd.to_datetime(X["pickup_date"]).dt.dayofweek
        X["is_weekend"] = X["pickup_dayofweek"].isin([5, 6]).astype(int)
        X.drop(
            columns=["pickup_date"],
            inplace=True,
            errors="ignore",
        )

        return X




pipeline_weather = Pipeline(
    steps=[
        ("pickup_time_features", PickupTimeFeatures()),
        ("num_scaler", Num_Scaler()),
        ("weather", OneHotWeather())
    ]
)
pipeline_no_weather = Pipeline(
    steps=[
        ("pickup_time_features", PickupTimeFeatures()),
        ("num_scaler", Num_Scaler())
    ]
)

class PipelineCreator:
    """
    Creates a pipeline for processing taxi trip data.
    The pipeline includes feature extraction, scaling, and encoding.
    """
    def __init__(self):
        self.features = ["PULocationID", "DOLocationID", "trip_distance", "pickup_date", "weather"]

    def create_pipeline(self, df):
        new_data = df[self.features].copy()
        pipel_weather_fitted = pipeline_weather.fit(new_data)
        pipel_no_weather_fitted = pipeline_no_weather.fit(new_data)
        joblib.dump(pipel_weather_fitted, "trained_pipeline_weather.pkl")
        joblib.dump(pipel_no_weather_fitted, "trained_pipeline_no_weather.pkl")