import numpy as np
import warnings
import pandas as pd
from pipeline.pipeline_preproc import run_pipeline
from IPython.display import display

from sklearn.metrics import (
    mean_absolute_error,
    median_absolute_error,
    root_mean_squared_error,
    mean_squared_log_error,
    r2_score,
    explained_variance_score,
)

from sklearn.preprocessing import OneHotEncoder, PowerTransformer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.multioutput import MultiOutputRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor


def run_pipeline_chunks(df_raw, chunk_size=50_000):
    """
    Run the data processing pipeline on the raw DataFrame.
    Args:
        df_raw (pd.DataFrame): Raw DataFrame containing the data to be processed.
        chunk_size (int): Size of each chunk to process.
    Returns:
        pd.DataFrame: Cleaned DataFrame after processing.
    """
    chunk_size = chunk_size if chunk_size > 0 else 50_000
    df_clean = pd.DataFrame()
    print(f"Processing data in chunks of size {chunk_size}...")
    for start in range(0, len(df_raw), chunk_size):
        try:
            chunk = df_raw.iloc[start : start + chunk_size]

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                warnings.simplefilter("ignore", UserWarning)  # sklearn, pandas, etc.

                df_clean_chunk = run_pipeline(chunk)

            df_clean = pd.concat([df_clean, df_clean_chunk], ignore_index=False)

        except Exception as e:
            print(f"[✗] Error in chunk {start}-{start + chunk_size}: {e}")
            continue
    print(f"\n[✓] Processed {len(df_clean)} rows successfully.")
    print(f"Final df_clean shape: {df_clean.shape}")
    return df_clean


def check_metadata_and_missing_values(df_raw, df_zone):
    """
    Check metadata consistency and count missing values in the DataFrame.
    Args:
        df_raw (pd.DataFrame): Raw DataFrame containing the data to be checked.
        df_zone (pd.DataFrame): DataFrame containing zone metadata.
    """
    # Zones in metadata but not in May data
    missing_PUids = set(df_zone["LocationID"]) - set(df_raw["PULocationID"])
    print("Missing PULocationIDs:", list(missing_PUids))
    missing_DOids = set(df_zone["LocationID"]) - set(df_raw["DOLocationID"])
    print("Missing DOLocationIDs:", list(missing_DOids), end="\n\n")
    # Count missing values
    dfs = {
        "df_raw": df_raw,
        "df_zone": df_zone,
    }
    for name, df in dfs.items():
        if not isinstance(df, pd.DataFrame):
            print(f"{name} is not a DataFrame.")
            continue
        if df.empty:
            print(f"{name} is empty.")
            continue
        print(f"Missing values in {name}:")
        col_null = [column for column in df.columns if df[column].isnull().sum() > 0]
        print(df[col_null].isnull().sum(), end="\n\n")


def assign_tertiles(df_clean, df_zone, ID_columns=["PULocationID", "DOLocationID"]):
    result_dict = {}

    for col in ID_columns:
        print(f"\n=== 📌 {col} Summary ===")

        counts = (
            df_clean[col]
            .value_counts()
            .rename_axis("LocationID")
            .reset_index(name="trip_count")
        )

        merged_df = pd.merge(counts, df_zone, on="LocationID", how="left")

        tertiles = merged_df["trip_count"].quantile([1 / 3, 2 / 3])

        def assign_tertile(x):
            if x <= tertiles[1 / 3]:
                return "T1"
            elif x <= tertiles[2 / 3]:
                return "T2"
            else:
                return "T3"

        merged_df["tertile"] = merged_df["trip_count"].apply(assign_tertile)

        # Separate displays
        thresholds_df = tertiles.round(0).astype(int).to_frame(name="Thresholds")
        top5_df = counts.head(5).rename(columns={"trip_count": "Top 5 Trip Count"})
        bottom5_df = counts.tail(5).rename(
            columns={"trip_count": "Bottom 5 Trip Count"}
        )

        print("• Tertile thresholds:")
        display(thresholds_df)

        print("• Top 5 zones by trip count:")
        display(pd.merge(top5_df, df_zone, on="LocationID", how="left"))

        print("• Bottom 5 zones by trip count:")
        display(pd.merge(bottom5_df, df_zone, on="LocationID", how="left"))

        result_dict[col] = merged_df

    return result_dict


def compute_borough_distributions(df, df_zone, label=""):
    """Computes the distribution of trips by borough for pickup and dropoff locations.
    Args:
        df (DataFrame): DataFrame containing 'PULocationID' and 'DOLocationID'.
        df_zone (DataFrame): DataFrame containing 'LocationID' and 'Borough'.
        label (str): Optional label for the distribution.
    Returns:
        DataFrame: Distribution of trips by borough for pickup and dropoff locations.
    """
    df = df.copy()
    df_zone = df_zone[["LocationID", "Borough"]]

    df = df.merge(
        df_zone.rename(columns={"LocationID": "PULocationID", "Borough": "PU_Borough"}),
        on="PULocationID",
        how="left",
    )
    df = df.merge(
        df_zone.rename(columns={"LocationID": "DOLocationID", "Borough": "DO_Borough"}),
        on="DOLocationID",
        how="left",
    )

    borough_dist = pd.concat(
        [
            df["PU_Borough"].value_counts(normalize=True).rename("Pickup"),
            df["DO_Borough"].value_counts(normalize=True).rename("Dropoff"),
        ],
        axis=1,
    ).fillna(0)
    borough_dist.index.name = "Borough"
    return borough_dist


def setup_data_for_modeling(df_clean, features=None, targets=None, tranf_method=None):
    """Prepare the data for modeling by loading the cleaned trip data,
    defining features and targets, and splitting into train/test sets.
    Args:
        df_clean (pd.DataFrame): Cleaned DataFrame containing the data to be used for modeling.
        features (list): List of feature column names to be used in the model.
        targets (list): List of target column names to be predicted.
        tranf_method (str): Transformation method to apply to the target variables.
    """
    # Prepare input (X) and output (y) matrices
    X = df_clean[features].copy()
    y = df_clean[targets].copy()
    y = y.round(1)  # Round target values to 1 decimal places

    # Mark categorical features (only works with XGBoost >= 1.6)
    for col in ["PULocationID", "DOLocationID"]:
        if col in features:
            X[col] = X[col].astype("category")

    # One-hot encode the weather feature
    if "weather" in features:
        encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
        X_weather_encoded = pd.DataFrame(
            encoder.fit_transform(X[["weather"]]),
            index=X.index,
            columns=encoder.get_feature_names_out(["weather"]),
        )
        # Combine the encoded weather with the rest of the features
        X = pd.concat([X.drop(columns="weather"), X_weather_encoded], axis=1)

    if tranf_method is None:
        # No transformation, use raw
        pass
    elif tranf_method == "log":
        # Apply log transformation to fare_amount and trip_time_seconds
        y = np.log1p(y)
    elif tranf_method == "box-cox" or tranf_method == "yeo-johnson":
        # Apply Box-Cox or Yeo-Johnson transformation
        # Note: Box-Cox requires positive values, Yeo-Johnson can handle zeros and negatives
        if tranf_method == "box-cox":
            assert (y > 0).all().all(), "Box-Cox requires positive values"
        pt = PowerTransformer(method=tranf_method, standardize=False)
        y = pd.DataFrame(pt.fit_transform(y), columns=targets, index=y.index)
    else:
        raise ValueError(
            "Unsupported transformation method. Use 'log', 'box-cox', or 'yeo-johnson'."
        )

    # Divide in train and test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    return X_train, X_test, y_train, y_test


def evaluate_model(
    y_true, y_pred, target_names=["Fare Amount ($)", "Trip Duration (s)"]
):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    assert y_true.shape == y_pred.shape, "Shapes of y_true and y_pred must match"

    n_targets = y_true.shape[1]

    for i in range(n_targets):
        true = y_true[:, i]
        pred = y_pred[:, i]
        name = target_names[i]

        mae = mean_absolute_error(true, pred)
        rmse = root_mean_squared_error(true, pred)
        medae = median_absolute_error(true, pred)
        evs = explained_variance_score(true, pred)

        try:
            msle = mean_squared_log_error(true, pred)
        except ValueError:
            msle = np.nan

        r2 = r2_score(true, pred)
        residuals = true - pred

        print(f"\n--- {name} ---")
        print(f"MAE:   {mae:.4f}")
        print(f"MedAE: {medae:.4f}")
        print(f"RMSE:  {rmse:.4f}")
        print(f"MSLE:  {msle:.4f}")
        print(f"R²:    {r2:.4f}")
        print(f"Explained Variance: {evs:.4f}")


import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor


def report_overfitting_gap(
    pred_train,
    pred_test,
    y_train,
    y_test,
    retrain=False,
    params=None,
    X_train=None,
    X_test=None,
):
    """
    Report RMSE gap (overfitting) per target and optionally plot per-iteration RMSE if retrain=True.
    Assumes model is MultiOutputRegressor(XGBRegressor).
    Args:
        pred_train (np.ndarray): Predictions on training set.
        pred_test (np.ndarray): Predictions on test set.
        y_train (pd.DataFrame): True values for training set.
        y_test (pd.DataFrame): True values for test set.
        retrain (bool): Whether to retrain models to get per-iteration RMSE.
    """
    target_names = y_train.columns.tolist()
    print("Overfitting Report per Target:\n")

    for i, name in enumerate(target_names):
        rmse_train = np.sqrt(mean_squared_error(y_train.iloc[:, i], pred_train[:, i]))
        rmse_test = np.sqrt(mean_squared_error(y_test.iloc[:, i], pred_test[:, i]))
        overfit_rate = 100 * (rmse_test - rmse_train) / rmse_train

        print(f"Target: {name}")
        print(f"  RMSE (Train): {rmse_train:.4f}")
        print(f"  RMSE (Test):  {rmse_test:.4f}")
        print(f"  Overfitting Rate: {overfit_rate:+.2f}%\n")

    # Optional RMSE per-iteration plots via retraining
    if retrain:
        if params is None:
            params = {
                "n_estimators": 100,
                "objective": "reg:squarederror",
                "random_state": 42,
                "enable_categorical": True,
            }
        evals_result = {}
        for name in target_names:
            xgb = XGBRegressor(**params)
            xgb.fit(
                X_train,
                y_train[name],
                eval_set=[(X_train, y_train[name]), (X_test, y_test[name])],
                verbose=False,
            )
            evals_result[name] = xgb.evals_result()

        for name in target_names:
            plt.figure()
            plt.plot(evals_result[name]["validation_0"]["rmse"], label="Train")
            plt.plot(evals_result[name]["validation_1"]["rmse"], label="Test")
            plt.title(f"XGBoost RMSE per iteration - {name}")
            plt.xlabel("Boosting Round")
            plt.ylabel("RMSE")
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.show()


def train_and_evaluate_model(X_train, X_test, y_train, y_test, model_type="XGB"):
    """Train and evaluate a regression model on the provided data.
    Supports XGBoost, LightGBM, CatBoost, and Linear Regression.
    """
    if model_type == "XGB":
        model = MultiOutputRegressor(
            XGBRegressor(
                n_estimators=200,
                objective="reg:squarederror",
                random_state=42,
                enable_categorical=True,
            )
        )
    elif model_type == "LGBM":
        model = MultiOutputRegressor(
            LGBMRegressor(n_estimators=200, objective="regression", random_state=42)
        )
    elif model_type == "CatBoost":
        cat_features = ["PULocationID", "DOLocationID"]
        model = MultiOutputRegressor(
            CatBoostRegressor(
                iterations=200,
                learning_rate=0.1,
                depth=6,
                random_seed=42,
                verbose=0,  # suppress training output
            )
        )
    elif model_type == "Linear":
        model = MultiOutputRegressor(LinearRegression())
    else:
        raise ValueError(
            "Unsupported model type. Choose from: XGB, LGBM, CatBoost, Linear."
        )

    # Train the model
    if model_type == "CatBoost":
        model.fit(X_train, y_train, cat_features=cat_features)
    else:
        model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)

    # Evaluate the model
    evaluate_model(y_test, y_pred)

    return model, y_pred


def evaluate_models_leaderboard(models, X_tests, y_tests):
    leaderboard = []

    for name, model, X_test, y_test in zip(
        models.keys(), models.values(), X_tests, y_tests
    ):
        # Predict
        y_pred = model.predict(X_test)
        y_true = y_test.copy()

        # Extract true and predicted
        y_fare_true = y_true["fare_amount"]
        y_duration_true = y_true["trip_time_seconds"]
        y_fare_pred = y_pred[:, 0]
        y_duration_pred = y_pred[:, 1]

        # Metrics for fare
        mae_fare = mean_absolute_error(y_fare_true, y_fare_pred)
        medae_fare = median_absolute_error(y_fare_true, y_fare_pred)
        rmse_fare = root_mean_squared_error(y_fare_true, y_fare_pred)
        evs_fare = explained_variance_score(y_fare_true, y_fare_pred)
        r2_fare = r2_score(y_fare_true, y_fare_pred)
        try:
            msle_fare = mean_squared_log_error(y_fare_true, y_fare_pred)
        except ValueError:
            msle_fare = np.nan

        # Metrics for duration
        mae_time = mean_absolute_error(y_duration_true, y_duration_pred)
        medae_time = median_absolute_error(y_duration_true, y_duration_pred)
        rmse_time = root_mean_squared_error(y_duration_true, y_duration_pred)
        evs_time = explained_variance_score(y_duration_true, y_duration_pred)
        r2_time = r2_score(y_duration_true, y_duration_pred)
        try:
            msle_time = mean_squared_log_error(y_duration_true, y_duration_pred)
        except ValueError:
            msle_time = np.nan

        leaderboard.append(
            {
                "Model": name,
                "MAE - Fare": mae_fare,
                "MedAE - Fare": medae_fare,
                "RMSE - Fare": rmse_fare,
                "MSLE - Fare": msle_fare,
                "ExplainedVar - Fare": evs_fare,
                "R² - Fare": r2_fare,
                "MAE - Duration": mae_time,
                "MedAE - Duration": medae_time,
                "RMSE - Duration": rmse_time,
                "MSLE - Duration": msle_time,
                "ExplainedVar - Duration": evs_time,
                "R² - Duration": r2_time,
            }
        )

    df_leaderboard = pd.DataFrame(leaderboard).sort_values("RMSE - Duration")
    return df_leaderboard


def prepare_features_for_prediction(df, features):
    """Prepare raw feature DataFrame for prediction with trained model."""
    X = df[features].copy()

    # Ensure categorical types for location IDs
    for col in ["PULocationID", "DOLocationID"]:
        if col in features:
            X[col] = X[col].astype("category")

    if "weather" in features:
        # One-hot encode the weather feature
        encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
        X_weather_encoded = pd.DataFrame(
            encoder.fit_transform(X[["weather"]]),
            index=X.index,
            columns=encoder.get_feature_names_out(["weather"]),
        )
        # Combine the encoded weather with the rest of the features
        X = pd.concat([X.drop(columns="weather"), X_weather_encoded], axis=1)

    return X
