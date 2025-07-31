import pandas as pd
import numpy as np
from scipy.spatial.distance import mahalanobis
from scipy.stats import chi2
from sklearn.preprocessing import StandardScaler, MinMaxScaler


def one_hot_encode(dataframe):
    """
    One-hot encodes categorical columns in a DataFrame.
    Args:
        dataframe (pd.DataFrame): The input DataFrame with categorical columns.
    Returns:
        pd.DataFrame: A DataFrame with one-hot encoded categorical columns.
    """
    cat_cols = dataframe.select_dtypes(include="object").columns.tolist()
    df = pd.get_dummies(dataframe, columns=cat_cols, drop_first=True)
    return df

# Done
def add_trip_duration(dataframe):
    """Adds trip duration and trip time in seconds to the DataFrame.
    Args:
        dataframe (pd.DataFrame): The input DataFrame with pickup and dropoff datetime columns.
    Returns:
        pd.DataFrame: A DataFrame with trip duration and trip time in seconds added.
    """

    dataframe["trip_duration"] = (
        dataframe["tpep_dropoff_datetime"] - dataframe["tpep_pickup_datetime"]
    )
    dataframe["trip_time_seconds"] = dataframe["trip_duration"].dt.total_seconds()


# Used only on target columns fare_amount y trip_time_seconds
def clean_outliers(
    dataframe, targets=["fare_amount", "trip_time_seconds"], method="IQR"
):
    """
    Cleans outliers from the DataFrame based on specified targets and method.
    Used to remove outliers from fare and trip duration columns (targets).
    Args:
        dataframe (pd.DataFrame): The input DataFrame with potential outliers.
        targets (list): List of target columns to check for outliers.
        method (str): Method to use for outlier detection. Options are 'IQR', 'mean_std'.
    Returns:
        pd.DataFrame: A DataFrame with outliers removed.
    """
    df_clean = dataframe.copy()
    conditions = []

    for target in targets:
        if target not in df_clean.columns:
            raise ValueError(f"Target column '{target}' not found in the DataFrame.")

        if method == "IQR":
            Q1 = df_clean[target].quantile(0.25)
            Q3 = df_clean[target].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR
        elif method == "mean_std":
            mean = df_clean[target].mean()
            std = df_clean[target].std()
            lower = mean - 3 * std
            upper = mean + 3 * std
        else:
            raise ValueError("Method must be 'IQR' or 'mean_std'.")

        # Build condition for this target
        cond = (
            (df_clean[target] >= lower)
            & (df_clean[target] <= upper)
            & (df_clean[target] > 0)
        )
        conditions.append(cond)

    # Combine all conditions (AND)
    if conditions:
        final_condition = np.logical_and.reduce(conditions)
        df_clean = df_clean[final_condition]

    return df_clean


def detect_outliers_mahalanobis(dataframe, columns, alpha=0.999):
    """
    Detects multivariate outliers using the Mahalanobis distance.

    Args:
      dataframe: The pandas DataFrame to analyze.
      columns: A list of column names to consider for outlier detection.

    Returns:
      A list of indices of potential outliers.
    """
    # Select the specified columns
    df_subset = dataframe[columns].copy()

    # Handle missing values by dropping rows with NaNs in the subset
    df_subset.dropna(inplace=True)  # Note: There are no NaNs in the selected columns.

    if df_subset.empty:
        print("Warning: DataFrame subset is empty after dropping NaNs.")
        return []

    # Calculate the covariance matrix
    try:
        cov_matrix = df_subset.cov()
        # Calculate the inverse of the covariance matrix
        inv_cov_matrix = np.linalg.inv(cov_matrix)
    except np.linalg.LinAlgError:
        print(
            "Warning: Could not compute the inverse covariance matrix. Skipping Mahalanobis distance."
        )
        return []

    # Calculate the mean of each column
    mean = df_subset.mean()

    # Calculate the Mahalanobis distance for each data point
    mahalanobis_distances = []
    for index, row in df_subset.iterrows():
        try:
            md = mahalanobis(row, mean, inv_cov_matrix)
            mahalanobis_distances.append((index, md))
        except ValueError as e:
            print(f"Error calculating Mahalanobis distance for index {index}: {e}")
            mahalanobis_distances.append(
                (index, np.nan)
            )  # Append NaN for failed calculations

    # Convert to a DataFrame for easier sorting and thresholding
    md_df = pd.DataFrame(
        mahalanobis_distances, columns=["index", "mahalanobis_distance"]
    ).dropna()

    if md_df.empty:
        print("Warning: No valid Mahalanobis distances calculated.")
        return []

    # Determine a threshold for outlier detection (e.g., using chi-squared distribution)
    # The degrees of freedom are the number of variables (columns)

    k = len(columns)
    # A common significance level is 0.001 for outlier detection
    threshold = chi2.ppf(alpha, k)

    # Identify potential outliers
    outlier_indices = md_df[md_df["mahalanobis_distance"] > threshold]["index"].tolist()

    return outlier_indices


def standardize_data(dataframe, features):
    """
    Standardizes specified features in a DataFrame to have mean 0 and variance 1.
    Args:
        dataframe (pd.DataFrame): The input DataFrame with numerical columns.
        features (list): List of feature names to standardize.
    Returns:
        pd.DataFrame: A DataFrame with standardized specified features.
    """
    df = dataframe.copy()
    scaler = StandardScaler()
    df[features] = scaler.fit_transform(df[features])
    return df


def normalize_data(dataframe, features):
    """
    Normalizes specified features in a DataFrame to a range of [0, 1].
    Args:
        dataframe (pd.DataFrame): The input DataFrame with numerical columns.
        features (list): List of feature names to normalize.
    Returns:
        pd.DataFrame: A DataFrame with normalized specified features.
    """
    df = dataframe.copy()
    scaler = MinMaxScaler()
    df[features] = scaler.fit_transform(df[features])
    return df
