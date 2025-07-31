import matplotlib.pyplot as plt
import seaborn as sns

from pandas import DataFrame
import pandas as pd
import geopandas as gpd
from matplotlib.gridspec import GridSpec

from src.utils import compute_borough_distributions


# Plotting the fare amount against trip duration
def plot_fare_vs_duration(
    df: DataFrame, y_fare: str = "fare_amount", y_duration: str = "trip_duration"
):
    """
    Plots the normalized fare amount against normalized trip duration.
    Args:
        df (DataFrame): DataFrame containing 'fare_amount' and 'trip_duration' columns.
        y_fare (str): Column name for fare amount.
        y_duration (str): Column name for trip duration.
    """
    # Drop NA rows in both columns
    df = df[[y_fare, y_duration]].dropna()

    # Min-max normalization
    df["norm_fare"] = (df[y_fare] - df[y_fare].min()) / (
        df[y_fare].max() - df[y_fare].min()
    )
    df["norm_duration"] = (df[y_duration] - df[y_duration].min()) / (
        df[y_duration].max() - df[y_duration].min()
    )

    # Plot
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x="norm_duration", y="norm_fare", alpha=0.5)
    plt.title("Fare Amount vs Trip Duration")
    plt.xlabel("Trip Duration (seconds)")
    plt.ylabel("Fare Amount ($)")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_trip_counts_by_tertile(merged_df: DataFrame, ID_column="PULocationID"):
    """
    Plots the trip count per PULocationID for each tertile.
    Args:
        merged_df (DataFrame): DataFrame containing 'LocationID', 'trip_count', and 'tertile' columns.
    """
    # Ensure quartiles are sorted correctly
    quartiles = ["T1", "T2", "T3"]
    for i, q in enumerate(quartiles):
        # Filter data for the current quartile and sort by trip count
        data_q = merged_df[merged_df["tertile"] == q].sort_values(
            "trip_count", ascending=False
        )

        # Create a new figure for each quartile
        plt.figure(figsize=(12, 4))
        plt.bar(data_q["LocationID"].astype(str), data_q["trip_count"], color=f"C{i}")
        plt.title(f"Trip Count per {ID_column} - {q}", fontsize=16)
        plt.xlabel(f"{ID_column}")
        plt.ylabel("Number of Trips")
        plt.xticks(rotation=90)
        plt.grid(axis="y", linestyle="--", alpha=0.5)
        plt.tight_layout()
        plt.show()


def plot_geospatial_data(
    zones_gdf: gpd.GeoDataFrame,
    df_pu: pd.DataFrame,
    df_do: pd.DataFrame,
    title: str = "NYC Trip Count Tertiles",
):
    """
    Plots geospatial maps of NYC with tertile information for PULocationID and DOLocationID side by side.

    Args:
        zones_gdf (GeoDataFrame): GeoDataFrame with geometry and 'LocationID'.
        df_pu (DataFrame): DataFrame with 'LocationID', 'tertile', and 'Borough' for PULocationID.
        df_do (DataFrame): DataFrame with 'LocationID', 'tertile', and 'Borough' for DOLocationID.
        title (str): Overall plot title.
    """
    # Ensure LocationID is int
    zones_gdf = zones_gdf.copy()
    zones_gdf["LocationID"] = zones_gdf["LocationID"].astype(int)

    # Merge separately for pickup and dropoff
    pu_gdf = zones_gdf.merge(
        df_pu[["LocationID", "tertile", "Borough"]], on="LocationID", how="left"
    )
    do_gdf = zones_gdf.merge(
        df_do[["LocationID", "tertile", "Borough"]], on="LocationID", how="left"
    )

    # Project to NYC coordinate system
    pu_gdf = pu_gdf.to_crs(epsg=2263)
    do_gdf = do_gdf.to_crs(epsg=2263)

    # Borough boundaries
    boroughs_gdf = pu_gdf.dissolve(by="Borough", as_index=False)
    boroughs_gdf["centroid"] = boroughs_gdf.geometry.centroid

    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 9))
    plt.suptitle(title, fontsize=18, y=1.02)

    for ax, gdf, label in zip(
        [ax1, ax2], [pu_gdf, do_gdf], ["Pick-up Tertiles", "Drop-off Tertiles"]
    ):
        gdf.plot(
            column="tertile",
            ax=ax,
            cmap="Set1",
            edgecolor="grey",
            linewidth=0.4,
            legend=True,
            legend_kwds={"title": "Tertile"},
        )
        boroughs_gdf.boundary.plot(ax=ax, color="black", linewidth=1)
        for _, row in boroughs_gdf.iterrows():
            c = row["centroid"]
            ax.text(
                c.x,
                c.y,
                row["Borough"],
                color="black",
                fontsize=16,
                ha="center",
                va="center",
                weight="bold",
                alpha=0.8,
            )
        ax.set_title(label, fontsize=14)
        ax.axis("off")

    plt.tight_layout()
    plt.show()


def plot_nyc_zones_with_tertiles_and_location_ids(
    zones_gdf: gpd.GeoDataFrame,
    show_table: bool = False,
    table_rows: int = 10,
):
    """
    Plots NYC zones with tertile coloring and LocationID labels,
    optionally showing a table of top zones by tertile above the map.
    """
    # Use vertical layout: table (top), map (bottom)
    fig = plt.figure(figsize=(22, 20))
    gs = GridSpec(2, 1, height_ratios=[1, 3])  # Taller map, shorter table

    ax_table = fig.add_subplot(gs[0])
    ax_map = fig.add_subplot(gs[1])

    # Plot the map
    zones_gdf.plot(
        column="tertile",
        legend=True,
        ax=ax_map,
        cmap="Set1",
        edgecolor="black",
    )

    for idx, row in zones_gdf.iterrows():
        if not row["geometry"].is_empty:
            centroid = row["geometry"].centroid
            ax_map.text(
                centroid.x,
                centroid.y,
                str(row["LocationID"]),
                fontsize=10,
                ha="center",
                va="center",
                color="black",
            )

    ax_map.set_title("NYC Zones with Tertiles and LocationIDs", fontsize=16)
    ax_map.axis("off")

    # Table of top N zones by tertile
    if show_table:
        ax_table.axis("off")

        tertiles = ["T1", "T2", "T3"]
        tables = []

        for tertile in tertiles:
            subset = zones_gdf[zones_gdf["tertile"] == tertile]
            if "zone" in subset.columns:
                cols = ["LocationID", "zone", "tertile"]
            else:
                cols = ["LocationID", "tertile"]
            top_rows = subset[cols].dropna().head(table_rows).copy()
            top_rows.reset_index(drop=True, inplace=True)
            top_rows.columns = [f"{col} ({tertile})" for col in top_rows.columns]
            tables.append(top_rows)

        combined_table = pd.concat(tables, axis=1)

        table = ax_table.table(
            cellText=combined_table.values,
            colLabels=combined_table.columns,
            cellLoc="center",
            loc="center",
        )
        table.auto_set_font_size(True)
        # table.set_fontsize(10)
        table.scale(1.4, 1.4)
        ax_table.set_title("Top Zones by Tertile", fontsize=14)

    plt.tight_layout()
    plt.show()


def plot_borough_distribution_comparison(df_raw, df_clean, df_zone):
    """
    Compare pickup/dropoff probability distributions by borough before and after cleaning.
    Shows df_raw in subplot 1 and df_clean in subplot 2.
    """

    dist_raw = compute_borough_distributions(df_raw, df_zone, label="Raw")
    dist_clean = compute_borough_distributions(df_clean, df_zone, label="Cleaned")

    # Remove 'Unknown' borough if present
    if "Unknown" in dist_raw.index:
        dist_raw = dist_raw.drop("Unknown")
    if "Unknown" in dist_clean.index:
        dist_clean = dist_clean.drop("Unknown")

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(18, 6))
    plt.suptitle("Pickup/Dropoff Probability Distributions by Borough", fontsize=16)

    # Raw data plot
    dist_raw.plot(kind="bar", ax=axes[0], color=["skyblue", "salmon"])
    axes[0].set_title("Before Refilling (df_raw_cleaned)")
    axes[0].set_ylabel("Probability")
    axes[0].set_xlabel("Borough")
    axes[0].grid(axis="y", linestyle="--", alpha=0.5)

    # Cleaned data plot
    dist_clean.plot(kind="bar", ax=axes[1], color=["skyblue", "salmon"])
    axes[1].set_title("After Refilling (df_refilled)")
    axes[1].set_xlabel("Borough")
    axes[1].grid(axis="y", linestyle="--", alpha=0.5)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()


def plot_feature_importances(model, X_train):
    """Plot feature importances from the trained model."""

    # Extract and plot importances from first estimator (fare_amount)
    importance_df = pd.DataFrame(
        {
            "Feature": X_train.columns,
            "Importance (fare)": model.estimators_[0].feature_importances_,
            "Importance (duration)": model.estimators_[1].feature_importances_,
        }
    ).set_index("Feature")

    # Plot
    plt.figure(figsize=(14, 6))
    importance_df.sort_values("Importance (fare)", ascending=True)[
        "Importance (fare)"
    ].plot(kind="barh", color="skyblue")
    plt.title("Feature Importance for Fare Prediction (XGBoost)")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(14, 6))
    importance_df.sort_values("Importance (duration)", ascending=True)[
        "Importance (duration)"
    ].plot(kind="barh", color="salmon")
    plt.title("Feature Importance for Trip Duration Prediction (XGBoost)")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.show()


def plot_predictions_vs_actuals(y_pred, y_test):
    """Plot predicted vs actual values for fare and trip duration.
    Args:
        y_pred (np.ndarray): Predicted values from the model.
        y_test (pd.DataFrame): Actual target values from the test set.
    """
    true_duration = y_test["trip_time_seconds"].values
    true_fare = y_test["fare_amount"].values

    # Create a figure with 2 subplots: one for fare, one for duration
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # --- Plot 1: Fare Amount ---
    sns.scatterplot(
        x=true_fare, y=y_pred[:, 0], ax=axes[0], alpha=0.5, color="steelblue"
    )
    axes[0].plot(
        [y_pred[:, 0].min(), y_pred[:, 0].max()],
        [y_pred[:, 0].min(), true_fare.max()],
        "r--",
    )  # y=x line
    axes[0].set_title("Predicted vs Actual Fare Amount")
    axes[0].set_xlabel("Actual Fare Amount")
    axes[0].set_ylabel("Predicted Fare Amount")

    # --- Plot 2: Trip Duration ---
    sns.scatterplot(
        x=true_duration, y=y_pred[:, 1], ax=axes[1], alpha=0.5, color="darkgreen"
    )
    axes[1].plot(
        [y_pred[:, 1].min(), y_pred[:, 1].max()],
        [y_pred[:, 1].min(), y_pred[:, 1].max()],
        "r--",
    )
    axes[1].set_title("Predicted vs Actual Trip Duration (s)")
    axes[1].set_xlabel("Actual Trip Duration (s)")
    axes[1].set_ylabel("Predicted Trip Duration (s)")

    plt.tight_layout()
    plt.show()


def plot_leaderboard(df_leaderboard):

    df = df_leaderboard.set_index("Model")

    fig, axs = plt.subplots(
        2, 4, figsize=(20, 10)
    )  # 2 rows (fare & duration), 4 metrics

    metrics = ["RMSE", "MAE", "MedAE", "R²"]
    targets = ["Fare", "Duration"]
    colors = ["viridis", "plasma"]

    for i, metric in enumerate(metrics):
        for j, target in enumerate(targets):
            ax = axs[j, i]
            col = f"{metric} - {target}"
            df[[col]].plot.bar(ax=ax, legend=False, colormap=colors[j])
            ax.set_title(f"{col} Comparison")
            ax.set_ylabel(col)
            ax.grid(True, axis="y", linestyle="--", alpha=0.7)
            ax.set_xlabel("")

    plt.suptitle("Model Performance Leaderboard", fontsize=16)
    plt.tight_layout()
    plt.show()


def plot_leaderboard_heatmap(df_leaderboard, normalize=True):
    import seaborn as sns
    import matplotlib.pyplot as plt

    df = df_leaderboard.set_index("Model").copy()

    # Optional normalization (column-wise)
    if normalize:
        df = (df - df.min()) / (df.max() - df.min())

    plt.figure(figsize=(12, 4))
    sns.heatmap(df, annot=True, fmt=".4f", cmap="coolwarm", linewidths=0.5, cbar=True)
    plt.title("Leaderboard Heatmap of All Metrics", fontsize=16)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.show()
