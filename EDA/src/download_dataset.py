import pandas as pd

# import settings
import geopandas as gpd
from api.utils import read_taxi_zones_dataset_from_shapefile
import os


def load_local_month_dataset(month: int, year: int) -> pd.DataFrame:
    """
    Loads the Yellow Taxi dataset for a given month/year from a local Parquet file.
    Assumes the file is named 'yellow_tripdata_YYYY-MM.parquet' and located in '../data/'.
    """
    filename = f"yellow_tripdata_{year}-{month:02d}.parquet"
    file_path = os.path.join("../data/trip_records", filename)
    print(f"Loading local file: {filename}")
    df = pd.read_parquet(file_path, engine="pyarrow")
    return df


def read_taxi_zones_dataset_from_shapefile() -> pd.DataFrame:
    """
    Download the NYC taxi zones dataset from a previously downloaded shapefile.

    returns:
        GeodataFrame: A GeoDataFrame containing the taxi zones data.
    """

    path_file_shapefile = "../data/taxi_zones.shp"
    # Check if the shapefile exists
    if not os.path.exists(path_file_shapefile):
        raise FileNotFoundError(
            f"The shapefile {path_file_shapefile} does not exist. Please download it first."
        )
    # Read the dataset to ensure it was downloaded correctly
    geo_df = gpd.read_file(
        path_file_shapefile
    )  # it returns a GeoDataFrame. don't be confused with pandas DataFrame

    # Set CRS if missing (based on your data)
    if geo_df.crs is None:
        geo_df.set_crs("EPSG:2263", inplace=True)

    # print(f"List of distinct location IDs: {sorted(geo_df['LocationID'].dropna().unique())}")  #This is used to check the distinct LocationIDs in the dataset

    return geo_df


def get_location_ids_with_few_records(
    df_baseline: pd.DataFrame, threshold: int
) -> list:
    location_counts = df_baseline["PULocationID"].value_counts()
    df_taxi_zones = read_taxi_zones_dataset_from_shapefile()
    locations_taxi_zones = df_taxi_zones["LocationID"].unique()

    locations_to_analyze = location_counts[location_counts < threshold].index.tolist()
    locations_to_analyze += [
        loc for loc in locations_taxi_zones if loc not in location_counts.index
    ]

    print(f"Locations with fewer than {threshold} records: {locations_to_analyze}")
    return locations_to_analyze


def refill_data_using_other_months_data(
    df_baseline: pd.DataFrame, month: int, year: int, locations_to_analyze: list
) -> pd.DataFrame:
    df_new_data = load_local_month_dataset(month, year)
    df_filtered = df_new_data[df_new_data["PULocationID"].isin(locations_to_analyze)]
    df_combined = pd.concat([df_baseline, df_filtered], ignore_index=True)
    return df_combined


def download_dataset(threshold):
    df_baseline = load_local_month_dataset(month=5, year=2022)
    print(f"Initial records in May 2022: {len(df_baseline)}")

    locations_to_analyze = get_location_ids_with_few_records(df_baseline, threshold)

    for month in range(1, 13):
        if month == 5:
            continue
        df_baseline = refill_data_using_other_months_data(
            df_baseline, month, 2022, locations_to_analyze
        )
        print(f"After month {month}: {len(df_baseline)} records")

        locations_to_analyze = get_location_ids_with_few_records(
            df_baseline, threshold=20
        )

    df_baseline.to_csv("refilled_nyc_yellow_taxi_trip_data.csv", index=False)
    print("Final dataset saved to refilled_nyc_yellow_taxi_trip_data.csv")


if __name__ == "__main__":
    download_dataset()
