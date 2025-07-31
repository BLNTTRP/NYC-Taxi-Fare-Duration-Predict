import requests
import pandas as pd
import settings
from typing import Tuple
from utils import read_taxi_zones_dataset_from_shapefile


def convert_period_to_datetime_string(month: int, year: int) -> Tuple[str, str]:
    """
    Converts a month and year into a date range string for querying the NYC Yellow Taxi Trip dataset.
    Args:
        month (int): The month to convert (1-12).
        year (int): The year to convert.
    Returns:
        Tuple[str, str]: A tuple containing the start and end date strings in the format 'YYYY-MM-DD'.
    """
    date_from = f"{year}-{month:02d}-01T00:00:00"
    date_to = f"{year}-{month:02d}-{pd.Period(year=year, month=month, freq='M').days_in_month}T23:59:59"
    return date_from, date_to


def download_nyc_dataset_per_month(
    month: int = 5, year: int = 2022, number_limit: int = 5000000
) -> pd.DataFrame:
    """
    Downloads the NYC Yellow Taxi Trip dataset for a specific month and year from the NYC Open Data portal.
    Args:
        month (int): The month to download (1-12).
        year (int): The year to download.
        number_limit (int): The maximum number of records to download. Default is 5 million.
    Returns:
        pd.DataFrame: A DataFrame containing the downloaded dataset.
    """

    headers = {
        "Accept": "application/json",
        "X-App-Token": settings.NYC_OPENDATA_APP_TOKEN,
    }

    # Convert month and year to get start and end date strings
    start_date, end_date = convert_period_to_datetime_string(month, year)
    uri = (
        settings.URI_YELLOW_TAXI_TRIP_DATASET
        + f"?$where=tpep_pickup_datetime between '{start_date}' and '{end_date}'&$limit={number_limit}"
    )  # Limit the number of records
    print(f"uri to download dataset: {uri}")

    response = requests.get(uri, headers=headers)

    if response.status_code != 200:
        raise Exception(
            f"Failed to download dataset from NYC Open Data. Message error: {response.text} ; status code: {response.status_code}"
        )

    # In case of success, convert the response to a DataFrame
    data = response.json()
    df_data = pd.DataFrame(data)

    df_data.columns = [
        "VendorID",
        "tpep_pickup_datetime",
        "tpep_dropoff_datetime",
        "passenger_count",
        "trip_distance",
        "RatecodeID",
        "store_and_fwd_flag",
        "PULocationID",
        "DOLocationID",
        "payment_type",
        "fare_amount",
        "extra",
        "mta_tax",
        "tip_amount",
        "tolls_amount",
        "improvement_surcharge",
        "total_amount",
        "congestion_surcharge",
        "airport_fee"
    ]

    return df_data


# read May 2022 dataset as baseline and then, check if there are LocationID values with fewer than 20 records. Get the list of LocationID values with fewer than 20 records
def get_location_ids_with_few_records(
    df_baseline: pd.DataFrame, threshold: int
) -> list:

    location_counts = df_baseline["PULocationID"].value_counts()
    df_taxi_zones = read_taxi_zones_dataset_from_shapefile()
    locations_taxi_zones = df_taxi_zones[
        "LocationID"
    ].unique()  # Get the list of all LocationIDs in the taxi zones dataset

    locations_to_analyze = location_counts[location_counts < threshold].index.tolist()

    print(f"Locations with fewer than {threshold} records: {locations_to_analyze}")

    return locations_to_analyze


def refill_data_using_other_months_data(
    df_baseline: pd.DataFrame, month: int, year: int, locations_to_analyze: list
) -> pd.DataFrame:

    # First, get the dataset for the specified month and year
    df_new_data = download_nyc_dataset_per_month(month=month, year=year)
    df_filtered = df_new_data[
        df_new_data["PULocationID"].isin(locations_to_analyze)
    ]  # Filter only the rows with the specified PULocationIDs
    # Concatenate the new data with the baseline data
    df_combined = pd.concat([df_baseline, df_filtered], ignore_index=True)
    return df_combined


def download_dataset():

    df_baseline = download_nyc_dataset_per_month(month=5, year=2022)
    print(f"Number of records in May 2022 dataset: {len(df_baseline)}")
    locations_to_analyze = get_location_ids_with_few_records(df_baseline, threshold=20)

    # for each month in 2022, refill the data for the locations with fewer than 20 records
    for month in range(1, 13):
        if month == 5:  # Skip May since it's the baseline
            continue
        df_baseline = refill_data_using_other_months_data(
            df_baseline, month, 2022, locations_to_analyze
        )
        print(
            f"Number of records after refilling data for month {month}: {len(df_baseline)}"
        )
        locations_to_analyze = get_location_ids_with_few_records(
            df_baseline, threshold=20
        )

    # Save the final DataFrame to a CSV file
    df_baseline.to_csv("refilled_nyc_yellow_taxi_trip_data.csv", index=False)


download_dataset()
