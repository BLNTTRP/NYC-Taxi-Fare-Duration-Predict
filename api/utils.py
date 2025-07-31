import pandas as pd
import settings as settings
import os

import kagglehub 
from typing import Tuple

import geopandas as gpd
from shapely import wkt
from shapely.geometry import Point


def read_taxi_zones_dataset() -> pd.DataFrame:
    """
    Download the NYC taxi zones dataset from Kaggle.
    Then, it reads the downloaded dataset and returns it as a pandas DataFrame.

    Returns:
        str: The path to the downloaded dataset file.
    """
    path = kagglehub.dataset_download(settings.KAGGLE_DATA_ZONES_PATH)

    path_file_destination = settings.PATH_TAXI_ZONES

    # Move the downloaded file to the destination folder. Check if the file exists
    if os.path.exists(path_file_destination):
        print(f"File {path_file_destination} already exists.")
    else:
        os.rename(path, path_file_destination)

    # Read the dataset
    df_zones = pd.read_csv(path_file_destination)

    # Convert the geometry column from WKT to Shapely geometries
    df_zones[settings.KAGGLE_GEOMETRY_COLUMN] = df_zones[
        settings.KAGGLE_GEOMETRY_COLUMN
    ].apply(wkt.loads)
    # Convert to a GeoDataFrame
    gdf = gpd.GeoDataFrame(
        df_zones, geometry=settings.KAGGLE_GEOMETRY_COLUMN, crs="EPSG:4326"
    )

    return gdf


def read_taxi_zones_dataset_from_shapefile() -> pd.DataFrame:
    """
    Download the NYC taxi zones dataset from a previously downloaded shapefile.

    returns:
        GeodataFrame: A GeoDataFrame containing the taxi zones data.
    """

    path_file_shapefile = settings.PATH_TAXI_ZONES_SHAPEFILE
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

    return geo_df


def find_point_location(latitude: float, longitude: float) -> Point:
    """
    Find the location of a point given its latitude and longitude.

    Args:
        latitude (float): Latitude of the point.
        longitude (float): Longitude of the point.

    Returns:
        matching_zone: a 1 row GeoDataFrame containing the taxi zone that contains the point.
        From this GeoDataFrame, you can extract the 'zone', 'LocationID' and 'borough' columns.
    """
    gdf_matching_zone = None

    try:
        gdf = read_taxi_zones_dataset_from_shapefile()  # read the shapefile dataset

        # Create a Point object from the latitude and longitude
        point = Point(
            longitude, latitude
        )  # Note: Point(x, y) => Point(longitude, latitude)
        # Convert the point to a GeoDataFrame and reproject to match zone CRS
        pickup_gdf = gpd.GeoDataFrame(geometry=[point], crs="EPSG:4326")
        pickup_gdf = pickup_gdf.to_crs(gdf.crs)
        pickup_projected = pickup_gdf.geometry.iloc[0]

        # Find the matching zone. If this point is not in any zone, it will return an empty GeoDataFrame (Or None)
        gdf_matching_zone = gdf[gdf.contains(pickup_projected)]
    except Exception as e:
        print(f"An error occurred while finding the point location: {e}")

    return gdf_matching_zone


def get_taxi_zones(
    latitude_from: float, latitude_to: float, longitude_from: float, longitude_to: float
) -> Tuple[int, int]:
    """
    Given a latitude and longitude from different points, find the taxi zones codes (for both pickup and dropoff)

    Args:
        latitude_from (float): The minimum latitude of the pickup location.
        latitude_to (float): The maximum latitude of the dropoff location.
        longitude_from (float): The minimum longitude of the pickup location.
        longitude_to (float): The maximum longitude of the dropoff location.

    Returns:
        tuple: A tuple containing the pickup and dropoff taxi zone codes [int, int].
    """
    gdf_pickup_location = find_point_location(latitude_from, longitude_from)
    gdf_dropoff_location = find_point_location(latitude_to, longitude_to)

    # Check if the pickup and dropoff points are in any taxi zone
    if gdf_pickup_location.empty:
        raise ValueError(
            f"The pickup point {gdf_pickup_location} is not in any taxi zone."
        )
    if gdf_dropoff_location.empty:
        raise ValueError(
            f"The dropoff point {gdf_dropoff_location} is not in any taxi zone."
        )

    # extract the LocationID for pickup and dropoff zones
    pickup_location_id = gdf_pickup_location.iloc[0]["LocationID"]
    dropoff_location_id = gdf_dropoff_location.iloc[0]["LocationID"]

    return pickup_location_id, dropoff_location_id


def running_in_docker():
    # Method 1: buscar archivo /.dockerenv
    if os.path.exists("/.dockerenv"):
        return True
    # Method 2: revisar contenido de cgroup
    try:
        with open("/proc/1/cgroup", "rt") as f:
            return any("docker" in line or "containerd" in line for line in f)
    except FileNotFoundError:
        return False
