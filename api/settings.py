import os

# import dotenv
from dotenv import load_dotenv
import pathlib

# Load environment variables from .env file
load_dotenv(".env")

# Run API in Debug mode
API_DEBUG = True

# We will store images uploaded by the user on this folder
UPLOAD_FOLDER = "uploads/"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# REDIS settings
# Queue name
REDIS_QUEUE = "service_queue"
# Port
REDIS_PORT = 6379
# DB Id
REDIS_DB_ID = 0
# Host IP
REDIS_IP = os.getenv("REDIS_IP", "redis")
# Sleep parameters which manages the
# interval between requests to our redis queue
API_SLEEP = 0.05




# Database settings
DATABASE_USERNAME = "postgres"
DATABASE_PASSWORD = "adlibitum"
DATABASE_HOST = "db"
DATABASE_NAME = "sp3"
SECRET_KEY = os.getenv("SECRET_KEY", "S09WWWHXBAJDIUEREHCN3752346572452VGGGVWWW526194")

print(f"Conectando a host={DATABASE_HOST}, db={DATABASE_NAME}, user={DATABASE_USERNAME}")

# KAGGLE
KAGGLE_DATA_ZONES_PATH = "mxruedag/tlc-nyc-taxi-zones"
KAGGLE_GEOMETRY_COLUMN = "the_geom"

# PATHS
BASE_DIR = pathlib.Path(__file__).parent  # Ajusta según tu estructura

PATH_TAXI_ZONES = BASE_DIR / "data" / "taxi_zones.csv"
PATH_TAXI_ZONES_SHAPEFILE = BASE_DIR / "data" / "taxi_zones.shp"

# NYC Open Data API
NYC_OPENDATA_APP_TOKEN = "asas"
URI_YELLOW_TAXI_TRIP_DATASET = "https://data.cityofnewyork.us/resource/qp3b-zxtp.json"
