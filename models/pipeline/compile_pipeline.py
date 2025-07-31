from pipeline import PipelineCreator
import joblib
import pandas as pd

df = pd.read_parquet('cleaned_data_with_weather.parquet')

df.rename(columns={'tpep_dropoff_datetime': 'pickup_date'}, inplace=True)
features = ["PULocationID", "DOLocationID", "trip_distance", "pickup_date", "weather"]


pipeline = PipelineCreator()
pipeline.create_pipeline(df)


