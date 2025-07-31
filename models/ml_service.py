import json
import os
import time

import numpy as np
import redis
import settings
import pandas as pd

from joblib import load
from utils import running_in_docker

# Make use of settings.py module to get Redis settings like host, port, etc.
REDIS_IP = settings.REDIS_IP if running_in_docker() else 'localhost'
db = redis.Redis(host= REDIS_IP, port= settings.REDIS_PORT, db=settings.REDIS_DB_ID)

#This function loads a pre-trained and simple model. Just for testing purposes.
def load_model(with_weather=True):
    # Load the dataset and train a simple model
    if with_weather:
        model = load('model_weather.pkl')
        pipeline_pkl = load('pipeline/trained_pipeline_weather.pkl')
    else:
        model = load('trained_model.pkl')
        pipeline_pkl = load('pipeline/trained_pipeline_no_weather.pkl')

        
    return model, pipeline_pkl

def predict(input_fields):
    """
    Loop indefinitely asking Redis for new jobs.
    When a new job arrives, takes it from the Redis queue, uses the loaded ML model to get predictions and stores the results back in Redis using
    the original job ID so other services can see it was processed and access the results.

    """
    print(f"Features: {input_fields}")
    
    model, pipeline_pkl = load_model()  # Load the pre-trained model
    
    fare, duration = None, None
    results = {}
    try:

        df = pd.DataFrame([input_fields])
        
        print(df.dtypes)
        print(df.head())
        
        df['PULocationID'] = df['PULocationID'].astype('category')
        df['DOLocationID'] = df['DOLocationID'].astype('category')
        
        df = pipeline_pkl.transform(df)
        
        print("Después del pipeline:")
        print(df.head())
        
        # Predict
        preds = model.predict(df)        
    
        print(f"Predictions: {preds}")
        print(f"fare: {preds[:, 0]}")
        print(f"duration: {preds[:, 1]}")
        
        fare = float(preds[:, 0])
        duration = float(preds[:, 1])
    
    except Exception as e:
        raise Exception(f"Error in predict {e}")
        
    results['fare'] = fare
    results['duration'] = duration
    
    print("Getting results")
    print(results)
    
    return results

def classify_process():
    
    while True:

        # Take a new job from Redis
        job_data = db.brpop("taxi_predict")
        
        if not job_data:
            raise Exception(f"Not jobs in queue")

        # Decode the JSON data for the given job
        job_json = job_data[1] # 
        try:
            job = json.loads(job_json)
        except json.JSONDecodeError as e:
            raise Exception(f"Error decoding job data {e} {job_data}")
            
        # Important! Get and keep the original job ID
        job_id = job.get("id")
        
        fare, duration = "", 0.0

        # Run the loaded ml model (use the predict() function)
        try:
            features = job.get("features")
            results = predict(features)
            
            fare_pred = results['fare']  # fare_amount predictions
            fare_pred = round(np.exp(fare_pred), 2)  # Round to 2 decimal places

            duration_pred_secs = results['duration']  # Assuming the second column is trip_duration in seconds
            duration_pred_secs = np.exp(duration_pred_secs)  # Round to nearest second
            predicted_durations = round(duration_pred_secs/60, 2)    

        except Exception as e:
            raise Exception(f"Error running the prediction for job {job_id}: {e}")
    
        # Prepare a new JSON with the results
        
        output = {"fare": float(fare_pred), "duration": float(predicted_durations)}

        # Store the job results on Redis using the original
        # job ID as the key
        db.set(f"completed_jobs:{job_id}",json.dumps(output))
        # Sleep for a bit
        time.sleep(settings.SERVER_SLEEP)


if __name__ == "__main__":
    # Now launch process
    print("Launching ML service...")
    print(f"http://{REDIS_IP}:{settings.REDIS_PORT}/")
    classify_process()
