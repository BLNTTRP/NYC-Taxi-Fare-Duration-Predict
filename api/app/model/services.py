import json
import time
from uuid import uuid4
from utils import running_in_docker

import redis

from api import settings

# Connect to Redis and assign to variable `db`
REDIS_IP = settings.REDIS_IP if running_in_docker() else 'localhost'
db = redis.StrictRedis(host=REDIS_IP, port=settings.REDIS_PORT, db=settings.REDIS_DB_ID)


async def model_predict(features):
    print(f"Processing model predict ...")
    print(f"Features: {features}")
    """
    Receives an image name and queues the job into Redis.
    Will loop until getting the answer from our ML service.

    Parameters
    ----------
    image_name : str
        Name for the image uploaded by the user.

    Returns
    -------
    prediction, score : tuple(str, float)
        Model predicted class as a string and the corresponding confidence
        score as a number.
    """
    fare = None
    duration = None
    success = True

    # Assign an unique ID for this job and add it to the queue.
    # We need to assing this ID because we must be able to keep track
    # of this particular job across all the services
    # DONE
    job_id = str(uuid4())

    # Create a dict with the job data we will send through Redis having the
    # following shape:
    # {
    #    "id": str,
    #    "image_name": str,
    # }
    try:
        job_data = {"id": job_id, "features": features}
    except Exception as e:
        print(e)

    # Send the job to the model service using Redis
    db.lpush("taxi_predict", json.dumps(job_data))

    # Loop until we received the response from our ML model
    while True:
        # Attempt to get model predictions using job_id
        
        output = db.get(f'completed_jobs:{job_id}')

        # Check if the text was correctly processed by our ML model
        # Don't modify the code below, it should work as expected
        if output is not None:
            output = json.loads(output.decode("utf-8"))
            fare = output["fare"]
            duration = output["duration"]

            db.delete(job_id)
            break

        # Sleep some time waiting for model results
        time.sleep(settings.API_SLEEP)

    return success, fare, duration
