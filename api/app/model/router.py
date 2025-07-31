import os
from typing import List

from app import db
from api import settings as config
from app.auth.jwt import get_current_user
from app.model.schema import PredictRequest, PredictResponse, InputModel
from app.model.services import model_predict
from fastapi import APIRouter, Depends, HTTPException, status, Request
from sqlalchemy.orm import Session
from api import utils

router = APIRouter(tags=["Model"], prefix="/model")

# TODO: Remove fake predictions
@router.post("/predict_fare_duration")
async def predict_fare_duration(request: Request): #, current_user=Depends(get_current_user)):
    rpse = {"success": False, "fare": None, "duration": None}
    """ Endpoint to predict the the fare and duration """
    data = await request.json()
    print(data)
    
    try:
        PULocationID = None
        DOLocationID = None
        pick_hour = None
        trip_distance = None
        pickup_date = None

        features = []

        if 'start_location' in data:
            lat, lon = data['start_location']['latitude'], data['start_location']['longitude']
            print('Asignando PULocationID')
            try:
                PULocationID = utils.find_point_location(float(lat), float(lon))
            except Exception as e:
                raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Invalid start location: {e}")
         
        if 'end_location' in data:
            lat, lon = data['end_location']['latitude'], data['end_location']['longitude']
            DOLocationID = utils.find_point_location(float(lat), float(lon))
            print('Asignando DOLocationID')
        
        if 'pickup_date' in data:
            pickup_date = data['pickup_date']
        
        pick_hour = data['pickup_hour']
        trip_distance = data['trip_distance']
        weather = data.get('weather', None)
        
        PULocationID = (PULocationID["LocationID"].iloc[0])
        DOLocationID = (DOLocationID["LocationID"].iloc[0])

        print(f"PULocationID: {PULocationID}, DOLocationID: {DOLocationID}")
        print(f"pickup_date: {pickup_date}, pick_hour: {pick_hour}, trip_distance: {trip_distance}, weather: {weather}")
        
        features = {
           "PULocationID": int(PULocationID),
           "DOLocationID": int(DOLocationID),
           "trip_distance":float(trip_distance),
           "pickup_hour": pick_hour,
           "is_weekend": 0,
           "pickup_date": pickup_date,
           "weather": weather
        }
        
        success, fare, duration = await model_predict(features=features)
        
        rpse['success'] = success
        rpse["fare"] = fare
        rpse['duration'] = duration
        
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"No se pudo realizar la prediccion: {e}")
        
    return PredictResponse(**rpse)
