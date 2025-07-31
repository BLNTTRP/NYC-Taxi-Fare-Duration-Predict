from pydantic import BaseModel
from datetime import date, time

class InputModel(BaseModel):
    PULocationID: list
    DOLocationID: list
    pickup_hour: str
    trip_distance: str
    
        
class PredictRequest(BaseModel):
    file: str


class PredictResponse(BaseModel):
    success: bool
    fare: float
    duration: float
