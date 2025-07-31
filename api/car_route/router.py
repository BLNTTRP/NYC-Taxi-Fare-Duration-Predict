import openrouteservice as ors
from fastapi import APIRouter, Depends, HTTPException, status, Request
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session
import os
import pandas as pd
from car_route import settings

router = APIRouter(tags=["OpenRoute"], prefix="/route_service")

# Ensure the API key is set in the environment variable
os.environ['ORS_API_KEY'] = settings.API_KEY
client = ors.Client(key = os.environ['ORS_API_KEY'])

@router.post("/autocomplete")
async def get_autocomplete(request: Request):
    """ 
        Autocomplete endpoint for location search.
        
        Args:
            request with the text to autocomplete
            
        Returns a list of suggestions
    """

    data = await request.json()
    text = data.get("text", "") # type: ignore
    if not text:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Text parameter is required")
    # Request autocomplete suggestions
    response = client.pelias_autocomplete(text)
    suggestions = response.get('features', [])
    if not suggestions:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="No suggestions found")
    # Extract relevant information from suggestions
    results = []
    for suggestion in suggestions:
        properties = suggestion.get('properties', {})
        result = {
            'name': properties.get('label', ''),
            'latitude': suggestion['geometry']['coordinates'][1],
            'longitude': suggestion['geometry']['coordinates'][0],
            'country': properties.get('country', ''),
            'city': properties.get('locality', ''),
            'state': properties.get('region', '')
        }
        results.append(result)
    return results

@router.post("/coordinates")
async def get_coordinates(request: Request):
    """ 
        Get the coordinates of a location
        
        Args:
            request with the text to get the coordinates
            
        Returns the coordinates
    """
    data = await request.json()
    text = data.get("text", "") # type: ignore
    if not text:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Text parameter is required")       
    # Request coordinates
    response = client.pelias_search(text)
    features = response.get('features', [])
    if not features:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="No coordinates found")
    # Extract the first feature's coordinates
    coordinates = features[0]['geometry']['coordinates']       
    print("Received coordinates:", coordinates) 
    return {
        'latitude': coordinates[1],
        'longitude': coordinates[0]
    }

@router.post("/get_route")
async def get_route(request: Request):
    data = await request.json()
    coords = data.get("coordinates")
    if not coords or not isinstance(coords, list) or len(coords) != 2:
        raise HTTPException(status_code=400, detail="Invalid coordinates format")
    response = client.directions(coords, profile='driving-car', format='geojson')
    geometry = response['features'][0]['geometry']
    return geometry

    
@router.post("/get_distance_duration")
async def get_distance_duration(request: Request):
    """ 
        Give the the distance in km and duration in from a location to
        another
        
        Args:
            start:
            end:
        
    """
    data = await request.json()
    coords = data.get("coordinates")


    # Request route
    response = client.directions(coords, profile='driving-car', format='geojson', units='km')
    
    distance = response['features'][0]['properties']['segments'][0]['distance']
    duration = response['features'][0]['properties']['segments'][0]['duration']
    

    return {"distance": distance, "duration": duration}
    
    