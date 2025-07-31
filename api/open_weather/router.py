from fastapi import APIRouter, HTTPException
import os
import requests
from open_weather import settings

router = APIRouter(tags=["OpenWeather"], prefix="/open_weather")

# Configura la API key de OpenWeather
API_KEY = settings.APIKEY
BASE_URL = settings.ENDPOINT

@router.post("/get_weather")
async def get_current_nyc_weather():
    """
    Consulta el estado del clima actual para la ciudad de New York.
    """

    try:
        response = requests.get(BASE_URL)
        response.raise_for_status()
        data = response.json()
        print(data)
        return {
            "city": data.get("name"),
            "weather": data.get("weather", [{}])[0].get("description"),
            "temperature": data.get("main", {}).get("temp"),
            "humidity": data.get("main", {}).get("humidity"),
            "wind_speed": data.get("wind", {}).get("speed"),
            "icon_url": f"https://openweathermap.org/img/wn/{data.get('weather', [{}])[0].get('icon')}@2x.png"
        }
    except requests.RequestException as e:
        raise HTTPException(status_code=500, detail=f"Error al consultar el clima: {str(e)}")