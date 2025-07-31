import os

API_HOST = os.getenv("API_HOST", "localhost")
API_PORT = os.getenv("API_PORT", 5000)
API_BASE_URL = f"http://{API_HOST}:{API_PORT}"

CITYID = 5128581
APIKEY = 'e6f61255d454b47d690e5b3bd6a5981b'
ENDPOINT = f"https://api.openweathermap.org/data/2.5/weather?id={CITYID}&units=metric&appid={APIKEY}"
