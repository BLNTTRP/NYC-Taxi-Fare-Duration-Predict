from app.model import router as model_router
from app.auth.router import router as auth_router
from app.user import router as user_router
from car_route import router as car_router
from open_weather import router as weather_router
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="NYC Taxi Fare and Trip Duration Prediction", version="1.0.0")

# Configura CORS para permitir cualquier origen
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Permite todos los orígenes
    allow_credentials=True,
    allow_methods=["*"],  # Permite todos los métodos (GET, POST, PUT, DELETE, etc.)
    allow_headers=["*"],  # Permite todos los encabezados
)

app.include_router(auth_router)
app.include_router(model_router.router)
app.include_router(user_router.router)
app.include_router(car_router.router)
app.include_router(weather_router.router)

