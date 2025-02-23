
import torch
import requests
from fastapi import FastAPI, Query
import sys
import os

# Add ml folder to system path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../ml")))

from model import FloodPredictor

# Initialize FastAPI app
app = FastAPI()

# OpenWeather API key
API_KEY = "b4e7b0f7f2d3290d396f7fc6d256d23e"  # Replace with your actual API key

# Load Model
model = FloodPredictor(use_gpu=True)
model.thresholds = torch.load("ml/thresholds.pth", map_location=model.device)
model.Q_tables = torch.load("ml/Q_tables.pth", map_location=model.device)

def get_weather_data(city: str):
    """Fetch weather data from OpenWeather API."""
    url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={API_KEY}&units=metric"
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()
    return {"error": f"Failed to fetch weather data: {response.status_code}"}

def process_weather_data(data):
    """Extract required weather parameters and convert to model format."""
    try:
        weather_info = {
            "city": data["name"],
            "temperature_max": data["main"]["temp_max"],
            "temperature_min": data["main"]["temp_min"],
            "humidity": data["main"]["humidity"],
            "wind_speed": data["wind"]["speed"],
            "cloud_coverage": data.get("clouds", {}).get("all", 0),
            "rainfall": data.get("rain", {}).get("1h", 0)
        }

        model_input = {
            "Max_Temp": torch.tensor([weather_info["temperature_max"]], dtype=torch.float32, device=model.device),
            "Min_Temp": torch.tensor([weather_info["temperature_min"]], dtype=torch.float32, device=model.device),
            "Rainfall": torch.tensor([weather_info["rainfall"]], dtype=torch.float32, device=model.device),
            "Relative_Humidity": torch.tensor([weather_info["humidity"]], dtype=torch.float32, device=model.device),
            "Wind_Speed": torch.tensor([weather_info["wind_speed"]], dtype=torch.float32, device=model.device),
            "Cloud_Coverage": torch.tensor([weather_info["cloud_coverage"]], dtype=torch.float32, device=model.device),
            "Month": torch.tensor([2], dtype=torch.float32, device=model.device)  # Replace with dynamic month if needed
        }

        return weather_info, model_input

    except KeyError as e:
        return {"error": f"Missing data: {str(e)}"}, None

@app.get("/")
def home():
    return {"message": "Welcome to the Flood Prediction API!"}

@app.get("/weather")
def fetch_weather(city: str = Query(..., description="Enter city name")):
    """Get current weather details for a given city."""
    weather_data = get_weather_data(city)
    if "error" in weather_data:
        return weather_data
    return weather_data

@app.get("/predict")
def predict_flood(city: str = Query(..., description="Enter city name")):
    """Fetch weather data, process it, and predict flood probability."""
    weather_data = get_weather_data(city)

    if "error" in weather_data:
        return weather_data

    weather_info, processed_data = process_weather_data(weather_data)

    if "error" in weather_info:
        return weather_info

    flood_probability = model.rule_based_flood_probability(processed_data)

    return {
        "city": weather_info["city"],
        "weather": weather_info,
        "flood_probability": flood_probability.item()
    }
