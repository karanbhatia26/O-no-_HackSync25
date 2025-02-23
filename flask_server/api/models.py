
import torch
import requests
from fastapi import FastAPI, Query
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
import sys
import os
import json
import re
from dotenv import load_dotenv

load_dotenv(dotenv_path="../ml/.env")

# Add ml folder to system path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../ml")))

from model import FloodPredictor

# Initialize FastAPI app
app = FastAPI()

# OpenWeather API key
API_KEY = "b4e7b0f7f2d3290d396f7fc6d256d23e"  # Replace with your actual API key

# Load Model
model = FloodPredictor(use_gpu=True)
model.thresholds = torch.load("../ml/thresholds.pth", map_location=model.device)
model.Q_tables = torch.load("../ml/Q_tables.pth", map_location=model.device)


llm = ChatGroq(
    temperature=0,
    groq_api_key=os.getenv("GROQ_API_KEY"),
    model_name="llama-3.1-8b-instant"
)

# Define prompt template for suggestions
prompt = PromptTemplate.from_template("""
User Data:
{user_data}

Weather Analysis:
  Temperature:
    • Maximum: {Max_Temp}°C
    • Minimum: {Min_Temp}°C
    • Range: from {Min_Temp}°C to {Max_Temp}°C

  Precipitation:
    • Rainfall: {Rainfall}mm
    • Cloud Cover: {Cloud_Coverage}/10

  Atmospheric Conditions:
    • Humidity: {Relative_Humidity}%
    • Wind Speed: {Wind_Speed}m/s
    • Month: {Month}

Risk Assessment:
  • Flood Status: {flood_occurred}
  • Flood Probability: {probability}

User Query:
{chat_message}

In all cases, format your answer using:
  - Clear bullet points.
  - **Bold text** for any important warnings.
  - Specific numbers and context from the data provided.
  - Short, actionable, and direct recommendations.

Response (NO PREAMBLE):
""")


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

def clean_and_parse_json(response_str):
    """
    Cleans and formats the plain text response into a structured JSON.
    Divides the response into keys like 'adv1', 'adv2', etc.
    """
    # Debugging: Print raw response for inspection
    print(f"Raw response: {response_str}")

    # Remove markdown-like bold text
    response_str = re.sub(r"\*\*([^*]+)\*\*", r"\1", response_str)

    # Strip leading/trailing whitespaces and split the response into bullet points
    response_str = response_str.strip()
    
    # Assuming each bullet point starts with "•", split the string by the bullet point
    adv_list = response_str.split('\n• ')
    
    # Remove the first bullet point from the first entry, as it doesn't need it
    adv_list[0] = adv_list[0].replace('• ', '')

    # Construct the JSON object with structured advice
    adv_dict = {f"adv{i+1}": adv_list[i] for i in range(len(adv_list))}
    
    # Wrap the result into a structured JSON
    json_response = {
        "chat_response": adv_dict
    }
    
    return json_response


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

@app.get("/suggestions")
def suggestions(city: str = Query(..., description="Enter city name")):
    """Give suggestions based on weather data for a given city."""
    weather_data = get_weather_data(city)
    
    if "error" in weather_data:
        return weather_data

    weather_info, processed_data = process_weather_data(weather_data)

    if "error" in weather_info:
        return weather_info

    flood_probability = model.rule_based_flood_probability(processed_data)
    flood_occurred = flood_probability.item() > 0.5

    # Prepare the prompt input for the language model
    prompt_inputs = {
        "user_data": json.dumps({"city": weather_info["city"]}),
        "Max_Temp": weather_info["temperature_max"],
        "Min_Temp": weather_info["temperature_min"],
        "Rainfall": weather_info["rainfall"],
        "Relative_Humidity": weather_info["humidity"],
        "Wind_Speed": weather_info["wind_speed"],
        "Cloud_Coverage": weather_info["cloud_coverage"],
        "Month": 2,  # Replace with dynamic month if needed
        "flood_occurred": "Yes" if flood_occurred else "No",
        "probability": f"{flood_probability.item():.2f}",
        "chat_message": 'What Precautions should I take today?'
    }

    # Generate the response from the language model
    chain = prompt | llm
    response = chain.invoke(prompt_inputs)
    cleaned_response = clean_and_parse_json(response.content)

    # Return only necessary information
    return {
        "city": weather_info["city"],
        "flood_probability": flood_probability.item(),
        "chat_response": cleaned_response
    }