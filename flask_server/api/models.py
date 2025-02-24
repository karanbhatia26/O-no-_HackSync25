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
from fastapi.middleware.cors import CORSMiddleware
from twilio.rest import Client
from typing import List, Dict

load_dotenv()
required_env_vars = [
    "TWILIO_ACCOUNT_SID",
    "TWILIO_AUTH_TOKEN",
    "TWILIO_PHONE_NUMBER",
    "GROQ_API_KEY" 
]

missing_vars = [var for var in required_env_vars if not os.getenv(var)]
if missing_vars:
    raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}")
# Add ml folder to system path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../ml")))

from model import FloodPredictor

# Initialize FastAPI app
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # React Frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



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


Instructions:
If the user's message is a greeting or general question (like "hello", "hi", "how are you"):
  - Respond in a friendly, conversational tone
  - Mention that you're a weather and flood safety assistant
  - Offer to help with weather and flood-related information

For weather or flood-related questions:
  - Use bullet points
  - Include **Bold text** for important warnings
  - Reference specific numbers from the weather data
  - Give actionable recommendations

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
TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")
TWILIO_PHONE_NUMBER = os.getenv("TWILIO_PHONE_NUMBER")

# Initialize Twilio client
twilio_client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)

def send_flood_alert(phone_numbers: List[str], city: str, probability: float) -> Dict:
    """
    Send SMS alerts to provided phone numbers when flood risk is high
    """
    try:
        message_body = (
            f"⚠️ FLOOD ALERT for {city}\n"
            f"Current flood probability: {probability:.1%}\n"
            "Please take necessary precautions and stay safe."
        )
        
        sent_messages = []
        for phone_number in phone_numbers:
            message = twilio_client.messages.create(
                body=message_body,
                from_=TWILIO_PHONE_NUMBER,
                to=phone_number
            )
            sent_messages.append(message.sid)
            
        return {"status": "success", "message_ids": sent_messages}
    
    except Exception as e:
        return {"status": "error", "message": str(e)}

# Modify your existing predict_flood endpoint
@app.get("/sms")
def predict_flood(
    city: str = Query(..., description="Enter city name"),
    alert_numbers: List[str] = Query(None, description="Phone numbers to alert")
):
    """Fetch weather data, process it, and predict flood probability."""
    weather_data = get_weather_data(city)

    if "error" in weather_data:
        return weather_data

    weather_info, processed_data = process_weather_data(weather_data)

    if "error" in weather_info:
        return weather_info

    flood_probability = model.rule_based_flood_probability(processed_data)
    
    # Send alerts if probability is high and phone numbers are provided
    alert_response = None
    if alert_numbers:  # Remove probability threshold for testing
        message = "🟢 NO FLOOD RISK" if flood_probability.item() <= 0.7 else "🔴 HIGH FLOOD RISK"
        alert_response = send_flood_alert(
            alert_numbers, 
            weather_info["city"], 
            flood_probability.item()
        )

    return {
        "city": weather_info["city"],
        "weather": weather_info,
        "flood_probability": flood_probability.item(),
        "alert_status": alert_response
    }
@app.get("/chat")
def chat_with_model(
    city: str = Query(..., description="Enter city name"),
    user_message: str = Query(..., description="Enter your question")
):
    """Chat interface for custom weather and flood-related queries."""
    weather_data = get_weather_data(city)
    
    if "error" in weather_data:
        return weather_data

    weather_info, processed_data = process_weather_data(weather_data)

    if "error" in weather_info:
        return weather_info

    flood_probability = model.rule_based_flood_probability(processed_data)
    flood_occurred = flood_probability.item() > 0.5

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
        "chat_message": user_message
    }
    chain = prompt | llm
    response = chain.invoke(prompt_inputs)
    cleaned_response = clean_and_parse_json(response.content)

    return {
        "city": weather_info["city"],
        "flood_probability": flood_probability.item(),
        "chat_response": cleaned_response,
        "user_query": user_message
    }
from fastapi.responses import HTMLResponse
import os

@app.get("/api/evacuation-route", response_class=HTMLResponse)
async def get_evacuation_route():
    try:
        file_path = "../ml/evacuation_route.html"
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
        return HTMLResponse(content=content)
    except Exception as e:
        return HTMLResponse(content=f"Error loading map: {str(e)}", status_code=500)