import os
import torch
import json
from fastapi import FastAPI, Request, HTTPException
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
load_dotenv()
class FloodPredictor:
    def __init__(self, device=None):
        self.device = device or torch.device("gpu")
        self.thresholds = {}
        self.Q_tables = {}

    def rule_based_flood_probability(self, features):
        # For demonstration: sum the feature values and normalize.
        score = sum(t.item() for t in features.values())
        prob = max(0.0, min(score / (len(features) * 100), 1.0))
        return torch.tensor([prob], device=self.device)

# Load separate thresholds and Q_tables files from floodprediction2.ipynb context
thresholds_path = 'thresholds.pth'
q_tables_path = 'Q_tables.pth'
flood_model = FloodPredictor()
flood_model.thresholds = torch.load(thresholds_path, map_location="gpu")
flood_model.Q_tables = torch.load(q_tables_path, map_location="gpu")

llm = ChatGroq(
    temperature=0,
    groq_api_key=os.getenv("GROQ_API_KEY"),
    model_name="llama-3.1-8b-instant"
)
prompt = PromptTemplate.from_template(
    """
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
    
    Instructions for Response:
    First, check the flood probability:
    
    If the flood probability is HIGH (greater than 0.5):
      - Provide a list of immediate safety measures (e.g., "Move to higher ground immediately", "Turn off utilities if safe", "Follow evacuation routes").
      - Include clear step-by-step evacuation guidelines and emergency contact numbers.
      - Mention any specific weather data that supports these urgent measures.
    
    If the flood probability is LOW (0.5 or below):
      - Provide general weather safety advice.
      - For Rainfall: If Rainfall > 5mm, advise: "Carry an umbrella or wear rainproof clothing (Rainfall: {Rainfall}mm)."
      - For Temperature: 
          * If Maximum Temperature > 30°C, advise: "Wear light, breathable clothing and stay cool (Max Temperature: {Max_Temp}°C)."
          * If Minimum Temperature < 20°C, advise: "Consider carrying a light jacket (Min Temperature: {Min_Temp}°C)."
      - For Wind Speed: If Wind Speed > 10m/s, advise: "Avoid prolonged outdoor activities, strong winds may affect travel (Wind Speed: {Wind_Speed}m/s)."
      - For Humidity: If Humidity > 70%, advise: "Stay hydrated and consider carrying water (Humidity: {Relative_Humidity}%)."
      - Add any time-sensitive suggestions if applicable (e.g., "Since it's afternoon/evening, plan accordingly").
    
    In all cases, format your answer using:
      - Clear bullet points.
      - **Bold text** for any important warnings.
      - Specific numbers and context from the data provided.
      - Short, actionable, and direct recommendations.
    
    Response (NO PREAMBLE):
    """
)
app = FastAPI()

@app.post("/chat")
async def chat(request: Request):
    try:
        payload = await request.json()
        required_features = ["Max_Temp", "Min_Temp", "Rainfall",
                             "Relative_Humidity", "Wind_Speed",
                             "Cloud_Coverage", "Month"]
        if "flood_data" not in payload:
            raise HTTPException(status_code=400, detail="Missing field: flood_data")
        flood_data = payload["flood_data"]
        input_features = {}
        for feature in required_features:
            if feature not in flood_data:
                raise HTTPException(status_code=400, detail=f"Missing flood feature: {feature}")
            input_features[feature] = torch.tensor([float(flood_data[feature])], device=flood_model.device)
        
        prediction = flood_model.rule_based_flood_probability(input_features)
        flood_occurred = prediction.item() > 0.5
        
        user_data = payload.get("user_data", "No user data provided.")
        chat_message = payload.get("chat_message", "No query provided.")
        
        prompt_inputs = {
            "user_data": json.dumps(user_data),  # convert dict to string
            "Max_Temp": flood_data["Max_Temp"],
            "Min_Temp": flood_data["Min_Temp"],
            "Rainfall": flood_data["Rainfall"],
            "Relative_Humidity": flood_data["Relative_Humidity"],
            "Wind_Speed": flood_data["Wind_Speed"],
            "Cloud_Coverage": flood_data["Cloud_Coverage"],
            "Month": flood_data["Month"],
            "flood_occurred": "Yes" if flood_occurred else "No",
            "probability": f"{prediction.item():.2f}",
            "chat_message": chat_message
        }
        
        chain = prompt | llm
        response = chain.invoke(prompt_inputs)
        
        return {
            "chat_response": response.content,
            "flood_prediction": {
                "flood_occurred": flood_occurred,
                "probability": prediction.item()
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


