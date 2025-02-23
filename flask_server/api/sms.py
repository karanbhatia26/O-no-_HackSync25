import requests

def test_flood_alert():
    # Test with a high-risk scenario
    url = "http://localhost:8000/predict"
    
    # Replace with your actual phone number in E.164 format
    test_phone = "+911234567890"  # Example format
    
    params = {
        "city": "Mumbai",  # Use a city name that might trigger high flood probability
        "alert_numbers": [test_phone]
    }
    
    response = requests.get(url, params=params)
    print("Response:", response.json())

if __name__ == "__main__":
    test_flood_alert()