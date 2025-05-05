# crime_reporter.py
import os
import requests
from dotenv import load_dotenv


load_dotenv()


def get_emergency_level(score: float) -> str:
    """Maps confidence score to emergency level."""
    if score > 0.85:
        return "HIGH"
    elif score > 0.65:
        return "MEDIUM"
    else:
        return "LOW"
    
CRIME_TYPE_MAP = {
    "Stealing": "STEALING",
    "Robbery": "ROBBERY",
    "Arson": "ARSON",
    "Shooting": "SHOOTING",
    "Vandalism": "VANDALISM",
    "Assault": "ASSAULT",
    "Shoplifting": "SHOPLIFTING",
    "Abuse": "ABUSE",
    "Explosion": "EXPLOSION",
    "Arrest": "ARREST",
    "Fighting": "FIGHTING",
    "RoadAccidents": "ROAD_ACCIDENTS", 
    "Burglary": "BURGLARY",
}   

def send_crime_to_api(camera_name: str, crime_type: str, emergency_level: str, image_url: str, location_id: str,location_name):
    """Sends crime data to the backend API."""
    payload = {
        "crimeDescription": f"Detected {crime_type} from {camera_name} in {location_name}",
        "crimeType": crime_type.upper(),
        "crimeLocation": location_id,  
        "emergencyLevel": emergency_level,
        "supportingImage": image_url,
    }

    headers = {
        "model-api-key": os.getenv("MODEL_API_KEY"), 
        "Content-Type": "application/json",
    }

    try:
        response = requests.post("https://smart-surveillance-system.onrender.com/api/crimes", json=payload, headers=headers)
        print(f"📤 Crime sent - Status: {response.status_code}")
        print(response.json())
    except Exception as e:
        print(f"❌ Error sending crime: {e}")
