"""
Run this ONCE to add lat/lng to all doctors who don't have it yet.
Usage: python geocode_doctors.py
"""
import time
import requests
from pymongo import MongoClient
from dotenv import load_dotenv
import os

load_dotenv()
client = MongoClient(os.getenv("MONGO_URI"))
db = client["medical_ai_db"]
doctors_collection = db["doctors"]

def geocode(address, city):
    query = f"{address}, {city}, India"
    url = "https://nominatim.openstreetmap.org/search"
    params = {"q": query, "format": "json", "limit": 1, "countrycodes": "in"}
    headers = {"User-Agent": "EasyScanAI/1.0"}
    try:
        r = requests.get(url, params=params, headers=headers, timeout=8)
        data = r.json()
        if data:
            return float(data[0]["lat"]), float(data[0]["lon"])
    except Exception as e:
        print(f"  Error: {e}")
    return None, None

doctors = list(doctors_collection.find({"$or": [{"lat": {"$exists": False}}, {"lat": None}]}))
print(f"Found {len(doctors)} doctors without coordinates.\n")

for doc in doctors:
    name = doc.get("name", "?")
    address = doc.get("address", "")
    city = doc.get("city", "")
    print(f"Geocoding Dr. {name} — {address}, {city}...")

    lat, lng = geocode(address, city)
    if lat and lng:
        doctors_collection.update_one(
            {"_id": doc["_id"]},
            {"$set": {"lat": lat, "lng": lng}}
        )
        print(f"  ✅ Saved: {lat}, {lng}")
    else:
        print(f"  ❌ Could not find coordinates. Try a more specific address.")

    time.sleep(1.2)  # Nominatim rate limit: 1 req/sec

print("\nDone! Restart Flask and test location sorting.")
