from flask import Flask, render_template, request, redirect, url_for, session, flash, jsonify
from pymongo import MongoClient
from flask_bcrypt import Bcrypt
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import os
from datetime import timedelta
from dotenv import load_dotenv
from google import genai
from bson import ObjectId
import math
import requests
import threading

load_dotenv()

gemini_client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET", "fallback_secret_key")
app.permanent_session_lifetime = timedelta(minutes=30)
bcrypt = Bcrypt(app)

client = MongoClient(os.getenv("MONGO_URI"))
db = client["medical_ai_db"]
users_collection = db["users"]
doctors_collection = db["doctors"]
appointments_collection = db["appointments"]

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "best_model.h5")
try:
    cnn_model = tf.keras.models.load_model(MODEL_PATH)
except:
    cnn_model = None

IMAGE_SIZE = (224, 224)
UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), "static")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def predict_image(img_path):
    if not cnn_model:
        return "Model Error", 0.0
    img = image.load_img(img_path, target_size=IMAGE_SIZE)
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    prediction = cnn_model.predict(img_array)[0][0]
    if prediction > 0.5:
        return "PNEUMONIA", float(prediction)
    else:
        return "NORMAL", float(1 - prediction)

@app.route("/", methods=["GET", "POST"])
def index():
    if "user_email" not in session:
        return redirect(url_for("login"))
    user_data = users_collection.find_one({"email": session["user_email"]})
    user_name = user_data["name"] if user_data and "name" in user_data else session["user_email"]
    if request.method == "POST":
        if "file" in request.files:
            file = request.files["file"]
            if file.filename != "":
                filename = file.filename
                filepath = os.path.join(UPLOAD_FOLDER, filename)
                file.save(filepath)
                result, confidence = predict_image(filepath)
                confidence_pct = round(confidence * 100)
                users_collection.update_one(
                    {"email": session["user_email"]},
                    {"$push": {"scan_history": {"filename": filename, "result": result, "confidence": confidence_pct}}}
                )
                # Save result in session so it persists across page visits
                session["last_result"] = result
                session["last_confidence"] = confidence_pct
                session["last_image"] = filename
    # Load from session if no new upload
    result = session.get("last_result")
    confidence = session.get("last_confidence")
    uploaded_image = session.get("last_image")
    return render_template("index.html", user=user_name, result=result, confidence=confidence, uploaded_image=uploaded_image)

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form.get("email")
        password = request.form.get("password")
        user = users_collection.find_one({"email": email})
        if user and bcrypt.check_password_hash(user["password"], password):
            session.permanent = True
            session["user_email"] = email
            return redirect(url_for("index"))
        flash("Invalid email or password")
    return render_template("login.html")

@app.route("/signup", methods=["GET", "POST"])
def signup():
    if request.method == "POST":
        name = request.form.get("name")
        email = request.form.get("email")
        password = request.form.get("password")
        if users_collection.find_one({"email": email}):
            flash("User already exists!")
            return redirect(url_for("signup"))
        hashed_password = bcrypt.generate_password_hash(password).decode("utf-8")
        users_collection.insert_one({"name": name, "email": email, "password": hashed_password, "scan_history": [], "chat_history": []})
        flash("Signup successful! Please login.")
        return redirect(url_for("login"))
    return render_template("signup.html")

@app.route("/forgot_password", methods=["GET", "POST"])
def forgot_password():
    if request.method == "POST":
        email = request.form.get("email")
        new_password = request.form.get("new_password")
        user = users_collection.find_one({"email": email})
        if user:
            hashed = bcrypt.generate_password_hash(new_password).decode("utf-8")
            users_collection.update_one({"email": email}, {"$set": {"password": hashed}})
            flash("Password updated! Login now.")
            return redirect(url_for("login"))
        flash("Email not found.")
    return render_template("forgot_password.html")

@app.route("/profile")
def profile():
    if "user_email" not in session:
        return redirect(url_for("login"))
    user = users_collection.find_one({"email": session["user_email"]})
    appointments = list(appointments_collection.find({"user_email": session["user_email"]}).sort("date", -1))
    return render_template("profile.html", user=user, appointments=appointments)

@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))

@app.route("/chatbot")
def chatbot():
    return render_template("chatbot.html")

@app.route("/chat", methods=["POST"])
def chat():
    try:
        data = request.get_json()
        user_msg = data["message"].lower().strip()
        responses = {
            "what is pneumonia": "Pneumonia is a lung infection that inflames the air sacs in one or both lungs.",
            "what causes pneumonia": "Pneumonia can be caused by bacteria, viruses, or fungi.",
            "what are the symptoms of pneumonia": "Common symptoms: chest pain, cough with phlegm, fatigue, fever, shortness of breath.",
            "how is pneumonia treated": "Bacterial pneumonia is treated with antibiotics. Rest and fluids also help.",
            "hi": "Hello! I am your AI Medical Assistant. Ask me anything about pneumonia!",
            "hello": "Hi there! How can I help you today?",
        }
        for key, answer in responses.items():
            if key in user_msg:
                return {"reply": answer}
        if any(w in user_msg for w in ["symptom", "sign", "feel"]):
            return {"reply": responses["what are the symptoms of pneumonia"]}
        elif any(w in user_msg for w in ["treat", "cure", "medicine"]):
            return {"reply": responses["how is pneumonia treated"]}
        else:
            return {"reply": "I can answer questions about pneumonia. Try: 'What is pneumonia?' or 'What are the symptoms?'"}
    except Exception as e:
        return {"reply": str(e)}

# ── NEARBY DOCTORS ──────────────────────────

@app.route("/nearby_doctors")
def nearby_doctors():
    if "user_email" not in session:
        return redirect(url_for("login"))
    return render_template("nearby_doctors.html")

def haversine(lat1, lon1, lat2, lon2):
    """Returns distance in km between two coordinates."""
    R = 6371
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat/2)**2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon/2)**2
    return R * 2 * math.asin(math.sqrt(a))

@app.route("/api/doctors")
def api_doctors():
    city = request.args.get("city", "").strip().lower()
    user_lat = request.args.get("lat", type=float)
    user_lng = request.args.get("lng", type=float)

    if not city or city == "all":
        doctors = list(doctors_collection.find())
    else:
        doctors = list(doctors_collection.find({"city": {"$regex": city, "$options": "i"}}))

    result = []
    for d in doctors:
        dist = None
        if user_lat and user_lng and d.get("lat") and d.get("lng"):
            dist = round(haversine(user_lat, user_lng, d["lat"], d["lng"]), 1)
        result.append({
            "id": str(d["_id"]),
            "name": d.get("name", ""),
            "clinic": d.get("clinic", ""),
            "phone": d.get("phone", ""),
            "city": d.get("city", ""),
            "address": d.get("address", ""),
            "specialization": d.get("specialization", "Pulmonologist"),
            "experience": d.get("experience", ""),
            "rating": d.get("rating", 0),
            "total_ratings": d.get("total_ratings", 0),
            "available": d.get("available", True),
            "distance": dist,
        })

    # Sort: if geo available sort by distance, else by rating
    if user_lat and user_lng:
        result.sort(key=lambda x: (x["distance"] is None, x["distance"] or 9999))
    else:
        result.sort(key=lambda x: -x["rating"])

    return jsonify(result)

@app.route("/doctor_profile_view/<doctor_id>")
def doctor_profile_view(doctor_id):
    if "user_email" not in session:
        return redirect(url_for("login"))
    try:
        doctor = doctors_collection.find_one({"_id": ObjectId(doctor_id)})
    except:
        flash("Doctor not found.")
        return redirect(url_for("nearby_doctors"))
    if not doctor:
        flash("Doctor not found.")
        return redirect(url_for("nearby_doctors"))
    doctor["id"] = str(doctor["_id"])
    user_data = users_collection.find_one({"email": session["user_email"]})
    # doctor_id can be stored as string in different formats — check both
    completed = appointments_collection.find_one({
        "user_email": session["user_email"],
        "doctor_id": {"$in": [doctor_id, str(doctor["_id"])]},
        "status": "completed"
    })
    already_rated = appointments_collection.find_one({
        "user_email": session["user_email"],
        "doctor_id": {"$in": [doctor_id, str(doctor["_id"])]},
        "rated": True
    })
    return render_template("doctor_profile.html", doctor=doctor, user=user_data,
                           can_rate=bool(completed and not already_rated),
                           already_rated=bool(already_rated))

@app.route("/book_appointment", methods=["POST"])
def book_appointment():
    if "user_email" not in session:
        return jsonify({"success": False, "message": "Not logged in"}), 401
    data = request.get_json()
    doctor_id = data.get("doctor_id")
    date = data.get("date")
    time_slot = data.get("time_slot")
    notes = data.get("notes", "")
    user = users_collection.find_one({"email": session["user_email"]})
    doctor = doctors_collection.find_one({"_id": ObjectId(doctor_id)})
    if not doctor:
        return jsonify({"success": False, "message": "Doctor not found"}), 404
    appointments_collection.insert_one({
        "user_email": session["user_email"],
        "user_name": user.get("name", ""),
        "doctor_id": doctor_id,
        "doctor_name": doctor.get("name", ""),
        "clinic": doctor.get("clinic", ""),
        "city": doctor.get("city", ""),
        "date": date,
        "time_slot": time_slot,
        "notes": notes,
        "status": "pending"
    })
    return jsonify({"success": True, "message": "Appointment booked successfully!"})

@app.route("/rate_doctor", methods=["POST"])
def rate_doctor():
    if "user_email" not in session:
        return jsonify({"success": False}), 401
    data = request.get_json()
    doctor_id = data.get("doctor_id")
    rating = float(data.get("rating", 0))
    if rating < 1 or rating > 5:
        return jsonify({"success": False, "message": "Invalid rating"}), 400
    try:
        doctor = doctors_collection.find_one({"_id": ObjectId(doctor_id)})
    except:
        return jsonify({"success": False, "message": "Invalid doctor ID"}), 400
    if not doctor:
        return jsonify({"success": False}), 404
    # Verify user actually has a completed appointment
    completed = appointments_collection.find_one({
        "user_email": session["user_email"],
        "doctor_id": {"$in": [doctor_id, str(doctor["_id"])]},
        "status": "completed"
    })
    if not completed:
        return jsonify({"success": False, "message": "You can only rate after a completed appointment"}), 403
    already_rated = appointments_collection.find_one({
        "user_email": session["user_email"],
        "doctor_id": {"$in": [doctor_id, str(doctor["_id"])]},
        "rated": True
    })
    if already_rated:
        return jsonify({"success": False, "message": "You have already rated this doctor"}), 403
    old_total = doctor.get("total_ratings", 0)
    old_rating = doctor.get("rating", 0)
    new_total = old_total + 1
    new_rating = round(((old_rating * old_total) + rating) / new_total, 1)
    doctors_collection.update_one(
        {"_id": ObjectId(doctor_id)},
        {"$set": {"rating": new_rating, "total_ratings": new_total}}
    )
    # Mark the appointment as rated using the actual _id we found
    appointments_collection.update_one(
        {"_id": completed["_id"]},
        {"$set": {"rated": True}}
    )
    return jsonify({"success": True, "new_rating": new_rating})

# ── DOCTOR AUTH ──────────────────────────────

@app.route("/doctor_login", methods=["GET", "POST"])
def doctor_login():
    if request.method == "POST":
        email = request.form.get("email")
        password = request.form.get("password")
        doctor = doctors_collection.find_one({"email": email})
        if doctor and bcrypt.check_password_hash(doctor["password"], password):
            session["doctor_email"] = email
            session["doctor_id"] = str(doctor["_id"])
            return redirect(url_for("doctor_dashboard"))
        flash("Invalid email or password")
    return render_template("doctor_login.html")

@app.route("/doctor_signup", methods=["GET", "POST"])
def doctor_signup():
    if request.method == "POST":
        name = request.form.get("name")
        email = request.form.get("email")
        password = request.form.get("password")
        clinic = request.form.get("clinic")
        phone = request.form.get("phone")
        city = request.form.get("city", "").strip().lower()
        address = request.form.get("address")
        specialization = request.form.get("specialization", "Pulmonologist")
        experience = request.form.get("experience", "")
        if doctors_collection.find_one({"email": email}):
            flash("Doctor already registered with this email!")
            return redirect(url_for("doctor_signup"))
        hashed = bcrypt.generate_password_hash(password).decode("utf-8")
        lat = request.form.get("lat", type=float)
        lng = request.form.get("lng", type=float)
        doctors_collection.insert_one({
            "name": name, "email": email, "password": hashed,
            "clinic": clinic, "phone": phone, "city": city,
            "address": address, "specialization": specialization,
            "experience": experience, "rating": 0, "total_ratings": 0,
            "available": True, "lat": lat, "lng": lng
        })
        flash("Registration successful! Please login.")
        return redirect(url_for("doctor_login"))
    return render_template("doctor_signup.html")

@app.route("/doctor_logout")
def doctor_logout():
    session.pop("doctor_email", None)
    session.pop("doctor_id", None)
    return redirect(url_for("doctor_login"))

# ── DOCTOR DASHBOARD ─────────────────────────

@app.route("/doctor_dashboard")
def doctor_dashboard():
    if "doctor_email" not in session:
        return redirect(url_for("doctor_login"))
    doctor = doctors_collection.find_one({"email": session["doctor_email"]})
    doctor["id"] = str(doctor["_id"])
    appointments = list(appointments_collection.find({"doctor_id": str(doctor["_id"])}).sort("date", 1))
    return render_template("doctor_dashboard.html", doctor=doctor, appointments=appointments)

@app.route("/update_appointment_status", methods=["POST"])
def update_appointment_status():
    if "doctor_email" not in session:
        return jsonify({"success": False}), 401
    data = request.get_json()
    appointments_collection.update_one(
        {"_id": ObjectId(data.get("appointment_id"))},
        {"$set": {"status": data.get("status")}}
    )
    return jsonify({"success": True})

@app.route("/toggle_availability", methods=["POST"])
def toggle_availability():
    if "doctor_email" not in session:
        return jsonify({"success": False}), 401
    doctor = doctors_collection.find_one({"email": session["doctor_email"]})
    new_status = not doctor.get("available", True)
    doctors_collection.update_one({"email": session["doctor_email"]}, {"$set": {"available": new_status}})
    return jsonify({"success": True, "available": new_status})


# ── ADMIN: GEOCODE EXISTING DOCTORS ──────────
@app.route("/admin/geocode_doctors")
def admin_geocode_doctors():
    """
    Visit /admin/geocode_doctors once to add lat/lng to all existing doctors.
    Runs in background — check terminal for progress.
    """
    def run_geocode():
        headers = {"User-Agent": "EasyScanAI/1.0"}
        all_docs = list(doctors_collection.find(
            {"$or": [{"lat": {"$exists": False}}, {"lat": None}]}
        ))
        print(f"[Geocode] Found {len(all_docs)} doctors to geocode.")
        for doc in all_docs:
            address = doc.get("address", "")
            city = doc.get("city", "")
            query = f"{address}, {city}, India"
            try:
                r = requests.get(
                    "https://nominatim.openstreetmap.org/search",
                    params={"q": query, "format": "json", "limit": 1, "countrycodes": "in"},
                    headers=headers, timeout=8
                )
                data = r.json()
                if data:
                    lat, lng = float(data[0]["lat"]), float(data[0]["lon"])
                    doctors_collection.update_one(
                        {"_id": doc["_id"]},
                        {"$set": {"lat": lat, "lng": lng}}
                    )
                    print(f"[Geocode] Dr. {doc.get('name')} → {lat:.4f}, {lng:.4f}")
                else:
                    print(f"[Geocode] Dr. {doc.get('name')} — NOT FOUND for: {query}")
            except Exception as e:
                print(f"[Geocode] Error for {doc.get('name')}: {e}")
            import time; time.sleep(1.2)
        print("[Geocode] Done!")

    t = threading.Thread(target=run_geocode, daemon=True)
    t.start()
    return "<h2>✅ Geocoding started in background.</h2><p>Check your terminal for progress. Refresh <a href='/nearby_doctors'>Nearby Doctors</a> in ~30 seconds.</p>"

if __name__ == "__main__":
    app.run(debug=True)
