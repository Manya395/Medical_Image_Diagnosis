from flask import Flask, render_template, request, redirect, url_for, session, flash
from pymongo import MongoClient
from flask_bcrypt import Bcrypt
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import os
from datetime import timedelta

# ---------------- APP CONFIG ----------------
app = Flask(__name__)
app.secret_key = "super_secure_manya_key"
app.permanent_session_lifetime = timedelta(minutes=30)

bcrypt = Bcrypt(app)

# ---------------- MONGODB ----------------
client = MongoClient(
    "mongodb+srv://Medical_image_diag:Manya12345@cluster0.arsp9te.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
)
db = client["medical_ai_db"]
users_collection = db["users"]

# ---------------- LOAD MODEL ----------------
MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "best_model.h5")
try:
    model = tf.keras.models.load_model(MODEL_PATH)
except:
    model = None

IMAGE_SIZE = (224, 224)
UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), "static")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def predict_image(img_path):
    if not model: return "Model Error", 0.0
    img = image.load_img(img_path, target_size=IMAGE_SIZE)
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    prediction = model.predict(img_array)[0][0]
    if prediction > 0.5:
        return "PNEUMONIA", float(prediction)
    else:
        return "NORMAL", float(1 - prediction)

# ================= ROUTING =================

@app.route("/", methods=["GET", "POST"])
def index():
    if "user_email" not in session:
        return redirect(url_for("login"))

    result = None
    confidence = None
    uploaded_image = None

    if request.method == "POST":
        if "file" in request.files:
            file = request.files["file"]
            if file.filename != "":
                filename = file.filename
                filepath = os.path.join(UPLOAD_FOLDER, filename)
                file.save(filepath)
                
                result, confidence = predict_image(filepath)

                users_collection.update_one(
                    {"email": session["user_email"]},
                    {"$push": {"scan_history": {
                        "filename": filename,
                        "result": result,
                        "confidence": round(confidence * 100, 2)
                    }}}
                )
                uploaded_image = filename

    return render_template("index.html", user=session["user_email"], result=result, confidence=confidence, uploaded_image=uploaded_image)

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
        users_collection.insert_one({
            "name": name,
            "email": email,
            "password": hashed_password,
            "scan_history": [],
            "chat_history": []
        })
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
    return render_template("profile.html", user=user)

@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))

if __name__ == "__main__":
    app.run(debug=True)