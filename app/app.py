import os
import numpy as np
import tensorflow as tf
from flask import Flask, render_template, request
from tensorflow.keras.preprocessing import image

app = Flask(__name__)

# Load model once at startup
MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "best_model.h5")
model = tf.keras.models.load_model(MODEL_PATH)

IMAGE_SIZE = (224, 224)

UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), "static")

# Make sure static folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


def predict_image(img_path):
    img = image.load_img(img_path, target_size=IMAGE_SIZE)
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)[0][0]

    if prediction > 0.5:
        return "PNEUMONIA", float(prediction)
    else:
        return "NORMAL", float(1 - prediction)


@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    confidence = None
    filename = None

    if request.method == "POST":
        file = request.files["file"]
        filename = file.filename

        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)

        result, confidence = predict_image(filepath)

    return render_template(
        "index.html",
        result=result,
        confidence=confidence,
        filename=filename
    )


if __name__ == "__main__":
    app.run(debug=True)