import os
import io
import base64

import numpy as np
from PIL import Image
from flask import Flask, render_template, request, jsonify

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.efficientnet import preprocess_input

from dotenv import load_dotenv
from google import genai

# ----------------------------------------------------
# Load Environment Variables
# ----------------------------------------------------

load_dotenv()

app = Flask(__name__)

UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ----------------------------------------------------
# Gemini Client
# ----------------------------------------------------

client = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY")
)

# ----------------------------------------------------
# Load Model
# ----------------------------------------------------

MODEL_PATH = "model/efficientnetb0_disaster_classifier.keras"

model = load_model(MODEL_PATH)

IMG_SIZE = (224, 224)

CLASS_NAMES = [

    "earthquake",

    "fire",

    "flood",

    "normal"

]

last_prediction = {

    "label": None,

    "confidence": None

}

# ----------------------------------------------------
# Image Preprocessing
# ----------------------------------------------------

def preprocess_image(image):

    image = image.convert("RGB")

    image = image.resize(IMG_SIZE)

    image = img_to_array(image)

    image = np.expand_dims(image, axis=0)

    image = preprocess_input(image)

    return image


def decode_base64_image(data):

    if "," in data:

        data = data.split(",")[1]

    return Image.open(

        io.BytesIO(

            base64.b64decode(data)

        )

    )

# ----------------------------------------------------
# HOME
# ----------------------------------------------------

@app.route("/")
def home():

    return render_template("index.html")

# ----------------------------------------------------
# PREDICT
# ----------------------------------------------------

@app.route("/predict", methods=["POST"])
def predict():

    global last_prediction

    try:

        if "file" in request.files:

            file = request.files["file"]

            if file.filename == "":

                return jsonify({

                    "error":"No image selected."

                }),400

            image = Image.open(file.stream)

        else:

            data = request.get_json()

            if not data or "image" not in data:

                return jsonify({

                    "error":"No camera image received."

                }),400

            image = decode_base64_image(

                data["image"]

            )

        processed = preprocess_image(image)

        prediction = model.predict(

            processed,

            verbose=0

        )[0]

        predicted_index = np.argmax(prediction)

        predicted_label = CLASS_NAMES[predicted_index]

        confidence = float(prediction[predicted_index])

        last_prediction["label"] = predicted_label

        last_prediction["confidence"] = confidence

        probabilities = {

            CLASS_NAMES[i]:

            float(prediction[i])

            for i in range(len(CLASS_NAMES))

        }

        return jsonify({

            "label": predicted_label,

            "confidence": round(

                confidence*100,

                4

            ),

            "all_probs": probabilities

        })

    except Exception as e:

        return jsonify({

            "error":str(e)

        }),500

# ----------------------------------------------------
# GEMINI CHATBOT
# ----------------------------------------------------

@app.route("/chat", methods=["POST"])
def chat():

    global last_prediction

    try:

        data=request.get_json()

        if not data:

            return jsonify({

                "error":"No message."

            }),400

        question=data.get(

            "message",""

        ).strip()

        if question=="":

            return jsonify({

                "error":"Empty message."

            }),400

        if last_prediction["label"]:

            context=f"""

The uploaded image was classified.

Prediction:

{last_prediction['label']}

Confidence:

{last_prediction['confidence']*100:.5f}%

"""

        else:

            context="No prediction available."

        prompt=f"""

You are an AI Disaster Assistant.

Answer only disaster related questions.

Explain predictions simply.

Prediction Context:

{context}

User Question:

{question}

"""

        response = client.models.generate_content(

            model="gemini-3-flash-preview",

            contents=prompt

        )

        return jsonify({

            "reply": response.text

        })

    except Exception as e:

        return jsonify({

            "error": str(e)

        }),500

# ----------------------------------------------------
# MAIN
# ----------------------------------------------------

if __name__=="__main__":

    app.run(

        debug=True,

        host="0.0.0.0",

        port=5000

    )