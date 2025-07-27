from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import numpy as np
import cv2
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)
CORS(app)

# Load the Trained Model
MODEL_PATH = "skin_disease_model.h5"
model = tf.keras.models.load_model(MODEL_PATH)

# Define Allowed Extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
UPLOAD_FOLDER = "uploads"
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Define Class Labels (Updated to Match Dataset)
CLASS_LABELS = [
    "Acne", "Actinic Keratoses", "Basal Cell Carcinoma", "Benign Keratosis",
    "Dermatofibroma", "Impetigo", "Melanoma", "Psoriasis", "Rash", "Vascular Skin Lesion"
]

# Disease Details (Example)
DISEASE_DETAILS = {
    "Acne": {
        "description": "A common skin condition that causes pimples, blackheads, and whiteheads due to clogged pores.",
        "medications": [
            {"name": "Benzoyl Peroxide", "description": "Kills bacteria and dries excess oil.", "dosage": "Apply once daily."},
            {"name": "Clindamycin", "description": "Topical antibiotic to reduce inflammation.", "dosage": "Apply twice daily."}
        ],
        "precautions": ["Wash face twice a day.", "Avoid touching your face.", "Use non-comedogenic skincare products."]
    },
    "Actinic Keratoses": {
        "description": "Precancerous skin growths caused by prolonged sun exposure, appearing as rough, scaly patches.",
        "medications": [
            {"name": "Fluorouracil Cream", "description": "Destroys precancerous cells.", "dosage": "Apply once daily for 2-4 weeks."}
        ],
        "precautions": ["Avoid sun exposure.", "Wear protective clothing.", "Apply SPF 50+ sunscreen daily."]
    },
    "Psoriasis": {
        "description": "A chronic autoimmune skin disease that causes red, scaly patches and inflammation.",
        "medications": [
            {"name": "Topical Corticosteroids", "description": "Reduces inflammation and scaling.", "dosage": "Apply to affected areas twice daily."},
            {"name": "Calcipotriene", "description": "Vitamin D analog to slow skin cell growth.", "dosage": "Apply once daily."}
        ],
        "precautions": ["Avoid scratching.", "Use moisturizer regularly.", "Limit exposure to triggers like stress and cold weather."]
    },
    "Melanoma": {
        "description": "A serious type of skin cancer that develops from pigment-producing cells (melanocytes).",
        "medications": [
            {"name": "Immunotherapy", "description": "Boosts the body's immune response against cancer cells.", "dosage": "As prescribed by an oncologist."},
            {"name": "Targeted Therapy", "description": "Blocks cancer cell growth.", "dosage": "Taken orally as per medical guidance."}
        ],
        "precautions": ["Avoid excessive sun exposure.", "Wear SPF 50+ sunscreen.", "Get regular skin check-ups."]
    },
    "Benign Keratosis": {
        "description": "Non-cancerous skin growths that appear as rough, warty patches on the skin.",
        "medications": [
            {
                "name": "Salicylic Acid (Topical)",
                "dosage": "Apply a thin layer to the affected area once daily.",
                "description": "Helps exfoliate and remove keratotic skin."
            },
            {
                "name": "Urea Cream (Topical)",
                "dosage": "Apply twice daily to affected areas.",
                "description": "Softens rough, thickened skin."
            },
            {
                "name": "Tretinoin (Topical Retinoid)",
                "dosage": "Apply at night before sleep.",
                "description": "Encourages cell turnover and reduces thickened skin."
            },
            {
                "name": "Hydrocortisone Cream (Mild Steroid)",
                "dosage": "Apply 1-2 times daily if irritation occurs.",
                "description": "Reduces inflammation and discomfort."
            }
        ],
        "precautions": [
            "Monitor for changes in size or color.",
            "No treatment required unless symptomatic.",
            "Avoid unnecessary irritation."
        ]
    },
    "Dermatofibroma": {
        "description": "A benign, firm, and usually painless skin growth caused by excess collagen production.",
        "medications": [
            {
                "name": "Hydrocortisone Cream",
                "dosage": "Apply a thin layer to the affected area 1-2 times daily if irritation or itching occurs.",
                "description": "Helps reduce inflammation and mild discomfort."
            },
            {
                "name": "Salicylic Acid (Topical)",
                "dosage": "Apply once daily as needed.",
                "description": "May help soften the lesion over time, but not a guaranteed treatment."
            },
            {
                "name": "Imiquimod Cream",
                "dosage": "Apply 3 times per week for several weeks if advised by a dermatologist.",
                "description": "Can stimulate the immune system to reduce the size of the lesion."
            },
            {
                "name": "Ibuprofen or Acetaminophen",
                "dosage": "Take as needed for pain relief (consult a doctor for proper dosage).",
                "description": "Used for managing pain if the lesion becomes tender."
            }
        ],
        "precautions": [
            "No specific treatment needed.",
            "Avoid unnecessary irritation.",
            "Consult a dermatologist if painful."
        ]
    },
    "Rash": {
        "description": "A general term for inflamed, red, or itchy skin caused by irritation, allergies, or infections.",
        "medications": [
            {"name": "Hydrocortisone Cream", "description": "Reduces itching and inflammation.", "dosage": "Apply twice daily."},
            {"name": "Antihistamines", "description": "Relieves allergic reactions.", "dosage": "Take once daily if needed."}
        ],
        "precautions": ["Avoid scratching.", "Use mild, fragrance-free soaps.", "Identify and avoid irritants."]
    },
    "Impetigo": {
        "description": "A contagious bacterial skin infection causing red sores and blisters, often affecting children.",
        "medications": [
            {"name": "Mupirocin", "description": "Topical antibiotic to kill bacteria.", "dosage": "Apply three times daily."},
            {"name": "Oral Antibiotics", "description": "Prescribed for widespread infections.", "dosage": "Take as directed by a doctor."}
        ],
        "precautions": ["Wash hands frequently.", "Avoid touching affected areas.", "Keep skin clean and dry."]
    },
    "Basal Cell Carcinoma": {
        "description": "A common type of skin cancer that appears as a waxy bump, often due to sun exposure.",
        "medications": [
            {"name": "Fluorouracil Cream", "description": "Destroys abnormal skin cells.", "dosage": "Apply once daily for several weeks."},
            {"name": "Imiquimod", "description": "Boosts immune response to fight cancer cells.", "dosage": "Apply five times a week for six weeks."}
        ],
        "precautions": ["Avoid sun exposure.", "Wear protective clothing.", "Regular dermatology check-ups."]
    },
    "Vascular Skin Lesion": {
        "description": "Abnormal blood vessel formations on the skin, such as birthmarks or spider veins.",
        "medications": [
            {"name": "Laser Therapy", "description": "Helps remove visible blood vessels.", "dosage": "Performed by a dermatologist."}
        ],
        "precautions": ["Avoid excessive sun exposure.", "Use sunscreen to prevent worsening.", "Monitor for changes."]
    }
}

# Check File Extension
def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route("/diagnose", methods=["POST"])
def diagnose():
    if "file" not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(file_path)

        # Read Image & Preprocess
        img = cv2.imread(file_path)
        if img is None:
            return jsonify({"error": "Invalid image format"}), 400

        img = cv2.resize(img, (224, 224))
        img = img / 255.0  # Normalize
        img = np.expand_dims(img, axis=0)  # Add Batch Dimension

        # Make Prediction
        predictions = model.predict(img)
        predicted_index = np.argmax(predictions[0])
        confidence = float(predictions[0][predicted_index])

        # Confidence Threshold
        if confidence < 0.50:  # Avoid incorrect predictions
            return jsonify({"error": "Low confidence. Unable to determine disease."}), 400

        # Get Disease Details
        disease_name = CLASS_LABELS[predicted_index]
        medications = DISEASE_DETAILS.get(disease_name, {}).get("medications", [])
        precautions = DISEASE_DETAILS.get(disease_name, {}).get("precautions", [])

        response_data = {
            "predicted_disease": disease_name,
            "confidence": confidence,
            "description": DISEASE_DETAILS.get(disease_name, {}).get("description", "No description available."),
            "medications": DISEASE_DETAILS.get(disease_name, {}).get("medications", ["No specific medications available."]),
            "precautions": precautions
        }


        return jsonify(response_data)

    return jsonify({"error": "Invalid file format"}), 400

if __name__ == "__main__":
    app.run(debug=True)
