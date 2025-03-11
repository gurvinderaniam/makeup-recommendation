from flask import Flask, render_template, request, jsonify, url_for
import joblib
import pandas as pd
import os

app = Flask(__name__)

# Load trained model
model = joblib.load("makeup_recommendation_model.pkl")

# Upload folder setup
UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Face shape encoding & makeup map
face_shape_map = {"Oval": 0, "Round": 1, "Heart": 2, "Square": 3}
makeup_map = {
    0: "Neutral & Warm Shades",
    1: "Bright Colors & Contouring",
    2: "Soft Pinks & Mauves",
    3: "Bold & Defined Colors"
}

@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        if "file" not in request.files:
            return jsonify({"error": "No file uploaded!"}), 400

        file = request.files["file"]
        if file.filename == "":
            return jsonify({"error": "No file selected!"}), 400

        # Ensure the filename is clean and remove double extensions
        filename = os.path.splitext(file.filename)[0] + os.path.splitext(file.filename)[1]
        file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(file_path)

        print(f"✅ Uploaded file saved at: {file_path}")

        # Mock skin tone & face shape
        skin_tone_r, skin_tone_g, skin_tone_b = 120, 85, 60
        face_shape = "Oval"

        # Encode face shape
        face_shape_encoded = face_shape_map.get(face_shape, -1)
        if face_shape_encoded == -1:
            return jsonify({"error": "Invalid face shape"}), 400

        # Prepare features for the model
        features = pd.DataFrame([[skin_tone_r, skin_tone_g, skin_tone_b, face_shape_encoded]],
                                columns=["Skin_Tone_R", "Skin_Tone_G", "Skin_Tone_B", "Face_Shape"])

        # Predict makeup recommendation
        prediction = model.predict(features)[0]
        recommendation = makeup_map.get(prediction, "No Recommendation Found")

        uploaded_file = os.path.join("uploads", filename)

        return render_template("index.html", recommendation=recommendation, uploaded_file=uploaded_file)

    except Exception as e:
        print("❌ Error:", str(e))
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
